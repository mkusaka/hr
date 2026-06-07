use crate::ccr::{content_hash, marker_for_hash, CcrStore};
use crate::stats;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeSet;
use tracing::{debug, trace, warn};

/// Per-content byte threshold below which a live-zone string is not
/// compressed. Mirrors the reference per-content-type thresholds
/// (`crates/headroom-core/src/transforms/live_zone.rs` `THRESHOLD_*`,
/// all 512 bytes).
pub const LIVE_ZONE_MIN_BYTES: usize = 512;

/// Responses `*_output` items must clear this floor before the
/// content threshold even runs, mirroring
/// `crates/headroom-proxy/src/responses_items.rs`
/// (`OUTPUT_ITEM_MIN_BYTES`) and `live_zone.rs`
/// (`RESPONSES_OUTPUT_MIN_BYTES`).
const RESPONSES_OUTPUT_MIN_BYTES: usize = 512;

/// Block types inside an Anthropic live-zone message that stay in the
/// cache hot zone even when the message itself is in the live zone.
/// Mirrors `live_zone.rs` `HOT_ZONE_BLOCK_TYPES`.
const HOT_ZONE_BLOCK_TYPES: &[&str] = &["tool_use", "thinking", "redacted_thinking", "compaction"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiShape {
    OpenAiChatCompletions,
    OpenAiResponses,
    AnthropicMessages,
    AnthropicMessageBatches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestAuthMode {
    Payg,
    OAuth,
    Subscription,
}

#[derive(Clone, Copy)]
pub struct CompressOptions<'a> {
    pub store: &'a dyn CcrStore,
    pub min_bytes: usize,
}

/// Request-scoped compression context.
#[derive(Clone, Copy)]
pub struct CompressContext<'a> {
    pub store: &'a dyn CcrStore,
    pub auth_mode: RequestAuthMode,
    /// Provider-confirmed cached-prefix floor for this session.
    /// Message indices below this are never mutated. Mirrors the
    /// reference `PrefixCacheTracker` floor
    /// (`headroom/cache/prefix_tracker.py`).
    pub frozen_message_count: usize,
    /// Opt-in user/plain-text compression. The reference default
    /// protects user text and compresses tool output only
    /// (`headroom/proxy/models.py` `compress_user_messages = False`).
    pub compress_user_text: bool,
    /// Apply provider cache metadata (tool/schema sort, cache_control
    /// placement, `prompt_cache_key`). HTTP requests mirror the
    /// reference handlers and keep this on; Codex WS `response.create`
    /// frames mirror `_compress_openai_responses_payload`
    /// (`openai.py:1212`, `1749-1770`), which compresses live units
    /// and injects the CCR retrieve tool but never reorders tools or
    /// synthesizes cache keys.
    pub provider_metadata: bool,
}

impl<'a> CompressContext<'a> {
    pub fn new(store: &'a dyn CcrStore, auth_mode: RequestAuthMode) -> Self {
        Self {
            store,
            auth_mode,
            frozen_message_count: 0,
            compress_user_text: false,
            provider_metadata: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressResult {
    pub output: String,
    pub hash: Option<String>,
    pub marker: Option<String>,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub stored: bool,
    pub skipped_reason: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestCompression {
    pub body: Vec<u8>,
    pub compressed: bool,
    pub skipped_reason: Option<String>,
    pub hashes: Vec<String>,
    pub bytes_before: usize,
    pub bytes_after: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

pub fn compress(input: &str, options: CompressOptions<'_>) -> CompressResult {
    let original_bytes = input.len();
    let original_tokens = estimate_tokens(input);

    if input.trim().is_empty() {
        return skipped(input, "empty", original_bytes, original_tokens);
    }

    if original_bytes < options.min_bytes {
        return skipped(input, "below_min_bytes", original_bytes, original_tokens);
    }

    let hash = content_hash(input);
    let marker = marker_for_hash(&hash);

    match options.store.put(&hash, input) {
        Ok(inserted) => {
            if inserted {
                stats::record_ccr_entry_inserted();
            }
            CompressResult {
                compressed_bytes: marker.len(),
                compressed_tokens: estimate_tokens(&marker),
                output: marker.clone(),
                hash: Some(hash),
                marker: Some(marker),
                original_bytes,
                original_tokens,
                stored: true,
                skipped_reason: None,
                error: None,
            }
        }
        Err(err) => CompressResult {
            output: input.to_string(),
            hash: Some(hash),
            marker: None,
            original_bytes,
            compressed_bytes: original_bytes,
            original_tokens,
            compressed_tokens: original_tokens,
            stored: false,
            skipped_reason: Some("store_error".to_string()),
            error: Some(err.to_string()),
        },
    }
}

pub fn compress_json_request(
    body: &[u8],
    shape: ApiShape,
    store: &dyn CcrStore,
) -> RequestCompression {
    compress_json_request_with_auth(body, shape, store, RequestAuthMode::Payg)
}

pub fn compress_json_request_with_auth(
    body: &[u8],
    shape: ApiShape,
    store: &dyn CcrStore,
    auth_mode: RequestAuthMode,
) -> RequestCompression {
    compress_json_request_ctx(body, shape, CompressContext::new(store, auth_mode))
}

pub fn compress_json_request_ctx(
    body: &[u8],
    shape: ApiShape,
    ctx: CompressContext<'_>,
) -> RequestCompression {
    let bytes_before = body.len();
    let tokens_before = estimate_tokens_bytes(body);
    let passthrough = |reason: &str| RequestCompression {
        body: body.to_vec(),
        compressed: false,
        skipped_reason: Some(reason.to_string()),
        hashes: Vec::new(),
        bytes_before,
        bytes_after: bytes_before,
        tokens_before,
        tokens_after: tokens_before,
    };

    let Ok(parsed) = serde_json::from_slice::<Value>(body) else {
        return passthrough("invalid_json");
    };
    if std::str::from_utf8(body).is_err() {
        return passthrough("invalid_json");
    }

    // `n > 1` requests pass through verbatim, mirroring the reference
    // pre-dispatch skip (`live_zone_openai.rs` `should_skip_compression`).
    if shape == ApiShape::OpenAiChatCompletions
        && parsed
            .get("n")
            .and_then(Value::as_u64)
            .is_some_and(|n| n > 1)
    {
        return passthrough("multi_choice_n");
    }

    let marker_was_present = value_contains_ccr_marker(&parsed);

    // Metadata pre-pass (PAYG only): deterministic tool-array sort,
    // canonical schema bytes and Anthropic cache_control
    // auto-placement, applied as byte-splices confined to the tools
    // subtree. The JSON effect mirrors the reference normalization
    // (`live_zone_openai.rs:373-449`, `anthropic_cache_control.rs`);
    // the reference re-serializes the whole body when it applies
    // (`live_zone_openai.rs:436`) — hr keeps every byte outside the
    // tools subtree verbatim instead.
    let mut metadata_splices: Vec<Replacement> = Vec::new();
    if ctx.provider_metadata && ctx.auth_mode == RequestAuthMode::Payg {
        if let Ok(body_str) = std::str::from_utf8(body) {
            plan_provider_metadata_splices(body_str, &parsed, shape, &mut metadata_splices);
        }
    }
    let metadata_mutated = !metadata_splices.is_empty();
    let current: Vec<u8> = if metadata_mutated {
        apply_replacements(body, &mut metadata_splices)
    } else {
        body.to_vec()
    };
    let working: Value = if metadata_mutated {
        match serde_json::from_slice(&current) {
            Ok(value) => value,
            Err(_) => return passthrough("invalid_json"),
        }
    } else {
        parsed
    };
    // `current` came from valid UTF-8 JSON either way.
    let Ok(current_str) = std::str::from_utf8(&current) else {
        return passthrough("invalid_json");
    };

    // Live-zone planning over the current bytes.
    let mut slots: Vec<PlanSlot> = Vec::new();
    let plan_reason = match shape {
        ApiShape::OpenAiChatCompletions => {
            plan_openai_chat(current_str, &working, &ctx, &mut slots)
        }
        ApiShape::OpenAiResponses => {
            plan_openai_responses(current_str, &working, ctx.frozen_message_count, &mut slots)
        }
        ApiShape::AnthropicMessages => plan_anthropic_segment(
            current_str,
            0,
            ctx.frozen_message_count,
            ctx.compress_user_text,
            None,
            &mut slots,
        ),
        ApiShape::AnthropicMessageBatches => plan_anthropic_batches(current_str, &ctx, &mut slots),
    };

    // Gate + compress each planned slot into a byte-range replacement.
    let mut replacements: Vec<Replacement> = Vec::new();
    let mut hashes: Vec<String> = Vec::new();
    let mut compressed_batch_requests: BTreeSet<usize> = BTreeSet::new();
    for slot in &slots {
        if let Some((replacement, hash)) = compress_slot(slot, ctx.store) {
            replacements.push(replacement);
            hashes.push(hash);
            if let Some(index) = slot.batch_request_index {
                compressed_batch_requests.insert(index);
            }
        }
    }
    let compressed_blocks = replacements.len();

    // Byte-range surgery: every byte outside the replaced spans is
    // copied verbatim from the input (mirrors `live_zone.rs`
    // `apply_replacements`).
    let spliced: Vec<u8> = if replacements.is_empty() {
        current.clone()
    } else {
        apply_replacements(&current, &mut replacements)
    };

    // Metadata post-pass, stage 1: retrieve-tool injection as a
    // byte-splice so all other bytes stay verbatim; the JSON effect
    // matches the reference, which mutates the parsed body and
    // re-serializes (`headroom/proxy/handlers/anthropic.py:2560`).
    let mut retrieve_splices: Vec<Replacement> = Vec::new();
    let mut retrieve_tool_injected = false;
    if let Ok(spliced_str) = std::str::from_utf8(&spliced) {
        if let Ok(spliced_value) = serde_json::from_str::<Value>(spliced_str) {
            let inject_retrieve = compressed_blocks > 0 || marker_was_present;
            if shape == ApiShape::AnthropicMessageBatches {
                // Reference: the retrieve tool is injected per batch
                // request, and only into requests that were actually
                // compressed (`anthropic.py:2560-2574`,
                // `tokens_saved > 0`).
                retrieve_tool_injected = plan_batch_retrieve_tool_injection(
                    spliced_str,
                    &spliced_value,
                    &compressed_batch_requests,
                    &mut retrieve_splices,
                );
            } else if inject_retrieve {
                retrieve_tool_injected = plan_retrieve_tool_injection(
                    spliced_str,
                    &spliced_value,
                    shape,
                    &mut retrieve_splices,
                );
            }
        }
    }
    let injected = if retrieve_splices.is_empty() {
        spliced
    } else {
        apply_replacements(&spliced, &mut retrieve_splices)
    };

    // Stage 2: `prompt_cache_key`, derived from the POST-injection
    // body so the key reflects the final tools array on the wire —
    // the item-5 invariant requires the key to vary when the tools
    // change (`headroom-proxy/src/proxy.rs:1197` runs the inject on
    // the final dispatch body).
    let mut pck_splices: Vec<Replacement> = Vec::new();
    let mut prompt_cache_key_injected = false;
    if ctx.provider_metadata
        && ctx.auth_mode == RequestAuthMode::Payg
        && matches!(
            shape,
            ApiShape::OpenAiChatCompletions | ApiShape::OpenAiResponses
        )
    {
        if let Ok(injected_str) = std::str::from_utf8(&injected) {
            if let Ok(injected_value) = serde_json::from_str::<Value>(injected_str) {
                prompt_cache_key_injected = plan_prompt_cache_key_injection(
                    injected_str,
                    &injected_value,
                    shape,
                    &mut pck_splices,
                );
            }
        }
    }
    let final_body = if pck_splices.is_empty() {
        injected
    } else {
        apply_replacements(&injected, &mut pck_splices)
    };

    if compressed_blocks == 0
        && !metadata_mutated
        && !retrieve_tool_injected
        && !prompt_cache_key_injected
    {
        return RequestCompression {
            body: body.to_vec(),
            compressed: false,
            skipped_reason: Some(plan_reason.unwrap_or("no_live_zone_text").to_string()),
            hashes,
            bytes_before,
            bytes_after: bytes_before,
            tokens_before,
            tokens_after: tokens_before,
        };
    }

    let bytes_after = final_body.len();
    let tokens_after = estimate_tokens_bytes(&final_body);
    RequestCompression {
        body: final_body,
        compressed: compressed_blocks > 0,
        skipped_reason: (compressed_blocks == 0).then(|| "metadata_injected_only".to_string()),
        hashes,
        bytes_before,
        bytes_after,
        tokens_before,
        tokens_after,
    }
}

pub fn estimate_tokens(input: &str) -> usize {
    let chars = input.chars().count();
    let whitespace_units = input.split_whitespace().count();
    chars.div_ceil(4).max(whitespace_units).max(1)
}

fn estimate_tokens_bytes(input: &[u8]) -> usize {
    std::str::from_utf8(input)
        .map(estimate_tokens)
        .unwrap_or_else(|_| input.len().div_ceil(4).max(1))
}

fn skipped(
    input: &str,
    reason: &str,
    original_bytes: usize,
    original_tokens: usize,
) -> CompressResult {
    CompressResult {
        output: input.to_string(),
        hash: None,
        marker: None,
        original_bytes,
        compressed_bytes: original_bytes,
        original_tokens,
        compressed_tokens: original_tokens,
        stored: false,
        skipped_reason: Some(reason.to_string()),
        error: None,
    }
}

// ─── Byte-range surgery ────────────────────────────────────────────────

/// One byte-range replacement (or pure insertion when
/// `range.0 == range.1`). Sorted by ascending start before splicing.
struct Replacement {
    range: (usize, usize),
    bytes: Vec<u8>,
}

/// Apply all `replacements` to `original`, copying every byte outside
/// the replaced ranges verbatim. Mirrors `live_zone.rs`
/// `apply_replacements`.
fn apply_replacements(original: &[u8], replacements: &mut [Replacement]) -> Vec<u8> {
    replacements.sort_by_key(|replacement| replacement.range.0);

    let removed: usize = replacements
        .iter()
        .map(|replacement| replacement.range.1 - replacement.range.0)
        .sum();
    let added: usize = replacements
        .iter()
        .map(|replacement| replacement.bytes.len())
        .sum();
    let mut out = Vec::with_capacity(original.len().saturating_sub(removed) + added);

    let mut cursor = 0usize;
    for replacement in replacements.iter() {
        debug_assert!(
            cursor <= replacement.range.0,
            "overlapping replacement spans: cursor {} past range start {}",
            cursor,
            replacement.range.0
        );
        out.extend_from_slice(&original[cursor..replacement.range.0]);
        out.extend_from_slice(&replacement.bytes);
        cursor = replacement.range.1;
    }
    out.extend_from_slice(&original[cursor..]);
    out
}

/// Byte offset of `child` within `parent` when both are `&str` views
/// into the same backing buffer (serde_json `&RawValue` borrows point
/// into the input). Mirrors `live_zone.rs` `bytes_offset_of`.
fn offset_of(parent: &str, child: &str) -> Option<usize> {
    let parent_start = parent.as_ptr() as usize;
    let parent_end = parent_start + parent.len();
    let child_start = child.as_ptr() as usize;
    if child_start < parent_start || child_start + child.len() > parent_end {
        return None;
    }
    Some(child_start - parent_start)
}

// ─── Live-zone planning ────────────────────────────────────────────────

/// One compressible JSON-string slot located in the request bytes.
/// `range` covers the JSON string token including its quotes.
struct PlanSlot {
    range: (usize, usize),
    text: String,
    min_bytes: usize,
    batch_request_index: Option<usize>,
    label: &'static str,
}

#[derive(Deserialize)]
struct MessagesView<'a> {
    #[serde(borrow, default)]
    messages: Option<Vec<&'a RawValue>>,
}

#[derive(Deserialize)]
struct MessageHeader<'a> {
    #[serde(borrow, default)]
    role: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
    #[serde(borrow, default)]
    name: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    tool_call_id: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct BlockHeader<'a> {
    #[serde(borrow, default)]
    r#type: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
    #[serde(borrow, default)]
    text: Option<&'a RawValue>,
    #[serde(borrow, default)]
    cache_control: Option<&'a RawValue>,
}

/// Plan live-zone slots for one Anthropic `messages` object occupying
/// `segment` (a slice of the full body starting at byte `base`).
///
/// Candidate rule mirrors the reference content router
/// (`headroom/transforms/content_router.py:1999-2004, 2318-2394`):
/// every message at an index `>= floor` is walked; `tool_result`
/// string content compresses by default; `text` blocks and string
/// user content compress only when user-text compression is opted in;
/// blocks carrying `cache_control` and cache-hot block types never
/// compress.
fn plan_anthropic_segment(
    segment: &str,
    base: usize,
    extra_floor: usize,
    compress_user_text: bool,
    batch_request_index: Option<usize>,
    slots: &mut Vec<PlanSlot>,
) -> Option<&'static str> {
    let Ok(view) = serde_json::from_str::<MessagesView<'_>>(segment) else {
        return Some("missing_messages");
    };
    let Some(messages) = view.messages else {
        return Some("missing_messages");
    };
    let Ok(segment_value) = serde_json::from_str::<Value>(segment) else {
        return Some("missing_messages");
    };
    let cache_control_floor = segment_value
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| anthropic_frozen_message_count(messages))
        .unwrap_or(0);
    let floor = extra_floor.max(cache_control_floor);

    if floor >= messages.len() {
        return Some("no_live_zone_message");
    }

    let before = slots.len();
    for (index, message_raw) in messages.iter().enumerate() {
        if index < floor {
            continue;
        }
        let Ok(header) = serde_json::from_str::<MessageHeader<'_>>(message_raw.get()) else {
            continue;
        };
        let role = header.role.as_deref().unwrap_or("");
        let Some(content_raw) = header.content else {
            continue;
        };
        let content_str = content_raw.get();
        let Some(content_offset) = offset_of(segment, content_str) else {
            continue;
        };

        if content_str.starts_with('"') {
            // Legacy string-shaped content is user text; protected by
            // default per the reference role gate.
            if role == "user" && compress_user_text {
                if let Ok(text) = serde_json::from_str::<String>(content_str) {
                    slots.push(PlanSlot {
                        range: (
                            base + content_offset,
                            base + content_offset + content_str.len(),
                        ),
                        text,
                        min_bytes: LIVE_ZONE_MIN_BYTES,
                        batch_request_index,
                        label: "anthropic.string_content",
                    });
                }
            }
            continue;
        }
        if !content_str.starts_with('[') {
            continue;
        }
        let Ok(blocks) = serde_json::from_str::<Vec<&RawValue>>(content_str) else {
            continue;
        };
        for block_raw in blocks {
            let Ok(block) = serde_json::from_str::<BlockHeader<'_>>(block_raw.get()) else {
                continue;
            };
            if block.cache_control.is_some() {
                debug!("skipped cache_control block");
                continue;
            }
            let block_type = block.r#type.as_deref().unwrap_or("");
            if HOT_ZONE_BLOCK_TYPES.contains(&block_type) {
                trace!(block_type, "skipped hot-zone block");
                continue;
            }
            let field_raw = match block_type {
                "tool_result" => block.content,
                "text" => {
                    if role != "user" || !compress_user_text {
                        continue;
                    }
                    block.text
                }
                _ => continue,
            };
            let Some(field_raw) = field_raw else {
                continue;
            };
            let field_str = field_raw.get();
            // Only the string shape is compressed; structured-array
            // content round-trips byte-equal (mirrors
            // `live_zone.rs:1150`).
            if !field_str.starts_with('"') {
                continue;
            }
            let Some(field_offset) = offset_of(segment, field_str) else {
                continue;
            };
            let Ok(text) = serde_json::from_str::<String>(field_str) else {
                continue;
            };
            slots.push(PlanSlot {
                range: (base + field_offset, base + field_offset + field_str.len()),
                text,
                min_bytes: LIVE_ZONE_MIN_BYTES,
                batch_request_index,
                label: if block_type == "tool_result" {
                    "anthropic.tool_result"
                } else {
                    "anthropic.text"
                },
            });
        }
    }

    if slots.len() == before {
        Some("no_live_zone_text")
    } else {
        None
    }
}

#[derive(Deserialize)]
struct BatchView<'a> {
    #[serde(borrow, default)]
    requests: Option<Vec<&'a RawValue>>,
}

#[derive(Deserialize)]
struct BatchRequestView<'a> {
    #[serde(borrow, default)]
    params: Option<&'a RawValue>,
}

fn plan_anthropic_batches(
    body_str: &str,
    ctx: &CompressContext<'_>,
    slots: &mut Vec<PlanSlot>,
) -> Option<&'static str> {
    let Ok(view) = serde_json::from_str::<BatchView<'_>>(body_str) else {
        return Some("missing_requests");
    };
    let Some(requests) = view.requests else {
        return Some("missing_requests");
    };

    let before = slots.len();
    for (index, request_raw) in requests.iter().enumerate() {
        let Ok(request) = serde_json::from_str::<BatchRequestView<'_>>(request_raw.get()) else {
            continue;
        };
        let Some(params_raw) = request.params else {
            continue;
        };
        let Some(params_offset) = offset_of(body_str, params_raw.get()) else {
            continue;
        };
        let _ = plan_anthropic_segment(
            params_raw.get(),
            params_offset,
            0,
            ctx.compress_user_text,
            Some(index),
            slots,
        );
    }

    if slots.len() == before {
        Some("no_live_zone_text")
    } else {
        None
    }
}

/// Plan live-zone slots for an OpenAI Chat Completions request:
/// the latest `role == "tool"` message (excluding `headroom_retrieve`
/// results) and — only when user-text compression is opted in — the
/// latest `role == "user"` message, both at indices `>= floor`.
/// Mirrors `live_zone.rs` `compress_openai_chat_live_zone` with the
/// reference role gate from `content_router.py:2318-2378`.
fn plan_openai_chat(
    body_str: &str,
    parsed: &Value,
    ctx: &CompressContext<'_>,
    slots: &mut Vec<PlanSlot>,
) -> Option<&'static str> {
    let Ok(view) = serde_json::from_str::<MessagesView<'_>>(body_str) else {
        return Some("missing_messages");
    };
    let Some(messages) = view.messages else {
        return Some("missing_messages");
    };
    let retrieve_tool_call_ids = parsed
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| headroom_retrieve_tool_call_ids(messages))
        .unwrap_or_default();

    let floor = ctx.frozen_message_count;
    let mut latest_tool: Option<(usize, MessageHeader<'_>)> = None;
    let mut latest_user: Option<(usize, MessageHeader<'_>)> = None;
    for (index, message_raw) in messages.iter().enumerate().rev() {
        if index < floor {
            break;
        }
        let Ok(header) = serde_json::from_str::<MessageHeader<'_>>(message_raw.get()) else {
            continue;
        };
        match header.role.as_deref() {
            Some("tool") if latest_tool.is_none() => {
                let name_is_retrieve = header.name.as_deref().is_some_and(is_retrieve_tool_name);
                let call_is_retrieve = header
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| retrieve_tool_call_ids.iter().any(|known| known == id));
                if !(name_is_retrieve || call_is_retrieve) {
                    latest_tool = Some((index, header));
                }
            }
            Some("user") if latest_user.is_none() => {
                latest_user = Some((index, header));
            }
            _ => {}
        }
        if latest_tool.is_some() && latest_user.is_some() {
            break;
        }
    }

    if latest_tool.is_none() && latest_user.is_none() {
        return Some("no_live_zone_message");
    }

    let before = slots.len();
    if let Some((_, header)) = latest_tool {
        // Tool output compresses freely (reference role gate). Only
        // the string shape is planned, mirroring
        // `plan_openai_tool_message`.
        if let Some(content_raw) = header.content {
            let content_str = content_raw.get();
            if content_str.starts_with('"') {
                if let (Some(content_offset), Ok(text)) = (
                    offset_of(body_str, content_str),
                    serde_json::from_str::<String>(content_str),
                ) {
                    slots.push(PlanSlot {
                        range: (content_offset, content_offset + content_str.len()),
                        text,
                        min_bytes: LIVE_ZONE_MIN_BYTES,
                        batch_request_index: None,
                        label: "openai.tool_content",
                    });
                }
            }
        }
    }
    if ctx.compress_user_text {
        if let Some((_, header)) = latest_user {
            if let Some(content_raw) = header.content {
                plan_openai_user_content(body_str, content_raw, slots);
            }
        }
    }

    if slots.len() == before {
        Some("no_live_zone_text")
    } else {
        None
    }
}

/// Plan the latest user message's text content for OpenAI chat:
/// string content as one slot, or every `{type: "text", text}` part.
/// Mirrors `live_zone.rs` `plan_openai_user_message`.
fn plan_openai_user_content(body_str: &str, content_raw: &RawValue, slots: &mut Vec<PlanSlot>) {
    let content_str = content_raw.get();
    if content_str.starts_with('"') {
        if let (Some(content_offset), Ok(text)) = (
            offset_of(body_str, content_str),
            serde_json::from_str::<String>(content_str),
        ) {
            slots.push(PlanSlot {
                range: (content_offset, content_offset + content_str.len()),
                text,
                min_bytes: LIVE_ZONE_MIN_BYTES,
                batch_request_index: None,
                label: "openai.user_string",
            });
        }
        return;
    }
    if !content_str.starts_with('[') {
        return;
    }
    let Ok(parts) = serde_json::from_str::<Vec<&RawValue>>(content_str) else {
        return;
    };
    for part_raw in parts {
        let Ok(part) = serde_json::from_str::<BlockHeader<'_>>(part_raw.get()) else {
            continue;
        };
        if part.r#type.as_deref() != Some("text") {
            continue;
        }
        let Some(text_raw) = part.text else {
            continue;
        };
        let text_str = text_raw.get();
        if !text_str.starts_with('"') {
            continue;
        }
        let (Some(text_offset), Ok(text)) = (
            offset_of(body_str, text_str),
            serde_json::from_str::<String>(text_str),
        ) else {
            continue;
        };
        slots.push(PlanSlot {
            range: (text_offset, text_offset + text_str.len()),
            text,
            min_bytes: LIVE_ZONE_MIN_BYTES,
            batch_request_index: None,
            label: "openai.user_text",
        });
    }
}

#[derive(Deserialize)]
struct ResponsesBodyView<'a> {
    #[serde(borrow, default)]
    input: Option<&'a RawValue>,
    #[serde(borrow, default)]
    messages: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct ResponsesItemHeader<'a> {
    #[serde(borrow, default)]
    r#type: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    call_id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    output: Option<&'a RawValue>,
}

/// Plan live-zone slots for an OpenAI Responses request: every
/// current-frame `function_call_output` / `local_shell_call_output` /
/// `apply_patch_call_output` item's string `output`, excluding
/// `headroom_retrieve` outputs. Message text is never compressed and
/// non-array `input` passes through, mirroring `live_zone.rs`
/// `compress_openai_responses_live_zone`.
fn plan_openai_responses(
    body_str: &str,
    parsed: &Value,
    frozen_item_count: usize,
    slots: &mut Vec<PlanSlot>,
) -> Option<&'static str> {
    let Ok(view) = serde_json::from_str::<ResponsesBodyView<'_>>(body_str) else {
        return Some("missing_input");
    };
    let Some(items_raw) = view.input.or(view.messages) else {
        return Some("missing_input");
    };
    if !items_raw.get().starts_with('[') {
        return Some("unsupported_input");
    }
    let Ok(items) = serde_json::from_str::<Vec<&RawValue>>(items_raw.get()) else {
        return Some("unsupported_input");
    };

    let retrieve_call_ids = parsed
        .get("input")
        .or_else(|| parsed.get("messages"))
        .and_then(Value::as_array)
        .map(|items| responses_retrieve_call_ids(items))
        .unwrap_or_default();

    // The tracker floor counts the synthesized walk list (a leading
    // `instructions` entry plus the input items —
    // `session::estimate_responses_request_tokens`); shift it back to
    // item indices before applying it.
    let frozen_item_count =
        frozen_item_count.saturating_sub(crate::session::responses_estimate_prefix_len(parsed));

    let before = slots.len();
    for (index, item_raw) in items.iter().enumerate() {
        // Items inside the provider-confirmed cached prefix are never
        // mutated (checklist 3: the session tracker floor applies to
        // the Responses surface too).
        if index < frozen_item_count {
            continue;
        }
        let Ok(header) = serde_json::from_str::<ResponsesItemHeader<'_>>(item_raw.get()) else {
            continue;
        };
        let kind = header.r#type.as_deref().unwrap_or("");
        if !is_responses_output_item(kind) {
            continue;
        }
        if header
            .call_id
            .as_deref()
            .is_some_and(|id| retrieve_call_ids.iter().any(|known| known == id))
        {
            continue;
        }
        let Some(output_raw) = header.output else {
            continue;
        };
        let output_str = output_raw.get();
        if !output_str.starts_with('"') {
            continue;
        }
        let (Some(output_offset), Ok(text)) = (
            offset_of(body_str, output_str),
            serde_json::from_str::<String>(output_str),
        ) else {
            continue;
        };
        slots.push(PlanSlot {
            range: (output_offset, output_offset + output_str.len()),
            text,
            min_bytes: RESPONSES_OUTPUT_MIN_BYTES.max(LIVE_ZONE_MIN_BYTES),
            batch_request_index: None,
            label: "responses.output",
        });
    }

    if slots.len() == before {
        Some("no_live_zone_text")
    } else {
        None
    }
}

/// Gate one planned slot and produce the marker replacement.
///
/// Gate order mirrors `live_zone.rs` `compress_one_block`: byte
/// threshold first, then the token-shrink check; the original is
/// persisted to the CCR store only after both gates admit the
/// compression.
fn compress_slot(slot: &PlanSlot, store: &dyn CcrStore) -> Option<(Replacement, String)> {
    if slot.text.contains("<<ccr:") {
        debug!(label = slot.label, "skipped existing ccr marker");
        return None;
    }
    if slot.text.len() < slot.min_bytes {
        trace!(
            label = slot.label,
            bytes = slot.text.len(),
            threshold = slot.min_bytes,
            "skipped below byte threshold"
        );
        return None;
    }

    let hash = content_hash(&slot.text);
    let marker = marker_for_hash(&hash);
    if estimate_tokens(&marker) >= estimate_tokens(&slot.text) {
        debug!(label = slot.label, "skipped: marker not smaller in tokens");
        return None;
    }

    match store.put(&hash, &slot.text) {
        Ok(inserted) => {
            if inserted {
                stats::record_ccr_entry_inserted();
            }
        }
        Err(err) => {
            warn!(label = slot.label, error = %err, "ccr store put failed; block kept");
            return None;
        }
    }

    let replacement = serde_json::to_vec(&Value::String(marker)).expect("string serializes");
    debug!(
        label = slot.label,
        hash,
        original_bytes = slot.text.len(),
        compressed_bytes = replacement.len(),
        "compressed live-zone block"
    );
    trace!(label = slot.label, snippet = %snippet(&slot.text), "compressed snippet");
    Some((
        Replacement {
            range: slot.range,
            bytes: replacement,
        },
        hash,
    ))
}

// ─── Retrieve-tool / metadata helpers ──────────────────────────────────

fn headroom_retrieve_tool_call_ids(messages: &[Value]) -> Vec<String> {
    let mut ids = Vec::new();
    for message in messages {
        let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for tool_call in tool_calls {
            let name = tool_call
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if is_retrieve_tool_name(name) {
                if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
                    ids.push(id.to_string());
                }
            }
        }
    }
    ids
}

fn responses_retrieve_call_ids(items: &[Value]) -> Vec<String> {
    let mut ids = Vec::new();
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            continue;
        }
        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
        if is_retrieve_tool_name(name) {
            if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                ids.push(call_id.to_string());
            }
        }
    }
    ids
}

fn is_responses_output_item(kind: &str) -> bool {
    matches!(
        kind,
        "function_call_output" | "local_shell_call_output" | "apply_patch_call_output"
    )
}

fn value_contains_ccr_marker(value: &Value) -> bool {
    match value {
        Value::String(text) => text.contains("<<ccr:"),
        Value::Array(items) => items.iter().any(value_contains_ccr_marker),
        Value::Object(map) => map.values().any(value_contains_ccr_marker),
        _ => false,
    }
}

/// Plan the PAYG metadata splices (tool-array sort, canonical schema
/// bytes, Anthropic `cache_control` auto-placement) for the tools
/// subtree(s) of the request.
fn plan_provider_metadata_splices(
    body_str: &str,
    parsed: &Value,
    shape: ApiShape,
    replacements: &mut Vec<Replacement>,
) {
    match shape {
        ApiShape::OpenAiChatCompletions | ApiShape::OpenAiResponses => {
            plan_tools_normalization(body_str, 0, parsed, ToolShape::OpenAi, replacements);
        }
        ApiShape::AnthropicMessages => {
            plan_tools_normalization(body_str, 0, parsed, ToolShape::Anthropic, replacements);
        }
        ApiShape::AnthropicMessageBatches => {
            let Ok(view) = serde_json::from_str::<BatchView<'_>>(body_str) else {
                return;
            };
            let Some(requests) = view.requests else {
                return;
            };
            let parsed_requests = parsed.get("requests").and_then(Value::as_array);
            for (index, request_raw) in requests.iter().enumerate() {
                let Ok(request) = serde_json::from_str::<BatchRequestView<'_>>(request_raw.get())
                else {
                    continue;
                };
                let Some(params_raw) = request.params else {
                    continue;
                };
                let Some(params_offset) = offset_of(body_str, params_raw.get()) else {
                    continue;
                };
                let Some(params_value) = parsed_requests
                    .and_then(|requests| requests.get(index))
                    .and_then(|request| request.get("params"))
                else {
                    continue;
                };
                plan_tools_normalization(
                    params_raw.get(),
                    params_offset,
                    params_value,
                    ToolShape::Anthropic,
                    replacements,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ToolShape {
    OpenAi,
    Anthropic,
}

#[derive(Deserialize)]
struct ToolsArrayView<'a> {
    #[serde(borrow, default)]
    tools: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct ToolSchemaView<'a> {
    #[serde(borrow, default)]
    function: Option<FunctionSchemaView<'a>>,
    #[serde(borrow, default)]
    parameters: Option<&'a RawValue>,
    #[serde(borrow, default)]
    input_schema: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct FunctionSchemaView<'a> {
    #[serde(borrow, default)]
    parameters: Option<&'a RawValue>,
}

/// Deterministic tool-array sort + canonical schema bytes for one
/// object's `tools` array, planned as byte-splices.
///
/// - An existing `cache_control` marker on any tool skips the array
///   ordering ONLY — schema canonicalization still runs, mirroring
///   `live_zone_openai.rs:373-430`.
/// - The sort mirrors `tool_def_normalize.rs`
///   `sort_tools_deterministically`: stable sort keyed by tool name
///   (`name`, then `function.name`) with an MD5-of-canonical-JSON
///   fallback for unnamed tools; equal keys keep relative order, and
///   the array is rewritten only when the key sequence moves.
/// - Schema canonicalization mirrors `sort_schema_keys_recursive`:
///   each schema subtree is rewritten to its canonical (recursively
///   key-sorted, compact) bytes when it isn't already canonical, while
///   the tool object's own key order and interior bytes are preserved
///   (the reference preserves key order via IndexMap and sorts only
///   the schema), in both the reordered and order-untouched paths.
/// - For Anthropic shapes, `cache_control: {"type": "ephemeral"}` is
///   auto-placed on the last tool when no marker exists anywhere
///   (mirrors `anthropic_cache_control.rs`
///   `auto_place_anthropic_cache_control`).
fn plan_tools_normalization(
    segment: &str,
    base: usize,
    object_value: &Value,
    shape: ToolShape,
    replacements: &mut Vec<Replacement>,
) {
    let Some(tools) = object_value.get("tools").and_then(Value::as_array) else {
        return;
    };
    if tools.is_empty() {
        return;
    }
    let Ok(view) = serde_json::from_str::<ToolsArrayView<'_>>(segment) else {
        return;
    };
    let Some(tools_raw) = view.tools else {
        return;
    };
    let tools_str = tools_raw.get();
    let Some(tools_offset) = offset_of(segment, tools_str) else {
        return;
    };
    let Ok(tool_raws) = serde_json::from_str::<Vec<&RawValue>>(tools_str) else {
        return;
    };
    if tool_raws.len() != tools.len() {
        return;
    }

    let marker_present = tools.iter().any(|tool| tool.get("cache_control").is_some());
    let place_cache_control =
        matches!(shape, ToolShape::Anthropic) && !anthropic_cache_control_already_set(object_value);

    if !marker_present {
        let before: Vec<String> = tools.iter().map(tool_sort_key).collect();
        let mut order: Vec<usize> = (0..tools.len()).collect();
        order.sort_by_key(|&index| before[index].clone());
        if order
            .iter()
            .enumerate()
            .any(|(position, &index)| position != index)
        {
            // The array moved: reorder the original tool slices so each
            // tool keeps its own key order and interior bytes (the
            // reference preserves key order via IndexMap and only sorts
            // the schema subtree), canonicalizing each tool's schema the
            // same way as the order-untouched path. Fold the Anthropic
            // cache_control placement into the rewrite.
            let mut pieces: Vec<String> = order
                .iter()
                .map(|&index| canonicalized_tool_piece(tool_raws[index].get(), shape))
                .collect();
            if place_cache_control {
                if let Some(last) = pieces.last_mut() {
                    append_cache_control_to_piece(last);
                }
            }
            let mut bytes = Vec::with_capacity(tools_str.len() + 48);
            bytes.push(b'[');
            for (position, piece) in pieces.iter().enumerate() {
                if position > 0 {
                    bytes.push(b',');
                }
                bytes.extend_from_slice(piece.as_bytes());
            }
            bytes.push(b']');
            replacements.push(Replacement {
                range: (base + tools_offset, base + tools_offset + tools_str.len()),
                bytes,
            });
            return;
        }
    }

    // Order untouched: canonicalize each tool schema in place.
    for tool_raw in &tool_raws {
        let Ok(schema_view) = serde_json::from_str::<ToolSchemaView<'_>>(tool_raw.get()) else {
            continue;
        };
        let schema_raw = match shape {
            ToolShape::OpenAi => schema_view
                .function
                .and_then(|function| function.parameters)
                .or(schema_view.parameters),
            ToolShape::Anthropic => schema_view.input_schema,
        };
        let Some(schema_raw) = schema_raw else {
            continue;
        };
        let schema_str = schema_raw.get();
        let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) else {
            continue;
        };
        let Ok(canonical) = serde_json::to_vec(&schema_value) else {
            continue;
        };
        if canonical != schema_str.as_bytes() {
            let Some(schema_offset) = offset_of(segment, schema_str) else {
                continue;
            };
            replacements.push(Replacement {
                range: (
                    base + schema_offset,
                    base + schema_offset + schema_str.len(),
                ),
                bytes: canonical,
            });
        }
    }

    if place_cache_control {
        if let Some(last_raw) = tool_raws.last() {
            let last_str = last_raw.get();
            if last_str.ends_with('}') {
                let Some(last_offset) = offset_of(segment, last_str) else {
                    return;
                };
                let insert_at = base + last_offset + last_str.len() - 1;
                let interior_empty = last_str
                    .strip_prefix('{')
                    .map(|rest| rest.trim_start().starts_with('}'))
                    .unwrap_or(false);
                let mut bytes = Vec::with_capacity(48);
                if !interior_empty {
                    bytes.push(b',');
                }
                bytes.extend_from_slice(b"\"cache_control\":{\"type\":\"ephemeral\"}");
                replacements.push(Replacement {
                    range: (insert_at, insert_at),
                    bytes,
                });
            }
        }
    }
}

/// Returns the tool's original slice with only its schema subtree
/// canonicalized. Used when the tool array is reordered so each tool
/// keeps its own key order and interior bytes, matching the
/// order-untouched path (the reference preserves tool key order and
/// sorts only the schema subtree).
fn canonicalized_tool_piece(tool_str: &str, shape: ToolShape) -> String {
    let Ok(schema_view) = serde_json::from_str::<ToolSchemaView<'_>>(tool_str) else {
        return tool_str.to_string();
    };
    let schema_raw = match shape {
        ToolShape::OpenAi => schema_view
            .function
            .and_then(|function| function.parameters)
            .or(schema_view.parameters),
        ToolShape::Anthropic => schema_view.input_schema,
    };
    let Some(schema_raw) = schema_raw else {
        return tool_str.to_string();
    };
    let schema_str = schema_raw.get();
    let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) else {
        return tool_str.to_string();
    };
    let Ok(canonical) = serde_json::to_string(&schema_value) else {
        return tool_str.to_string();
    };
    if canonical == schema_str {
        return tool_str.to_string();
    }
    let Some(offset) = offset_of(tool_str, schema_str) else {
        return tool_str.to_string();
    };
    let mut piece = String::with_capacity(tool_str.len() + canonical.len());
    piece.push_str(&tool_str[..offset]);
    piece.push_str(&canonical);
    piece.push_str(&tool_str[offset + schema_str.len()..]);
    piece
}

/// Splices `"cache_control":{"type":"ephemeral"}` before the closing
/// brace of a tool piece, mirroring the order-untouched insertion path.
fn append_cache_control_to_piece(piece: &mut String) {
    if !piece.ends_with('}') {
        return;
    }
    let interior_empty = piece
        .strip_prefix('{')
        .map(|rest| rest.trim_start().starts_with('}'))
        .unwrap_or(false);
    let insert_at = piece.len() - 1;
    let insertion = if interior_empty {
        "\"cache_control\":{\"type\":\"ephemeral\"}"
    } else {
        ",\"cache_control\":{\"type\":\"ephemeral\"}"
    };
    piece.insert_str(insert_at, insertion);
}

/// Sort key for the deterministic tool sort, mirroring
/// `tool_def_normalize.rs` `sort_key`: `name` wins, then
/// `function.name`, then MD5 of the canonical JSON serialization.
fn tool_sort_key(tool: &Value) -> String {
    if let Some(name) = tool.get("name").and_then(Value::as_str) {
        return name.to_string();
    }
    if let Some(name) = tool
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
    {
        return name.to_string();
    }
    let serialized = serde_json::to_vec(tool).unwrap_or_default();
    let digest = md5::Md5::digest(&serialized);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Plan a byte-splice that appends the per-shape retrieve tool to the
/// top-level `tools` array (creating it when missing).
fn plan_retrieve_tool_injection(
    body_str: &str,
    parsed: &Value,
    shape: ApiShape,
    replacements: &mut Vec<Replacement>,
) -> bool {
    let (tool, exists): (Value, fn(&Value) -> bool) = match shape {
        ApiShape::OpenAiChatCompletions => (openai_chat_retrieve_tool(), |tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .is_some_and(is_retrieve_tool_name)
        }),
        ApiShape::OpenAiResponses => (openai_responses_retrieve_tool(), |tool| {
            tool.get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    tool.get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                })
                .is_some_and(is_retrieve_tool_name)
        }),
        ApiShape::AnthropicMessages | ApiShape::AnthropicMessageBatches => {
            (anthropic_retrieve_tool(), |tool| {
                tool.get("name")
                    .and_then(Value::as_str)
                    .is_some_and(is_retrieve_tool_name)
            })
        }
    };
    plan_tool_append(body_str, 0, parsed, tool, exists, replacements)
}

/// Per-request retrieve-tool injection for Anthropic batch creates:
/// only requests that were actually compressed receive the tool,
/// mirroring `anthropic.py:2560-2574` (`tokens_saved > 0`).
fn plan_batch_retrieve_tool_injection(
    body_str: &str,
    parsed: &Value,
    compressed_requests: &BTreeSet<usize>,
    replacements: &mut Vec<Replacement>,
) -> bool {
    if compressed_requests.is_empty() {
        return false;
    }
    let Ok(view) = serde_json::from_str::<BatchView<'_>>(body_str) else {
        return false;
    };
    let Some(requests) = view.requests else {
        return false;
    };
    let parsed_requests = parsed.get("requests").and_then(Value::as_array);

    let mut injected = false;
    for index in compressed_requests {
        let Some(request_raw) = requests.get(*index) else {
            continue;
        };
        let Ok(request) = serde_json::from_str::<BatchRequestView<'_>>(request_raw.get()) else {
            continue;
        };
        let Some(params_raw) = request.params else {
            continue;
        };
        let Some(params_offset) = offset_of(body_str, params_raw.get()) else {
            continue;
        };
        let Some(params_value) = parsed_requests
            .and_then(|requests| requests.get(*index))
            .and_then(|request| request.get("params"))
        else {
            continue;
        };
        injected |= plan_tool_append(
            params_raw.get(),
            params_offset,
            params_value,
            anthropic_retrieve_tool(),
            |tool| {
                tool.get("name")
                    .and_then(Value::as_str)
                    .is_some_and(is_retrieve_tool_name)
            },
            replacements,
        );
    }
    injected
}

/// Plan the splice that appends `tool` to the `tools` array of the
/// object occupying `segment` (at byte `base` of the full body), or
/// creates `"tools": [tool]` when the field is missing. Returns false
/// (and plans nothing) when the tool is already present or the
/// `tools` field has a non-array shape.
fn plan_tool_append(
    segment: &str,
    base: usize,
    object_value: &Value,
    tool: Value,
    exists: impl Fn(&Value) -> bool,
    replacements: &mut Vec<Replacement>,
) -> bool {
    let Ok(tool_bytes) = serde_json::to_vec(&tool) else {
        return false;
    };

    match object_value.get("tools") {
        Some(Value::Array(tools)) => {
            if tools.iter().any(exists) {
                return false;
            }
            #[derive(Deserialize)]
            struct ToolsFieldView<'a> {
                #[serde(borrow, default)]
                tools: Option<&'a RawValue>,
            }
            let Ok(view) = serde_json::from_str::<ToolsFieldView<'_>>(segment) else {
                return false;
            };
            let Some(tools_raw) = view.tools else {
                return false;
            };
            let tools_str = tools_raw.get();
            if !tools_str.ends_with(']') {
                return false;
            }
            let Some(tools_offset) = offset_of(segment, tools_str) else {
                return false;
            };
            let insert_at = base + tools_offset + tools_str.len() - 1;
            let mut bytes = Vec::with_capacity(tool_bytes.len() + 1);
            if !tools.is_empty() {
                bytes.push(b',');
            }
            bytes.extend_from_slice(&tool_bytes);
            replacements.push(Replacement {
                range: (insert_at, insert_at),
                bytes,
            });
            true
        }
        Some(_) => false,
        None => {
            let trimmed = segment.trim_end();
            if !trimmed.ends_with('}') {
                return false;
            }
            let insert_at = base + trimmed.len() - 1;
            let interior_empty = trimmed
                .strip_prefix('{')
                .map(|rest| rest.trim_start().starts_with('}'))
                .unwrap_or(false);
            let mut bytes = Vec::with_capacity(tool_bytes.len() + 12);
            if !interior_empty {
                bytes.push(b',');
            }
            bytes.extend_from_slice(b"\"tools\":[");
            bytes.extend_from_slice(&tool_bytes);
            bytes.push(b']');
            replacements.push(Replacement {
                range: (insert_at, insert_at),
                bytes,
            });
            true
        }
    }
}

#[derive(Deserialize)]
struct PromptCacheKeyView<'a> {
    #[serde(borrow, default)]
    prompt_cache_key: Option<&'a RawValue>,
}

/// Plan the `prompt_cache_key` splice. Skipped when a non-empty
/// string key is already present (customer value wins); an existing
/// empty/non-string value is treated as absent and REPLACED in place
/// (the reference `map.insert(...)` overwrites it —
/// `openai_cache_key.rs` `inject_prompt_cache_key`); a missing field
/// is inserted before the top-level closing brace.
fn plan_prompt_cache_key_injection(
    body_str: &str,
    parsed: &Value,
    shape: ApiShape,
    replacements: &mut Vec<Replacement>,
) -> bool {
    if !parsed.is_object() {
        return false;
    }
    if parsed
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .is_some_and(|key| !key.is_empty())
    {
        return false;
    }

    let key = derive_openai_prompt_cache_key(parsed, shape);
    let key_token = format!("\"{key}\"");

    // Existing-but-absent-like value (e.g. `""`): replace its value
    // span in place so the object never carries duplicate keys.
    if parsed.get("prompt_cache_key").is_some() {
        let Ok(view) = serde_json::from_str::<PromptCacheKeyView<'_>>(body_str) else {
            return false;
        };
        let Some(value_raw) = view.prompt_cache_key else {
            return false;
        };
        let value_str = value_raw.get();
        let Some(value_offset) = offset_of(body_str, value_str) else {
            return false;
        };
        replacements.push(Replacement {
            range: (value_offset, value_offset + value_str.len()),
            bytes: key_token.into_bytes(),
        });
        return true;
    }

    let trimmed = body_str.trim_end();
    if !trimmed.ends_with('}') {
        return false;
    }
    let insert_at = trimmed.len() - 1;
    let interior_empty = trimmed
        .strip_prefix('{')
        .map(|rest| rest.trim_start().starts_with('}'))
        .unwrap_or(false);
    let mut bytes = Vec::with_capacity(key.len() + 24);
    if !interior_empty {
        bytes.push(b',');
    }
    bytes.extend_from_slice(b"\"prompt_cache_key\":");
    bytes.extend_from_slice(key_token.as_bytes());
    replacements.push(Replacement {
        range: (insert_at, insert_at),
        bytes,
    });
    true
}

/// Derive the deterministic OpenAI `prompt_cache_key`: model bytes,
/// the system *content*'s full SHA-256 hex and the tools field's full
/// SHA-256 hex, NUL-separated, hashed and truncated to 32 hex chars.
/// Mirrors `openai_cache_key.rs` `derive_key`.
fn derive_openai_prompt_cache_key(value: &Value, shape: ApiShape) -> String {
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let system = openai_system_value(value, shape);
    let tools = value.get("tools").cloned().unwrap_or(Value::Null);
    let system_hash = canonical_json_sha256(&system);
    let tools_hash = canonical_json_sha256(&tools);

    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update([0]);
    hasher.update(system_hash.as_bytes());
    hasher.update([0]);
    hasher.update(tools_hash.as_bytes());
    let digest = hasher.finalize();
    digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

/// System content for the cache key. Responses checks `instructions`
/// first, then a system message in `input`, then one in `messages`,
/// mirroring `openai_cache_key.rs` `extract_system`.
fn openai_system_value(value: &Value, shape: ApiShape) -> Value {
    if shape == ApiShape::OpenAiResponses {
        if let Some(instructions) = value.get("instructions") {
            return instructions.clone();
        }
        if let Some(content) = value.get("input").and_then(first_system_content) {
            return content;
        }
        return value
            .get("messages")
            .and_then(first_system_content)
            .unwrap_or(Value::Null);
    }

    value
        .get("messages")
        .and_then(first_system_content)
        .unwrap_or(Value::Null)
}

fn first_system_content(items: &Value) -> Option<Value> {
    let items = items.as_array()?;
    for item in items {
        if item.get("role").and_then(Value::as_str) == Some("system") {
            return Some(item.get("content").cloned().unwrap_or(Value::Null));
        }
    }
    None
}

fn canonical_json_sha256(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"null".to_vec());
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn openai_chat_retrieve_tool() -> Value {
    json!({
        "type": "function",
        "function": retrieve_function_definition(),
    })
}

fn openai_responses_retrieve_tool() -> Value {
    let function = retrieve_function_definition();
    json!({
        "type": "function",
        "name": function["name"].clone(),
        "description": function["description"].clone(),
        "parameters": function["parameters"].clone(),
    })
}

fn anthropic_retrieve_tool() -> Value {
    let function = retrieve_function_definition();
    json!({
        "name": function["name"].clone(),
        "description": function["description"].clone(),
        "input_schema": function["parameters"].clone(),
    })
}

fn retrieve_function_definition() -> Value {
    json!({
        "name": "headroom_retrieve",
        "description": "Retrieve original uncompressed content that was compressed to save tokens. Use this when a <<ccr:HASH>> marker is present and more detail is needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "hash": {
                    "type": "string",
                    "description": "Hash from a <<ccr:HASH>> marker."
                },
                "query": {
                    "type": "string",
                    "description": "Optional search query used to return matching lines instead of the full original content."
                }
            },
            "required": ["hash"]
        }
    })
}

fn is_retrieve_tool_name(name: &str) -> bool {
    name == "headroom_retrieve" || name.ends_with("__headroom_retrieve")
}

fn anthropic_cache_control_already_set(value: &Value) -> bool {
    value
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(|tool| tool.get("cache_control").is_some()))
        || anthropic_system_has_cache_control(value.get("system"))
        || value
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages
                    .iter()
                    .any(|message| anthropic_content_has_cache_control(message.get("content")))
            })
}

fn anthropic_system_has_cache_control(system: Option<&Value>) -> bool {
    match system {
        Some(Value::Array(blocks)) => blocks
            .iter()
            .any(|block| block.get("cache_control").is_some()),
        Some(Value::Object(block)) => block.get("cache_control").is_some(),
        _ => false,
    }
}

fn anthropic_content_has_cache_control(content: Option<&Value>) -> bool {
    match content {
        Some(Value::Array(blocks)) => blocks
            .iter()
            .any(|block| block.get("cache_control").is_some()),
        Some(Value::Object(block)) => block.get("cache_control").is_some(),
        _ => false,
    }
}

fn anthropic_frozen_message_count(messages: &[Value]) -> usize {
    messages
        .iter()
        .enumerate()
        .filter(|(_, message)| anthropic_content_has_cache_control(message.get("content")))
        .map(|(index, _)| index + 1)
        .max()
        .unwrap_or(0)
}

fn snippet(input: &str) -> String {
    const LIMIT: usize = 120;
    let mut snippet: String = input.chars().take(LIMIT).collect();
    if input.chars().count() > LIMIT {
        snippet.push_str("...");
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::{decompress_text, SqliteStore};
    use sha2::{Digest, Sha256};

    fn long_text(label: &str) -> String {
        format!("{label} {}", "0123456789 abcdefghij ".repeat(32))
    }

    fn ctx<'a>(store: &'a SqliteStore, auth_mode: RequestAuthMode) -> CompressContext<'a> {
        CompressContext::new(store, auth_mode)
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn json_string_token(text: &str) -> String {
        serde_json::to_string(text).unwrap()
    }

    /// Expected effect of the retrieve-tool injection splice when the
    /// body has no `tools` field: `,"tools":[<tool>]` inserted before
    /// the top-level closing brace (trailing whitespace preserved).
    fn with_top_level_tools_splice(body: &str, tool: &Value) -> String {
        let tool_json = serde_json::to_string(tool).unwrap();
        let trimmed = body.trim_end();
        assert!(trimmed.ends_with('}'));
        format!(
            "{},\"tools\":[{}]{}",
            &trimmed[..trimmed.len() - 1],
            tool_json,
            &body[trimmed.len() - 1..]
        )
    }

    #[test]
    fn compress_emits_ccr_marker_and_stores_original() {
        let _guard = crate::stats::test_lock();
        crate::stats::reset_for_tests();
        let store = SqliteStore::in_memory().unwrap();

        let result = compress(
            "large live content",
            CompressOptions {
                store: &store,
                min_bytes: 1,
            },
        );

        assert!(result.stored);
        assert_eq!(
            result.output,
            marker_for_hash(result.hash.as_ref().unwrap())
        );
        assert_eq!(
            decompress_text(&result.output, &store).output,
            "large live content"
        );
    }

    #[test]
    fn anthropic_tool_result_compresses_by_default_and_bytes_outside_span_are_preserved() {
        let store = SqliteStore::in_memory().unwrap();
        let bulk = long_text("tool output");
        let bulk_token = json_string_token(&bulk);
        // Irregular whitespace, non-alphabetical key order, escaped and
        // multi-byte strings — all must round-trip byte-identically.
        let body = format!(
            "{{\n  \"model\" :\t\"claude-test\",\n  \"messages\": [ {{\"content\": [ {{\"content\": {bulk_token}, \"type\": \"tool_result\", \"tool_use_id\": \"t\\u00e9st_\\\"id\\\"_🦀\"}} ],\"role\":\"user\"}} ]\n}}\n"
        );

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::AnthropicMessages,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(result.compressed);
        assert_eq!(result.hashes.len(), 1);
        let marker_token = json_string_token(&marker_for_hash(&result.hashes[0]));
        let expected = with_top_level_tools_splice(
            &body.replacen(&bulk_token, &marker_token, 1),
            &anthropic_retrieve_tool(),
        );
        assert_eq!(std::str::from_utf8(&result.body).unwrap(), expected);
        // Prefix bytes before the compressed span are byte-identical.
        let prefix_len = body.find(&bulk_token).unwrap();
        assert_eq!(
            sha256(&body.as_bytes()[..prefix_len]),
            sha256(&result.body[..prefix_len])
        );
    }

    #[test]
    fn anthropic_user_text_protected_by_default_and_compressed_on_opt_in() {
        let store = SqliteStore::in_memory().unwrap();
        let text = long_text("latest user");
        let body = serde_json::to_vec(&json!({
            "model": "claude-test",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ]
        }))
        .unwrap();

        // Default: user text is protected; nothing to compress; the
        // request is forwarded byte-identical.
        let result = compress_json_request_ctx(
            &body,
            ApiShape::AnthropicMessages,
            ctx(&store, RequestAuthMode::Subscription),
        );
        assert!(!result.compressed);
        assert_eq!(result.body, body);
        assert_eq!(result.skipped_reason.as_deref(), Some("no_live_zone_text"));

        // Opt-in restores user-text compression.
        let mut opt_in = ctx(&store, RequestAuthMode::Subscription);
        opt_in.compress_user_text = true;
        let result = compress_json_request_ctx(&body, ApiShape::AnthropicMessages, opt_in);
        assert!(result.compressed);
        let value: Value = serde_json::from_slice(&result.body).unwrap();
        assert!(value["messages"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn anthropic_skips_hot_zone_cache_control_and_frozen_messages() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t0", "content": long_text("frozen")},
                    {"type": "text", "text": "breakpoint", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": long_text("secret"), "signature": "sig"},
                    {"type": "text", "text": long_text("assistant text")}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_use", "id": "t1", "name": "tool", "input": {}},
                    {"type": "tool_result", "tool_use_id": "t2", "content": long_text("live two")},
                    {"type": "text", "text": long_text("user text stays")}
                ]}
            ]
        }))
        .unwrap();

        let result = compress_json_request_ctx(
            &body,
            ApiShape::AnthropicMessages,
            ctx(&store, RequestAuthMode::Subscription),
        );
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        // Message 0 is below the cache_control floor: untouched.
        assert!(value["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with("frozen"));
        // Assistant text and thinking are protected.
        assert!(value["messages"][1]["content"][1]["text"]
            .as_str()
            .unwrap()
            .starts_with("assistant"));
        // The live tool_result compressed; hot-zone tool_use and the
        // default-protected user text did not.
        assert_eq!(result.hashes.len(), 1);
        assert!(value["messages"][2]["content"][1]["content"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
        assert!(value["messages"][2]["content"][2]["text"]
            .as_str()
            .unwrap()
            .starts_with("user text stays"));
    }

    #[test]
    fn anthropic_session_floor_freezes_prefix_bytes() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t0", "content": long_text("cached prefix")}
                ]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": long_text("live")}
                ]}
            ]
        }))
        .unwrap();

        let mut with_floor = ctx(&store, RequestAuthMode::Subscription);
        with_floor.frozen_message_count = 2;
        let result = compress_json_request_ctx(&body, ApiShape::AnthropicMessages, with_floor);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert!(value["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with("cached prefix"));
        assert!(value["messages"][2]["content"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn openai_chat_compresses_latest_tool_message_and_protects_user_text() {
        let store = SqliteStore::in_memory().unwrap();
        let tool_output = long_text("tool output");
        let tool_token = json_string_token(&tool_output);
        let user_text = long_text("user text");
        let user_token = json_string_token(&user_text);
        let body = format!(
            "{{ \"messages\":[ {{\"role\":\"tool\",\"tool_call_id\":\"c0\",\"content\":\"early\"}},\n\t{{\"role\":\"user\" , \"content\": {user_token}}},\n  {{\"content\": {tool_token},\"tool_call_id\":\"c1\",\"role\":\"tool\"}} ],\"model\":\"gpt-test\"}}"
        );

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(result.compressed);
        assert_eq!(result.hashes.len(), 1);
        let marker_token = json_string_token(&marker_for_hash(&result.hashes[0]));
        let expected = with_top_level_tools_splice(
            &body.replacen(&tool_token, &marker_token, 1),
            &openai_chat_retrieve_tool(),
        );
        assert_eq!(std::str::from_utf8(&result.body).unwrap(), expected);
        // User text stayed verbatim.
        assert!(std::str::from_utf8(&result.body)
            .unwrap()
            .contains(&user_token));
    }

    #[test]
    fn openai_chat_opt_in_compresses_latest_user_text() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "user", "content": long_text("latest user")}
            ]
        }))
        .unwrap();

        let mut opt_in = ctx(&store, RequestAuthMode::Subscription);
        opt_in.compress_user_text = true;
        let result = compress_json_request_ctx(&body, ApiShape::OpenAiChatCompletions, opt_in);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert_eq!(value["messages"][0]["content"], "old");
        assert!(value["messages"][1]["content"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn openai_chat_multi_choice_passthrough_is_byte_identical() {
        let store = SqliteStore::in_memory().unwrap();
        let body = format!(
            "{{\"n\": 2, \"messages\":[{{\"role\":\"tool\",\"tool_call_id\":\"c\",\"content\":{}}}]}}",
            json_string_token(&long_text("tool"))
        );

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );

        assert!(!result.compressed);
        assert_eq!(result.skipped_reason.as_deref(), Some("multi_choice_n"));
        assert_eq!(result.body, body.as_bytes());
    }

    #[test]
    fn responses_compresses_all_output_items_and_preserves_other_bytes() {
        let store = SqliteStore::in_memory().unwrap();
        let out1 = long_text("first output");
        let out2 = long_text("second output");
        let token1 = json_string_token(&out1);
        let token2 = json_string_token(&out2);
        let body = format!(
            "{{\"input\": [\n {{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":{token1}}} ,\n {{\"type\":\"reasoning\",\"encrypted_content\":\"opaque-blob-\\ud83d\\ude00\"}},\n {{\"output\": {token2},\"call_id\":\"c2\",\"type\":\"local_shell_call_output\"}}\n],\"model\":\"gpt-test\"}}"
        );

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiResponses,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(result.compressed);
        assert_eq!(result.hashes.len(), 2);
        let marker1 = json_string_token(&marker_for_hash(&result.hashes[0]));
        let marker2 = json_string_token(&marker_for_hash(&result.hashes[1]));
        let expected = with_top_level_tools_splice(
            &body
                .replacen(&token1, &marker1, 1)
                .replacen(&token2, &marker2, 1),
            &openai_responses_retrieve_tool(),
        );
        assert_eq!(std::str::from_utf8(&result.body).unwrap(), expected);
        // Prefix bytes before the first compressed span are identical.
        let prefix_len = body.find(&token1).unwrap();
        assert_eq!(
            sha256(&body.as_bytes()[..prefix_len]),
            sha256(&result.body[..prefix_len])
        );
    }

    #[test]
    fn responses_string_input_and_message_text_pass_through_byte_identical() {
        let store = SqliteStore::in_memory().unwrap();

        // String `input` is not an items array: passthrough, mirroring
        // the reference `NoMessagesArray` path (`live_zone.rs:2261-2265`).
        let body =
            serde_json::to_vec(&json!({"input": long_text("string input"), "model": "gpt-test"}))
                .unwrap();
        let result = compress_json_request_ctx(
            &body,
            ApiShape::OpenAiResponses,
            ctx(&store, RequestAuthMode::Subscription),
        );
        assert!(!result.compressed);
        assert_eq!(result.skipped_reason.as_deref(), Some("unsupported_input"));
        assert_eq!(result.body, body);

        // Message text inside the items array is never compressed
        // (`live_zone.rs:2293` keeps `latest_message = None`).
        let body = serde_json::to_vec(&json!({
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": long_text("user text")}]}
            ]
        }))
        .unwrap();
        let result = compress_json_request_ctx(
            &body,
            ApiShape::OpenAiResponses,
            ctx(&store, RequestAuthMode::Subscription),
        );
        assert!(!result.compressed);
        assert_eq!(result.body, body);
    }

    #[test]
    fn responses_skips_retrieve_outputs_and_sub_floor_outputs() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "input": [
                {"type": "function_call", "call_id": "r1", "name": "headroom_retrieve",
                 "arguments": "{\"hash\":\"abc\"}"},
                {"type": "function_call_output", "call_id": "r1",
                 "output": long_text("retrieved original")},
                {"type": "function_call_output", "call_id": "c1", "output": "small"}
            ]
        }))
        .unwrap();

        let result = compress_json_request_ctx(
            &body,
            ApiShape::OpenAiResponses,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(!result.compressed);
        assert_eq!(result.body, body);
    }

    #[test]
    fn batch_create_compresses_per_request_and_injects_tool_only_into_compressed_requests() {
        let store = SqliteStore::in_memory().unwrap();
        let bulk = long_text("batch tool result");
        let bulk_token = json_string_token(&bulk);
        let body = format!(
            "{{\"requests\": [\n {{\"custom_id\":\"req_small\",\"params\":{{\"model\":\"claude-test\",\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"tool_result\",\"tool_use_id\":\"a\",\"content\":\"tiny\"}}]}}]}}}},\n {{\"custom_id\":\"req_big\",\"params\":{{\"model\":\"claude-test\",\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"tool_result\",\"tool_use_id\":\"b\",\"content\":{bulk_token}}}]}}]}}}}\n]}}"
        );

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::AnthropicMessageBatches,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(result.compressed);
        assert_eq!(result.hashes.len(), 1);
        let value: Value = serde_json::from_slice(&result.body).unwrap();
        // Only the compressed request receives the retrieve tool.
        assert!(value["requests"][0]["params"].get("tools").is_none());
        let tools = value["requests"][1]["params"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "headroom_retrieve");
        // Everything outside the compressed span and the injected
        // tools field is byte-identical: the output equals the input
        // with the span swapped and the tools splice added before the
        // second params object's closing brace.
        let marker_token = json_string_token(&marker_for_hash(&result.hashes[0]));
        let tool_json = serde_json::to_string(&anthropic_retrieve_tool()).unwrap();
        let expected = body.replacen(&bulk_token, &marker_token, 1).replacen(
            "]}}\n]}",
            &format!("],\"tools\":[{tool_json}]}}}}\n]}}"),
            1,
        );
        assert_eq!(std::str::from_utf8(&result.body).unwrap(), expected);
        // Prefix up to the compressed span must be byte-identical.
        let prefix_len = body.find(&bulk_token).unwrap();
        assert_eq!(
            sha256(&body.as_bytes()[..prefix_len]),
            sha256(&result.body[..prefix_len])
        );
    }

    #[test]
    fn skip_and_noop_requests_forward_byte_identical_for_all_shapes() {
        let store = SqliteStore::in_memory().unwrap();
        let fixtures: Vec<(ApiShape, String)> = vec![
            (
                ApiShape::OpenAiChatCompletions,
                "{ \"messages\" : [ {\"role\":\"user\",\"content\":\"hi \\u00e9🦀\"} ] }"
                    .to_string(),
            ),
            (
                ApiShape::OpenAiResponses,
                "{\n\"input\":[ {\"type\":\"function_call_output\",\"call_id\":\"c\",\"output\":\"small\"} ]\n}".to_string(),
            ),
            (
                ApiShape::AnthropicMessages,
                "{\t\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"tiny\"}]}]}".to_string(),
            ),
            (
                ApiShape::AnthropicMessageBatches,
                "{ \"requests\": [ {\"custom_id\":\"r\",\"params\":{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}} ] }".to_string(),
            ),
        ];

        for (shape, body) in fixtures {
            let result = compress_json_request_ctx(
                body.as_bytes(),
                shape,
                ctx(&store, RequestAuthMode::Subscription),
            );
            assert!(!result.compressed, "shape {shape:?} must not compress");
            assert_eq!(
                result.body,
                body.as_bytes(),
                "shape {shape:?} must forward byte-identical"
            );
        }
    }

    #[test]
    fn retrieve_tool_injection_is_a_pure_splice_when_tools_exist() {
        let store = SqliteStore::in_memory().unwrap();
        let bulk = long_text("tool output");
        let bulk_token = json_string_token(&bulk);
        let body = format!(
            "{{\"messages\":[{{\"role\":\"tool\",\"tool_call_id\":\"c\",\"content\":{bulk_token}}}],\n\"tools\": [ {{\"type\":\"function\",\"function\":{{\"name\":\"existing\"}}}} ]\t}}"
        );

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(result.compressed);
        let marker_token = json_string_token(&marker_for_hash(&result.hashes[0]));
        let tool_json = serde_json::to_string(&openai_chat_retrieve_tool()).unwrap();
        // The splice inserts `,<tool>` immediately before the array's
        // closing bracket; all original bytes (including the interior
        // spacing) are preserved.
        let expected = body.replacen(&bulk_token, &marker_token, 1).replacen(
            "{\"type\":\"function\",\"function\":{\"name\":\"existing\"}} ]",
            &format!(
                "{{\"type\":\"function\",\"function\":{{\"name\":\"existing\"}}}} ,{tool_json}]"
            ),
            1,
        );
        assert_eq!(std::str::from_utf8(&result.body).unwrap(), expected);
    }

    #[test]
    fn payg_metadata_reserializes_and_sorts_tools_like_the_reference() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "zebra", "parameters": {"type": "object", "properties": {"b": {}, "a": {}}}}},
                {"type": "function", "function": {"name": "apple"}}
            ]
        }))
        .unwrap();

        let result = compress_json_request_ctx(
            &body,
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(!result.compressed);
        assert_eq!(
            result.skipped_reason.as_deref(),
            Some("metadata_injected_only")
        );
        assert_eq!(value["tools"][0]["function"]["name"], "apple");
        assert_eq!(value["tools"][1]["function"]["name"], "zebra");
        let properties = value["tools"][1]["function"]["parameters"]["properties"]
            .as_object()
            .unwrap();
        let keys: Vec<_> = properties.keys().collect();
        assert_eq!(keys, vec!["a", "b"]);
        assert!(value.get("prompt_cache_key").is_some());
    }

    #[test]
    fn marker_bearing_tool_arrays_keep_order_but_schemas_still_sort() {
        let store = SqliteStore::in_memory().unwrap();
        // Raw bytes with reverse-alphabetical tool order, an existing
        // cache_control marker, and a schema whose RAW key order is
        // non-canonical: the marker pins the array order, but the
        // schema span is still rewritten to canonical bytes.
        let body = "{\"model\":\"claude-test\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"name\":\"zebra\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"b\":{},\"a\":{}}},\"cache_control\":{\"type\":\"ephemeral\"}},{\"name\":\"apple\",\"input_schema\":{}}]}";

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::AnthropicMessages,
            ctx(&store, RequestAuthMode::Payg),
        );
        let actual = std::str::from_utf8(&result.body).unwrap();
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        // Order preserved (marker present), schema canonicalized on
        // the wire (key order, not just the parsed view).
        assert_eq!(value["tools"][0]["name"], "zebra");
        assert_eq!(value["tools"][1]["name"], "apple");
        assert!(actual.contains("{\"properties\":{\"a\":{},\"b\":{}},\"type\":\"object\"}"));
        assert!(!actual.contains("\"properties\":{\"b\":{},\"a\":{}}"));
        // No extra cache_control auto-placement (marker already set).
        assert!(value["tools"][1].get("cache_control").is_none());
    }

    #[test]
    fn sorted_tool_arrays_keep_per_tool_key_order_like_the_reference() {
        let store = SqliteStore::in_memory().unwrap();
        // Reverse-alphabetical tool order forces the sort path. Each
        // tool's top-level keys are deliberately non-alphabetical: the
        // reference preserves tool key order (IndexMap) and sorts only
        // the schema subtree, so the rewrite must not re-sort tool keys.
        let body = "{\"model\":\"claude-test\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"name\":\"zebra\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"b\":{},\"a\":{}}},\"description\":\"z tool\"},{\"name\":\"apple\",\"input_schema\":{},\"description\":\"a tool\"}]}";

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::AnthropicMessages,
            ctx(&store, RequestAuthMode::Payg),
        );
        let actual = std::str::from_utf8(&result.body).unwrap();

        // Sorted order, original per-tool key order, canonical schema,
        // and cache_control appended to the (sorted) last tool.
        assert!(
            actual.contains("{\"name\":\"apple\",\"input_schema\":{},\"description\":\"a tool\"}")
        );
        assert!(actual.contains(
            "{\"name\":\"zebra\",\"input_schema\":{\"properties\":{\"a\":{},\"b\":{}},\"type\":\"object\"},\"description\":\"z tool\",\"cache_control\":{\"type\":\"ephemeral\"}}"
        ));
        let value: Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(value["tools"][0]["name"], "apple");
        assert_eq!(value["tools"][1]["name"], "zebra");

        // Same invariant for the OpenAI shape (`type` precedes
        // `function` in the raw bytes and must stay that way).
        let body = "{\"model\":\"gpt-test\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"zebra\",\"parameters\":{\"type\":\"object\",\"properties\":{\"b\":{},\"a\":{}}}}},{\"type\":\"function\",\"function\":{\"name\":\"apple\"}}]}";
        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        let actual = std::str::from_utf8(&result.body).unwrap();
        assert!(actual.contains("{\"type\":\"function\",\"function\":{\"name\":\"apple\"}}"));
        assert!(actual.contains(
            "{\"type\":\"function\",\"function\":{\"name\":\"zebra\",\"parameters\":{\"properties\":{\"a\":{},\"b\":{}},\"type\":\"object\"}}}"
        ));
    }

    #[test]
    fn empty_prompt_cache_key_is_replaced_in_place_not_duplicated() {
        let store = SqliteStore::in_memory().unwrap();
        // `prompt_cache_key: ""` is treated as absent but must be
        // REPLACED in place (the reference `map.insert` overwrites it,
        // `openai_cache_key.rs`); a naive append would emit a
        // duplicate top-level key.
        let body = "{\"model\":\"gpt-test\", \"prompt_cache_key\" : \"\" ,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        let actual = std::str::from_utf8(&result.body).unwrap();

        assert_eq!(actual.matches("prompt_cache_key").count(), 1);
        let value: Value = serde_json::from_slice(&result.body).unwrap();
        let key = value["prompt_cache_key"].as_str().unwrap();
        assert_eq!(key.len(), 32);
        assert!(key.bytes().all(|byte| byte.is_ascii_hexdigit()));
        // Everything outside the replaced value span is verbatim
        // (irregular spacing around the field preserved).
        assert!(actual.starts_with("{\"model\":\"gpt-test\", \"prompt_cache_key\" : \""));
        assert!(actual.contains("\" ,\"messages\""));
    }

    #[test]
    fn prompt_cache_key_invariant_under_user_content_variant_under_prefix_changes() {
        let key_of =
            |value: &Value| derive_openai_prompt_cache_key(value, ApiShape::OpenAiChatCompletions);
        let base = json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "stable system"},
                {"role": "user", "content": "turn one"}
            ],
            "tools": [{"type": "function", "function": {"name": "t"}}]
        });

        let mut user_changed = base.clone();
        user_changed["messages"][1]["content"] = json!("turn two");
        assert_eq!(key_of(&base), key_of(&user_changed));

        let mut model_changed = base.clone();
        model_changed["model"] = json!("gpt-other");
        assert_ne!(key_of(&base), key_of(&model_changed));

        let mut system_changed = base.clone();
        system_changed["messages"][0]["content"] = json!("different system");
        assert_ne!(key_of(&base), key_of(&system_changed));

        let mut tools_changed = base.clone();
        tools_changed["tools"] = json!([{"type": "function", "function": {"name": "u"}}]);
        assert_ne!(key_of(&base), key_of(&tools_changed));

        // 32 lowercase hex chars.
        let key = key_of(&base);
        assert_eq!(key.len(), 32);
        assert!(key.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn prompt_cache_key_reflects_injected_retrieve_tool() {
        // Two compressed PAYG requests whose FINAL tools arrays are
        // identical must share a key: request A already carries the
        // retrieve tool (no injection), request B receives it via
        // injection. Deriving the key before injection would key B on
        // the pre-injection tools and break the item-5 invariant
        // ("key varies when tools change").
        let store = SqliteStore::in_memory().unwrap();
        let tool_message = json!({
            "role": "tool", "tool_call_id": "call_bulk", "content": long_text("bulk")
        });
        let body_with_tool = serde_json::to_vec(&json!({
            "model": "gpt-test",
            "messages": [tool_message.clone()],
            "tools": [
                {"type": "function", "function": {"name": "aaa"}},
                openai_chat_retrieve_tool()
            ]
        }))
        .unwrap();
        let body_without_tool = serde_json::to_vec(&json!({
            "model": "gpt-test",
            "messages": [tool_message],
            "tools": [{"type": "function", "function": {"name": "aaa"}}]
        }))
        .unwrap();

        let with_tool = compress_json_request_ctx(
            &body_with_tool,
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        let without_tool = compress_json_request_ctx(
            &body_without_tool,
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        assert!(with_tool.compressed && without_tool.compressed);

        let with_tool: Value = serde_json::from_slice(&with_tool.body).unwrap();
        let without_tool: Value = serde_json::from_slice(&without_tool.body).unwrap();
        // Both ended up with the same final tools array...
        assert_eq!(with_tool["tools"], without_tool["tools"]);
        // ...so they share the same prompt_cache_key.
        assert_eq!(
            with_tool["prompt_cache_key"],
            without_tool["prompt_cache_key"]
        );
    }

    #[test]
    fn responses_frozen_floor_shifts_past_the_instructions_entry() {
        // The tracker floor counts the synthesized walk list
        // ([instructions] + items); with instructions present, a
        // floor of 2 freezes only input item 0.
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "instructions": "be terse",
            "input": [
                {"type": "function_call_output", "call_id": "c0",
                 "output": long_text("cached prefix output")},
                {"type": "function_call_output", "call_id": "c1",
                 "output": long_text("live output")}
            ]
        }))
        .unwrap();

        let mut with_floor = ctx(&store, RequestAuthMode::Subscription);
        with_floor.frozen_message_count = 2;
        let result = compress_json_request_ctx(&body, ApiShape::OpenAiResponses, with_floor);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert!(value["input"][0]["output"]
            .as_str()
            .unwrap()
            .starts_with("cached prefix output"));
        assert!(value["input"][1]["output"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn responses_cache_key_prefers_instructions_then_input_then_messages() {
        let key_of =
            |value: &Value| derive_openai_prompt_cache_key(value, ApiShape::OpenAiResponses);
        let with_instructions = json!({"model": "m", "instructions": "sys"});
        let with_input_system =
            json!({"model": "m", "input": [{"role": "system", "content": "sys"}]});
        let with_messages_system = json!({
            "model": "m",
            "input": [{"role": "user", "content": "u"}],
            "messages": [{"role": "system", "content": "sys"}]
        });
        let with_nothing = json!({"model": "m", "input": [{"role": "user", "content": "u"}]});

        assert_eq!(key_of(&with_instructions), key_of(&with_input_system));
        assert_eq!(key_of(&with_instructions), key_of(&with_messages_system));
        assert_ne!(key_of(&with_instructions), key_of(&with_nothing));
    }

    #[test]
    fn existing_marker_triggers_tool_injection_without_other_byte_changes() {
        let store = SqliteStore::in_memory().unwrap();
        let body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"<<ccr:aaaaaaaaaaaaaaaaaaaaaaaa>>\"}]}]}".to_string();

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::AnthropicMessages,
            ctx(&store, RequestAuthMode::Subscription),
        );

        assert!(!result.compressed);
        assert_eq!(
            result.skipped_reason.as_deref(),
            Some("metadata_injected_only")
        );
        let tool_json = serde_json::to_string(&anthropic_retrieve_tool()).unwrap();
        let expected = format!(
            "{},\"tools\":[{tool_json}]}}",
            body.trim_end().trim_end_matches('}')
        );
        assert_eq!(std::str::from_utf8(&result.body).unwrap(), expected);
    }

    #[test]
    fn responses_frozen_floor_protects_prefix_items() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "input": [
                {"type": "function_call_output", "call_id": "c0",
                 "output": long_text("cached prefix output")},
                {"type": "function_call_output", "call_id": "c1",
                 "output": long_text("live output")}
            ]
        }))
        .unwrap();

        let mut with_floor = ctx(&store, RequestAuthMode::Subscription);
        with_floor.frozen_message_count = 1;
        let result = compress_json_request_ctx(&body, ApiShape::OpenAiResponses, with_floor);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert_eq!(result.hashes.len(), 1);
        assert!(value["input"][0]["output"]
            .as_str()
            .unwrap()
            .starts_with("cached prefix output"));
        assert!(value["input"][1]["output"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn tool_sort_is_stable_with_name_then_function_name_then_digest_key() {
        let store = SqliteStore::in_memory().unwrap();
        // Two same-named tools must keep their relative order (stable
        // sort by key only); unnamed tools fall back to the canonical
        // JSON digest key. Mirrors `tool_def_normalize.rs` `sort_key`.
        let body = serde_json::to_vec(&json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "zeta", "description": "first"}},
                {"type": "function", "function": {"name": "alpha"}},
                {"type": "function", "function": {"name": "zeta", "description": "second"}}
            ]
        }))
        .unwrap();

        let result = compress_json_request_ctx(
            &body,
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        let value: Value = serde_json::from_slice(&result.body).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["function"]["name"], "alpha");
        assert_eq!(tools[1]["function"]["name"], "zeta");
        assert_eq!(tools[1]["function"]["description"], "first");
        assert_eq!(tools[2]["function"]["description"], "second");

        // Unnamed tools get a deterministic digest key: both input
        // permutations converge to the same tool order, and re-running
        // on the normalized output is a byte-level no-op.
        let normalize = |tools: Value| {
            let body = serde_json::to_vec(&json!({
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": tools
            }))
            .unwrap();
            let store = SqliteStore::in_memory().unwrap();
            compress_json_request_ctx(
                &body,
                ApiShape::OpenAiChatCompletions,
                ctx(&store, RequestAuthMode::Payg),
            )
            .body
        };
        let unnamed = json!({"type": "function"});
        let named = json!({"name": "a"});
        let out_a = normalize(json!([unnamed.clone(), named.clone()]));
        let out_b = normalize(json!([named, unnamed]));
        let tools_a: Value = serde_json::from_slice(&out_a).unwrap();
        let tools_b: Value = serde_json::from_slice(&out_b).unwrap();
        assert_eq!(tools_a["tools"], tools_b["tools"]);
    }

    #[test]
    fn schema_canonicalization_splices_bytes_even_without_tool_reorder() {
        let store = SqliteStore::in_memory().unwrap();
        // Tools already in name order; one schema has non-canonical
        // key order and whitespace in the RAW bytes. The schema span
        // must be rewritten to canonical bytes while everything else
        // (including the irregular whitespace outside the schema)
        // stays verbatim.
        let body = "{ \"model\":\"gpt-test\",\n  \"messages\": [ {\"role\":\"user\",\"content\":\"hi\"} ],\n  \"tools\": [ {\"type\":\"function\",\"function\":{\"name\":\"alpha\",\"parameters\":{ \"type\" : \"object\", \"properties\": { \"b\": {}, \"a\": {} } }}} ]\t}";

        let result = compress_json_request_ctx(
            body.as_bytes(),
            ApiShape::OpenAiChatCompletions,
            ctx(&store, RequestAuthMode::Payg),
        );
        let actual = std::str::from_utf8(&result.body).unwrap();

        assert_eq!(
            result.skipped_reason.as_deref(),
            Some("metadata_injected_only")
        );
        // Canonical schema bytes landed on the wire...
        let canonical_schema = "{\"properties\":{\"a\":{},\"b\":{}},\"type\":\"object\"}";
        assert!(
            actual.contains(canonical_schema),
            "schema must be canonicalized in the forwarded bytes: {actual}"
        );
        // ...and the bytes before the schema span are untouched.
        let schema_start = body.find("{ \"type\" :").unwrap();
        assert_eq!(&actual[..schema_start], &body[..schema_start]);
    }

    #[test]
    fn ws_frame_context_skips_provider_metadata_but_keeps_retrieve_injection() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "model": "gpt-test",
            "input": [
                {"type": "function_call_output", "call_id": "c1",
                 "output": long_text("ws frame output")}
            ],
            "tools": [
                {"type": "function", "name": "zebra"},
                {"type": "function", "name": "apple"}
            ]
        }))
        .unwrap();

        let mut ws_ctx = ctx(&store, RequestAuthMode::Payg);
        ws_ctx.provider_metadata = false;
        let result = compress_json_request_ctx(&body, ApiShape::OpenAiResponses, ws_ctx);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        // No prompt_cache_key, no tool reordering...
        assert!(value.get("prompt_cache_key").is_none());
        assert_eq!(value["tools"][0]["name"], "zebra");
        assert_eq!(value["tools"][1]["name"], "apple");
        // ...but the CCR retrieve tool is still appended
        // (`openai.py:1749-1770`).
        assert_eq!(value["tools"][2]["name"], "headroom_retrieve");
    }

    #[test]
    fn token_shrink_gate_rejects_markers_that_do_not_shrink() {
        let store = SqliteStore::in_memory().unwrap();
        // 600 identical bytes with no whitespace: ~150 estimated
        // tokens, marker ~8 — compresses. A pathological short-token
        // string is covered by the floor; emulate the gate directly.
        let slot = PlanSlot {
            range: (0, 2),
            text: "<<ccr".to_string(),
            min_bytes: 0,
            batch_request_index: None,
            label: "test",
        };
        assert!(compress_slot(&slot, &store).is_none());

        let tiny = PlanSlot {
            range: (0, 2),
            // 32 chars => 8 estimated tokens; the marker is also 8 — not smaller.
            text: "abcdefghijklmnopqrstuvwxyzabcdef".to_string(),
            min_bytes: 0,
            batch_request_index: None,
            label: "test",
        };
        assert!(compress_slot(&tiny, &store).is_none());
    }
}

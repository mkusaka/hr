//! Session-scoped proxy state: provider-confirmed prefix-cache
//! tracking and the session-sticky `anthropic-beta` header merge.
//!
//! Mirrors `headroom/cache/prefix_tracker.py` (`PrefixCacheTracker`,
//! `SessionTrackerStore`) and `headroom/proxy/helpers.py`
//! (`SessionBetaTracker`).

use axum::http::HeaderMap;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Minimum provider-confirmed cached tokens before the frozen floor
/// activates (`PrefixFreezeConfig.min_cached_tokens`).
pub const MIN_CACHED_TOKENS: u64 = 1024;
/// Idle TTL after which a session tracker expires
/// (`PrefixFreezeConfig.session_ttl_seconds`).
const SESSION_TTL: Duration = Duration::from_secs(600);
/// Expired-session sweep cadence (`SessionTrackerStore._cleanup_interval`).
const CLEANUP_INTERVAL: Duration = Duration::from_secs(60);
/// Beta-tracker LRU bound (`SessionBetaTracker` default `max_sessions`).
const MAX_BETA_SESSIONS: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionProvider {
    Anthropic,
    OpenAi,
}

#[derive(Debug)]
struct PrefixState {
    cached_token_count: u64,
    cached_message_count: usize,
    turn_number: u64,
    last_activity: Instant,
}

#[derive(Debug, Default)]
struct Inner {
    prefixes: HashMap<(SessionProvider, String), PrefixState>,
    last_cleanup: Option<Instant>,
    /// Per-(provider, session) ordered beta-token lists, LRU-ordered
    /// front-to-back (front = oldest).
    betas: Vec<((SessionProvider, String), Vec<String>)>,
}

/// Shared session state. Cheap to clone (Arc inside).
#[derive(Debug, Clone, Default)]
pub struct SessionTrackers {
    inner: Arc<Mutex<Inner>>,
}

impl SessionTrackers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Derive the session identity for a request, mirroring
    /// `SessionTrackerStore.compute_session_id`: the explicit
    /// `x-headroom-session-id` header wins; otherwise hash
    /// `"{model}:{system_content[:500]}"` where the system content is
    /// the first system-role message's string content (or its first
    /// `text` block). The reference hashes with MD5; hr uses SHA-256
    /// truncated to the same 16 hex chars — the ID never leaves the
    /// proxy, so only stability matters.
    pub fn compute_session_id(headers: &HeaderMap, body: &Value) -> String {
        if let Some(session) = explicit_session_header(headers) {
            return session;
        }
        // Chat / Anthropic shape: the reference walks the body's own
        // `messages` for the first system-role message
        // (`prefix_tracker.py:302-335`).
        let system_content = body
            .get("messages")
            .and_then(Value::as_array)
            .and_then(|messages| first_system_message_text(messages))
            .unwrap_or_default();
        hash_session_id(body, &system_content)
    }

    /// Session id for the `/v1/responses` surface. The reference
    /// synthesizes the session-shaping `messages` from a truthy
    /// STRING `instructions` (and a string `input`) only
    /// (`openai.py:2596-2603`) — list-typed `instructions`, system
    /// items inside the `input` array, and any legacy top-level
    /// `messages` alias are deliberately NOT part of the session
    /// hash. The reference accepts that coarseness, and hr mirrors it
    /// rather than inventing finer session separation.
    pub fn compute_responses_session_id(headers: &HeaderMap, body: &Value) -> String {
        if let Some(session) = explicit_session_header(headers) {
            return session;
        }
        let system_content = match body.get("instructions") {
            Some(Value::String(instructions)) if !instructions.is_empty() => {
                truncate_chars(instructions, 500)
            }
            _ => String::new(),
        };
        hash_session_id(body, &system_content)
    }

    /// The frozen floor for the next request in this session: 0 on
    /// cold start or while confirmed cached tokens stay below
    /// [`MIN_CACHED_TOKENS`], otherwise the message count covered by
    /// the provider-confirmed cached prefix. Mirrors
    /// `PrefixCacheTracker.get_frozen_message_count`.
    pub fn frozen_message_count(&self, provider: SessionProvider, session_id: &str) -> usize {
        let mut inner = self.inner.lock().expect("session tracker lock poisoned");
        Self::maybe_cleanup(&mut inner);
        let Some(state) = inner.prefixes.get_mut(&(provider, session_id.to_string())) else {
            return 0;
        };
        state.last_activity = Instant::now();
        if state.turn_number == 0 || state.cached_token_count < MIN_CACHED_TOKENS {
            return 0;
        }
        state.cached_message_count
    }

    /// Record provider-confirmed cache usage for this session and
    /// recompute the frozen floor: walk the per-message token
    /// estimates from the start, freezing every message that fits
    /// inside `cache_read + cache_write` tokens. Mirrors
    /// `PrefixCacheTracker.update_from_response`.
    pub fn update_from_response(
        &self,
        provider: SessionProvider,
        session_id: &str,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        message_token_estimates: &[u64],
    ) {
        let mut inner = self.inner.lock().expect("session tracker lock poisoned");
        Self::maybe_cleanup(&mut inner);
        let state = inner
            .prefixes
            .entry((provider, session_id.to_string()))
            .or_insert(PrefixState {
                cached_token_count: 0,
                cached_message_count: 0,
                turn_number: 0,
                last_activity: Instant::now(),
            });
        state.last_activity = Instant::now();
        state.turn_number += 1;

        let total_cached = cache_read_tokens.saturating_add(cache_write_tokens);
        if total_cached == 0 {
            state.cached_token_count = 0;
            state.cached_message_count = 0;
            return;
        }

        let mut accumulated = 0u64;
        let mut frozen_count = 0usize;
        for (index, tokens) in message_token_estimates.iter().enumerate() {
            accumulated = accumulated.saturating_add(*tokens);
            if accumulated <= total_cached {
                frozen_count = index + 1;
            } else {
                break;
            }
        }

        state.cached_token_count = total_cached;
        state.cached_message_count = frozen_count;
    }

    /// Union the client's `anthropic-beta` tokens with the tokens
    /// previously seen for this session and return the merged
    /// comma-separated value: stored tokens first (first-seen order),
    /// new client tokens appended, case-insensitive dedupe with
    /// first-seen casing. Sticky-on: tokens are never removed.
    /// Mirrors `SessionBetaTracker.record_and_get_sticky_betas`.
    pub fn sticky_betas(
        &self,
        provider: SessionProvider,
        session_id: &str,
        client_value: Option<&str>,
    ) -> String {
        let client_tokens: Vec<String> = client_value
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        let key = (provider, session_id.to_string());
        let mut inner = self.inner.lock().expect("session tracker lock poisoned");

        let mut merged = match inner.betas.iter().position(|(entry, _)| *entry == key) {
            Some(position) => inner.betas.remove(position).1,
            None => Vec::new(),
        };
        for token in client_tokens {
            if !merged.iter().any(|seen| seen.eq_ignore_ascii_case(&token)) {
                merged.push(token);
            }
        }
        let value = merged.join(",");
        inner.betas.push((key, merged));
        while inner.betas.len() > MAX_BETA_SESSIONS {
            inner.betas.remove(0);
        }
        value
    }

    fn maybe_cleanup(inner: &mut Inner) {
        let now = Instant::now();
        if inner
            .last_cleanup
            .is_some_and(|last| now.duration_since(last) < CLEANUP_INTERVAL)
        {
            return;
        }
        inner
            .prefixes
            .retain(|_, state| now.duration_since(state.last_activity) <= SESSION_TTL);
        inner.last_cleanup = Some(now);
    }
}

fn truncate_chars(input: &str, limit: usize) -> String {
    input.chars().take(limit).collect()
}

fn explicit_session_header(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-headroom-session-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn hash_session_id(body: &Value, system_content: &str) -> String {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");
    let digest = Sha256::digest(format!("{model}:{system_content}").as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Text of the first system-role message in a chat `messages` array:
/// string content directly, or the first `text` block of an
/// array-shaped content — the exact reference rule
/// (`SessionTrackerStore.compute_session_id`,
/// `prefix_tracker.py:317-332`), truncated to 500 chars.
fn first_system_message_text(messages: &[Value]) -> Option<String> {
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("system") {
            continue;
        }
        return match message.get("content") {
            Some(Value::String(content)) => Some(truncate_chars(content, 500)),
            Some(Value::Array(blocks)) => blocks
                .iter()
                .find(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .and_then(|block| block.get("text").and_then(Value::as_str))
                .map(|text| truncate_chars(text, 500)),
            _ => None,
        };
    }
    None
}

/// Per-message token estimates used to map provider-confirmed cached
/// tokens back onto a message-count floor. Mirrors
/// `PrefixCacheTracker._estimate_message_tokens` (chars / 3.5 with a
/// 20-char structural overhead; counts text, tool_result content and
/// tool_use input).
pub fn estimate_message_tokens(messages: &[Value]) -> Vec<u64> {
    messages
        .iter()
        .map(|message| {
            let mut chars = 0usize;
            match message.get("content") {
                Some(Value::String(content)) => chars += content.chars().count(),
                Some(Value::Array(blocks)) => {
                    for block in blocks {
                        let Some(block) = block.as_object() else {
                            continue;
                        };
                        match block.get("type").and_then(Value::as_str).unwrap_or("") {
                            "text" => {
                                chars += block
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .map(|text| text.chars().count())
                                    .unwrap_or(0);
                            }
                            "tool_result" => match block.get("content") {
                                Some(Value::String(inner)) => chars += inner.chars().count(),
                                Some(Value::Array(parts)) => {
                                    chars += parts
                                        .iter()
                                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                                        .map(|text| text.chars().count())
                                        .sum::<usize>();
                                }
                                _ => {}
                            },
                            "tool_use" => match block.get("input") {
                                Some(Value::String(input)) => chars += input.chars().count(),
                                Some(input @ Value::Object(_)) => {
                                    chars += serde_json::to_string(input)
                                        .map(|json| json.chars().count())
                                        .unwrap_or(0);
                                }
                                _ => {}
                            },
                            _ => {
                                chars += block
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .map(|text| text.chars().count())
                                    .unwrap_or(0);
                            }
                        }
                    }
                }
                _ => {}
            }
            chars += 20;
            ((chars as f64) / 3.5).floor().max(1.0) as u64
        })
        .collect()
}

/// Token estimates for the walk list of an OpenAI Responses request,
/// mirroring the reference's synthesized messages
/// (`openai.py:2596-2603`: `instructions` becomes a leading system
/// entry, a string `input` becomes a user entry) followed by the
/// per-item estimates of an array-shaped `input`. The leading
/// `instructions` entry keeps the floor walk aligned with what the
/// provider actually cached; [`responses_estimate_prefix_len`]
/// reports how many leading entries are NOT input items.
pub fn estimate_responses_request_tokens(body: &Value) -> Vec<u64> {
    let mut estimates = Vec::new();
    // The reference adds the synthetic system entry only when
    // `instructions` is truthy (`openai.py:2599-2601`: empty strings
    // and empty arrays add nothing) and counts it with the normal
    // message estimator (`prefix_tracker.py:230`).
    match body.get("instructions") {
        Some(Value::String(instructions)) if !instructions.is_empty() => {
            estimates.push(text_token_estimate(instructions.chars().count()));
        }
        Some(instructions @ Value::Array(items)) if !items.is_empty() => {
            let synthetic = serde_json::json!({"role": "system", "content": instructions});
            estimates.extend(estimate_message_tokens(std::slice::from_ref(&synthetic)));
        }
        _ => {}
    }
    match body.get("input") {
        Some(Value::String(input)) => {
            estimates.push(text_token_estimate(input.chars().count()));
        }
        Some(Value::Array(items)) => {
            estimates.extend(estimate_response_item_tokens(items));
        }
        // The live zone accepts a legacy `messages` alias for
        // Responses items (`live_zone.rs:2249`); estimate those with
        // the item estimator too so the floor walk sees
        // `function_call_output.output` payloads.
        _ => {
            if let Some(Value::Array(items)) = body.get("messages") {
                estimates.extend(estimate_response_item_tokens(items));
            }
        }
    }
    estimates
}

/// Number of leading entries in [`estimate_responses_request_tokens`]
/// output that do not correspond to `input` array items (today: the
/// synthesized `instructions` entry). The frozen floor must be
/// shifted by this amount before it is applied to item indices.
pub fn responses_estimate_prefix_len(body: &Value) -> usize {
    usize::from(match body.get("instructions") {
        Some(Value::String(instructions)) => !instructions.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        _ => false,
    })
}

fn text_token_estimate(chars: usize) -> u64 {
    (((chars + 20) as f64) / 3.5).floor().max(1.0) as u64
}

/// Per-item token estimates for an OpenAI Responses `input` array,
/// used to map provider-confirmed cached tokens back onto an
/// item-count floor for the Responses surface. Same shape as
/// [`estimate_message_tokens`]: chars / 3.5 with a 20-char structural
/// overhead, counting the string payload fields the wire shape
/// carries (`output`, `text`, `arguments`, string/array `content`).
pub fn estimate_response_item_tokens(items: &[Value]) -> Vec<u64> {
    items
        .iter()
        .map(|item| {
            let mut chars = 0usize;
            for field in ["output", "text", "arguments"] {
                if let Some(text) = item.get(field).and_then(Value::as_str) {
                    chars += text.chars().count();
                }
            }
            match item.get("content") {
                Some(Value::String(content)) => chars += content.chars().count(),
                Some(Value::Array(parts)) => {
                    chars += parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .map(|text| text.chars().count())
                        .sum::<usize>();
                }
                _ => {}
            }
            chars += 20;
            ((chars as f64) / 3.5).floor().max(1.0) as u64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_id_prefers_header_then_model_and_system_hash() {
        let mut headers = HeaderMap::new();
        headers.insert("x-headroom-session-id", "explicit".parse().unwrap());
        let body = json!({"model": "m", "messages": []});
        assert_eq!(
            SessionTrackers::compute_session_id(&headers, &body),
            "explicit"
        );

        let headers = HeaderMap::new();
        let a = SessionTrackers::compute_session_id(
            &headers,
            &json!({"model": "m", "messages": [{"role": "system", "content": "sys"}]}),
        );
        let b = SessionTrackers::compute_session_id(
            &headers,
            &json!({"model": "m", "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": "different user turn"}
            ]}),
        );
        assert_eq!(a, b, "same model + system prefix must share a session");
        assert_eq!(a.len(), 16);

        let c = SessionTrackers::compute_session_id(
            &headers,
            &json!({"model": "other", "messages": [{"role": "system", "content": "sys"}]}),
        );
        assert_ne!(a, c);
    }

    #[test]
    fn responses_session_hash_uses_string_instructions_only() {
        // Mirrors the reference exactly (`openai.py:2596-2603` +
        // `prefix_tracker.py:302-335`): only a truthy STRING
        // `instructions` joins the session hash on the Responses
        // surface; list-typed `instructions`, system items inside
        // `input`, and any legacy top-level `messages` alias are NOT
        // hashed (the reference accepts that coarseness).
        let headers = HeaderMap::new();
        let no_system =
            SessionTrackers::compute_responses_session_id(&headers, &json!({"model": "m"}));

        let string_a = SessionTrackers::compute_responses_session_id(
            &headers,
            &json!({"model": "m", "instructions": "persona A"}),
        );
        let string_b = SessionTrackers::compute_responses_session_id(
            &headers,
            &json!({"model": "m", "instructions": "persona B"}),
        );
        assert_ne!(string_a, string_b);
        assert_ne!(string_a, no_system);

        // Array-shaped instructions collapse to the no-system session,
        // exactly like the reference synthesis (which only handles
        // `isinstance(instructions, str)` shapes).
        let array_a = SessionTrackers::compute_responses_session_id(
            &headers,
            &json!({"model": "m", "instructions": [
                {"type": "message", "role": "system",
                 "content": [{"type": "input_text", "text": "persona A"}]}
            ]}),
        );
        assert_eq!(array_a, no_system);

        // System items inside `input` are likewise not hashed.
        let input_system = SessionTrackers::compute_responses_session_id(
            &headers,
            &json!({"model": "m", "input": [
                {"type": "message", "role": "system",
                 "content": [{"type": "input_text", "text": "persona A"}]},
                {"type": "message", "role": "user", "content": "hi"}
            ]}),
        );
        assert_eq!(input_system, no_system);

        // A legacy `messages` alias is a Responses item container,
        // not a chat system source — it never joins the hash.
        let messages_alias = SessionTrackers::compute_responses_session_id(
            &headers,
            &json!({"model": "m",
                    "messages": [{"role": "system", "content": "other"}]}),
        );
        assert_eq!(messages_alias, no_system);

        // The chat derivation still walks `messages`, so the two
        // surfaces hash the same body differently.
        let chat_messages = SessionTrackers::compute_session_id(
            &headers,
            &json!({"model": "m",
                    "messages": [{"role": "system", "content": "other"}]}),
        );
        assert_ne!(chat_messages, messages_alias);
    }

    #[test]
    fn responses_request_estimates_cover_instructions_and_string_input() {
        // Array input with instructions: a leading instructions entry
        // plus per-item estimates, with the prefix length reported so
        // the floor can be shifted back to item indices.
        let body = json!({
            "model": "m",
            "instructions": "i".repeat(330),
            "input": [
                {"type": "function_call_output", "call_id": "c", "output": "o".repeat(120)}
            ]
        });
        let estimates = estimate_responses_request_tokens(&body);
        assert_eq!(estimates.len(), 2);
        assert_eq!(estimates[0], ((330 + 20) as f64 / 3.5) as u64);
        assert_eq!(estimates[1], ((120 + 20) as f64 / 3.5) as u64);
        assert_eq!(responses_estimate_prefix_len(&body), 1);

        // String input: synthesized as one user entry; no prefix when
        // instructions are absent.
        let body = json!({"model": "m", "input": "u".repeat(150)});
        let estimates = estimate_responses_request_tokens(&body);
        assert_eq!(estimates, vec![((150 + 20) as f64 / 3.5) as u64]);
        assert_eq!(responses_estimate_prefix_len(&body), 0);

        // Falsy instructions add no synthetic entry, mirroring the
        // reference truthiness check (`openai.py:2599-2601`).
        for falsy in [json!(""), json!([])] {
            let body = json!({"model": "m", "instructions": falsy, "input": [
                {"type": "function_call_output", "call_id": "c", "output": "o".repeat(120)}
            ]});
            let estimates = estimate_responses_request_tokens(&body);
            assert_eq!(estimates, vec![((120 + 20) as f64 / 3.5) as u64]);
            assert_eq!(responses_estimate_prefix_len(&body), 0);
        }

        // A legacy `messages` alias holds Responses ITEMS and is
        // counted with the item estimator (`live_zone.rs:2249`
        // accepts the alias), so `function_call_output.output`
        // payloads still feed the floor walk.
        let body = json!({"model": "m", "messages": [
            {"type": "function_call_output", "call_id": "c", "output": "o".repeat(330)}
        ]});
        let estimates = estimate_responses_request_tokens(&body);
        assert_eq!(estimates, vec![((330 + 20) as f64 / 3.5) as u64]);
        assert_eq!(responses_estimate_prefix_len(&body), 0);

        // Non-empty array instructions are counted with the normal
        // message estimator over the synthetic system message (each
        // top-level item is a content block; nested message items
        // carry no direct `text`, so only the structural overhead
        // counts — exactly like `_estimate_message_tokens`).
        let body = json!({"model": "m", "instructions": [
            {"type": "message", "role": "system",
             "content": [{"type": "input_text", "text": "x".repeat(700)}]}
        ]});
        let estimates = estimate_responses_request_tokens(&body);
        assert_eq!(estimates.len(), 1);
        assert_eq!(estimates[0], (20.0_f64 / 3.5) as u64);
        assert_eq!(responses_estimate_prefix_len(&body), 1);
    }

    #[test]
    fn frozen_floor_requires_min_cached_tokens_and_walks_estimates() {
        let trackers = SessionTrackers::new();
        // Cold start: no floor.
        assert_eq!(
            trackers.frozen_message_count(SessionProvider::Anthropic, "s"),
            0
        );

        // Below the 1024-token activation threshold: still no floor.
        trackers.update_from_response(SessionProvider::Anthropic, "s", 100, 0, &[50, 60, 70]);
        assert_eq!(
            trackers.frozen_message_count(SessionProvider::Anthropic, "s"),
            0
        );

        // 1500 cached tokens over estimates [1000, 400, 700]: the
        // first two messages fit (1400 <= 1500); the third does not.
        trackers.update_from_response(
            SessionProvider::Anthropic,
            "s",
            1200,
            300,
            &[1000, 400, 700],
        );
        assert_eq!(
            trackers.frozen_message_count(SessionProvider::Anthropic, "s"),
            2
        );

        // Providers are independent.
        assert_eq!(
            trackers.frozen_message_count(SessionProvider::OpenAi, "s"),
            0
        );

        // A zero-cache response resets the floor.
        trackers.update_from_response(SessionProvider::Anthropic, "s", 0, 0, &[1000, 400]);
        assert_eq!(
            trackers.frozen_message_count(SessionProvider::Anthropic, "s"),
            0
        );
    }

    #[test]
    fn sticky_betas_merge_dedupe_and_persist() {
        let trackers = SessionTrackers::new();
        assert_eq!(
            trackers.sticky_betas(
                SessionProvider::Anthropic,
                "s",
                Some("interleaved-thinking-2025-05-14")
            ),
            "interleaved-thinking-2025-05-14"
        );
        // Later request without the header still gets the sticky token;
        // a new token appends; case-insensitive dedupe keeps the
        // first-seen casing.
        assert_eq!(
            trackers.sticky_betas(
                SessionProvider::Anthropic,
                "s",
                Some("INTERLEAVED-THINKING-2025-05-14, context-1m-2025-08-07")
            ),
            "interleaved-thinking-2025-05-14,context-1m-2025-08-07"
        );
        assert_eq!(
            trackers.sticky_betas(SessionProvider::Anthropic, "s", None),
            "interleaved-thinking-2025-05-14,context-1m-2025-08-07"
        );
        // Sessions are independent.
        assert_eq!(
            trackers.sticky_betas(SessionProvider::Anthropic, "other", None),
            ""
        );
    }

    #[test]
    fn message_token_estimates_mirror_reference_shapes() {
        let messages = vec![
            json!({"role": "user", "content": "x".repeat(330)}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "y".repeat(50)},
                {"type": "tool_result", "tool_use_id": "t", "content": "z".repeat(120)},
                {"type": "tool_use", "id": "u", "name": "n", "input": {"k": "v"}}
            ]}),
            json!({"role": "assistant"}),
        ];
        let estimates = estimate_message_tokens(&messages);
        assert_eq!(estimates.len(), 3);
        assert_eq!(estimates[0], ((330 + 20) as f64 / 3.5) as u64);
        // 50 + 120 + len("{\"k\":\"v\"}")=9 + 20 overhead.
        assert_eq!(estimates[1], ((50 + 120 + 9 + 20) as f64 / 3.5) as u64);
        assert_eq!(estimates[2], (20.0_f64 / 3.5) as u64);
    }
}

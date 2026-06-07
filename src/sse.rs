//! Byte-level SSE framing and the off-byte-path telemetry state
//! machines.
//!
//! The framer mirrors `crates/headroom-proxy/src/sse/framing.rs`:
//! bytes accumulate until a blank-line terminator, UTF-8 is decoded
//! once per complete event (so multi-byte codepoints split across
//! chunks rejoin correctly), comment lines are dropped, multiple
//! `data:` lines join with `\n` and `data: [DONE]` is the OpenAI
//! end-of-stream sentinel.
//!
//! The state machines run in a spawned task fed through a bounded
//! mpsc tee (`crates/headroom-proxy/src/proxy.rs`
//! `SSE_PARSER_QUEUE_DEPTH` / `run_sse_state_machine`): the client
//! byte path never waits on parsing, and a saturated queue drops
//! telemetry chunks, never client bytes.

use crate::session::{SessionProvider, SessionTrackers};
use crate::stats;
use bytes::Bytes;
use serde_json::Value;
use tracing::{info, trace};

/// Bound on the in-flight queue between the byte passthrough and the
/// SSE state-machine task. Mirrors `headroom-proxy/src/proxy.rs`
/// `SSE_PARSER_QUEUE_DEPTH`.
pub const SSE_PARSER_QUEUE_DEPTH: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseKind {
    OpenAiChat,
    OpenAiResponses,
    Anthropic,
}

impl SseKind {
    /// Provider label for the per-session cache-hit-rate metric,
    /// mirroring `observability/cache_hit_rate.rs` `provider`.
    fn provider_label(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAiChat => "openai_chat",
            Self::OpenAiResponses => "openai_responses",
        }
    }
}

/// One framed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event_name: Option<String>,
    pub data: Vec<u8>,
}

impl SseEvent {
    pub fn data_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }

    /// True iff the data field is the literal `[DONE]` sentinel.
    pub fn is_done_sentinel(&self) -> bool {
        self.data == b"[DONE]"
    }
}

/// Stateful byte-level SSE framer.
#[derive(Debug, Default)]
pub struct SseFramer {
    buf: Vec<u8>,
    done_seen: bool,
}

impl SseFramer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn done_seen(&self) -> bool {
        self.done_seen
    }

    /// Append inbound bytes. Chunks may straddle event boundaries,
    /// line boundaries, or multi-byte UTF-8 codepoints.
    pub fn push(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        self.buf.extend_from_slice(chunk);
    }

    /// Number of buffered (un-framed) bytes.
    pub fn buffered_len(&self) -> usize {
        self.buf.len()
    }

    /// Drain the next complete event, skipping comment-only and empty
    /// blocks. Returns `None` until a blank-line terminator arrives.
    pub fn next_event(&mut self) -> Option<SseEvent> {
        loop {
            let (end, term_len) = find_double_newline(&self.buf)?;
            let block: Vec<u8> = self.buf.drain(..end + term_len).take(end).collect();
            match parse_event_block(&block) {
                Some(event) => {
                    if event.is_done_sentinel() {
                        self.done_seen = true;
                    }
                    return Some(event);
                }
                // Comment-only / empty block (e.g. `: ping`). Keep
                // consuming.
                None => continue,
            }
        }
    }
}

/// Find the first `\n\n` or `\r\n\r\n` terminator, returning
/// `(block_end, terminator_len)`.
fn find_double_newline(buf: &[u8]) -> Option<(usize, usize)> {
    let mut index = 0;
    while index + 1 < buf.len() {
        if buf[index] == b'\n' && buf[index + 1] == b'\n' {
            return Some((index, 2));
        }
        if index + 3 < buf.len()
            && buf[index] == b'\r'
            && buf[index + 1] == b'\n'
            && buf[index + 2] == b'\r'
            && buf[index + 3] == b'\n'
        {
            return Some((index, 4));
        }
        index += 1;
    }
    None
}

/// Parse one event block (bytes between terminators). Returns `None`
/// when the block has no `data:` lines.
fn parse_event_block(block: &[u8]) -> Option<SseEvent> {
    let mut event_name: Option<String> = None;
    let mut data_parts: Vec<&[u8]> = Vec::new();

    for line in block.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        if line[0] == b':' {
            // Comment line (`: ping` keepalives). Dropped silently.
            continue;
        }
        let (field, value) = match line.iter().position(|byte| *byte == b':') {
            Some(position) => (&line[..position], &line[position + 1..]),
            None => (line, &line[line.len()..]),
        };
        let value = value.strip_prefix(b" ").unwrap_or(value);
        match field {
            b"event" => {
                // The SSE spec requires ASCII event names; tolerate
                // (and skip) anything else.
                if let Ok(name) = std::str::from_utf8(value) {
                    event_name = Some(name.to_string());
                }
            }
            b"data" => data_parts.push(value),
            _ => continue,
        }
    }

    if data_parts.is_empty() {
        return None;
    }
    let data = data_parts.join(&b"\n"[..]);
    Some(SseEvent { event_name, data })
}

/// Session context used to feed provider-confirmed cache usage back
/// into the prefix tracker when the stream reports it.
#[derive(Debug, Clone)]
pub struct SseSessionCtx {
    pub trackers: SessionTrackers,
    pub provider: SessionProvider,
    pub session_id: String,
    /// Per-message token estimates of the forwarded request messages.
    pub message_token_estimates: Vec<u64>,
}

/// Drive the per-provider state machine over a stream of byte chunks.
/// Lives in its own task; the byte path never waits on it.
pub async fn run_sse_state_machine(
    kind: SseKind,
    mut rx: tokio::sync::mpsc::Receiver<Bytes>,
    session: Option<SseSessionCtx>,
) {
    let mut framer = SseFramer::new();
    let mut state = SseStreamState::new(kind, session);
    while let Some(chunk) = rx.recv().await {
        framer.push(&chunk);
        while let Some(event) = framer.next_event() {
            state.apply(&event);
        }
    }
    state.finish();
}

/// Per-provider SSE telemetry state machine. Usage and cache-hit-rate
/// samples are emitted only on terminal events (Chat final usage
/// chunk, Responses `response.completed|failed|incomplete`, Anthropic
/// `message_stop`); the Responses terminal-status counter is bumped
/// once per stream with the last status seen.
#[derive(Debug)]
pub struct SseStreamState {
    kind: SseKind,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
    emitted: bool,
    responses_last_status: Option<&'static str>,
    /// Accumulated character count of the assistant message being
    /// streamed (Anthropic only): the reference appends the parsed
    /// assistant message to the forwarded messages before updating the
    /// prefix tracker (`headroom/proxy/handlers/streaming.py:738-748`).
    assistant_chars: usize,
    session: Option<SseSessionCtx>,
}

impl SseStreamState {
    pub fn new(kind: SseKind, session: Option<SseSessionCtx>) -> Self {
        Self {
            kind,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            emitted: false,
            responses_last_status: None,
            assistant_chars: 0,
            session,
        }
    }

    pub fn apply(&mut self, event: &SseEvent) {
        if event.is_done_sentinel() {
            return;
        }
        let Some(data) = event.data_str() else {
            return;
        };
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return;
        };

        // Streaming CCR parity with the reference proxy: retrieval
        // tool calls in SSE responses are resolved by the client
        // (`headroom/proxy/handlers/streaming.py:1208-1218,
        // 1478-1484`); the proxy only records them for observability.
        if ccr_stream_tool_call_detected(self.kind, &value) {
            stats::record_ccr_stream_tool_call();
            info!(
                provider = ?self.kind,
                "ccr retrieval tool call detected in SSE stream (client-resolved)"
            );
        }

        match self.kind {
            SseKind::OpenAiChat => {
                // OpenAI emits a final usage chunk only with
                // `stream_options.include_usage`; intermediate chunks
                // carry `usage: null`, which must not trip emission.
                if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
                    self.absorb_usage(usage);
                    self.emit_final();
                }
            }
            SseKind::OpenAiResponses => {
                let kind = value
                    .get("type")
                    .and_then(Value::as_str)
                    .or(event.event_name.as_deref())
                    .unwrap_or("");
                // Terminal statuses mirror the reference state machine
                // (`sse/openai_responses.rs` `terminal_status`): only
                // `completed` / `failed` / `incomplete` are emitted.
                let status = match kind {
                    "response.completed" => Some(response_status::COMPLETED),
                    "response.failed" => Some(response_status::FAILED),
                    "response.incomplete" => Some(response_status::INCOMPLETE),
                    _ => None,
                };
                if let Some(status) = status {
                    self.responses_last_status = Some(status);
                    if let Some(usage) = value
                        .get("usage")
                        .or_else(|| {
                            value
                                .get("response")
                                .and_then(|response| response.get("usage"))
                        })
                        .filter(|usage| usage.is_object())
                    {
                        self.absorb_usage(usage);
                    }
                    if let Some(tier) = value
                        .get("response")
                        .and_then(|response| response.get("service_tier"))
                        .or_else(|| value.get("service_tier"))
                        .and_then(Value::as_str)
                    {
                        stats::record_service_tier(service_tier::validate(tier));
                    }
                    self.emit_final();
                }
            }
            SseKind::Anthropic => {
                let kind = value
                    .get("type")
                    .and_then(Value::as_str)
                    .or(event.event_name.as_deref());
                match kind {
                    Some("message_start") => {
                        if let Some(usage) = value
                            .get("message")
                            .and_then(|message| message.get("usage"))
                            .or_else(|| value.get("usage"))
                            .filter(|usage| usage.is_object())
                        {
                            self.absorb_usage(usage);
                        }
                    }
                    Some("message_delta") => {
                        if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
                            self.absorb_usage(usage);
                        }
                    }
                    Some("content_block_delta") => {
                        if let Some(delta) = value.get("delta") {
                            for field in ["text", "partial_json", "thinking"] {
                                if let Some(text) = delta.get(field).and_then(Value::as_str) {
                                    self.assistant_chars += text.chars().count();
                                }
                            }
                        }
                    }
                    Some("message_stop") => self.emit_final(),
                    _ => {}
                }
            }
        }
    }

    /// Stream end: bump the Responses terminal-status counter when (and
    /// only when) a terminal status was observed, mirroring the
    /// reference emit site (`headroom-proxy/src/proxy.rs:1545-1556`,
    /// gated on `terminal_status().is_some()`). Streams that close
    /// mid-flight record nothing.
    pub fn finish(&mut self) {
        if self.kind == SseKind::OpenAiResponses {
            if let Some(status) = self.responses_last_status {
                stats::record_response_status(status);
            }
        }
    }

    fn absorb_usage(&mut self, usage: &Value) {
        let input = usage_u64(usage, &["prompt_tokens", "input_tokens"]);
        let output = usage_u64(usage, &["completion_tokens", "output_tokens"]);
        let details = usage
            .get("prompt_tokens_details")
            .or_else(|| usage.get("input_tokens_details"));
        let cache_read = details
            .and_then(|details| usage_u64_opt(details, &["cached_tokens"]))
            .or_else(|| usage_u64_opt(usage, &["cache_read_input_tokens"]))
            .unwrap_or_default();
        let cache_creation = usage_u64(usage, &["cache_creation_input_tokens"]);

        self.input_tokens = self.input_tokens.max(input);
        self.output_tokens = self.output_tokens.max(output);
        self.cache_read_input_tokens = self.cache_read_input_tokens.max(cache_read);
        self.cache_creation_input_tokens = self.cache_creation_input_tokens.max(cache_creation);
    }

    fn emit_final(&mut self) {
        if self.emitted {
            return;
        }
        self.emitted = true;
        if self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_read_input_tokens == 0
            && self.cache_creation_input_tokens == 0
        {
            return;
        }

        stats::record_sse_usage(
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        );

        // OpenAI reports cache reads but no write counter; the
        // uncached input portion is the best write-volume proxy
        // (`headroom/proxy/handlers/openai.py:334-344`). When the
        // upstream reports an explicit creation counter, it wins
        // (`openai.py:2097-2110`).
        let cache_write = match self.kind {
            SseKind::OpenAiChat | SseKind::OpenAiResponses => {
                if self.cache_creation_input_tokens > 0 {
                    self.cache_creation_input_tokens
                } else {
                    let inferred = self
                        .input_tokens
                        .saturating_sub(self.cache_read_input_tokens);
                    stats::record_inferred_cache_write_tokens(inferred);
                    inferred
                }
            }
            SseKind::Anthropic => self.cache_creation_input_tokens,
        };

        let denominator = match self.kind {
            SseKind::Anthropic => self
                .input_tokens
                .saturating_add(self.cache_read_input_tokens)
                .saturating_add(self.cache_creation_input_tokens),
            // OpenAI `prompt_tokens` / `input_tokens` already include
            // the cached portion.
            SseKind::OpenAiChat | SseKind::OpenAiResponses => self.input_tokens,
        };
        if denominator > 0 && self.cache_read_input_tokens <= denominator {
            stats::record_sse_cache_hit_rate(
                self.kind.provider_label(),
                self.cache_read_input_tokens as f64 / denominator as f64,
            );
        }

        // Feed the provider-confirmed cached prefix back into the
        // session tracker so the next request freezes it
        // (`headroom/proxy/handlers/streaming.py:720-755`). For
        // Anthropic the streamed assistant message is appended to the
        // walked estimates, mirroring `streaming.py:738-748`.
        if let Some(session) = &self.session {
            let mut estimates = session.message_token_estimates.clone();
            if self.kind == SseKind::Anthropic && self.assistant_chars > 0 {
                estimates.push(
                    ((self.assistant_chars as f64 + 20.0) / 3.5)
                        .floor()
                        .max(1.0) as u64,
                );
            }
            session.trackers.update_from_response(
                session.provider,
                &session.session_id,
                self.cache_read_input_tokens,
                cache_write,
                &estimates,
            );
        }
        trace!(kind = ?self.kind, "sse terminal usage recorded");
    }
}

/// Detect a `headroom_retrieve` tool call inside one SSE event without
/// mutating the stream.
pub fn ccr_stream_tool_call_detected(kind: SseKind, value: &Value) -> bool {
    match kind {
        SseKind::Anthropic => {
            value.get("type").and_then(Value::as_str) == Some("content_block_start")
                && value.get("content_block").is_some_and(|block| {
                    block.get("type").and_then(Value::as_str) == Some("tool_use")
                        && block
                            .get("name")
                            .and_then(Value::as_str)
                            .is_some_and(is_retrieve_tool_name)
                })
        }
        SseKind::OpenAiChat => {
            value
                .get("choices")
                .and_then(Value::as_array)
                .is_some_and(|choices| {
                    choices.iter().any(|choice| {
                        choice
                            .get("delta")
                            .and_then(|delta| delta.get("tool_calls"))
                            .and_then(Value::as_array)
                            .is_some_and(|calls| {
                                calls.iter().any(|call| {
                                    call.get("function")
                                        .and_then(|function| function.get("name"))
                                        .and_then(Value::as_str)
                                        .is_some_and(is_retrieve_tool_name)
                                })
                            })
                    })
                })
        }
        SseKind::OpenAiResponses => {
            value.get("type").and_then(Value::as_str) == Some("response.output_item.added")
                && value.get("item").is_some_and(|item| {
                    item.get("type").and_then(Value::as_str) == Some("function_call")
                        && item
                            .get("name")
                            .and_then(Value::as_str)
                            .is_some_and(is_retrieve_tool_name)
                })
        }
    }
}

fn is_retrieve_tool_name(name: &str) -> bool {
    name == "headroom_retrieve" || name.ends_with("__headroom_retrieve")
}

pub(crate) fn usage_u64(value: &Value, keys: &[&str]) -> u64 {
    usage_u64_opt(value, keys).unwrap_or_default()
}

pub(crate) fn usage_u64_opt(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_u64))
}

/// Bounded OpenAI Responses `service_tier` label vocabulary, mirroring
/// `metric_names.rs::service_tier` (unknown values bucket to `other`
/// so arbitrary inbound JSON cannot blow up label cardinality).
pub mod service_tier {
    pub const AUTO: &str = "auto";
    pub const DEFAULT: &str = "default";
    pub const FLEX: &str = "flex";
    pub const ON_DEMAND: &str = "on_demand";
    pub const PRIORITY: &str = "priority";
    pub const SCALE: &str = "scale";
    pub const OTHER: &str = "other";

    pub fn validate(raw: &str) -> &'static str {
        match raw {
            AUTO => AUTO,
            DEFAULT => DEFAULT,
            FLEX => FLEX,
            ON_DEMAND => ON_DEMAND,
            PRIORITY => PRIORITY,
            SCALE => SCALE,
            _ => {
                tracing::warn!(
                    raw = %raw,
                    bucket = OTHER,
                    "unknown service_tier value bucketed to 'other' to bound cardinality"
                );
                OTHER
            }
        }
    }
}

/// OpenAI Responses terminal-status vocabulary, mirroring
/// `metric_names.rs::response_status`.
pub mod response_status {
    pub const COMPLETED: &str = "completed";
    pub const INCOMPLETE: &str = "incomplete";
    pub const FAILED: &str = "failed";
    pub const CANCELLED: &str = "cancelled";
    pub const IN_PROGRESS: &str = "in_progress";
}

/// Extract upstream rate-limit headers, accepting both the Anthropic
/// (`anthropic-ratelimit-*`) and OpenAI (`x-ratelimit-*`) families.
/// Mirrors `observability/proxy_metrics.rs`
/// `extract_rate_limit_snapshot`.
pub fn extract_rate_limit_snapshot(headers: &axum::http::HeaderMap) -> stats::RateLimitSnapshot {
    let parse_i64 = |name: &str| -> Option<i64> {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<i64>().ok())
    };
    stats::RateLimitSnapshot {
        remaining_requests: parse_i64("anthropic-ratelimit-requests-remaining")
            .or_else(|| parse_i64("x-ratelimit-remaining-requests")),
        remaining_tokens: parse_i64("anthropic-ratelimit-tokens-remaining")
            .or_else(|| parse_i64("x-ratelimit-remaining-tokens")),
        remaining_input_tokens: parse_i64("anthropic-ratelimit-input-tokens-remaining"),
        remaining_output_tokens: parse_i64("anthropic-ratelimit-output-tokens-remaining"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn feed(framer: &mut SseFramer, chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            framer.push(chunk);
            while let Some(event) = framer.next_event() {
                events.push(event);
            }
        }
        events
    }

    #[test]
    fn framer_joins_split_utf8_and_multi_data_lines() {
        let mut framer = SseFramer::new();
        // 🦀 = F0 9F A6 80, split mid-codepoint across chunks.
        let events = feed(
            &mut framer,
            &[
                b"event: message_start\ndata: {\"emoji\":\"\xf0\x9f",
                b"\xa6\x80\"}\n\ndata: a\nda",
                b"ta: b\n\n: ping\n\ndata: [DONE]\n\n",
            ],
        );
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_name.as_deref(), Some("message_start"));
        assert_eq!(events[0].data_str(), Some("{\"emoji\":\"🦀\"}"));
        assert_eq!(events[1].event_name, None);
        assert_eq!(events[1].data_str(), Some("a\nb"));
        assert!(events[2].is_done_sentinel());
        assert!(framer.done_seen());
        assert_eq!(framer.buffered_len(), 0);
    }

    #[test]
    fn framer_handles_crlf_and_comments() {
        let mut framer = SseFramer::new();
        let events = feed(
            &mut framer,
            &[b": keepalive\r\n\r\nevent: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n"],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_name.as_deref(), Some("message_stop"));
        assert_eq!(events[0].data_str(), Some("{\"type\":\"message_stop\"}"));
    }

    #[test]
    fn chat_null_usage_does_not_emit_until_final_chunk() {
        let _guard = stats::test_lock();
        stats::reset_for_tests();
        let mut state = SseStreamState::new(SseKind::OpenAiChat, None);
        state.apply(&SseEvent {
            event_name: None,
            data: br#"{"choices":[{"delta":{"content":"x"}}],"usage":null}"#.to_vec(),
        });
        assert_eq!(stats::stats().sse_input_tokens, 0);
        state.apply(&SseEvent {
            event_name: None,
            data: br#"{"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":10}}}"#.to_vec(),
        });
        state.finish();
        let snapshot = stats::stats();
        assert_eq!(snapshot.sse_input_tokens, 40);
        assert_eq!(snapshot.sse_output_tokens, 3);
        assert_eq!(snapshot.sse_cache_read_input_tokens, 10);
        // Inferred OpenAI cache write: 40 - 10.
        assert_eq!(snapshot.sse_inferred_cache_write_tokens, 30);
        let chat = snapshot.sse_cache_hit_rates.get("openai_chat").unwrap();
        assert_eq!(chat.count, 1);
        assert!((chat.sum - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn responses_records_service_tier_and_terminal_status() {
        let _guard = stats::test_lock();
        stats::reset_for_tests();
        let mut state = SseStreamState::new(SseKind::OpenAiResponses, None);
        state.apply(&SseEvent {
            event_name: Some("response.in_progress".to_string()),
            data: br#"{"type":"response.in_progress","response":{}}"#.to_vec(),
        });
        assert_eq!(stats::stats().sse_input_tokens, 0);
        state.apply(&SseEvent {
            event_name: Some("response.completed".to_string()),
            data: br#"{"type":"response.completed","response":{"service_tier":"default","usage":{"input_tokens":20,"output_tokens":5,"input_tokens_details":{"cached_tokens":8}}}}"#.to_vec(),
        });
        state.finish();
        let snapshot = stats::stats();
        assert_eq!(snapshot.sse_input_tokens, 20);
        assert_eq!(snapshot.service_tier_counts.get("default"), Some(&1));
        assert_eq!(snapshot.response_status_counts.get("completed"), Some(&1));
        let responses = snapshot
            .sse_cache_hit_rates
            .get("openai_responses")
            .unwrap();
        assert_eq!(responses.count, 1);
        assert!((responses.sum - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn responses_stream_closed_mid_flight_records_no_status() {
        let _guard = stats::test_lock();
        stats::reset_for_tests();
        let mut state = SseStreamState::new(SseKind::OpenAiResponses, None);
        state.apply(&SseEvent {
            event_name: None,
            data: br#"{"type":"response.output_text.delta","delta":"x"}"#.to_vec(),
        });
        state.finish();
        // No terminal event was observed, so nothing is emitted —
        // mirroring the reference gate on `terminal_status().is_some()`.
        assert!(stats::stats().response_status_counts.is_empty());
    }

    #[test]
    fn unknown_service_tier_buckets_to_other() {
        assert_eq!(service_tier::validate("default"), "default");
        assert_eq!(service_tier::validate("scale"), "scale");
        assert_eq!(service_tier::validate("totally-new"), "other");
        // Case-sensitive: drift, not the same tier.
        assert_eq!(service_tier::validate("Default"), "other");
    }

    #[test]
    fn anthropic_emits_usage_and_updates_session_tracker_at_message_stop() {
        let _guard = stats::test_lock();
        stats::reset_for_tests();
        let trackers = SessionTrackers::new();
        let mut state = SseStreamState::new(
            SseKind::Anthropic,
            Some(SseSessionCtx {
                trackers: trackers.clone(),
                provider: SessionProvider::Anthropic,
                session_id: "sse-session".to_string(),
                message_token_estimates: vec![900, 300, 400],
            }),
        );
        state.apply(&SseEvent {
            event_name: Some("message_start".to_string()),
            data: br#"{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":1900,"cache_creation_input_tokens":700}}}"#.to_vec(),
        });
        // The streamed assistant message (3480 chars ≈ 1000 estimated
        // tokens) joins the tracker walk, mirroring
        // `streaming.py:738-748`.
        let chunk = "x".repeat(1160);
        state.apply(&SseEvent {
            event_name: Some("content_block_delta".to_string()),
            data: format!(
                "{{\"type\":\"content_block_delta\",\"delta\":{{\"text\":\"{chunk}\"}}}}"
            )
            .into_bytes(),
        });
        state.apply(&SseEvent {
            event_name: Some("content_block_delta".to_string()),
            data: format!(
                "{{\"type\":\"content_block_delta\",\"delta\":{{\"partial_json\":\"{chunk}\"}}}}"
            )
            .into_bytes(),
        });
        state.apply(&SseEvent {
            event_name: Some("content_block_delta".to_string()),
            data: format!(
                "{{\"type\":\"content_block_delta\",\"delta\":{{\"thinking\":\"{chunk}\"}}}}"
            )
            .into_bytes(),
        });
        state.apply(&SseEvent {
            event_name: Some("message_delta".to_string()),
            data: br#"{"type":"message_delta","usage":{"output_tokens":9}}"#.to_vec(),
        });
        assert_eq!(stats::stats().sse_input_tokens, 0);
        state.apply(&SseEvent {
            event_name: Some("message_stop".to_string()),
            data: br#"{"type":"message_stop"}"#.to_vec(),
        });
        state.finish();

        let snapshot = stats::stats();
        assert_eq!(snapshot.sse_input_tokens, 100);
        assert_eq!(snapshot.sse_output_tokens, 9);
        assert_eq!(snapshot.sse_cache_read_input_tokens, 1900);
        assert_eq!(snapshot.sse_cache_creation_input_tokens, 700);
        let anthropic = snapshot.sse_cache_hit_rates.get("anthropic").unwrap();
        assert_eq!(anthropic.count, 1);
        assert!((anthropic.sum - (1900.0 / 2700.0)).abs() < f64::EPSILON);

        // 2600 confirmed cached tokens over [900, 300, 400] + the
        // streamed assistant message (3480 chars + 20 over 3.5 = 1000)
        // freezes all four positions on the next turn — without the
        // assistant append the walk would stop at 3.
        assert_eq!(
            trackers.frozen_message_count(SessionProvider::Anthropic, "sse-session"),
            4
        );
    }

    #[test]
    fn detects_ccr_stream_tool_calls_per_provider() {
        assert!(ccr_stream_tool_call_detected(
            SseKind::Anthropic,
            &json!({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "headroom_retrieve"}
            })
        ));
        assert!(!ccr_stream_tool_call_detected(
            SseKind::Anthropic,
            &json!({
                "type": "content_block_start",
                "content_block": {"type": "text", "text": "headroom_retrieve"}
            })
        ));
        assert!(ccr_stream_tool_call_detected(
            SseKind::OpenAiChat,
            &json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "headroom_retrieve"}}]
                    }
                }]
            })
        ));
        assert!(ccr_stream_tool_call_detected(
            SseKind::OpenAiResponses,
            &json!({
                "type": "response.output_item.added",
                "item": {"type": "function_call", "name": "mcp__headroom_retrieve"}
            })
        ));
    }

    #[test]
    fn rate_limit_snapshot_parses_both_header_families() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "anthropic-ratelimit-requests-remaining",
            "99".parse().unwrap(),
        );
        headers.insert(
            "anthropic-ratelimit-input-tokens-remaining",
            "5000".parse().unwrap(),
        );
        let snapshot = extract_rate_limit_snapshot(&headers);
        assert_eq!(snapshot.remaining_requests, Some(99));
        assert_eq!(snapshot.remaining_input_tokens, Some(5000));
        assert_eq!(snapshot.remaining_tokens, None);

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-ratelimit-remaining-requests", "7".parse().unwrap());
        headers.insert("x-ratelimit-remaining-tokens", "1234".parse().unwrap());
        let snapshot = extract_rate_limit_snapshot(&headers);
        assert_eq!(snapshot.remaining_requests, Some(7));
        assert_eq!(snapshot.remaining_tokens, Some(1234));
        assert_eq!(snapshot.remaining_output_tokens, None);
    }
}

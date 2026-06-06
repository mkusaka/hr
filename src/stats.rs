use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

static GLOBAL_STATS: LazyLock<Stats> = LazyLock::new(Stats::default);

#[derive(Debug, Default)]
pub(crate) struct Stats {
    total_requests: AtomicU64,
    compressed_requests: AtomicU64,
    bytes_before: AtomicU64,
    bytes_after: AtomicU64,
    tokens_before: AtomicU64,
    tokens_after: AtomicU64,
    ccr_entry_count: AtomicU64,
    decompress_hits: AtomicU64,
    decompress_misses: AtomicU64,
    websocket_sessions: AtomicU64,
    active_websocket_sessions: AtomicU64,
    sse_streams: AtomicU64,
    sse_input_tokens: AtomicU64,
    sse_output_tokens: AtomicU64,
    sse_cache_read_input_tokens: AtomicU64,
    sse_cache_creation_input_tokens: AtomicU64,
    sse_cache_hit_rates: Mutex<BTreeMap<String, CacheHitRateStats>>,
    skipped_requests: Mutex<BTreeMap<String, u64>>,
    ccr_continuation_rounds: AtomicU64,
    ccr_continuation_retrievals: AtomicU64,
    ccr_stream_tool_calls: AtomicU64,
    ccr_batch_results_processed: AtomicU64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CacheHitRateStats {
    pub count: u64,
    pub sum: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub compressed_requests: u64,
    pub skipped_requests: BTreeMap<String, u64>,
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub savings_ratio: f64,
    pub ccr_entry_count: u64,
    pub decompress_hits: u64,
    pub decompress_misses: u64,
    pub websocket_sessions: u64,
    pub active_websocket_sessions: u64,
    pub sse_streams: u64,
    pub sse_input_tokens: u64,
    pub sse_output_tokens: u64,
    pub sse_cache_read_input_tokens: u64,
    pub sse_cache_creation_input_tokens: u64,
    pub sse_cache_hit_rates: BTreeMap<String, CacheHitRateStats>,
    pub ccr_continuation_rounds: u64,
    pub ccr_continuation_retrievals: u64,
    pub ccr_stream_tool_calls: u64,
    pub ccr_batch_results_processed: u64,
}

pub fn stats() -> StatsSnapshot {
    GLOBAL_STATS.snapshot()
}

#[doc(hidden)]
pub fn stats_with_ccr_entry_count(ccr_entry_count: u64) -> StatsSnapshot {
    GLOBAL_STATS
        .ccr_entry_count
        .store(ccr_entry_count, Ordering::Relaxed);
    GLOBAL_STATS.snapshot()
}

pub(crate) fn record_request() {
    GLOBAL_STATS.total_requests.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_compressed_request(
    bytes_before: u64,
    bytes_after: u64,
    tokens_before: u64,
    tokens_after: u64,
) {
    GLOBAL_STATS
        .compressed_requests
        .fetch_add(1, Ordering::Relaxed);
    GLOBAL_STATS
        .bytes_before
        .fetch_add(bytes_before, Ordering::Relaxed);
    GLOBAL_STATS
        .bytes_after
        .fetch_add(bytes_after, Ordering::Relaxed);
    GLOBAL_STATS
        .tokens_before
        .fetch_add(tokens_before, Ordering::Relaxed);
    GLOBAL_STATS
        .tokens_after
        .fetch_add(tokens_after, Ordering::Relaxed);
}

pub(crate) fn record_skipped_request(reason: &str) {
    let mut skipped = GLOBAL_STATS
        .skipped_requests
        .lock()
        .expect("stats skipped_requests lock poisoned");
    *skipped.entry(reason.to_string()).or_insert(0) += 1;
}

pub(crate) fn record_ccr_entry_inserted() {
    GLOBAL_STATS.ccr_entry_count.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_decompress_hit() {
    GLOBAL_STATS.decompress_hits.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_decompress_miss() {
    GLOBAL_STATS
        .decompress_misses
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_websocket_open() {
    GLOBAL_STATS
        .websocket_sessions
        .fetch_add(1, Ordering::Relaxed);
    GLOBAL_STATS
        .active_websocket_sessions
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_websocket_close() {
    GLOBAL_STATS
        .active_websocket_sessions
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(1))
        })
        .ok();
}

pub(crate) fn record_sse_stream() {
    GLOBAL_STATS.sse_streams.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sse_usage(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
) {
    GLOBAL_STATS
        .sse_input_tokens
        .fetch_add(input_tokens, Ordering::Relaxed);
    GLOBAL_STATS
        .sse_output_tokens
        .fetch_add(output_tokens, Ordering::Relaxed);
    GLOBAL_STATS
        .sse_cache_read_input_tokens
        .fetch_add(cache_read_input_tokens, Ordering::Relaxed);
    GLOBAL_STATS
        .sse_cache_creation_input_tokens
        .fetch_add(cache_creation_input_tokens, Ordering::Relaxed);
}

pub(crate) fn record_ccr_continuation_round() {
    GLOBAL_STATS
        .ccr_continuation_rounds
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_ccr_continuation_retrievals(count: u64) {
    GLOBAL_STATS
        .ccr_continuation_retrievals
        .fetch_add(count, Ordering::Relaxed);
}

pub(crate) fn record_ccr_stream_tool_call() {
    GLOBAL_STATS
        .ccr_stream_tool_calls
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_ccr_batch_result_processed() {
    GLOBAL_STATS
        .ccr_batch_results_processed
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sse_cache_hit_rate(provider: &str, rate: f64) {
    if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
        return;
    }

    let mut rates = GLOBAL_STATS
        .sse_cache_hit_rates
        .lock()
        .expect("stats sse_cache_hit_rates lock poisoned");
    let entry = rates.entry(provider.to_string()).or_default();
    entry.count += 1;
    entry.sum += rate;
}

#[cfg(test)]
pub(crate) fn reset_for_tests() {
    GLOBAL_STATS.total_requests.store(0, Ordering::Relaxed);
    GLOBAL_STATS.compressed_requests.store(0, Ordering::Relaxed);
    GLOBAL_STATS.bytes_before.store(0, Ordering::Relaxed);
    GLOBAL_STATS.bytes_after.store(0, Ordering::Relaxed);
    GLOBAL_STATS.tokens_before.store(0, Ordering::Relaxed);
    GLOBAL_STATS.tokens_after.store(0, Ordering::Relaxed);
    GLOBAL_STATS.ccr_entry_count.store(0, Ordering::Relaxed);
    GLOBAL_STATS.decompress_hits.store(0, Ordering::Relaxed);
    GLOBAL_STATS.decompress_misses.store(0, Ordering::Relaxed);
    GLOBAL_STATS.websocket_sessions.store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .active_websocket_sessions
        .store(0, Ordering::Relaxed);
    GLOBAL_STATS.sse_streams.store(0, Ordering::Relaxed);
    GLOBAL_STATS.sse_input_tokens.store(0, Ordering::Relaxed);
    GLOBAL_STATS.sse_output_tokens.store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .sse_cache_read_input_tokens
        .store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .sse_cache_creation_input_tokens
        .store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .sse_cache_hit_rates
        .lock()
        .expect("stats sse_cache_hit_rates lock poisoned")
        .clear();
    GLOBAL_STATS
        .skipped_requests
        .lock()
        .expect("stats skipped_requests lock poisoned")
        .clear();
    GLOBAL_STATS
        .ccr_continuation_rounds
        .store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .ccr_continuation_retrievals
        .store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .ccr_stream_tool_calls
        .store(0, Ordering::Relaxed);
    GLOBAL_STATS
        .ccr_batch_results_processed
        .store(0, Ordering::Relaxed);
}

impl Stats {
    fn snapshot(&self) -> StatsSnapshot {
        let bytes_before = self.bytes_before.load(Ordering::Relaxed);
        let bytes_after = self.bytes_after.load(Ordering::Relaxed);
        let savings_ratio = if bytes_before == 0 {
            0.0
        } else {
            bytes_before.saturating_sub(bytes_after) as f64 / bytes_before as f64
        };

        StatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            compressed_requests: self.compressed_requests.load(Ordering::Relaxed),
            skipped_requests: self
                .skipped_requests
                .lock()
                .expect("stats skipped_requests lock poisoned")
                .clone(),
            bytes_before,
            bytes_after,
            tokens_before: self.tokens_before.load(Ordering::Relaxed),
            tokens_after: self.tokens_after.load(Ordering::Relaxed),
            savings_ratio,
            ccr_entry_count: self.ccr_entry_count.load(Ordering::Relaxed),
            decompress_hits: self.decompress_hits.load(Ordering::Relaxed),
            decompress_misses: self.decompress_misses.load(Ordering::Relaxed),
            websocket_sessions: self.websocket_sessions.load(Ordering::Relaxed),
            active_websocket_sessions: self.active_websocket_sessions.load(Ordering::Relaxed),
            sse_streams: self.sse_streams.load(Ordering::Relaxed),
            sse_input_tokens: self.sse_input_tokens.load(Ordering::Relaxed),
            sse_output_tokens: self.sse_output_tokens.load(Ordering::Relaxed),
            sse_cache_read_input_tokens: self.sse_cache_read_input_tokens.load(Ordering::Relaxed),
            sse_cache_creation_input_tokens: self
                .sse_cache_creation_input_tokens
                .load(Ordering::Relaxed),
            sse_cache_hit_rates: self
                .sse_cache_hit_rates
                .lock()
                .expect("stats sse_cache_hit_rates lock poisoned")
                .clone(),
            ccr_continuation_rounds: self.ccr_continuation_rounds.load(Ordering::Relaxed),
            ccr_continuation_retrievals: self.ccr_continuation_retrievals.load(Ordering::Relaxed),
            ccr_stream_tool_calls: self.ccr_stream_tool_calls.load(Ordering::Relaxed),
            ccr_batch_results_processed: self.ccr_batch_results_processed.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn savings_ratio_uses_bytes_before_after() {
        reset_for_tests();
        record_request();
        record_compressed_request(100, 25, 50, 10);

        let snapshot = stats();

        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.compressed_requests, 1);
        assert_eq!(snapshot.bytes_before, 100);
        assert_eq!(snapshot.bytes_after, 25);
        assert_eq!(snapshot.savings_ratio, 0.75);
    }
}

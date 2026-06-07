use crate::ccr::{CcrStore, SqliteStore, HASH_HEX_LEN};
use crate::compression::{
    compress_json_request_ctx, compress_json_request_with_auth, ApiShape, CompressContext,
    RequestAuthMode, RequestCompression,
};
use crate::session::{estimate_message_tokens, SessionProvider, SessionTrackers};
use crate::sse::{
    extract_rate_limit_snapshot, run_sse_state_machine, usage_u64, usage_u64_opt, SseKind,
    SseSessionCtx, SSE_PARSER_QUEUE_DEPTH,
};
use crate::stats;
use crate::{error, HrResult};
use axum::body::{to_bytes, Body};
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message as AxumWsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::header::{CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST, UPGRADE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as TungsteniteCloseFrame;
use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;
use tracing::{debug, info, trace, warn};
use url::Url;

pub const DEFAULT_MAX_BODY_BYTES: usize = 25 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(600);
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);
/// Maximum proxy-side `headroom_retrieve` continuation rounds per response,
/// mirroring the reference handler's `max_retrieval_rounds`.
const MAX_CCR_CONTINUATION_ROUNDS: usize = 3;
/// Batch contexts live for 24 hours in the reference Headroom proxy.
const BATCH_CONTEXT_TTL: Duration = Duration::from_secs(24 * 60 * 60);
/// Maximum batch contexts retained for CCR batch-result post-processing.
const MAX_BATCH_CONTEXTS: usize = 10_000;

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Per-batch CCR context: batch id paired with each request's compressed
/// params keyed by `custom_id`.
type BatchContexts = VecDeque<BatchContextEntry>;

#[derive(Debug, Clone)]
struct BatchContextEntry {
    batch_id: String,
    contexts: HashMap<String, Value>,
    expires_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub listen: SocketAddr,
    pub openai_upstream: Url,
    pub anthropic_upstream: Url,
    pub ccr_db: PathBuf,
    pub log_level: String,
    pub max_body_bytes: usize,
    pub compression_enabled: bool,
    pub compression_mode: CompressionMode,
    /// Opt-in user-text compression
    /// (`--compress-user-messages` / `HEADROOM_COMPRESS_USER_MESSAGES=1`,
    /// mirroring `headroom/proxy/models.py` `compress_user_messages`).
    pub compress_user_text: bool,
}

#[derive(Debug, Clone)]
pub struct ProxyState {
    client: reqwest::Client,
    openai_upstream: Url,
    anthropic_upstream: Url,
    store: SqliteStore,
    max_body_bytes: usize,
    compression_enabled: bool,
    compression_mode: CompressionMode,
    compress_user_text: bool,
    batch_contexts: Arc<Mutex<BatchContexts>>,
    sessions: SessionTrackers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UpstreamProvider {
    OpenAi,
    Anthropic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CompressionTarget {
    OpenAiChatCompletions,
    OpenAiResponses,
    AnthropicMessages,
    AnthropicMessageBatches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CompressionMode {
    Ccr,
    Passthrough,
    Off,
}

impl CompressionMode {
    pub fn parse(value: &str) -> HrResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ccr" | "compress" | "compression" | "on" | "auto" => Ok(Self::Ccr),
            "passthrough" | "pass-through" | "pass" => Ok(Self::Passthrough),
            "off" | "none" | "disabled" | "false" => Ok(Self::Off),
            other => Err(error(format!(
                "unsupported compression mode: {other}; expected ccr, passthrough, or off"
            ))),
        }
    }

    fn allows_compression(self) -> bool {
        matches!(self, Self::Ccr)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RequestClass {
    pub provider: UpstreamProvider,
    pub target: Option<CompressionTarget>,
    pub skipped_reason: Option<&'static str>,
}

pub fn build_router(state: ProxyState) -> Router {
    Router::new()
        .route("/healthz", get(healthz_handler))
        .route("/healthz/upstream", get(healthz_upstream_handler))
        .route("/livez", get(livez_handler))
        .route("/readyz", get(readyz_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/stats", get(stats_handler))
        .route("/v1/compress", post(compress_only_handler))
        .route("/v1/retrieve", post(retrieve_post_handler))
        .route("/v1/retrieve/stats", get(retrieve_stats_handler))
        .route("/v1/retrieve/tool_call", post(retrieve_tool_call_handler))
        .route("/v1/retrieve/{hash}", get(retrieve_get_handler))
        .route(
            "/{*path}",
            get(proxy_get_handler)
                .post(proxy_http_handler)
                .put(proxy_http_handler)
                .patch(proxy_http_handler)
                .delete(proxy_http_handler)
                .head(proxy_http_handler)
                .options(proxy_http_handler),
        )
        .fallback(proxy_http_handler)
        .with_state(state)
}

pub async fn serve_proxy(config: ProxyConfig) -> HrResult<()> {
    let store = SqliteStore::open(&config.ccr_db)?;
    let state = ProxyState::new(
        config.openai_upstream.clone(),
        config.anthropic_upstream.clone(),
        store,
    )
    .with_max_body_bytes(config.max_body_bytes)
    .with_compression_enabled(config.compression_enabled)
    .with_compression_mode(config.compression_mode)
    .with_compress_user_text(config.compress_user_text);
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(config.listen).await?;

    info!(
        listen = %config.listen,
        openai_upstream = %config.openai_upstream,
        anthropic_upstream = %config.anthropic_upstream,
        ccr_db = %config.ccr_db.display(),
        max_body_bytes = config.max_body_bytes,
        compression_enabled = config.compression_enabled,
        compression_mode = ?config.compression_mode,
        "proxy startup"
    );

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

pub fn classify_request(method: &Method, path: &str) -> RequestClass {
    let provider = if path == "/v1/messages" || path.starts_with("/v1/messages/") {
        UpstreamProvider::Anthropic
    } else {
        UpstreamProvider::OpenAi
    };

    if method != Method::POST {
        return RequestClass {
            provider,
            target: None,
            skipped_reason: Some("non_post_method"),
        };
    }

    let target = match path {
        "/v1/chat/completions" => Some(CompressionTarget::OpenAiChatCompletions),
        "/v1/responses"
        | "/v1/codex/responses"
        | "/backend-api/responses"
        | "/backend-api/codex/responses" => Some(CompressionTarget::OpenAiResponses),
        "/v1/messages/batches" => Some(CompressionTarget::AnthropicMessageBatches),
        "/v1/messages" => Some(CompressionTarget::AnthropicMessages),
        _ => None,
    };

    RequestClass {
        provider,
        target,
        skipped_reason: skipped_reason_for_path(path, target),
    }
}

fn skipped_reason_for_path(path: &str, target: Option<CompressionTarget>) -> Option<&'static str> {
    if target.is_some() {
        return None;
    }

    if path == "/v1/retrieve"
        || path == "/v1/retrieve/stats"
        || path == "/v1/retrieve/tool_call"
        || path.starts_with("/v1/retrieve/")
        || path == "/v1/compress"
    {
        return Some("reserved_proxy_endpoint");
    }

    if path == "/healthz"
        || path == "/healthz/upstream"
        || path == "/livez"
        || path == "/readyz"
        || path == "/health"
        || path == "/metrics"
        || path == "/stats"
    {
        return Some("reserved_proxy_endpoint");
    }

    if path == "/v1/conversations" || path.starts_with("/v1/conversations/") {
        return Some("conversations_passthrough");
    }

    Some("non_target_path")
}

impl ProxyState {
    pub fn new(openai_upstream: Url, anthropic_upstream: Url, store: SqliteStore) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(UPSTREAM_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .build()
            .expect("valid reqwest client configuration");
        Self {
            client,
            openai_upstream,
            anthropic_upstream,
            store,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            compression_enabled: true,
            compression_mode: CompressionMode::Ccr,
            compress_user_text: false,
            batch_contexts: Arc::new(Mutex::new(VecDeque::new())),
            sessions: SessionTrackers::new(),
        }
    }

    pub fn with_max_body_bytes(mut self, max_body_bytes: usize) -> Self {
        self.max_body_bytes = max_body_bytes;
        self
    }

    pub fn with_compression_enabled(mut self, compression_enabled: bool) -> Self {
        self.compression_enabled = compression_enabled;
        self
    }

    pub fn with_compression_mode(mut self, compression_mode: CompressionMode) -> Self {
        self.compression_mode = compression_mode;
        self
    }

    pub fn with_compress_user_text(mut self, compress_user_text: bool) -> Self {
        self.compress_user_text = compress_user_text;
        self
    }

    pub fn store(&self) -> &SqliteStore {
        &self.store
    }

    /// Records per-request compressed params from an Anthropic batch create
    /// so CCR tool calls in batch results can be continued later.
    fn record_batch_context(&self, batch_id: &str, compressed_body: &Value) {
        let Some(requests) = compressed_body.get("requests").and_then(Value::as_array) else {
            return;
        };
        let mut contexts = HashMap::new();
        for request in requests {
            let Some(custom_id) = request.get("custom_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(params) = request.get("params") else {
                continue;
            };
            contexts.insert(custom_id.to_string(), params.clone());
        }
        if contexts.is_empty() {
            return;
        }

        let now = SystemTime::now();
        let expires_at = now + BATCH_CONTEXT_TTL;
        let mut store = self
            .batch_contexts
            .lock()
            .expect("batch context lock poisoned");
        store.retain(|entry| entry.expires_at > now && entry.batch_id != batch_id);
        store.push_back(BatchContextEntry {
            batch_id: batch_id.to_string(),
            contexts,
            expires_at,
        });
        while store.len() > MAX_BATCH_CONTEXTS {
            store.pop_front();
        }
    }

    fn batch_context(&self, batch_id: &str) -> Option<HashMap<String, Value>> {
        let now = SystemTime::now();
        let mut store = self
            .batch_contexts
            .lock()
            .expect("batch context lock poisoned");
        store.retain(|entry| entry.expires_at > now);
        store
            .iter()
            .rev()
            .find(|entry| entry.batch_id == batch_id)
            .map(|entry| entry.contexts.clone())
    }
}

async fn healthz_handler() -> impl IntoResponse {
    axum::Json(json!({ "ok": true, "service": "hr-proxy" }))
}

async fn livez_handler() -> impl IntoResponse {
    axum::Json(json!({
        "service": "hr-proxy",
        "status": "healthy",
        "alive": true,
    }))
}

async fn readyz_handler(State(state): State<ProxyState>) -> Response<Body> {
    let openai = upstream_health(&state, UpstreamProvider::OpenAi).await;
    let anthropic = upstream_health(&state, UpstreamProvider::Anthropic).await;
    let ready = openai.ok && anthropic.ok;
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    json_response(
        status,
        json!({
            "service": "hr-proxy",
            "ready": ready,
            "openai": openai,
            "anthropic": anthropic,
        }),
    )
}

async fn health_handler(State(state): State<ProxyState>) -> Response<Body> {
    let ccr_entry_count = state.store.count().unwrap_or_default();
    let snapshot = stats::stats_with_ccr_entry_count(ccr_entry_count);
    json_response(
        StatusCode::OK,
        json!({
            "service": "hr-proxy",
            "status": "healthy",
            "ready": true,
            "stats": snapshot,
        }),
    )
}

async fn healthz_upstream_handler(State(state): State<ProxyState>) -> Response<Body> {
    let openai = upstream_health(&state, UpstreamProvider::OpenAi).await;
    let anthropic = upstream_health(&state, UpstreamProvider::Anthropic).await;
    let ok = openai.ok && anthropic.ok;
    let status = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    json_response(
        status,
        json!({
            "ok": ok,
            "openai": openai,
            "anthropic": anthropic,
        }),
    )
}

#[derive(Debug, Serialize)]
struct UpstreamHealth {
    ok: bool,
    status: Option<u16>,
    error: Option<String>,
}

async fn upstream_health(state: &ProxyState, provider: UpstreamProvider) -> UpstreamHealth {
    let mut url = match provider {
        UpstreamProvider::OpenAi => state.openai_upstream.clone(),
        UpstreamProvider::Anthropic => state.anthropic_upstream.clone(),
    };
    url.set_path("/healthz");
    url.set_query(None);

    match state.client.get(url).send().await {
        Ok(response) => UpstreamHealth {
            ok: response.status().is_success(),
            status: Some(response.status().as_u16()),
            error: None,
        },
        Err(err) => UpstreamHealth {
            ok: false,
            status: None,
            error: Some(err.to_string()),
        },
    }
}

async fn metrics_handler(State(state): State<ProxyState>) -> Response<Body> {
    let ccr_entry_count = state.store.count().unwrap_or_default();
    let snapshot = stats::stats_with_ccr_entry_count(ccr_entry_count);
    let skipped_total: u64 = snapshot.skipped_requests.values().sum();
    let body = format!(
        "\
# TYPE hr_total_requests counter
hr_total_requests {}
# TYPE hr_compressed_requests counter
hr_compressed_requests {}
# TYPE hr_skipped_requests counter
hr_skipped_requests {}
# TYPE hr_bytes_before counter
hr_bytes_before {}
# TYPE hr_bytes_after counter
hr_bytes_after {}
# TYPE hr_tokens_before counter
hr_tokens_before {}
# TYPE hr_tokens_after counter
hr_tokens_after {}
# TYPE hr_ccr_entry_count gauge
hr_ccr_entry_count {}
# TYPE hr_decompress_hits counter
hr_decompress_hits {}
# TYPE hr_decompress_misses counter
hr_decompress_misses {}
# TYPE hr_websocket_sessions counter
hr_websocket_sessions {}
# TYPE hr_sse_streams counter
hr_sse_streams {}
# TYPE hr_sse_input_tokens counter
hr_sse_input_tokens {}
# TYPE hr_sse_output_tokens counter
hr_sse_output_tokens {}
# TYPE hr_sse_cache_read_input_tokens counter
hr_sse_cache_read_input_tokens {}
# TYPE hr_sse_cache_creation_input_tokens counter
hr_sse_cache_creation_input_tokens {}
# TYPE hr_ccr_continuation_rounds counter
hr_ccr_continuation_rounds {}
# TYPE hr_ccr_continuation_retrievals counter
hr_ccr_continuation_retrievals {}
# TYPE hr_ccr_stream_tool_calls counter
hr_ccr_stream_tool_calls {}
# TYPE hr_ccr_batch_results_processed counter
hr_ccr_batch_results_processed {}
# TYPE hr_sse_inferred_cache_write_tokens counter
hr_sse_inferred_cache_write_tokens {}
",
        snapshot.total_requests,
        snapshot.compressed_requests,
        skipped_total,
        snapshot.bytes_before,
        snapshot.bytes_after,
        snapshot.tokens_before,
        snapshot.tokens_after,
        snapshot.ccr_entry_count,
        snapshot.decompress_hits,
        snapshot.decompress_misses,
        snapshot.websocket_sessions,
        snapshot.sse_streams,
        snapshot.sse_input_tokens,
        snapshot.sse_output_tokens,
        snapshot.sse_cache_read_input_tokens,
        snapshot.sse_cache_creation_input_tokens,
        snapshot.ccr_continuation_rounds,
        snapshot.ccr_continuation_retrievals,
        snapshot.ccr_stream_tool_calls,
        snapshot.ccr_batch_results_processed,
        snapshot.sse_inferred_cache_write_tokens,
    );
    let skipped_by_reason = snapshot
        .skipped_requests
        .iter()
        .map(|(reason, count)| {
            format!("hr_skipped_requests_by_reason{{reason=\"{reason}\"}} {count}\n")
        })
        .collect::<String>();
    let cache_hit_rates = snapshot
        .sse_cache_hit_rates
        .iter()
        .map(|(provider, stats)| {
            format!(
                "# TYPE proxy_cache_hit_rate_per_session summary\n\
proxy_cache_hit_rate_per_session_count{{provider=\"{provider}\"}} {}\n\
proxy_cache_hit_rate_per_session_sum{{provider=\"{provider}\"}} {}\n",
                stats.count, stats.sum
            )
        })
        .collect::<String>();
    // Metric names mirror the reference
    // (`observability/metric_names.rs`): bounded `tier` / `status`
    // label vocabularies, per-provider rate-limit gauges.
    let service_tiers = snapshot
        .service_tier_counts
        .iter()
        .map(|(tier, count)| format!("proxy_service_tier_count_total{{tier=\"{tier}\"}} {count}\n"))
        .collect::<String>();
    let response_statuses = snapshot
        .response_status_counts
        .iter()
        .map(|(status, count)| {
            format!("proxy_response_status_count_total{{status=\"{status}\"}} {count}\n")
        })
        .collect::<String>();
    let rate_limits = snapshot
        .rate_limit_remaining
        .iter()
        .map(|(provider, gauges)| {
            let mut lines = String::new();
            if let Some(value) = gauges.remaining_requests {
                lines.push_str(&format!(
                    "proxy_rate_limit_remaining_requests{{provider=\"{provider}\"}} {value}\n"
                ));
            }
            if let Some(value) = gauges.remaining_tokens {
                lines.push_str(&format!(
                    "proxy_rate_limit_remaining_tokens{{provider=\"{provider}\"}} {value}\n"
                ));
            }
            if let Some(value) = gauges.remaining_input_tokens {
                lines.push_str(&format!(
                    "proxy_rate_limit_remaining_input_tokens{{provider=\"{provider}\"}} {value}\n"
                ));
            }
            if let Some(value) = gauges.remaining_output_tokens {
                lines.push_str(&format!(
                    "proxy_rate_limit_remaining_output_tokens{{provider=\"{provider}\"}} {value}\n"
                ));
            }
            lines
        })
        .collect::<String>();
    let body = format!(
        "{body}{skipped_by_reason}{cache_hit_rates}{service_tiers}{response_statuses}{rate_limits}"
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(body))
        .unwrap_or_else(|err| error_response(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
}

async fn stats_handler(State(state): State<ProxyState>) -> impl IntoResponse {
    let ccr_entry_count = state.store.count().unwrap_or_default();
    let snapshot = stats::stats_with_ccr_entry_count(ccr_entry_count);
    info!(
        total_requests = snapshot.total_requests,
        compressed_requests = snapshot.compressed_requests,
        ccr_entry_count = snapshot.ccr_entry_count,
        savings_ratio = snapshot.savings_ratio,
        websocket_sessions = snapshot.websocket_sessions,
        "stats summary"
    );
    axum::Json(snapshot)
}

#[derive(Debug, Deserialize)]
struct RetrieveRequest {
    hash: String,
    query: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RetrieveQuery {
    query: Option<String>,
}

#[derive(Debug, Serialize)]
struct RetrieveSearchResult {
    line: usize,
    text: String,
}

async fn retrieve_post_handler(
    State(state): State<ProxyState>,
    axum::Json(request): axum::Json<RetrieveRequest>,
) -> Response<Body> {
    retrieve_response(&state.store, &request.hash, request.query.as_deref())
}

async fn retrieve_get_handler(
    State(state): State<ProxyState>,
    Path(hash): Path<String>,
    Query(query): Query<RetrieveQuery>,
) -> Response<Body> {
    retrieve_response(&state.store, &hash, query.query.as_deref())
}

async fn retrieve_stats_handler(State(state): State<ProxyState>) -> Response<Body> {
    let snapshot = stats::stats_with_ccr_entry_count(state.store.count().unwrap_or_default());
    json_response(
        StatusCode::OK,
        json!({
            "store": {
                "entries": snapshot.ccr_entry_count,
                "decompress_hits": snapshot.decompress_hits,
                "decompress_misses": snapshot.decompress_misses,
            },
            "recent_retrievals": [],
        }),
    )
}

async fn retrieve_tool_call_handler(
    State(state): State<ProxyState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> Response<Body> {
    let provider = request
        .get("provider")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("anthropic");
    let Some(tool_call) = request.get("tool_call") else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"success": false, "error": "tool_call required"}),
        );
    };

    let Some(parsed) = parse_retrieve_tool_call(tool_call, provider) else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({
                "success": false,
                "error": "invalid tool call or not a headroom_retrieve call"
            }),
        );
    };

    let data = retrieve_value(&state.store, &parsed.hash, parsed.query.as_deref());
    let success = data.get("error").is_none();
    let tool_result = format_tool_result(tool_call, provider, &data);

    json_response(
        StatusCode::OK,
        json!({
            "success": success,
            "tool_result": tool_result,
            "data": data,
        }),
    )
}

async fn compress_only_handler(
    State(state): State<ProxyState>,
    req: Request<Body>,
) -> Response<Body> {
    stats::record_request();

    let headers = req.headers().clone();
    let body = match read_limited_body(req, state.max_body_bytes).await {
        Ok(body) => body,
        Err(err) if err.to_string().contains("payload_too_large") => {
            return error_response(StatusCode::PAYLOAD_TOO_LARGE, err.to_string());
        }
        Err(err) => return error_response(StatusCode::BAD_REQUEST, err.to_string()),
    };

    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({
                "error": {
                    "type": "invalid_request",
                    "message": "Invalid JSON in request body."
                }
            }),
        );
    };

    if compression_bypass_requested(&headers) {
        stats::record_skipped_request("bypass_header");
        return json_response(
            StatusCode::OK,
            json!({
                "messages": value.get("messages").cloned().unwrap_or_else(|| json!([])),
                "tokens_before": 0,
                "tokens_after": 0,
                "tokens_saved": 0,
                "compression_ratio": 1.0,
                "transforms_applied": [],
                "ccr_hashes": [],
                "skipped_reason": "bypass_header",
            }),
        );
    }

    if value.get("messages").is_none() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({
                "error": {
                    "type": "invalid_request",
                    "message": "Missing required field: messages"
                }
            }),
        );
    }

    if value.get("model").is_none() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({
                "error": {
                    "type": "invalid_request",
                    "message": "Missing required field: model"
                }
            }),
        );
    }

    let mutation = compress_json_request_with_auth(
        &body,
        ApiShape::OpenAiChatCompletions,
        &state.store,
        classify_auth_mode(&headers),
    );
    record_mutation_stats(&mutation);
    let mutated = serde_json::from_slice::<serde_json::Value>(&mutation.body).unwrap_or(value);
    let tokens_saved = mutation.tokens_before.saturating_sub(mutation.tokens_after);
    let compression_ratio = if mutation.tokens_before == 0 {
        1.0
    } else {
        mutation.tokens_after as f64 / mutation.tokens_before as f64
    };

    let mut response = json!({
        "messages": mutated.get("messages").cloned().unwrap_or_else(|| json!([])),
        "tokens_before": mutation.tokens_before,
        "tokens_after": mutation.tokens_after,
        "tokens_saved": tokens_saved,
        "compression_ratio": compression_ratio,
        "transforms_applied": if mutation.compressed { vec!["ccr_live_zone"] } else { Vec::<&str>::new() },
        "ccr_hashes": mutation.hashes,
    });

    if let Some(reason) = mutation.skipped_reason {
        response["skipped_reason"] = json!(reason);
    }
    if let Some(tools) = mutated.get("tools") {
        response["tools"] = tools.clone();
    }

    json_response(StatusCode::OK, response)
}

#[derive(Debug)]
struct ParsedToolCall {
    hash: String,
    query: Option<String>,
}

fn parse_retrieve_tool_call(
    tool_call: &serde_json::Value,
    provider: &str,
) -> Option<ParsedToolCall> {
    if provider.eq_ignore_ascii_case("openai") {
        let function = tool_call.get("function")?;
        let name = function.get("name").and_then(serde_json::Value::as_str)?;
        if !is_retrieve_tool_name(name) {
            return None;
        }
        let arguments = function
            .get("arguments")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("{}");
        let arguments: serde_json::Value = serde_json::from_str(arguments).ok()?;
        parse_retrieve_arguments(&arguments)
    } else {
        let name = tool_call
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if !is_retrieve_tool_name(name) {
            return None;
        }
        let input = tool_call.get("input")?;
        parse_retrieve_arguments(input)
    }
}

fn is_retrieve_tool_name(name: &str) -> bool {
    name == "headroom_retrieve" || name.ends_with("__headroom_retrieve")
}

fn parse_retrieve_arguments(arguments: &serde_json::Value) -> Option<ParsedToolCall> {
    let hash = arguments
        .get("hash")
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let query = arguments
        .get("query")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    Some(ParsedToolCall { hash, query })
}

fn format_tool_result(
    tool_call: &serde_json::Value,
    provider: &str,
    data: &serde_json::Value,
) -> serde_json::Value {
    let content = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());
    if provider.eq_ignore_ascii_case("openai") {
        json!({
            "role": "tool",
            "tool_call_id": tool_call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(""),
            "content": content,
        })
    } else {
        json!({
            "type": "tool_result",
            "tool_use_id": tool_call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(""),
            "content": content,
        })
    }
}

fn retrieve_response(store: &SqliteStore, hash: &str, query: Option<&str>) -> Response<Body> {
    let value = retrieve_value(store, hash, query);
    let status = if value.get("error").is_some() {
        if !valid_hash(hash) {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::NOT_FOUND
        }
    } else {
        StatusCode::OK
    };
    json_response(status, value)
}

fn retrieve_value(store: &SqliteStore, hash: &str, query: Option<&str>) -> serde_json::Value {
    if !valid_hash(hash) {
        return json!({
            "hash": hash,
            "error": format!("invalid hash format; expected {HASH_HEX_LEN} hex characters"),
        });
    }

    let Some(original) = crate::decompress_hash(hash, store) else {
        return json!({
            "hash": hash,
            "error": "entry not found",
        });
    };

    if let Some(query) = query.filter(|query| !query.trim().is_empty()) {
        let results = search_content(&original, query);
        json!({
            "hash": hash,
            "query": query,
            "results": results,
            "count": results.len(),
        })
    } else {
        json!({
            "hash": hash,
            "original_content": original,
            "original_tokens": crate::estimate_tokens(&original),
            "original_item_count": 1,
            "compressed_item_count": 1,
            "tool_name": "headroom_retrieve",
            "retrieval_count": 1,
        })
    }
}

fn search_content(content: &str, query: &str) -> Vec<RetrieveSearchResult> {
    let query = query.to_lowercase();
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| line.to_lowercase().contains(&query))
        .map(|(index, line)| RetrieveSearchResult {
            line: index + 1,
            text: line.to_string(),
        })
        .collect()
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == HASH_HEX_LEN && hash.chars().all(|char| char.is_ascii_hexdigit())
}

async fn proxy_get_handler(
    State(state): State<ProxyState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    ws: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
    req: Request<Body>,
) -> Response<Body> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();
    let class = classify_request(&method, &path);

    debug!(
        method = %method,
        path,
        provider = ?class.provider,
        target = ?class.target,
        skipped_reason = ?class.skipped_reason,
        "request classification"
    );

    if let Ok(upgrade) = ws {
        stats::record_request();
        let headers = req.headers().clone();
        // Subprotocol negotiation: accept the client connection with
        // its first requested subprotocol so strict clients (Codex)
        // see it echoed; the raw `sec-websocket-protocol` header is
        // forwarded upstream by `copy_websocket_headers`. Mirrors
        // `headroom/proxy/handlers/openai.py:3341-3354`.
        let client_protocols = websocket_client_protocols(&headers);
        let upgrade = if client_protocols.is_empty() {
            upgrade
        } else {
            upgrade.protocols(client_protocols)
        };
        return upgrade
            .on_upgrade(move |socket| {
                proxy_websocket(
                    socket,
                    state,
                    uri,
                    headers,
                    class.provider,
                    Some(client_addr),
                )
            })
            .into_response();
    }

    proxy_http(state, req, class, Some(client_addr)).await
}

async fn proxy_http_handler(
    State(state): State<ProxyState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let class = classify_request(&method, &path);

    debug!(
        method = %method,
        path,
        provider = ?class.provider,
        target = ?class.target,
        skipped_reason = ?class.skipped_reason,
        "request classification"
    );

    proxy_http(state, req, class, Some(client_addr)).await
}

async fn proxy_http(
    state: ProxyState,
    req: Request<Body>,
    class: RequestClass,
    client_addr: Option<SocketAddr>,
) -> Response<Body> {
    stats::record_request();

    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let upstream_url = match upstream_url(&state, class.provider, &uri) {
        Ok(url) => url,
        Err(err) => return error_response(StatusCode::BAD_GATEWAY, err.to_string()),
    };

    if let Some(target) = class.target {
        if !state.compression_enabled || !state.compression_mode.allows_compression() {
            stats::record_skipped_request("compression_disabled");
            return match forward_streaming(
                state,
                method,
                headers,
                req.into_body(),
                upstream_url,
                uri.path(),
                client_addr,
            )
            .await
            {
                Ok(response) => response,
                Err(err) => upstream_error_response(err),
            };
        }

        if compression_bypass_requested(&headers) {
            stats::record_skipped_request("bypass_header");
            return match forward_streaming(
                state,
                method,
                headers,
                req.into_body(),
                upstream_url,
                uri.path(),
                client_addr,
            )
            .await
            {
                Ok(response) => response,
                Err(err) => upstream_error_response(err),
            };
        }

        if !is_application_json(&headers) {
            stats::record_skipped_request("non_json_content_type");
            return match forward_streaming(
                state,
                method,
                headers,
                req.into_body(),
                upstream_url,
                uri.path(),
                client_addr,
            )
            .await
            {
                Ok(response) => response,
                Err(err) => upstream_error_response(err),
            };
        }

        match buffer_and_forward_known_json(state, req, target, upstream_url, client_addr).await {
            Ok(response) => response,
            Err(err) if err.to_string().contains("payload_too_large") => {
                error_response(StatusCode::PAYLOAD_TOO_LARGE, err.to_string())
            }
            Err(err) => upstream_error_response(err),
        }
    } else {
        if let Some(reason) = class.skipped_reason {
            stats::record_skipped_request(reason);
        }

        if let Some(batch_id) = anthropic_batch_results_batch_id(&method, uri.path()) {
            if let Some(contexts) = state.batch_context(&batch_id) {
                return match forward_batch_results(
                    state,
                    headers,
                    upstream_url,
                    uri.path(),
                    client_addr,
                    contexts,
                )
                .await
                {
                    Ok(response) => response,
                    Err(err) => upstream_error_response(err),
                };
            }
        }

        match forward_streaming(
            state,
            method,
            headers,
            req.into_body(),
            upstream_url,
            uri.path(),
            client_addr,
        )
        .await
        {
            Ok(response) => response,
            Err(err) => upstream_error_response(err),
        }
    }
}

async fn buffer_and_forward_known_json(
    state: ProxyState,
    req: Request<Body>,
    target: CompressionTarget,
    upstream_url: Url,
    client_addr: Option<SocketAddr>,
) -> HrResult<Response<Body>> {
    let method = req.method().clone();
    let mut headers = req.headers().clone();
    // Telemetry keys off the CLIENT path: the upstream URL may carry a
    // configured base-path prefix that would break exact matching.
    let request_path = req.uri().path().to_string();
    let body = read_limited_body(req, state.max_body_bytes).await?;
    let parsed_request: Option<Value> = serde_json::from_slice(&body).ok();

    // Empty-batch validation, mirroring `anthropic.py:2458-2469`: a
    // missing or empty `requests` field is rejected with 400 before
    // anything is forwarded upstream.
    if target == CompressionTarget::AnthropicMessageBatches {
        if let Some(parsed) = &parsed_request {
            let requests_missing_or_empty = match parsed.get("requests") {
                None | Some(Value::Null) => true,
                Some(Value::Array(requests)) => requests.is_empty(),
                Some(_) => false,
            };
            if requests_missing_or_empty {
                stats::record_skipped_request("empty_batch");
                return Ok(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Missing or empty 'requests' field in batch request",
                        }
                    }),
                ));
            }
        }
    }

    // Session identity + provider-confirmed frozen floor for the
    // surfaces the reference tracks (`anthropic.py:885-892`,
    // `openai.py:1531-1576`).
    let session_provider = match target {
        CompressionTarget::AnthropicMessages => Some(SessionProvider::Anthropic),
        CompressionTarget::OpenAiChatCompletions | CompressionTarget::OpenAiResponses => {
            Some(SessionProvider::OpenAi)
        }
        CompressionTarget::AnthropicMessageBatches => None,
    };
    let session_id = session_provider.and_then(|provider| {
        parsed_request.as_ref().map(|parsed| {
            // The Responses surface has its own session shaping
            // (string `instructions` only); chat/Anthropic walk the
            // body's `messages`.
            let id = if target == CompressionTarget::OpenAiResponses {
                SessionTrackers::compute_responses_session_id(&headers, parsed)
            } else {
                SessionTrackers::compute_session_id(&headers, parsed)
            };
            (provider, id)
        })
    });
    let frozen_message_count = session_id
        .as_ref()
        .map(|(provider, id)| state.sessions.frozen_message_count(*provider, id))
        .unwrap_or(0);

    // Session-sticky beta-header merge: betas seen earlier in a
    // session are merged into later requests, case-insensitively
    // deduped, first-seen order. The reference applies this to
    // `anthropic-beta` on /v1/messages (`anthropic.py:895-938`) and to
    // `openai-beta` on the chat and Responses HTTP paths
    // (`openai.py:1535-1574`).
    let beta_header_name = match target {
        CompressionTarget::AnthropicMessages => Some("anthropic-beta"),
        CompressionTarget::OpenAiChatCompletions | CompressionTarget::OpenAiResponses => {
            Some("openai-beta")
        }
        CompressionTarget::AnthropicMessageBatches => None,
    };
    if let (Some(header_name), Some((provider, id))) = (beta_header_name, &session_id) {
        let client_value = headers
            .get(header_name)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        let merged = state
            .sessions
            .sticky_betas(*provider, id, client_value.as_deref());
        if !merged.is_empty() && Some(merged.as_str()) != client_value.as_deref() {
            if let Ok(value) = HeaderValue::from_str(&merged) {
                headers.insert(HeaderName::from_static(header_name), value);
            }
        }
    }

    let shape = match target {
        CompressionTarget::OpenAiChatCompletions => ApiShape::OpenAiChatCompletions,
        CompressionTarget::OpenAiResponses => ApiShape::OpenAiResponses,
        CompressionTarget::AnthropicMessages => ApiShape::AnthropicMessages,
        CompressionTarget::AnthropicMessageBatches => ApiShape::AnthropicMessageBatches,
    };
    let mut mutation = compress_json_request_ctx(
        &body,
        shape,
        CompressContext {
            store: &state.store,
            auth_mode: classify_auth_mode(&headers),
            frozen_message_count,
            compress_user_text: state.compress_user_text,
            provider_metadata: true,
        },
    );
    record_mutation_stats(&mutation);
    let body_to_forward = std::mem::take(&mut mutation.body);

    // Session context handed to the response side so provider-confirmed
    // cache usage feeds the tracker. Token estimates come from the
    // forwarded (post-compression) messages, mirroring the reference's
    // `next_forwarded_messages` (`anthropic.py:2216-2228`).
    let session = session_id.map(|(provider, id)| {
        let message_token_estimates = serde_json::from_slice::<Value>(&body_to_forward)
            .ok()
            .map(|forwarded| {
                // Route by surface, not body shape: a Responses body
                // carrying a legacy `messages` alias still holds
                // Responses ITEMS, which the chat estimator would
                // misread (missing `function_call_output.output`).
                if target == CompressionTarget::OpenAiResponses {
                    return crate::session::estimate_responses_request_tokens(&forwarded);
                }
                forwarded
                    .get("messages")
                    .and_then(Value::as_array)
                    .map(|messages| estimate_message_tokens(messages))
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        SseSessionCtx {
            trackers: state.sessions.clone(),
            provider,
            session_id: id,
            message_token_estimates,
        }
    });

    let mut response = match target {
        CompressionTarget::AnthropicMessages => {
            forward_with_ccr_continuation(
                &state,
                method,
                &headers,
                body_to_forward,
                upstream_url,
                &request_path,
                client_addr,
                CcrProvider::Anthropic,
                session,
            )
            .await?
        }
        CompressionTarget::OpenAiChatCompletions => {
            forward_with_ccr_continuation(
                &state,
                method,
                &headers,
                body_to_forward,
                upstream_url,
                &request_path,
                client_addr,
                CcrProvider::OpenAi,
                session,
            )
            .await?
        }
        CompressionTarget::AnthropicMessageBatches => {
            forward_batch_create(
                &state,
                method,
                &headers,
                body_to_forward,
                upstream_url,
                &request_path,
                client_addr,
            )
            .await?
        }
        CompressionTarget::OpenAiResponses => {
            forward_buffered(
                state,
                method,
                headers,
                body_to_forward,
                upstream_url,
                &request_path,
                client_addr,
                session,
            )
            .await?
        }
    };
    apply_compression_response_headers(response.headers_mut(), &mutation);
    Ok(response)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CcrProvider {
    Anthropic,
    OpenAi,
}

impl CcrProvider {
    fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }
}

/// Forwards a compressed target request and, when the upstream JSON response
/// contains `headroom_retrieve` tool calls, resolves them from the CCR store
/// and continues the conversation upstream until the response has no CCR tool
/// calls left (or the round limit is hit). Mirrors the reference proxy's
/// non-streaming CCR response handling.
#[allow(clippy::too_many_arguments)]
async fn forward_with_ccr_continuation(
    state: &ProxyState,
    method: Method,
    headers: &HeaderMap,
    body: Vec<u8>,
    upstream_url: Url,
    request_path: &str,
    client_addr: Option<SocketAddr>,
    provider: CcrProvider,
    session: Option<SseSessionCtx>,
) -> HrResult<Response<Body>> {
    let request_id = ensure_request_id(headers);
    let request_body: Option<Value> = serde_json::from_slice(&body).ok();
    let mut builder = state
        .client
        .request(method, upstream_url.clone())
        .body(body);
    builder = apply_headers(builder, headers, client_addr, &request_id);
    let response = builder.send().await?;
    info!(
        upstream = %upstream_url,
        status = %response.status(),
        "request forwarded"
    );

    let Some(request_body) = request_body else {
        return response_to_axum(response, &request_id, request_path, session);
    };
    if !ccr_response_buffer_eligible(&response) {
        return response_to_axum(response, &request_id, request_path, session);
    }

    let status = response.status();
    let response_headers = response.headers().clone();
    record_rate_limit_headers(&response_headers, request_path);
    let bytes = response.bytes().await?;
    let Ok(initial) = serde_json::from_slice::<Value>(&bytes) else {
        return buffered_response_to_axum(
            status,
            &response_headers,
            bytes.to_vec(),
            &request_id,
            false,
        );
    };
    if extract_ccr_tool_calls(&initial, provider).is_empty() {
        update_session_from_json_response(&session, &initial, provider);
        return buffered_response_to_axum(
            status,
            &response_headers,
            bytes.to_vec(),
            &request_id,
            false,
        );
    }

    let (final_response, changed) = run_ccr_continuation(
        state,
        &request_body,
        initial,
        &upstream_url,
        headers,
        client_addr,
        provider,
    )
    .await;
    update_session_from_json_response(&session, &final_response, provider);
    if !changed {
        // No continuation round succeeded — return the upstream bytes
        // untouched so the client can resolve the tool call itself.
        return buffered_response_to_axum(
            status,
            &response_headers,
            bytes.to_vec(),
            &request_id,
            false,
        );
    }
    let body = serde_json::to_vec(&final_response)?;
    buffered_response_to_axum(status, &response_headers, body, &request_id, true)
}

/// Whether the upstream response can be buffered for CCR tool call handling:
/// a successful, identity-encoded JSON body (SSE and encoded bodies stay on
/// the streaming passthrough path).
fn ccr_response_buffer_eligible(response: &reqwest::Response) -> bool {
    response.status() == StatusCode::OK
        && response.headers().get("content-encoding").is_none()
        && is_application_json(response.headers())
}

/// Runs the CCR continuation loop. Returns the latest upstream response and
/// whether it was changed by continuation or private-tool stripping.
async fn run_ccr_continuation(
    state: &ProxyState,
    request_body: &Value,
    initial: Value,
    upstream_url: &Url,
    headers: &HeaderMap,
    client_addr: Option<SocketAddr>,
    provider: CcrProvider,
) -> (Value, bool) {
    let mut messages = request_body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut current = initial;
    let mut changed = false;
    let mut rounds = 0;

    while rounds < MAX_CCR_CONTINUATION_ROUNDS {
        let tool_call_count = provider_tool_calls(&current, provider).len();
        let ccr_calls = extract_ccr_tool_calls(&current, provider);
        if ccr_calls.is_empty() {
            break;
        }
        if provider == CcrProvider::OpenAi && openai_choice_count(&current) != 1 {
            warn!(
                choices = openai_choice_count(&current),
                "ccr continuation skipped because multiple OpenAI choices are present"
            );
            if let Some(stripped) = strip_ccr_tool_calls(&current, provider) {
                current = stripped;
                changed = true;
            }
            break;
        }
        if tool_call_count != ccr_calls.len() {
            warn!(
                total_tool_calls = tool_call_count,
                ccr_tool_calls = ccr_calls.len(),
                provider = provider.as_str(),
                "ccr continuation skipped because non-headroom tool calls are present"
            );
            if let Some(stripped) = strip_ccr_tool_calls(&current, provider) {
                current = stripped;
                changed = true;
            }
            break;
        }
        rounds += 1;
        stats::record_ccr_continuation_retrievals(ccr_calls.len() as u64);
        info!(
            count = ccr_calls.len(),
            round = rounds,
            provider = provider.as_str(),
            "ccr continuation: resolving retrieval tool calls"
        );

        messages.push(extract_assistant_message(&current, provider));
        match provider {
            CcrProvider::Anthropic => {
                let blocks = ccr_calls
                    .iter()
                    .map(|(tool_call, parsed)| {
                        let data =
                            retrieve_value(&state.store, &parsed.hash, parsed.query.as_deref());
                        format_tool_result(tool_call, provider.as_str(), &data)
                    })
                    .collect::<Vec<_>>();
                messages.push(json!({"role": "user", "content": blocks}));
            }
            CcrProvider::OpenAi => {
                for (tool_call, parsed) in &ccr_calls {
                    let data = retrieve_value(&state.store, &parsed.hash, parsed.query.as_deref());
                    messages.push(format_tool_result(tool_call, provider.as_str(), &data));
                }
            }
        }

        let mut continuation_body = request_body.clone();
        continuation_body["messages"] = Value::Array(messages.clone());

        match send_ccr_continuation(
            state,
            &continuation_body,
            upstream_url,
            headers,
            client_addr,
        )
        .await
        {
            Ok(next) => {
                stats::record_ccr_continuation_round();
                changed = true;
                current = next;
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "ccr continuation request failed; returning previous response"
                );
                break;
            }
        }
    }

    if let Some(stripped) = strip_ccr_tool_calls(&current, provider) {
        current = stripped;
        changed = true;
    }

    if rounds >= MAX_CCR_CONTINUATION_ROUNDS
        && !extract_ccr_tool_calls(&current, provider).is_empty()
    {
        warn!("ccr continuation hit max rounds with unresolved retrieval tool calls");
    }
    (current, changed)
}

async fn send_ccr_continuation(
    state: &ProxyState,
    body: &Value,
    upstream_url: &Url,
    headers: &HeaderMap,
    client_addr: Option<SocketAddr>,
) -> HrResult<Value> {
    let request_id = generate_request_id();
    let payload = serde_json::to_vec(body)?;
    let mut sanitized = headers.clone();
    for name in ["content-encoding", "accept-encoding"] {
        sanitized.remove(name);
    }
    sanitized.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let mut builder = state
        .client
        .request(Method::POST, upstream_url.clone())
        .body(payload);
    builder = apply_headers(builder, &sanitized, client_addr, &request_id);
    let response = builder.send().await?;
    let status = response.status();
    if status != StatusCode::OK {
        return Err(error(format!(
            "ccr continuation upstream returned status {status}"
        )));
    }
    let bytes = response.bytes().await?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Extracts `headroom_retrieve` tool calls from a provider-shaped response.
fn extract_ccr_tool_calls(response: &Value, provider: CcrProvider) -> Vec<(Value, ParsedToolCall)> {
    provider_tool_calls(response, provider)
        .into_iter()
        .filter_map(|tool_call| {
            parse_retrieve_tool_call(tool_call, provider.as_str())
                .map(|parsed| (tool_call.clone(), parsed))
        })
        .collect()
}

fn provider_tool_calls(response: &Value, provider: CcrProvider) -> Vec<&Value> {
    match provider {
        CcrProvider::Anthropic => response
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
                    .collect()
            })
            .unwrap_or_default(),
        CcrProvider::OpenAi => response
            .get("choices")
            .and_then(Value::as_array)
            .map(|choices| {
                choices
                    .iter()
                    .filter_map(|choice| choice.get("message"))
                    .filter_map(|message| message.get("tool_calls"))
                    .filter_map(Value::as_array)
                    .flat_map(|calls| calls.iter())
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn openai_choice_count(response: &Value) -> usize {
    response
        .get("choices")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default()
}

/// Removes proxy-private CCR retrieval calls from a mixed tool-call response,
/// leaving client-owned tool calls visible to the caller.
fn strip_ccr_tool_calls(response: &Value, provider: CcrProvider) -> Option<Value> {
    match provider {
        CcrProvider::Anthropic => {
            let content = response.get("content").and_then(Value::as_array)?;
            let mut removed = false;
            let mut remaining_tool_calls = 0;
            let stripped_content = content
                .iter()
                .filter_map(|block| {
                    let is_tool_use = block.get("type").and_then(Value::as_str) == Some("tool_use");
                    if is_tool_use && parse_retrieve_tool_call(block, provider.as_str()).is_some() {
                        removed = true;
                        return None;
                    }
                    if is_tool_use {
                        remaining_tool_calls += 1;
                    }
                    Some(block.clone())
                })
                .collect::<Vec<_>>();

            // Never strip a response down to zero tool_use blocks: that would
            // leave `stop_reason: "tool_use"` with nothing to respond to.
            // Headroom-only responses are left to the continuation loop (or
            // returned as-is on failure/round limit).
            if !removed || remaining_tool_calls == 0 {
                return None;
            }
            let mut stripped = response.clone();
            stripped["content"] = Value::Array(stripped_content);
            Some(stripped)
        }
        CcrProvider::OpenAi => {
            let mut stripped = response.clone();
            let choices = stripped
                .get_mut("choices")
                .and_then(Value::as_array_mut)
                .filter(|choices| !choices.is_empty())?;
            let mut removed = false;
            let mut remaining_tool_calls = 0;
            for choice in choices.iter_mut() {
                let (original_len, stripped_len) = {
                    let Some(tool_calls) = choice
                        .get_mut("message")
                        .and_then(|message| message.get_mut("tool_calls"))
                        .and_then(Value::as_array_mut)
                    else {
                        continue;
                    };
                    let original_len = tool_calls.len();
                    tool_calls
                        .retain(|call| parse_retrieve_tool_call(call, provider.as_str()).is_none());
                    (original_len, tool_calls.len())
                };
                if stripped_len != original_len {
                    removed = true;
                }
                remaining_tool_calls += stripped_len;
                // Keep the choice in place so `index` stays aligned with array
                // position, but rewrite a now tool-free choice as a normal text
                // turn: an empty `tool_calls` array with `finish_reason:
                // "tool_calls"` would leave the client nothing to respond to.
                if original_len > 0 && stripped_len == 0 {
                    if let Some(message) = choice.get_mut("message").and_then(Value::as_object_mut)
                    {
                        message.remove("tool_calls");
                    }
                    if choice.get("finish_reason").and_then(Value::as_str) == Some("tool_calls") {
                        choice["finish_reason"] = Value::String("stop".to_string());
                    }
                }
            }
            // A response whose only tool calls were proxy-private is left to the
            // continuation loop (or returned as-is on failure/round limit).
            if !removed || remaining_tool_calls == 0 {
                return None;
            }
            Some(stripped)
        }
    }
}

/// Rebuilds the assistant turn from a provider response for continuation.
///
/// The OpenAI arm reads `choices[0]` only: `run_ccr_continuation` skips
/// continuation for multi-choice responses, so a single choice is guaranteed
/// here. Revisit this if that guard ever changes.
fn extract_assistant_message(response: &Value, provider: CcrProvider) -> Value {
    match provider {
        CcrProvider::Anthropic => json!({
            "role": "assistant",
            "content": response.get("content").cloned().unwrap_or_else(|| json!([])),
        }),
        CcrProvider::OpenAi => {
            let message = response
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            let mut assistant = json!({
                "role": "assistant",
                "content": message.get("content").cloned().unwrap_or(Value::Null),
            });
            if let Some(tool_calls) = message.get("tool_calls") {
                assistant["tool_calls"] = tool_calls.clone();
            }
            assistant
        }
    }
}

/// Forwards an Anthropic batch create and records per-request context from
/// the compressed body so CCR tool calls in batch results can be continued.
#[allow(clippy::too_many_arguments)]
async fn forward_batch_create(
    state: &ProxyState,
    method: Method,
    headers: &HeaderMap,
    body: Vec<u8>,
    upstream_url: Url,
    request_path: &str,
    client_addr: Option<SocketAddr>,
) -> HrResult<Response<Body>> {
    let request_id = ensure_request_id(headers);
    let compressed_body: Option<Value> = serde_json::from_slice(&body).ok();
    let mut builder = state
        .client
        .request(method, upstream_url.clone())
        .body(body);
    builder = apply_headers(builder, headers, client_addr, &request_id);
    let response = builder.send().await?;
    info!(
        upstream = %upstream_url,
        status = %response.status(),
        "request forwarded"
    );

    if !ccr_response_buffer_eligible(&response) {
        return response_to_axum(response, &request_id, request_path, None);
    }

    let status = response.status();
    let response_headers = response.headers().clone();
    let bytes = response.bytes().await?;
    if let (Some(compressed_body), Ok(response_json)) =
        (compressed_body, serde_json::from_slice::<Value>(&bytes))
    {
        if let Some(batch_id) = response_json.get("id").and_then(Value::as_str) {
            state.record_batch_context(batch_id, &compressed_body);
        }
    }
    buffered_response_to_axum(
        status,
        &response_headers,
        bytes.to_vec(),
        &request_id,
        false,
    )
}

fn anthropic_batch_results_batch_id(method: &Method, path: &str) -> Option<String> {
    if method != Method::GET {
        return None;
    }
    let rest = path.strip_prefix("/v1/messages/batches/")?;
    let (batch_id, tail) = rest.split_once('/')?;
    if tail != "results" || batch_id.is_empty() {
        return None;
    }
    Some(batch_id.to_string())
}

/// Fetches Anthropic batch results and post-processes each JSONL line whose
/// message contains `headroom_retrieve` tool calls, continuing those requests
/// against `/v1/messages` with the stored batch context.
async fn forward_batch_results(
    state: ProxyState,
    headers: HeaderMap,
    upstream_url: Url,
    request_path: &str,
    client_addr: Option<SocketAddr>,
    contexts: HashMap<String, Value>,
) -> HrResult<Response<Body>> {
    let request_id = ensure_request_id(&headers);
    let mut builder = state.client.request(Method::GET, upstream_url.clone());
    builder = apply_headers(builder, &headers, client_addr, &request_id);
    let response = builder.send().await?;
    info!(
        upstream = %upstream_url,
        status = %response.status(),
        "request forwarded"
    );

    if response.status() != StatusCode::OK || response.headers().get("content-encoding").is_some() {
        return response_to_axum(response, &request_id, request_path, None);
    }

    let status = response.status();
    let response_headers = response.headers().clone();
    let bytes = response.bytes().await?;
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return buffered_response_to_axum(
            status,
            &response_headers,
            bytes.to_vec(),
            &request_id,
            false,
        );
    };

    let mut messages_url = state.anthropic_upstream.clone();
    messages_url.set_path("/v1/messages");

    let mut processed_any = false;
    let mut lines_out: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(mut result) = serde_json::from_str::<Value>(line) else {
            lines_out.push(line.to_string());
            continue;
        };
        let custom_id = result
            .get("custom_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let message = result
            .get("result")
            .and_then(|inner| inner.get("message"))
            .cloned();
        let (Some(custom_id), Some(message)) = (custom_id, message) else {
            lines_out.push(line.to_string());
            continue;
        };
        let Some(params) = contexts.get(&custom_id) else {
            lines_out.push(line.to_string());
            continue;
        };
        if extract_ccr_tool_calls(&message, CcrProvider::Anthropic).is_empty() {
            lines_out.push(line.to_string());
            continue;
        }

        let (final_message, changed) = run_ccr_continuation(
            &state,
            params,
            message,
            &messages_url,
            &headers,
            client_addr,
            CcrProvider::Anthropic,
        )
        .await;
        if changed {
            result["result"]["message"] = final_message;
            stats::record_ccr_batch_result_processed();
            processed_any = true;
            info!(custom_id, "ccr batch result continuation complete");
            lines_out.push(serde_json::to_string(&result)?);
        } else {
            lines_out.push(line.to_string());
        }
    }

    if !processed_any {
        return buffered_response_to_axum(
            status,
            &response_headers,
            bytes.to_vec(),
            &request_id,
            false,
        );
    }
    let mut body = lines_out.join("\n");
    body.push('\n');
    buffered_response_to_axum(
        status,
        &response_headers,
        body.into_bytes(),
        &request_id,
        true,
    )
}

/// Builds a client response from buffered upstream bytes, mirroring
/// `response_to_axum` header filtering. When the body was replaced, stale
/// content-length/content-encoding headers are dropped.
fn buffered_response_to_axum(
    status: StatusCode,
    headers: &HeaderMap,
    body: Vec<u8>,
    request_id: &str,
    body_replaced: bool,
) -> HrResult<Response<Body>> {
    let upstream_request_id = upstream_request_id(headers);
    let mut builder = Response::builder().status(status);
    let connection_listed = connection_listed_headers(headers);

    for (name, value) in headers {
        if is_response_drop_header(name) {
            continue;
        }
        if connection_listed
            .iter()
            .any(|listed| listed.eq_ignore_ascii_case(name.as_str()))
        {
            continue;
        }
        if body_replaced
            && (name == CONTENT_LENGTH || name.as_str().eq_ignore_ascii_case("content-encoding"))
        {
            continue;
        }
        builder = builder.header(name, value);
    }

    let mut response = builder.body(Body::from(body))?;
    insert_header(response.headers_mut(), "x-request-id", request_id);
    if let Some(upstream_request_id) = upstream_request_id.as_deref() {
        insert_header(
            response.headers_mut(),
            "headroom-upstream-request-id",
            upstream_request_id,
        );
    }
    Ok(response)
}

async fn read_limited_body(req: Request<Body>, max_body_bytes: usize) -> HrResult<Vec<u8>> {
    let headers = req.headers();
    if let Some(content_length) = headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
    {
        if content_length > max_body_bytes {
            return Err(error(format!(
                "payload_too_large: Content-Length {content_length} exceeds compression buffer limit {max_body_bytes}"
            )));
        }
    }

    let body = to_bytes(req.into_body(), max_body_bytes)
        .await
        .map_err(|err| {
            error(format!(
                "payload_too_large: request body exceeds compression buffer limit {max_body_bytes}: {err}",
            ))
        })?;
    Ok(body.to_vec())
}

fn record_mutation_stats(mutation: &RequestCompression) {
    if mutation.compressed {
        debug!(
            bytes_before = mutation.bytes_before,
            bytes_after = mutation.bytes_after,
            tokens_before = mutation.tokens_before,
            tokens_after = mutation.tokens_after,
            "compression byte token counts"
        );
        stats::record_compressed_request(
            mutation.bytes_before as u64,
            mutation.bytes_after as u64,
            mutation.tokens_before as u64,
            mutation.tokens_after as u64,
        );
        info!(
            hashes = ?mutation.hashes,
            bytes_before = mutation.bytes_before,
            bytes_after = mutation.bytes_after,
            tokens_before = mutation.tokens_before,
            tokens_after = mutation.tokens_after,
            "compression decision"
        );
    } else if let Some(reason) = &mutation.skipped_reason {
        stats::record_skipped_request(reason);
        info!(reason, "compression skipped");
    }
}

#[allow(clippy::too_many_arguments)]
async fn forward_buffered(
    state: ProxyState,
    method: Method,
    headers: HeaderMap,
    body: Vec<u8>,
    upstream_url: Url,
    request_path: &str,
    client_addr: Option<SocketAddr>,
    session: Option<SseSessionCtx>,
) -> HrResult<Response<Body>> {
    let request_id = ensure_request_id(&headers);
    let mut builder = state
        .client
        .request(method, upstream_url.clone())
        .body(body);
    builder = apply_headers(builder, &headers, client_addr, &request_id);
    let response = builder.send().await?;
    info!(
        upstream = %upstream_url,
        status = %response.status(),
        "request forwarded"
    );

    // Buffered JSON responses feed the session prefix tracker, like
    // the reference Responses HTTP path (`openai.py:2318-2341` /
    // `3159-3176`): a non-streaming response confirming cached tokens
    // must advance the frozen floor for the next turn.
    if session.is_some() && ccr_response_buffer_eligible(&response) {
        let status = response.status();
        let response_headers = response.headers().clone();
        record_rate_limit_headers(&response_headers, request_path);
        let bytes = response.bytes().await?;
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            update_session_from_json_response(&session, &value, CcrProvider::OpenAi);
        }
        return buffered_response_to_axum(
            status,
            &response_headers,
            bytes.to_vec(),
            &request_id,
            false,
        );
    }
    response_to_axum(response, &request_id, request_path, session)
}

async fn forward_streaming(
    state: ProxyState,
    method: Method,
    headers: HeaderMap,
    body: Body,
    upstream_url: Url,
    request_path: &str,
    client_addr: Option<SocketAddr>,
) -> HrResult<Response<Body>> {
    let request_id = ensure_request_id(&headers);
    let stream = body.into_data_stream();
    let mut builder = state
        .client
        .request(method, upstream_url.clone())
        .body(reqwest::Body::wrap_stream(stream));
    builder = apply_headers(builder, &headers, client_addr, &request_id);
    let response = builder.send().await?;
    info!(
        upstream = %upstream_url,
        status = %response.status(),
        "request forwarded"
    );
    response_to_axum(response, &request_id, request_path, None)
}

fn apply_headers(
    mut builder: reqwest::RequestBuilder,
    headers: &HeaderMap,
    client_addr: Option<SocketAddr>,
    request_id: &str,
) -> reqwest::RequestBuilder {
    builder = builder.header("x-request-id", request_id);
    builder = builder.header("x-forwarded-proto", "http");
    if let Some(host) = headers.get(HOST) {
        builder = builder.header("x-forwarded-host", host);
    }
    if let Some(xff) = forwarded_for(headers, client_addr) {
        builder = builder.header("x-forwarded-for", xff);
    }
    if let Some(account_id) = chatgpt_account_id_from_authorization(headers) {
        builder = builder.header("ChatGPT-Account-ID", account_id);
    }

    let connection_listed = connection_listed_headers(headers);
    for (name, value) in headers {
        if is_request_drop_header(name) {
            continue;
        }
        if connection_listed
            .iter()
            .any(|listed| listed.eq_ignore_ascii_case(name.as_str()))
        {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder
}

fn response_to_axum(
    response: reqwest::Response,
    request_id: &str,
    request_path: &str,
    session: Option<SseSessionCtx>,
) -> HrResult<Response<Body>> {
    let status = response.status();
    let headers = response.headers().clone();
    let upstream_request_id = upstream_request_id(&headers);
    record_rate_limit_headers(&headers, request_path);
    let sse_kind = sse_stream_kind(&headers, request_path);
    let mut builder = Response::builder().status(status);
    let connection_listed = connection_listed_headers(&headers);

    for (name, value) in headers {
        let Some(name) = name else {
            continue;
        };
        if is_response_drop_header(&name) {
            continue;
        }
        if connection_listed
            .iter()
            .any(|listed| listed.eq_ignore_ascii_case(name.as_str()))
        {
            continue;
        }
        builder = builder.header(name, value);
    }

    let stream = response.bytes_stream();
    let body = if let Some(kind) = sse_kind {
        stats::record_sse_stream();
        // Tee each chunk into a bounded mpsc so the spawned
        // state-machine task can update telemetry without holding up
        // the client. `try_send` never blocks: if the parser falls
        // behind, the telemetry chunk is dropped and the client byte
        // path is unaffected (mirrors `headroom-proxy/src/proxy.rs`
        // PR-C1 contract).
        let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(SSE_PARSER_QUEUE_DEPTH);
        tokio::spawn(run_sse_state_machine(kind, rx, session));
        Body::from_stream(stream.map(move |chunk| {
            if let Ok(bytes) = &chunk {
                if let Err(err) = tx.try_send(bytes.clone()) {
                    debug!(
                        error = %err,
                        "sse parser queue full or closed; skipping telemetry chunk"
                    );
                }
            }
            chunk
        }))
    } else {
        Body::from_stream(stream)
    };
    let mut response = builder.body(body)?;
    insert_header(response.headers_mut(), "x-request-id", request_id);
    if let Some(upstream_request_id) = upstream_request_id.as_deref() {
        insert_header(
            response.headers_mut(),
            "headroom-upstream-request-id",
            upstream_request_id,
        );
    }
    Ok(response)
}

fn upstream_request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("request-id")
        .or_else(|| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WebSocketKind {
    CodexResponses,
    Passthrough,
}

const OPENAI_RESPONSES_WEBSOCKET_BETA: &str = "responses_websockets=2026-02-06";
/// First-frame wait bound for Codex Responses WebSocket connections
/// (`headroom/proxy/handlers/openai.py:331`).
const WS_FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(60);

fn sse_stream_kind(headers: &HeaderMap, request_path: &str) -> Option<SseKind> {
    let is_sse = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        });
    if !is_sse {
        return None;
    }

    if request_path == "/v1/messages" || request_path.starts_with("/v1/messages/") {
        Some(SseKind::Anthropic)
    } else if request_path == "/v1/chat/completions" {
        Some(SseKind::OpenAiChat)
    } else if request_path == "/v1/responses"
        || request_path == "/v1/codex/responses"
        || request_path == "/backend-api/responses"
        || request_path == "/backend-api/codex/responses"
    {
        Some(SseKind::OpenAiResponses)
    } else {
        None
    }
}

/// Feed provider-confirmed cache usage from a buffered JSON response
/// into the session prefix tracker, mirroring `anthropic.py:2195-2228`
/// (Anthropic `cache_read/creation_input_tokens`) and
/// `openai.py:2094-2116` (OpenAI `prompt_tokens_details.cached_tokens`
/// with the inferred cache write from `openai.py:334-344`).
fn update_session_from_json_response(
    session: &Option<SseSessionCtx>,
    response: &Value,
    provider: CcrProvider,
) {
    let Some(session) = session else {
        return;
    };
    let Some(usage) = response.get("usage").filter(|usage| usage.is_object()) else {
        return;
    };
    let mut estimates = session.message_token_estimates.clone();
    let (cache_read, cache_write) = match provider {
        CcrProvider::Anthropic => {
            // The reference appends the assistant message from the
            // response before walking the estimates
            // (`anthropic.py:2216-2228`).
            if let Some(content) = response.get("content") {
                let assistant = json!({"role": "assistant", "content": content.clone()});
                estimates.extend(estimate_message_tokens(std::slice::from_ref(&assistant)));
            }
            (
                usage_u64(usage, &["cache_read_input_tokens"]),
                usage_u64(usage, &["cache_creation_input_tokens"]),
            )
        }
        CcrProvider::OpenAi => {
            let cache_read = usage
                .get("prompt_tokens_details")
                .or_else(|| usage.get("input_tokens_details"))
                .and_then(|details| details.get("cached_tokens"))
                .and_then(Value::as_u64)
                .or_else(|| usage_u64_opt(usage, &["cache_read_input_tokens"]))
                .unwrap_or(0);
            let creation = usage_u64(usage, &["cache_creation_input_tokens"]);
            let write = if creation > 0 {
                creation
            } else {
                let input = usage_u64(usage, &["prompt_tokens", "input_tokens"]);
                input.saturating_sub(cache_read)
            };
            (cache_read, write)
        }
    };
    session.trackers.update_from_response(
        session.provider,
        &session.session_id,
        cache_read,
        cache_write,
        &estimates,
    );
}

/// Record upstream rate-limit gauges keyed by the provider label the
/// reference uses (`anthropic` / `openai_chat` / `openai_responses`),
/// derived from the request path. Mirrors
/// `headroom-proxy/src/proxy.rs:972-1013`.
fn record_rate_limit_headers(headers: &HeaderMap, request_path: &str) {
    let provider = if request_path == "/v1/messages" || request_path.starts_with("/v1/messages/") {
        "anthropic"
    } else if request_path == "/v1/chat/completions" {
        "openai_chat"
    } else if matches!(
        request_path,
        "/v1/responses"
            | "/v1/codex/responses"
            | "/backend-api/responses"
            | "/backend-api/codex/responses"
    ) {
        "openai_responses"
    } else {
        return;
    };
    stats::record_rate_limit_snapshot(provider, extract_rate_limit_snapshot(headers));
}

fn apply_compression_response_headers(headers: &mut HeaderMap, mutation: &RequestCompression) {
    let tokens_saved = mutation.tokens_before.saturating_sub(mutation.tokens_after);
    insert_header(
        headers,
        "x-headroom-tokens-before",
        &mutation.tokens_before.to_string(),
    );
    insert_header(
        headers,
        "x-headroom-tokens-after",
        &mutation.tokens_after.to_string(),
    );
    insert_header(
        headers,
        "x-headroom-tokens-saved",
        &tokens_saved.to_string(),
    );
    if mutation.compressed {
        insert_header(headers, "x-headroom-transforms", "ccr_live_zone");
    }
    if !mutation.hashes.is_empty() {
        insert_header(headers, "x-headroom-ccr-hashes", &mutation.hashes.join(","));
    }
    if let Some(reason) = &mutation.skipped_reason {
        insert_header(headers, "x-headroom-skipped-reason", reason);
    }
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn is_application_json(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json")
        })
        .unwrap_or(false)
}

fn compression_bypass_requested(headers: &HeaderMap) -> bool {
    let bypass = headers
        .get("x-headroom-bypass")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("true"));
    let passthrough_mode = headers
        .get("x-headroom-mode")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("passthrough"));
    bypass || passthrough_mode
}

fn ensure_request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(generate_request_id)
}

fn forwarded_for(headers: &HeaderMap, client_addr: Option<SocketAddr>) -> Option<String> {
    let existing = headers
        .get("x-forwarded-for")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty());
    match (existing, client_addr) {
        (Some(existing), Some(addr)) => Some(format!("{existing}, {}", addr.ip())),
        (Some(existing), None) => Some(existing.to_string()),
        (None, Some(addr)) => Some(addr.ip().to_string()),
        (None, None) => None,
    }
}

fn classify_auth_mode(headers: &HeaderMap) -> RequestAuthMode {
    let user_agent = headers
        .get("user-agent")
        .and_then(|value| value.to_str().ok())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if [
        "claude-cli/",
        "claude-code/",
        "codex-cli/",
        "cursor/",
        "claude-vscode/",
        "github-copilot/",
        "anthropic-cli/",
        "antigravity/",
    ]
    .iter()
    .any(|prefix| user_agent.contains(prefix))
    {
        return RequestAuthMode::Subscription;
    }

    let auth = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if let Some(token) = auth.strip_prefix("Bearer ") {
        if token.starts_with("sk-ant-oat-") {
            return RequestAuthMode::OAuth;
        }
        if token.starts_with("sk-ant-api") || token.starts_with("sk-") {
            return RequestAuthMode::Payg;
        }
        if token.split('.').count() >= 3 {
            return RequestAuthMode::OAuth;
        }
    } else if !auth.is_empty() {
        return RequestAuthMode::OAuth;
    }

    if headers.contains_key("x-api-key") || headers.contains_key("x-goog-api-key") {
        return RequestAuthMode::Payg;
    }

    RequestAuthMode::Payg
}

fn chatgpt_account_id_from_authorization(headers: &HeaderMap) -> Option<String> {
    if headers.get("chatgpt-account-id").is_some() {
        return None;
    }

    let auth = headers.get("authorization")?.to_str().ok()?;
    let (scheme, token) = auth.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let payload = token.split('.').nth(1)?;
    let decoded = decode_base64_url(payload)?;
    let value: Value = serde_json::from_slice(&decoded).ok()?;
    let account_id = value
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()?
        .trim();
    (!account_id.is_empty()).then(|| account_id.to_string())
}

fn decode_base64_url(input: &str) -> Option<Vec<u8>> {
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buffer = 0_u32;
    let mut bits = 0_u8;

    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        let value = base64_url_value(byte)? as u32;
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((buffer >> bits) & 0xff) as u8);
            buffer &= (1_u32 << bits) - 1;
        }
    }

    Some(output)
}

fn base64_url_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'-' | b'+' => Some(62),
        b'_' | b'/' => Some(63),
        _ => None,
    }
}

fn generate_request_id() -> String {
    let counter = REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("hr-{nanos}-{counter}")
}

fn connection_listed_headers(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn is_request_drop_header(name: &HeaderName) -> bool {
    is_hop_by_hop(name)
        || name == HOST
        || name == CONTENT_LENGTH
        || name.as_str().eq_ignore_ascii_case("x-request-id")
        || name.as_str().eq_ignore_ascii_case("x-forwarded-for")
        || name.as_str().eq_ignore_ascii_case("x-forwarded-proto")
        || name.as_str().eq_ignore_ascii_case("x-forwarded-host")
        || name.as_str().eq_ignore_ascii_case("x-headroom-bypass")
        || name.as_str().eq_ignore_ascii_case("x-headroom-mode")
        || name.as_str().starts_with("x-headroom-")
}

fn is_response_drop_header(name: &HeaderName) -> bool {
    is_hop_by_hop(name)
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str().to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

async fn proxy_websocket(
    client_socket: WebSocket,
    state: ProxyState,
    uri: Uri,
    headers: HeaderMap,
    provider: UpstreamProvider,
    client_addr: Option<SocketAddr>,
) {
    stats::record_websocket_open();
    info!(path = %uri.path(), provider = ?provider, "websocket open");

    let result = match websocket_kind(uri.path(), provider) {
        WebSocketKind::CodexResponses => {
            proxy_codex_responses_websocket(client_socket, state, uri.clone(), headers, client_addr)
                .await
        }
        WebSocketKind::Passthrough => {
            proxy_passthrough_websocket(
                client_socket,
                state,
                uri.clone(),
                headers,
                provider,
                client_addr,
            )
            .await
        }
    };

    if let Err(err) = result {
        info!(error = %err, "websocket proxy ended with error");
    }

    stats::record_websocket_close();
    info!(path = %uri.path(), provider = ?provider, "websocket close");
}

async fn proxy_passthrough_websocket(
    client_socket: WebSocket,
    state: ProxyState,
    uri: Uri,
    headers: HeaderMap,
    provider: UpstreamProvider,
    client_addr: Option<SocketAddr>,
) -> HrResult<()> {
    let url = websocket_url(&state, provider, &uri)?;
    let mut request = url.as_str().into_client_request()?;
    copy_websocket_headers(&headers, request.headers_mut(), client_addr);

    let (upstream_socket, _) = tokio_tungstenite::connect_async(request).await?;
    let (mut client_tx, mut client_rx) = client_socket.split();
    let (mut upstream_tx, mut upstream_rx) = upstream_socket.split();

    let client_to_upstream = async {
        while let Some(message) = client_rx.next().await {
            let message = message?;
            let (frame_kind, frame_bytes) = axum_ws_frame_summary(&message);
            trace!(
                kind = frame_kind,
                bytes = frame_bytes,
                "client websocket frame"
            );
            let Some(message) = axum_to_tungstenite(message) else {
                break;
            };
            upstream_tx.send(message).await?;
        }
        HrResult::<()>::Ok(())
    };

    let upstream_to_client = async {
        while let Some(message) = upstream_rx.next().await {
            let message = message?;
            let (frame_kind, frame_bytes) = tungstenite_ws_frame_summary(&message);
            trace!(
                kind = frame_kind,
                bytes = frame_bytes,
                "upstream websocket frame"
            );
            let Some(message) = tungstenite_to_axum(message) else {
                break;
            };
            client_tx.send(message).await?;
        }
        HrResult::<()>::Ok(())
    };

    tokio::select! {
        result = client_to_upstream => result,
        result = upstream_to_client => result,
    }
}

async fn proxy_codex_responses_websocket(
    mut client_socket: WebSocket,
    state: ProxyState,
    uri: Uri,
    mut headers: HeaderMap,
    client_addr: Option<SocketAddr>,
) -> HrResult<()> {
    // First-frame timeout, mirroring the reference WS handler
    // (`openai.py:331` `WS_FIRST_FRAME_TIMEOUT_SECONDS = 60`,
    // `openai.py:3579` close code 1001): a connection that never sends
    // a frame cannot hold a WS slot open indefinitely.
    let first_message =
        match tokio::time::timeout(WS_FIRST_FRAME_TIMEOUT, client_socket.next()).await {
            Ok(Some(message)) => message?,
            Ok(None) => return Ok(()),
            Err(_elapsed) => {
                let _ = client_socket
                    .send(AxumWsMessage::Close(Some(CloseFrame {
                        code: 1001,
                        reason: "first frame timeout".into(),
                    })))
                    .await;
                return Ok(());
            }
        };

    // Per-connection session-sticky `openai-beta` merge, mirroring
    // `openai.py:3496-3518` (the WS handler records the client value
    // under a per-connection session id, then layers the required
    // websockets beta on top).
    let connection_session = generate_request_id();
    let client_beta = headers
        .get("openai-beta")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let merged_beta = state.sessions.sticky_betas(
        SessionProvider::OpenAi,
        &connection_session,
        client_beta.as_deref(),
    );
    if !merged_beta.is_empty() && Some(merged_beta.as_str()) != client_beta.as_deref() {
        if let Ok(value) = HeaderValue::from_str(&merged_beta) {
            headers.insert(HeaderName::from_static("openai-beta"), value);
        }
    }

    let Some((first_message, first_text_for_fallback)) =
        prepare_client_codex_ws_message(first_message, &state, &headers)
    else {
        return Ok(());
    };

    let url = websocket_url(&state, UpstreamProvider::OpenAi, &uri)?;
    let mut request = url.as_str().into_client_request()?;
    copy_websocket_headers(&headers, request.headers_mut(), client_addr);
    ensure_openai_responses_websocket_beta(request.headers_mut());

    let upstream_socket = match tokio_tungstenite::connect_async(request).await {
        Ok((socket, _response)) => socket,
        Err(err) => {
            return codex_ws_http_fallback(
                client_socket,
                state,
                uri,
                headers,
                first_text_for_fallback,
                client_addr,
                &err.to_string(),
            )
            .await;
        }
    };

    let (mut client_tx, mut client_rx) = client_socket.split();
    let (mut upstream_tx, mut upstream_rx) = upstream_socket.split();
    upstream_tx.send(first_message).await?;

    let client_state = state.clone();
    let client_headers = headers.clone();
    let client_to_upstream = async move {
        while let Some(message) = client_rx.next().await {
            let message = message?;
            let (frame_kind, frame_bytes) = axum_ws_frame_summary(&message);
            trace!(
                kind = frame_kind,
                bytes = frame_bytes,
                "client websocket frame"
            );
            let Some((message, _fallback_text)) =
                prepare_client_codex_ws_message(message, &client_state, &client_headers)
            else {
                break;
            };
            upstream_tx.send(message).await?;
        }
        HrResult::<()>::Ok(())
    };

    let upstream_to_client = async move {
        let mut telemetry = OpenAiResponsesWsTelemetry;
        while let Some(message) = upstream_rx.next().await {
            let message = message?;
            let (frame_kind, frame_bytes) = tungstenite_ws_frame_summary(&message);
            trace!(
                kind = frame_kind,
                bytes = frame_bytes,
                "upstream websocket frame"
            );
            if let TungsteniteMessage::Text(text) = &message {
                telemetry.observe_text(text);
            }
            let Some(message) = tungstenite_to_axum(message) else {
                break;
            };
            client_tx.send(message).await?;
        }
        HrResult::<()>::Ok(())
    };

    tokio::select! {
        result = client_to_upstream => result,
        result = upstream_to_client => result,
    }
}

/// The client's requested WebSocket subprotocols, in offer order.
fn websocket_client_protocols(headers: &HeaderMap) -> Vec<String> {
    headers
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|protocol| !protocol.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn websocket_kind(path: &str, provider: UpstreamProvider) -> WebSocketKind {
    if provider == UpstreamProvider::OpenAi
        && matches!(
            path,
            "/v1/responses"
                | "/v1/codex/responses"
                | "/backend-api/responses"
                | "/backend-api/codex/responses"
        )
    {
        WebSocketKind::CodexResponses
    } else {
        WebSocketKind::Passthrough
    }
}

fn prepare_client_codex_ws_message(
    message: AxumWsMessage,
    state: &ProxyState,
    headers: &HeaderMap,
) -> Option<(TungsteniteMessage, Option<String>)> {
    let message = axum_to_tungstenite(message)?;
    if let TungsteniteMessage::Text(text) = message {
        let raw_text = text.to_string();
        let rewritten = maybe_compress_codex_response_create_frame(&raw_text, state, headers);
        Some((
            TungsteniteMessage::Text(rewritten.clone().into()),
            Some(rewritten),
        ))
    } else {
        Some((message, None))
    }
}

fn maybe_compress_codex_response_create_frame(
    raw_text: &str,
    state: &ProxyState,
    headers: &HeaderMap,
) -> String {
    let Ok(mut frame) = serde_json::from_str::<Value>(raw_text) else {
        return raw_text.to_string();
    };
    if frame
        .get("type")
        .and_then(Value::as_str)
        .is_none_or(|kind| kind != "response.create")
    {
        return raw_text.to_string();
    }

    if !state.compression_enabled || !state.compression_mode.allows_compression() {
        stats::record_skipped_request("compression_disabled");
        return raw_text.to_string();
    }
    if compression_bypass_requested(headers) {
        stats::record_skipped_request("bypass_header");
        return raw_text.to_string();
    }

    let wrapped = frame.get("response").is_some();
    let body_value = if wrapped {
        frame.get("response").cloned().unwrap_or(Value::Null)
    } else {
        frame.clone()
    };
    let Ok(body) = serde_json::to_vec(&body_value) else {
        return raw_text.to_string();
    };

    // WS `response.create` frames mirror the reference payload
    // compressor (`openai.py:1212`, `1749-1770`): live-zone
    // compression and CCR retrieve-tool injection only — no tool
    // sort, no cache_control placement, no `prompt_cache_key`.
    let mut mutation = compress_json_request_ctx(
        &body,
        ApiShape::OpenAiResponses,
        CompressContext {
            store: &state.store,
            auth_mode: classify_auth_mode(headers),
            frozen_message_count: 0,
            compress_user_text: state.compress_user_text,
            provider_metadata: false,
        },
    );
    record_mutation_stats(&mutation);
    // Forward the rewrite whenever the bytes changed: a frame whose
    // only mutation is the metadata-only retrieve-tool injection
    // (existing `<<ccr:...>>` markers, nothing newly compressed) must
    // still reach upstream with the tool available.
    if mutation.body == body {
        return raw_text.to_string();
    }

    if wrapped {
        match serde_json::from_slice::<Value>(&mutation.body) {
            Ok(mutated_response) => {
                frame["response"] = mutated_response;
                serde_json::to_string(&frame).unwrap_or_else(|_| raw_text.to_string())
            }
            Err(_) => raw_text.to_string(),
        }
    } else {
        String::from_utf8(std::mem::take(&mut mutation.body))
            .unwrap_or_else(|_| raw_text.to_string())
    }
}

async fn codex_ws_http_fallback(
    mut client_socket: WebSocket,
    state: ProxyState,
    uri: Uri,
    headers: HeaderMap,
    first_text: Option<String>,
    client_addr: Option<SocketAddr>,
    connect_error: &str,
) -> HrResult<()> {
    let Some(first_text) = first_text else {
        return Err(error(format!(
            "codex websocket upstream failed before a text response.create frame: {connect_error}"
        )));
    };
    let Some(body) = codex_ws_http_fallback_body(&first_text) else {
        return Err(error(format!(
            "codex websocket upstream failed and first frame cannot be converted to HTTP fallback: {connect_error}"
        )));
    };

    let upstream_url = upstream_url(&state, UpstreamProvider::OpenAi, &uri)?;
    let request_id = ensure_request_id(&headers);
    let mut fallback_headers = headers.clone();
    ensure_openai_responses_websocket_beta(&mut fallback_headers);
    let mut builder = state
        .client
        .post(upstream_url.clone())
        .header(CONTENT_TYPE, "application/json")
        .body(body);
    builder = apply_headers(builder, &fallback_headers, client_addr, &request_id);
    let response = builder.send().await?;

    info!(
        upstream = %upstream_url,
        status = %response.status(),
        reason = connect_error,
        "codex websocket using http streaming fallback"
    );

    if !response.status().is_success() {
        let status = response.status();
        // Error frame shape mirrors the reference fallback
        // (`headroom/proxy/handlers/openai.py:5540-5549`).
        let error_frame = json!({
            "type": "error",
            "error": {
                "type": "server_error",
                "message": format!("Upstream returned {}", status.as_u16()),
            }
        });
        client_socket
            .send(AxumWsMessage::Text(error_frame.to_string().into()))
            .await?;
        return Err(error(format!(
            "codex websocket http fallback returned {status}"
        )));
    }

    relay_sse_response_over_websocket(response, client_socket).await
}

fn codex_ws_http_fallback_body(first_text: &str) -> Option<Vec<u8>> {
    let frame = serde_json::from_str::<Value>(first_text).ok()?;
    if frame
        .get("type")
        .and_then(Value::as_str)
        .is_none_or(|kind| kind != "response.create")
    {
        return None;
    }

    let mut body = match frame.get("response").cloned() {
        Some(response) => response,
        None => frame,
    };
    if let Value::Object(object) = &mut body {
        object.remove("type");
        object.insert("stream".to_string(), Value::Bool(true));
    }
    serde_json::to_vec(&body).ok()
}

async fn relay_sse_response_over_websocket(
    response: reqwest::Response,
    mut client_socket: WebSocket,
) -> HrResult<()> {
    let mut telemetry = OpenAiResponsesWsTelemetry;
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::<u8>::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.extend_from_slice(&chunk);
        while let Some(line_end) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.drain(..=line_end).collect::<Vec<_>>();
            let Ok(line) = String::from_utf8(line) else {
                continue;
            };
            let line = line.trim_end_matches(['\r', '\n']);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim_start();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            telemetry.observe_text(data);
            client_socket
                .send(AxumWsMessage::Text(data.to_string().into()))
                .await?;
        }
    }

    Ok(())
}

/// Usage telemetry for the Codex Responses WebSocket path. Only
/// `response.completed` events carry usage
/// (`headroom/proxy/handlers/openai.py:347-378`, `_extract_responses_usage`
/// returns zeros for every other event type); one connection can
/// complete several responses, each recorded as it lands
/// (`openai.py:4865-4870`). The OpenAI cache-write inference from
/// `openai.py:334-344` applies per completed response.
#[derive(Debug, Default)]
struct OpenAiResponsesWsTelemetry;

impl OpenAiResponsesWsTelemetry {
    fn observe_text(&mut self, text: &str) {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return;
        };
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        let status = match kind {
            "response.completed" => crate::sse::response_status::COMPLETED,
            "response.failed" => crate::sse::response_status::FAILED,
            "response.incomplete" => crate::sse::response_status::INCOMPLETE,
            _ => return,
        };
        // Terminal-status and service-tier telemetry mirror the
        // reference WS metrics hook (`openai.py:4625` /
        // `_record_ws_response_metrics`, fired on
        // `response.completed|failed|incomplete`).
        stats::record_response_status(status);
        if let Some(tier) = value
            .get("response")
            .and_then(|response| response.get("service_tier"))
            .or_else(|| value.get("service_tier"))
            .and_then(Value::as_str)
        {
            stats::record_service_tier(crate::sse::service_tier::validate(tier));
        }
        if kind != "response.completed" {
            return;
        }
        let Some(usage) = value
            .get("usage")
            .or_else(|| {
                value
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })
            .filter(|usage| usage.is_object())
        else {
            return;
        };

        let input = usage_u64(usage, &["input_tokens", "prompt_tokens"]);
        let output = usage_u64(usage, &["output_tokens", "completion_tokens"]);
        let details = usage
            .get("input_tokens_details")
            .or_else(|| usage.get("prompt_tokens_details"));
        let cache_read = details
            .and_then(|details| usage_u64_opt(details, &["cached_tokens"]))
            .or_else(|| usage_u64_opt(usage, &["cache_read_input_tokens"]))
            .unwrap_or_default();
        if input == 0 && output == 0 && cache_read == 0 {
            return;
        }

        stats::record_sse_usage(input, output, cache_read, 0);
        stats::record_inferred_cache_write_tokens(input.saturating_sub(cache_read));
        if input > 0 && cache_read <= input {
            stats::record_sse_cache_hit_rate("openai_responses", cache_read as f64 / input as f64);
        }
    }
}

fn ensure_openai_responses_websocket_beta(headers: &mut HeaderMap) {
    let mut values = headers
        .get("openai-beta")
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if values
        .iter()
        .all(|value| !value.eq_ignore_ascii_case(OPENAI_RESPONSES_WEBSOCKET_BETA))
    {
        values.push(OPENAI_RESPONSES_WEBSOCKET_BETA.to_string());
    }
    if let Ok(value) = HeaderValue::from_str(&values.join(", ")) {
        headers.insert(HeaderName::from_static("openai-beta"), value);
    }
}

/// Frame kind and payload size for trace logging. Frame contents are never
/// logged: text frames carry prompt/response bodies.
fn axum_ws_frame_summary(message: &AxumWsMessage) -> (&'static str, usize) {
    match message {
        AxumWsMessage::Text(text) => ("text", text.len()),
        AxumWsMessage::Binary(bytes) => ("binary", bytes.len()),
        AxumWsMessage::Ping(bytes) => ("ping", bytes.len()),
        AxumWsMessage::Pong(bytes) => ("pong", bytes.len()),
        AxumWsMessage::Close(_) => ("close", 0),
    }
}

/// See [`axum_ws_frame_summary`].
fn tungstenite_ws_frame_summary(message: &TungsteniteMessage) -> (&'static str, usize) {
    match message {
        TungsteniteMessage::Text(text) => ("text", text.len()),
        TungsteniteMessage::Binary(bytes) => ("binary", bytes.len()),
        TungsteniteMessage::Ping(bytes) => ("ping", bytes.len()),
        TungsteniteMessage::Pong(bytes) => ("pong", bytes.len()),
        TungsteniteMessage::Close(_) => ("close", 0),
        TungsteniteMessage::Frame(frame) => ("frame", frame.len()),
    }
}

fn axum_to_tungstenite(message: AxumWsMessage) -> Option<TungsteniteMessage> {
    match message {
        AxumWsMessage::Text(text) => Some(TungsteniteMessage::Text(text.to_string().into())),
        AxumWsMessage::Binary(bytes) => Some(TungsteniteMessage::Binary(bytes)),
        AxumWsMessage::Ping(bytes) => Some(TungsteniteMessage::Ping(bytes)),
        AxumWsMessage::Pong(bytes) => Some(TungsteniteMessage::Pong(bytes)),
        AxumWsMessage::Close(Some(frame)) => {
            Some(TungsteniteMessage::Close(Some(TungsteniteCloseFrame {
                code: CloseCode::from(frame.code),
                reason: frame.reason.to_string().into(),
            })))
        }
        AxumWsMessage::Close(None) => Some(TungsteniteMessage::Close(None)),
    }
}

fn tungstenite_to_axum(message: TungsteniteMessage) -> Option<AxumWsMessage> {
    match message {
        TungsteniteMessage::Text(text) => Some(AxumWsMessage::Text(text.to_string().into())),
        TungsteniteMessage::Binary(bytes) => Some(AxumWsMessage::Binary(bytes)),
        TungsteniteMessage::Ping(bytes) => Some(AxumWsMessage::Ping(bytes)),
        TungsteniteMessage::Pong(bytes) => Some(AxumWsMessage::Pong(bytes)),
        TungsteniteMessage::Close(Some(frame)) => Some(AxumWsMessage::Close(Some(CloseFrame {
            code: frame.code.into(),
            reason: frame.reason.to_string().into(),
        }))),
        TungsteniteMessage::Close(None) => Some(AxumWsMessage::Close(None)),
        TungsteniteMessage::Frame(_) => None,
    }
}

fn copy_websocket_headers(from: &HeaderMap, to: &mut HeaderMap, client_addr: Option<SocketAddr>) {
    let request_id = ensure_request_id(from);
    if let Ok(value) = axum::http::HeaderValue::from_str(&request_id) {
        to.insert("x-request-id", value);
    }
    to.insert("x-forwarded-proto", HeaderValue::from_static("http"));
    if let Some(host) = from.get(HOST) {
        to.insert("x-forwarded-host", host.clone());
    }
    if let Some(xff) = forwarded_for(from, client_addr) {
        if let Ok(value) = axum::http::HeaderValue::from_str(&xff) {
            to.insert("x-forwarded-for", value);
        }
    }
    if let Some(account_id) = chatgpt_account_id_from_authorization(from) {
        if let Ok(value) = HeaderValue::from_str(&account_id) {
            to.insert("ChatGPT-Account-ID", value);
        }
    }
    for (name, value) in from {
        if should_skip_websocket_header(name) {
            continue;
        }
        to.insert(name.clone(), value.clone());
    }
}

fn should_skip_websocket_header(name: &HeaderName) -> bool {
    name == HOST
        || name == CONNECTION
        || name == UPGRADE
        || name.as_str().eq_ignore_ascii_case("sec-websocket-key")
        || name.as_str().eq_ignore_ascii_case("sec-websocket-version")
        || name
            .as_str()
            .eq_ignore_ascii_case("sec-websocket-extensions")
        || name == CONTENT_LENGTH
        || name.as_str().eq_ignore_ascii_case("x-request-id")
        || name.as_str().eq_ignore_ascii_case("x-forwarded-for")
        || name.as_str().eq_ignore_ascii_case("x-forwarded-proto")
        || name.as_str().eq_ignore_ascii_case("x-forwarded-host")
        || name.as_str().starts_with("x-headroom-")
}

fn upstream_url(state: &ProxyState, provider: UpstreamProvider, uri: &Uri) -> HrResult<Url> {
    let mut url = match provider {
        UpstreamProvider::OpenAi => state.openai_upstream.clone(),
        UpstreamProvider::Anthropic => state.anthropic_upstream.clone(),
    };
    apply_uri_path_and_query(&mut url, uri);
    Ok(url)
}

fn websocket_url(state: &ProxyState, provider: UpstreamProvider, uri: &Uri) -> HrResult<Url> {
    let mut url = upstream_url(state, provider, uri)?;
    let scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        "ws" | "wss" => url.scheme(),
        other => {
            return Err(error(format!(
                "unsupported upstream websocket scheme: {other}"
            )))
        }
    }
    .to_string();
    url.set_scheme(&scheme)
        .map_err(|_| error("failed to set websocket upstream scheme"))?;
    Ok(url)
}

fn apply_uri_path_and_query(url: &mut Url, uri: &Uri) {
    let base_path = url.path().trim_end_matches('/');
    let request_path = uri.path();
    let combined = if base_path.is_empty() || base_path == "/" {
        request_path.to_string()
    } else {
        format!("{base_path}{request_path}")
    };
    url.set_path(&combined);
    url.set_query(uri.query());
}

fn error_response(status: StatusCode, message: String) -> Response<Body> {
    (status, message).into_response()
}

fn upstream_error_response(err: Box<dyn std::error::Error + Send + Sync>) -> Response<Body> {
    let status = err
        .downcast_ref::<reqwest::Error>()
        .filter(|err| err.is_timeout())
        .map(|_| StatusCode::GATEWAY_TIMEOUT)
        .unwrap_or(StatusCode::BAD_GATEWAY);
    error_response(status, err.to_string())
}

fn json_response(status: StatusCode, value: serde_json::Value) -> Response<Body> {
    match serde_json::to_vec(&value) {
        Ok(body) => Response::builder()
            .status(status)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap_or_else(|err| {
                error_response(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
            }),
        Err(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_target_paths() {
        assert_eq!(
            classify_request(&Method::POST, "/v1/messages"),
            RequestClass {
                provider: UpstreamProvider::Anthropic,
                target: Some(CompressionTarget::AnthropicMessages),
                skipped_reason: None
            }
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/chat/completions").target,
            Some(CompressionTarget::OpenAiChatCompletions)
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/responses").target,
            Some(CompressionTarget::OpenAiResponses)
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/codex/responses").target,
            Some(CompressionTarget::OpenAiResponses)
        );
        assert_eq!(
            classify_request(&Method::POST, "/backend-api/codex/responses").target,
            Some(CompressionTarget::OpenAiResponses)
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/messages/batches").target,
            Some(CompressionTarget::AnthropicMessageBatches)
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/messages/batches/batch_123/results").target,
            None
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/messages/count_tokens").provider,
            UpstreamProvider::Anthropic
        );
        assert_eq!(
            classify_request(&Method::GET, "/v1/responses").skipped_reason,
            Some("non_post_method")
        );
        assert_eq!(
            classify_request(&Method::POST, "/healthz").skipped_reason,
            Some("reserved_proxy_endpoint")
        );
        assert_eq!(
            classify_request(&Method::POST, "/v1/conversations").skipped_reason,
            Some("conversations_passthrough")
        );
    }

    #[test]
    fn parses_compression_modes() {
        assert_eq!(CompressionMode::parse("ccr").unwrap(), CompressionMode::Ccr);
        assert_eq!(
            CompressionMode::parse("passthrough").unwrap(),
            CompressionMode::Passthrough
        );
        assert_eq!(CompressionMode::parse("off").unwrap(), CompressionMode::Off);
        assert!(CompressionMode::parse("unknown").is_err());
    }

    #[test]
    fn extracts_ccr_tool_calls_per_provider_shape() {
        let anthropic = json!({
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "toolu_1", "name": "headroom_retrieve",
                 "input": {"hash": "a".repeat(24)}},
                {"type": "tool_use", "id": "toolu_2", "name": "user_tool", "input": {}}
            ]
        });
        let calls = extract_ccr_tool_calls(&anthropic, CcrProvider::Anthropic);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.hash, "a".repeat(24));
        assert_eq!(
            provider_tool_calls(&anthropic, CcrProvider::Anthropic).len(),
            2
        );

        let openai = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "headroom_retrieve",
                            "arguments": "{\"hash\":\"bbbbbbbbbbbbbbbbbbbbbbbb\",\"query\":\"x\"}"
                        }
                    }]
                }
            }]
        });
        let calls = extract_ccr_tool_calls(&openai, CcrProvider::OpenAi);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.hash, "b".repeat(24));
        assert_eq!(calls[0].1.query.as_deref(), Some("x"));

        assert!(
            extract_ccr_tool_calls(&json!({"content": "text"}), CcrProvider::Anthropic).is_empty()
        );
    }

    #[test]
    fn strips_ccr_tool_calls_from_mixed_provider_responses() {
        let anthropic = json!({
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_headroom", "name": "headroom_retrieve",
                 "input": {"hash": "a".repeat(24)}},
                {"type": "tool_use", "id": "toolu_project", "name": "project_tool", "input": {}}
            ],
            "stop_reason": "tool_use"
        });
        let stripped = strip_ccr_tool_calls(&anthropic, CcrProvider::Anthropic).unwrap();
        let content = stripped["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["name"], "project_tool");

        let openai = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_headroom",
                            "type": "function",
                            "function": {
                                "name": "headroom_retrieve",
                                "arguments": "{\"hash\":\"bbbbbbbbbbbbbbbbbbbbbbbb\"}"
                            }
                        },
                        {
                            "id": "call_project",
                            "type": "function",
                            "function": {
                                "name": "project_tool",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            }]
        });
        let stripped = strip_ccr_tool_calls(&openai, CcrProvider::OpenAi).unwrap();
        let calls = stripped["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "project_tool");

        let headroom_only = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "headroom_retrieve",
                "input": {"hash": "c".repeat(24)}
            }]
        });
        assert!(strip_ccr_tool_calls(&headroom_only, CcrProvider::Anthropic).is_none());
    }

    #[test]
    fn strip_keeps_choice_positions_and_rewrites_tool_free_choices() {
        let headroom_call = json!({
            "id": "call_headroom",
            "type": "function",
            "function": {
                "name": "headroom_retrieve",
                "arguments": "{\"hash\":\"dddddddddddddddddddddddd\"}"
            }
        });
        let project_call = json!({
            "id": "call_project",
            "type": "function",
            "function": {"name": "project_tool", "arguments": "{}"}
        });

        // A multi-choice response where one choice carried only the private
        // retrieval call must keep both choices: dropping the choice would
        // desync `index` from array position and lose its other fields.
        let multi_choice = json!({
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": null,
                                "tool_calls": [headroom_call.clone()]},
                    "finish_reason": "tool_calls"
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": null,
                                "tool_calls": [project_call.clone()]},
                    "finish_reason": "tool_calls"
                }
            ]
        });
        let stripped = strip_ccr_tool_calls(&multi_choice, CcrProvider::OpenAi).unwrap();
        let choices = stripped["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        // The now tool-free choice loses its empty `tool_calls` array and its
        // `finish_reason: "tool_calls"` so the client is not left waiting to
        // answer a tool call that no longer exists.
        assert!(choices[0]["message"].get("tool_calls").is_none());
        assert_eq!(choices[0]["finish_reason"], "stop");
        assert_eq!(choices[1]["index"], 1);
        assert_eq!(
            choices[1]["message"]["tool_calls"][0]["function"]["name"],
            "project_tool"
        );
        assert_eq!(choices[1]["finish_reason"], "tool_calls");

        // A response whose only tool calls are private stays untouched so the
        // continuation loop (or the as-is fallback) can handle it.
        let headroom_only = json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null,
                            "tool_calls": [headroom_call]},
                "finish_reason": "tool_calls"
            }]
        });
        assert!(strip_ccr_tool_calls(&headroom_only, CcrProvider::OpenAi).is_none());
    }

    #[test]
    fn batch_context_replaces_duplicate_and_drops_expired_entries() {
        let url = Url::parse("http://127.0.0.1:1").unwrap();
        let state = ProxyState::new(url.clone(), url, SqliteStore::in_memory().unwrap());

        state.record_batch_context(
            "batch_1",
            &json!({
                "requests": [{
                    "custom_id": "req_1",
                    "params": {"model": "claude-test", "messages": []}
                }]
            }),
        );
        assert!(state
            .batch_context("batch_1")
            .unwrap()
            .contains_key("req_1"));

        state.record_batch_context(
            "batch_1",
            &json!({
                "requests": [{
                    "custom_id": "req_2",
                    "params": {"model": "claude-test", "messages": []}
                }]
            }),
        );
        let replacement = state.batch_context("batch_1").unwrap();
        assert!(!replacement.contains_key("req_1"));
        assert!(replacement.contains_key("req_2"));

        {
            let mut store = state.batch_contexts.lock().unwrap();
            store.push_back(BatchContextEntry {
                batch_id: "expired".to_string(),
                contexts: HashMap::new(),
                expires_at: SystemTime::now()
                    .checked_sub(Duration::from_secs(1))
                    .unwrap(),
            });
        }
        assert!(state.batch_context("expired").is_none());
        assert!(!state
            .batch_contexts
            .lock()
            .unwrap()
            .iter()
            .any(|entry| entry.batch_id == "expired"));
    }

    #[test]
    fn batch_context_sets_ttl_and_evicts_oldest_after_limit() {
        let url = Url::parse("http://127.0.0.1:1").unwrap();
        let state = ProxyState::new(url.clone(), url, SqliteStore::in_memory().unwrap());

        let recorded_at = SystemTime::now();
        state.record_batch_context(
            "ttl_batch",
            &json!({
                "requests": [{
                    "custom_id": "req",
                    "params": {"model": "claude-test", "messages": []}
                }]
            }),
        );
        let entry = state.batch_contexts.lock().unwrap().back().unwrap().clone();
        let ttl = entry.expires_at.duration_since(recorded_at).unwrap();
        assert!(ttl >= BATCH_CONTEXT_TTL.saturating_sub(Duration::from_secs(1)));
        assert!(ttl <= BATCH_CONTEXT_TTL + Duration::from_secs(1));

        {
            let mut store = state.batch_contexts.lock().unwrap();
            store.clear();
            for index in 0..MAX_BATCH_CONTEXTS {
                store.push_back(BatchContextEntry {
                    batch_id: format!("batch_{index}"),
                    contexts: HashMap::new(),
                    expires_at: SystemTime::now() + BATCH_CONTEXT_TTL,
                });
            }
        }
        state.record_batch_context(
            "batch_new",
            &json!({
                "requests": [{
                    "custom_id": "req_new",
                    "params": {"model": "claude-test", "messages": []}
                }]
            }),
        );

        let store = state.batch_contexts.lock().unwrap();
        assert_eq!(store.len(), MAX_BATCH_CONTEXTS);
        assert!(!store.iter().any(|entry| entry.batch_id == "batch_0"));
        assert!(store.iter().any(|entry| entry.batch_id == "batch_new"));
    }

    #[test]
    fn batch_results_path_is_recognized() {
        assert_eq!(
            anthropic_batch_results_batch_id(&Method::GET, "/v1/messages/batches/b_1/results"),
            Some("b_1".to_string())
        );
        assert_eq!(
            anthropic_batch_results_batch_id(&Method::POST, "/v1/messages/batches/b_1/results"),
            None
        );
        assert_eq!(
            anthropic_batch_results_batch_id(&Method::GET, "/v1/messages/batches/b_1"),
            None
        );
        assert_eq!(
            anthropic_batch_results_batch_id(&Method::GET, "/v1/messages/batches//results"),
            None
        );
    }
}

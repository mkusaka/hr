pub mod ccr;
pub mod compression;
pub mod proxy;
pub mod session;
pub mod sse;
pub mod stats;

pub use ccr::{
    content_hash, decompress_hash, decompress_text, marker_for_hash, CcrStore, DecompressResult,
    SqliteStore, HASH_HEX_LEN,
};
pub use compression::{
    compress, compress_json_request, compress_json_request_ctx, compress_json_request_with_auth,
    estimate_tokens, ApiShape, CompressContext, CompressOptions, CompressResult, RequestAuthMode,
    RequestCompression,
};
pub use proxy::{
    build_router, classify_request, serve_proxy, CompressionMode, CompressionTarget, ProxyConfig,
    ProxyState, RequestClass, UpstreamProvider, DEFAULT_MAX_BODY_BYTES,
};
pub use stats::{stats, StatsSnapshot};

pub type HrResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

#[doc(hidden)]
pub fn error(message: impl Into<String>) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::other(message.into()))
}

# hr

`hr` is a small Rust CLI and library for reversible CCR compression, direct
decompression, and an HTTP/WebSocket proxy for OpenAI/Codex and
Anthropic/Claude Code traffic.

The proxy compresses these JSON request shapes:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/codex/responses`
- `POST /backend-api/responses`
- `POST /backend-api/codex/responses`
- `POST /v1/messages`
- `POST /v1/messages/batches`

Anthropic subpaths such as `/v1/messages/count_tokens` and
batch read/result/cancel/delete paths are routed to the Anthropic upstream
without compression. Other paths route to the OpenAI upstream unless they are
reserved proxy-local endpoints.

It also reserves proxy-local endpoints for health, metrics, stats, and CCR
retrieval:

- `GET /healthz`
- `GET /healthz/upstream`
- `GET /livez`
- `GET /readyz`
- `GET /health`
- `GET /metrics`
- `GET /stats`
- `POST /v1/compress`
- `POST /v1/retrieve`
- `GET /v1/retrieve/{hash}`
- `GET /v1/retrieve/stats`
- `POST /v1/retrieve/tool_call`

## CLI

Compress a file or stdin. The original payload is stored in the default SQLite
CCR database (`$HR_CCR_DB`, or `~/.hr/ccr.sqlite` when unset), and stdout prints
a `<<ccr:HASH>>` marker.

```bash
hr compress --input prompt.txt
hr compress --input -
```

Decompress by direct hash lookup, or expand every known CCR marker in arbitrary
text.

```bash
hr decompress --hash 0123456789abcdef01234567
hr decompress --input response.txt
hr decompress --input -
```

Show process stats plus the default CCR database entry count.

```bash
hr stats
```

## Proxy

Run a proxy with separate upstreams for OpenAI/Codex and Anthropic/Claude Code:

```bash
hr proxy \
  --listen 127.0.0.1:8787 \
  --openai-upstream https://api.openai.com \
  --anthropic-upstream https://api.anthropic.com \
  --ccr-db ~/.hr/ccr.sqlite \
  --log-level info \
  --compression true \
  --compression-mode ccr \
  --max-body-bytes 26214400
```

Known JSON LLM request paths are buffered up to `--max-body-bytes`,
conservatively compressed, and then forwarded. Non-JSON bodies, non-target
paths, Conversations API paths (`/v1/conversations*`), and other HTTP requests
stream through without request mutation. WebSocket upgrades are passed through
to the selected upstream.

Use `--compression false`, `--compression-mode off`, or
`--compression-mode passthrough` to disable target request mutation. The
`--compression-max-body-bytes` alias is accepted for the same limit as
`--max-body-bytes`.

The forwarder strips hop-by-hop headers and proxy-internal `x-headroom-*`
headers before sending upstream, preserves or creates `x-request-id`, and adds
basic `x-forwarded-*` headers. Upstream responses are streamed back while
filtering hop-by-hop response headers. The proxy returns its request id as
`x-request-id`; if the upstream supplies `request-id` or `x-request-id`, that
provider id is also exposed as `headroom-upstream-request-id`. Compressed
requests add `x-headroom-tokens-before`, `x-headroom-tokens-after`,
`x-headroom-tokens-saved`, `x-headroom-transforms`, and
`x-headroom-ccr-hashes` response headers.
For Codex/ChatGPT backend requests, if `ChatGPT-Account-ID` is absent and the
OAuth bearer JWT payload contains
`chatgpt_account_id` under the `https://api.openai.com/auth` claim, the proxy
forwards that value as a best-effort routing hint; authentication is still left
to the upstream.

Set `x-headroom-bypass: true` or `x-headroom-mode: passthrough` to skip
compression for a target request while still stripping those internal headers
before the upstream call.

Compression mutates only live-zone user/tool/tool-result content of at least
512 bytes. It does not modify system prompts, tool definitions, historical
messages, reasoning blocks, thinking signatures, CCR retrieval outputs, or
content blocks carrying cache-control metadata. When a request contains a CCR
marker or compression creates one, the proxy injects a `headroom_retrieve` tool
definition in the provider's tool schema so the model can request the original
content.
Compressed OpenAI/Codex requests also receive a deterministic 32-hex
`prompt_cache_key` derived from model, system content, and tools when they do
not already provide one. This cache metadata is PAYG-only: OAuth and
subscription-shaped clients still get live-zone compression, but automatic
cache metadata mutation is skipped. Compressed Anthropic requests sort tools by
name and add ephemeral `cache_control` to the last tool when no cache-control
metadata is already present.

SSE responses are streamed byte-for-byte while a non-blocking parser records
usage and cache-read counters from OpenAI Chat/Responses and Anthropic event
streams into `/stats` and `/metrics`. `headroom_retrieve` tool calls appearing
in SSE streams are counted but left for the client to resolve.

When a non-streaming JSON response from `/v1/messages` or
`/v1/chat/completions` contains `headroom_retrieve` tool calls, the proxy
resolves them from the CCR store and continues the conversation upstream
(up to 3 rounds) so the client receives the final response without seeing the
retrieval tool call. If a continuation request fails or the round limit is
hit, the latest upstream response is returned as-is. Anthropic batch results
get the same treatment: the proxy records each batch create's compressed
params, and `GET /v1/messages/batches/{id}/results` lines whose message holds
a `headroom_retrieve` tool call are continued against `/v1/messages` using the
caller's auth headers before the JSONL is returned. Results for batches the
proxy did not create pass through unchanged.

Current live-zone policy:

- OpenAI Chat: skips `n > 1`, compresses the latest user message and latest
  non-`headroom_retrieve` tool result.
- OpenAI Responses: compresses current-frame output items such as
  `function_call_output`, `local_shell_call_output`, and
  `apply_patch_call_output`, while preserving `headroom_retrieve` outputs,
  encrypted reasoning, compaction, computer, and unknown items.
- Anthropic Messages: compresses the latest user message content and skips
  thinking/cache-control blocks.
- Anthropic Message Batches: compresses each create request's
  `requests[].params.messages`; batch subpaths stream through unchanged.

Compression-only mode is available without calling an upstream LLM:

```bash
curl -X POST http://127.0.0.1:8787/v1/compress \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-test","messages":[{"role":"user","content":"large content"}]}'
```

It returns compressed `messages`, token counters, `ccr_hashes`, and any injected
retrieve `tools`.

CCR retrieval can be called directly:

```bash
curl http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567
curl -X POST http://127.0.0.1:8787/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"hash":"0123456789abcdef01234567"}'
```

Logs are structured `tracing` JSON. Use `--log-level info`, `debug`, or `trace`
to control startup/request summaries, classification and byte/token decisions,
and truncated internal diagnostics.

## Library

The core API is exported from the `hr` crate:

```rust
use hr::{compress, decompress_hash, decompress_text, stats, CompressOptions, SqliteStore};

let store = SqliteStore::open("ccr.sqlite")?;
let result = compress("large content", CompressOptions { store: &store, min_bytes: 1 });
let original = decompress_hash(result.hash.as_deref().unwrap(), &store);
let expanded = decompress_text(&result.output, &store);
let snapshot = stats();
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

CCR hashes are deterministic SHA-256 prefixes encoded as 24 lowercase hex
characters. Markers use the `<<ccr:HASH>>` format.

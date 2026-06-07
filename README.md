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

Anthropic subpaths such as `/v1/messages/count_tokens` and batch
read/cancel/delete paths are routed to the Anthropic upstream without request
compression. Batch results pass through unchanged unless this proxy previously
created the batch and can post-process `headroom_retrieve` tool calls from its
stored batch context. Other paths route to the OpenAI upstream unless they are
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

### Request handling

- Known JSON LLM request paths are buffered up to `--max-body-bytes`,
  conservatively compressed, and then forwarded.
- Non-JSON bodies, non-target paths, Conversations API paths
  (`/v1/conversations*`), and other HTTP requests stream through without
  request mutation.
- Codex Responses WebSocket sessions on `/v1/responses` aliases compress
  client `response.create` frames and otherwise relay bidirectionally; other
  WebSocket upgrades pass through to the selected upstream.

### Flags

- `--compression false`, `--compression-mode off`, or
  `--compression-mode passthrough` disable target request mutation.
- `--compression-max-body-bytes` is accepted as an alias for
  `--max-body-bytes`.
- User-message text is protected by default; pass `--compress-user-messages`
  (or set `HEADROOM_COMPRESS_USER_MESSAGES=1`) to opt user text into
  compression as well.

### Headers

- Upstream-bound: hop-by-hop headers and proxy-internal `x-headroom-*` headers
  are stripped, `x-request-id` is preserved or created, and basic
  `x-forwarded-*` headers are added.
- Responses are streamed back with hop-by-hop response headers filtered. The
  proxy returns its request id as `x-request-id`; an upstream `request-id` /
  `x-request-id` is also exposed as `headroom-upstream-request-id`.
- Compressed requests add `x-headroom-tokens-before`,
  `x-headroom-tokens-after`, `x-headroom-tokens-saved`,
  `x-headroom-transforms`, and `x-headroom-ccr-hashes` response headers.
- Codex/ChatGPT backend requests: if `ChatGPT-Account-ID` is absent and the
  OAuth bearer JWT carries `chatgpt_account_id` under the
  `https://api.openai.com/auth` claim, that value is forwarded as a
  best-effort routing hint; authentication is still left to the upstream.
- Per-request opt-out: send `x-headroom-bypass: true` or
  `x-headroom-mode: passthrough` to skip compression while still stripping the
  internal headers before the upstream call.

### What compression touches

Only live-zone tool/tool-result content of at least 512 bytes is mutated.
Mutations are applied as byte-range splices: every byte outside the replaced
spans is forwarded exactly as the client sent it.

Never modified:

- user, system, and assistant text (user text becomes a candidate only with
  `--compress-user-messages`)
- system prompts, tool definitions, and historical messages
- reasoning blocks, thinking signatures, and CCR retrieval outputs
- content blocks carrying cache-control metadata
- prefixes the provider has already confirmed as cached for the session
  (tracked from usage fields in JSON and SSE responses)

When a request contains a CCR marker or compression creates one, the proxy
injects a `headroom_retrieve` tool definition in the provider's tool schema so
the model can request the original content.

### Cache metadata (PAYG only)

OAuth and subscription-shaped clients still get live-zone compression, but
automatic cache metadata mutation is skipped. For PAYG requests:

- Compressed OpenAI/Codex requests receive a deterministic 32-hex
  `prompt_cache_key` derived from model, system content, and tools when they
  do not already provide one.
- Compressed Anthropic requests sort tools by name and add ephemeral
  `cache_control` to the last tool when no cache-control metadata is already
  present.

### Live-zone policy by surface

- **OpenAI Chat**: skips `n > 1`, compresses the latest
  non-`headroom_retrieve` tool result; the latest user message is compressed
  only with `--compress-user-messages`.
- **OpenAI Responses**: compresses current-frame output items such as
  `function_call_output`, `local_shell_call_output`, and
  `apply_patch_call_output`, while preserving `headroom_retrieve` outputs,
  encrypted reasoning, compaction, computer, and unknown items.
- **Anthropic Messages**: compresses `tool_result` content and skips
  thinking/cache-control blocks; user, system, and assistant text is protected
  unless user text is opted in with `--compress-user-messages`.
- **Anthropic Message Batches**: compresses each create request's
  `requests[].params.messages`; batch subpaths stream through unchanged.

### Streaming and telemetry

SSE responses are streamed byte-for-byte. A bounded, non-blocking parser tee
records into `/stats` and `/metrics`:

- usage, cache-read, and inferred cache-write counters
- `service_tier` and terminal response status
- rate-limit header gauges

A saturated parser drops telemetry, never client bytes. Confirmed cache usage
also feeds the per-session frozen-prefix tracker. `headroom_retrieve` tool
calls appearing in SSE streams are counted but left for the client to resolve.

### CCR continuation for JSON responses

When a non-streaming JSON response from `/v1/messages` or
`/v1/chat/completions` contains `headroom_retrieve` tool calls, the proxy
resolves them from the CCR store and continues the conversation upstream (up
to 3 rounds) so the client receives the final response without seeing the
retrieval tool call.

- Continuation runs only when every tool call in the response is a
  `headroom_retrieve` call and, for OpenAI, the response has a single choice;
  multi-choice responses are never auto-continued.
- Mixed responses keep client-owned tool calls and strip the proxy-private
  retrieval calls instead — including when a mixed response arrives after a
  continuation round, a failed continuation request, or the round limit.
- An OpenAI choice left with no tool calls after stripping is rewritten as a
  normal text turn (`finish_reason: "stop"`).
- Only when the response still contains nothing but unresolved retrieval calls
  (after a failure, the round limit, or the multi-choice skip) is the latest
  upstream response returned as-is.

Anthropic batch results get the same treatment: the proxy records each batch
create's compressed params, and `GET /v1/messages/batches/{id}/results` lines
whose message holds only `headroom_retrieve` tool calls are continued against
`/v1/messages` using the caller's auth headers before the JSONL is returned;
mixed result lines have their private retrieval calls stripped the same way.
Batch contexts are retained in memory for 24 hours, up to 10000 batches.
Results for batches the proxy did not create, or whose context has expired,
pass through unchanged.

The proxy assumes a single trust boundary: every client sharing one `hr`
instance shares its CCR store and batch contexts, so run separate instances
for mutually untrusted clients.

### Local endpoints

Compression-only mode is available without calling an upstream LLM. It returns
compressed `messages`, token counters, `ccr_hashes`, and any injected retrieve
`tools`:

```bash
curl -X POST http://127.0.0.1:8787/v1/compress \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-test","messages":[{"role":"user","content":"large content"}]}'
```

CCR retrieval can be called directly:

```bash
curl http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567
curl -X POST http://127.0.0.1:8787/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"hash":"0123456789abcdef01234567"}'
```

### Logging

Logs are structured `tracing` JSON. Use `--log-level info`, `debug`, or
`trace` to control startup/request summaries, classification and byte/token
decisions, and internal diagnostics such as WebSocket frame kinds and sizes.
Request and frame contents are never logged at any level.

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

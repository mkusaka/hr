use axum::body::{to_bytes, Body};
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderValue, Request, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{any, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use hr::{build_router, stats, CcrStore, CompressionMode, ProxyState, SqliteStore};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use url::Url;

#[derive(Clone, Default)]
struct Capture {
    bodies: Arc<Mutex<Vec<Value>>>,
    raw_bodies: Arc<Mutex<Vec<Vec<u8>>>>,
    headers: Arc<Mutex<Vec<HeaderMap>>>,
    paths: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone, Default)]
struct WsCapture {
    frames: Arc<Mutex<Vec<String>>>,
    headers: Arc<Mutex<Vec<HeaderMap>>>,
}

#[tokio::test]
async fn http_passthrough_for_non_target_paths() {
    let capture = Capture::default();
    let upstream = spawn_upstream(capture.clone()).await;
    let proxy = spawn_proxy(upstream, upstream).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{proxy}/v1/files"))
        .json(&json!({"unchanged": true}))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        capture.bodies.lock().unwrap()[0],
        json!({"unchanged": true})
    );
    assert_eq!(capture.paths.lock().unwrap()[0], "/v1/files");
}

#[tokio::test]
async fn compresses_anthropic_messages() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("latest anthropic");

    post_json(
        proxy,
        "/v1/messages",
        json!({
            "system": "stable",
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": latest}
                ]}
            ]
        }),
    )
    .await;

    let bodies = anthropic_capture.bodies.lock().unwrap();
    let body = bodies.last().unwrap();
    assert_eq!(body["system"], "stable");
    assert_eq!(body["messages"][0]["content"], "old");
    assert!(body["messages"][2]["content"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert!(body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "headroom_retrieve"));
}

#[tokio::test]
async fn compresses_openai_chat_completions() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("latest openai");

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "ask"},
                {"role": "tool", "tool_call_id": "call_bulk", "content": latest}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = bodies.last().unwrap();
    assert_eq!(body["messages"][0]["content"], "stable");
    // User text is protected by default (reference role gate); the
    // latest tool output compresses.
    assert_eq!(body["messages"][1]["content"], "ask");
    assert!(body["messages"][2]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert!(body["tools"].as_array().unwrap().iter().any(|tool| {
        tool["type"] == "function" && tool["function"]["name"] == "headroom_retrieve"
    }));
}

#[tokio::test]
async fn compresses_openai_responses() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("latest responses");

    post_json(
        proxy,
        "/v1/responses",
        json!({
            "input": [
                {"type": "function_call_output", "call_id": "call_a", "output": latest}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = bodies.last().unwrap();
    assert!(body["input"][0]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert!(body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| { tool["type"] == "function" && tool["name"] == "headroom_retrieve" }));
}

#[tokio::test]
async fn compresses_codex_response_aliases() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("codex alias");

    post_json(
        proxy,
        "/backend-api/codex/responses",
        json!({
            "input": [
                {"type": "function_call_output", "call_id": "call_a", "output": latest}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    assert!(bodies[0]["input"][0]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
}

#[tokio::test]
async fn anthropic_subpaths_route_to_anthropic_upstream_without_compression() {
    let openai_capture = Capture::default();
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/messages/count_tokens",
        json!({"messages": [{"role": "user", "content": "count tokens"}]}),
    )
    .await;

    assert!(openai_capture.bodies.lock().unwrap().is_empty());
    assert_eq!(anthropic_capture.bodies.lock().unwrap().len(), 1);
    assert_eq!(
        anthropic_capture.paths.lock().unwrap()[0],
        "/v1/messages/count_tokens"
    );
}

#[tokio::test]
async fn openai_chat_mutates_latest_tool_skips_retrieve_output_and_protects_user_text() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let tool_output = long_text("tool output");
    let latest_user = long_text("latest user");

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "call_retrieve", "type": "function", "function": {"name": "headroom_retrieve", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_retrieve", "content": "retrieved original"},
                {"role": "tool", "tool_call_id": "call_other", "content": tool_output},
                {"role": "user", "content": latest_user}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    assert_eq!(body["messages"][1]["content"], "retrieved original");
    assert!(body["messages"][2]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    // User text is protected by default; opt-in is required to
    // compress it (reference `compress_user_messages = False`).
    assert_eq!(body["messages"][3]["content"], latest_user);
}

#[tokio::test]
async fn openai_chat_skips_multi_choice_requests() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({
            "n": 2,
            "messages": [{"role": "user", "content": "do not compress multi choice"}]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    assert_eq!(
        body["messages"][0]["content"],
        "do not compress multi choice"
    );
    assert!(body.get("tools").is_none());
}

#[tokio::test]
async fn responses_mutates_all_current_output_items_and_skips_retrieve_output() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let first = long_text("first output");
    let second = long_text("second output");

    post_json(
        proxy,
        "/v1/responses",
        json!({
            "input": [
                {"type": "function_call", "call_id": "call_retrieve", "name": "headroom_retrieve", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_a", "output": first},
                {"type": "function_call_output", "call_id": "call_retrieve", "output": "retrieved original"},
                {"type": "local_shell_call_output", "call_id": "call_b", "output": second}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    assert!(body["input"][1]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert_eq!(body["input"][2]["output"], "retrieved original");
    assert!(body["input"][3]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
}

#[tokio::test]
async fn responses_preserves_unknown_and_encrypted_items_while_compressing_supported_outputs() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/responses",
        json!({
            "input": [
                {"type": "reasoning", "encrypted_content": "sealed", "summary": []},
                {"type": "compaction", "payload": {"phase": "keep"}},
                {"type": "computer_call_output", "call_id": "computer", "output": long_text("computer output should stay")},
                {"type": "custom_unknown", "nested": {"value": "keep"}},
                {"type": "function_call_output", "call_id": "call_a", "output": long_text("function output should compress")}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    assert_eq!(body["input"][0]["encrypted_content"], "sealed");
    assert_eq!(body["input"][1]["payload"]["phase"], "keep");
    assert_eq!(
        body["input"][2]["output"],
        long_text("computer output should stay")
    );
    assert_eq!(body["input"][3]["nested"]["value"], "keep");
    assert!(body["input"][4]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
}

#[tokio::test]
async fn openai_requests_get_deterministic_prompt_cache_key_when_compressed() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let tools = json!([
        {"type": "function", "function": {"name": "z_tool", "parameters": {"type": "object"}}}
    ]);
    let body = json!({
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "stable system"},
            {"role": "user", "content": long_text("cache key")}
        ],
        "tools": tools.clone()
    });
    let body_with_different_user = json!({
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "stable system"},
            {"role": "user", "content": long_text("different user")}
        ],
        "tools": tools
    });

    post_json(proxy, "/v1/chat/completions", body.clone()).await;
    post_json(proxy, "/v1/chat/completions", body_with_different_user).await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let first = bodies[0]["prompt_cache_key"].as_str().unwrap();
    let second = bodies[1]["prompt_cache_key"].as_str().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 32);
    assert!(first.chars().all(|char| char.is_ascii_hexdigit()));
}

#[tokio::test]
async fn openai_prompt_cache_key_uses_system_content_not_system_object_metadata() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "name": "alpha", "content": "stable system"},
                {"role": "user", "content": long_text("same system content")}
            ]
        }),
    )
    .await;
    post_json(
        proxy,
        "/v1/chat/completions",
        json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "name": "beta", "content": "stable system"},
                {"role": "user", "content": long_text("different user content")}
            ]
        }),
    )
    .await;
    post_json(
        proxy,
        "/v1/chat/completions",
        json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "name": "alpha", "content": "changed system"},
                {"role": "user", "content": long_text("same system content")}
            ]
        }),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let first = bodies[0]["prompt_cache_key"].as_str().unwrap();
    let same_content = bodies[1]["prompt_cache_key"].as_str().unwrap();
    let changed_system = bodies[2]["prompt_cache_key"].as_str().unwrap();
    assert_eq!(first, same_content);
    assert_ne!(first, changed_system);
}

#[tokio::test]
async fn target_requests_below_live_zone_floor_are_not_compressed() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "small"}]}),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    assert_eq!(bodies[0]["messages"][0]["content"], "small");
    assert!(bodies[0].get("tools").is_none());
    assert!(bodies[0]["prompt_cache_key"]
        .as_str()
        .unwrap()
        .chars()
        .all(|char| char.is_ascii_hexdigit()));
}

#[tokio::test]
async fn oauth_openai_requests_skip_prompt_cache_key_metadata() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .header("authorization", "Bearer oauth.header.payload")
        .json(&json!({
            "messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": long_text("oauth request")}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bodies = openai_capture.bodies.lock().unwrap();
    assert!(bodies[0]["messages"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert!(bodies[0].get("prompt_cache_key").is_none());
}

#[tokio::test]
async fn compression_disabled_streams_target_request_without_mutation() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy_with_compression(openai, anthropic, false).await;

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": long_text("disabled compression")}]}),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    assert_eq!(
        bodies[0]["messages"][0]["content"],
        long_text("disabled compression")
    );
    assert!(bodies[0].get("tools").is_none());
    assert!(bodies[0].get("prompt_cache_key").is_none());
}

#[tokio::test]
async fn anthropic_batch_create_compresses_and_injects_tool_per_compressed_request() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/messages/batches",
        json!({
            "requests": [
                {
                    "custom_id": "one",
                    "params": {
                        "model": "claude-test",
                        "tools": [{"name": "z_user_tool", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "tr_1", "content": long_text("batch one")}
                        ]}]
                    }
                },
                {
                    "custom_id": "two",
                    "params": {
                        "model": "claude-test",
                        "messages": [{"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "tr_2", "content": "tiny"}
                        ]}]
                    }
                }
            ]
        }),
    )
    .await;

    let bodies = anthropic_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    assert_eq!(
        anthropic_capture.paths.lock().unwrap()[0],
        "/v1/messages/batches"
    );
    assert!(
        body["requests"][0]["params"]["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:")
    );
    // The sub-threshold request is forwarded unchanged and, per the
    // reference (`anthropic.py:2560-2574`), does NOT receive the
    // retrieve tool: injection is per-request, compressed only.
    assert_eq!(
        body["requests"][1]["params"]["messages"][0]["content"][0]["content"],
        "tiny"
    );
    assert!(body["requests"][1]["params"].get("tools").is_none());
    assert!(body["requests"][0]["params"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "headroom_retrieve"));
    assert!(body["requests"][0]["params"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "z_user_tool" && tool.get("cache_control").is_some()));
}

#[tokio::test]
async fn anthropic_batch_subpaths_passthrough_without_compression() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/messages/batches/batch_123/cancel",
        json!({"unchanged": "batch subpath"}),
    )
    .await;

    let bodies = anthropic_capture.bodies.lock().unwrap();
    assert_eq!(bodies[0], json!({"unchanged": "batch subpath"}));
    assert_eq!(
        anthropic_capture.paths.lock().unwrap()[0],
        "/v1/messages/batches/batch_123/cancel"
    );
}

#[tokio::test]
async fn anthropic_ccr_tool_call_is_resolved_by_proxy_continuation() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_ccr_upstream(anthropic_capture.clone(), 1).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("continuation anthropic");

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": latest}
                ]}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "final answer");

    let bodies = anthropic_capture.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let continuation = &bodies[1];
    assert_eq!(continuation["model"], "claude-test");
    assert_eq!(continuation["max_tokens"], 128);
    let messages = continuation["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert!(messages[0]["content"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["type"], "tool_use");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_1");
    assert!(messages[2]["content"][0]["content"]
        .as_str()
        .unwrap()
        .contains(&latest));
}

#[tokio::test]
async fn openai_ccr_tool_call_is_resolved_by_proxy_continuation() {
    let openai_capture = Capture::default();
    let openai = spawn_openai_ccr_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("continuation openai");

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "please"},
                {"role": "tool", "tool_call_id": "call_bulk", "content": latest}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "final answer");

    let bodies = openai_capture.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let continuation = &bodies[1];
    assert_eq!(continuation["model"], "gpt-test");
    let messages = continuation["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0]["content"], "please");
    assert!(messages[1]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(
        messages[2]["tool_calls"][0]["function"]["name"],
        "headroom_retrieve"
    );
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call_1");
    assert!(messages[3]["content"].as_str().unwrap().contains(&latest));
}

#[tokio::test]
async fn ccr_continuation_stops_at_max_rounds() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_ccr_upstream(anthropic_capture.clone(), usize::MAX).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("max rounds");

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": latest}
                ]}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    // The round limit was hit, so the unresolved tool call is surfaced to the
    // client (mirroring the reference handler).
    assert_eq!(body["content"][0]["type"], "tool_use");
    // Initial request plus exactly three continuation rounds.
    assert_eq!(anthropic_capture.bodies.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn ccr_continuation_strips_mixed_response_at_round_limit() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_ccr_then_mixed_upstream(anthropic_capture.clone(), 3).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": long_text("round limit mixed")}
                ]}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["name"], "project_tool");
    assert_eq!(anthropic_capture.bodies.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn ccr_continuation_strips_mixed_response_after_successful_round() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_ccr_then_mixed_upstream(anthropic_capture.clone(), 1).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": long_text("mixed after round")}
                ]}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["name"], "project_tool");
    // Initial request plus exactly one successful continuation round; the
    // mixed follow-up response breaks the loop instead of continuing.
    assert_eq!(anthropic_capture.bodies.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn mixed_anthropic_tool_calls_are_not_auto_continued() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_mixed_tool_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": long_text("mixed tools")}
                ]}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["name"], "project_tool");
    assert_eq!(anthropic_capture.bodies.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn mixed_openai_tool_calls_are_not_auto_continued() {
    let openai_capture = Capture::default();
    let openai = spawn_openai_mixed_tool_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-test",
            "messages": [
                {"role": "tool", "tool_call_id": "call_bulk", "content": long_text("mixed openai tools")}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    let first_tool_calls = choices[0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(first_tool_calls.len(), 1);
    assert_eq!(first_tool_calls[0]["function"]["name"], "project_tool");
    let second_tool_calls = choices[1]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(second_tool_calls.len(), 1);
    assert_eq!(second_tool_calls[0]["function"]["name"], "project_tool");
    assert_eq!(openai_capture.bodies.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn mixed_openai_single_choice_tool_calls_are_not_auto_continued() {
    let openai_capture = Capture::default();
    let openai = spawn_openai_single_choice_mixed_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-test",
            "messages": [
                {"role": "tool", "tool_call_id": "call_bulk", "content": long_text("single choice mixed")}
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    let tool_calls = choices[0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["function"]["name"], "project_tool");
    assert_eq!(choices[0]["finish_reason"], "tool_calls");
    assert_eq!(openai_capture.bodies.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn anthropic_batch_results_ccr_tool_calls_are_continued() {
    let capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_batch_ccr_upstream(capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let latest = long_text("batch continuation");

    post_json(
        proxy,
        "/v1/messages/batches",
        json!({
            "requests": [{
                "custom_id": "req-1",
                "params": {
                    "model": "claude-test",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "tr_1", "content": latest}
                    ]}]
                }
            }]
        }),
    )
    .await;

    let response = reqwest::Client::new()
        .get(format!(
            "http://{proxy}/v1/messages/batches/batch_test/results"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let text = response.text().await.unwrap();
    let line: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(line["custom_id"], "req-1");
    assert_eq!(
        line["result"]["message"]["content"][0]["text"],
        "final answer"
    );

    // The continuation reused the stored compressed batch params against
    // /v1/messages.
    let paths = capture.paths.lock().unwrap();
    let bodies = capture.bodies.lock().unwrap();
    let continuation = bodies
        .iter()
        .zip(paths.iter())
        .find(|(_, path)| *path == "/v1/messages")
        .map(|(body, _)| body.clone())
        .expect("continuation request reached /v1/messages");
    assert_eq!(continuation["model"], "claude-test");
    assert_eq!(continuation["max_tokens"], 64);
    let messages = continuation["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    assert!(messages[2]["content"][0]["content"]
        .as_str()
        .unwrap()
        .contains(&latest));
}

#[tokio::test]
async fn anthropic_batch_results_mixed_tool_calls_strip_private_retrieval() {
    let capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_batch_mixed_tool_upstream(capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/messages/batches",
        json!({
            "requests": [{
                "custom_id": "req-1",
                "params": {
                    "model": "claude-test",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "tr_1", "content": long_text("batch mixed")}
                    ]}]
                }
            }]
        }),
    )
    .await;

    let response = reqwest::Client::new()
        .get(format!(
            "http://{proxy}/v1/messages/batches/batch_test/results"
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = response.text().await.unwrap();
    let line: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(line["custom_id"], "req-1");
    let content = line["result"]["message"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["name"], "project_tool");

    let paths = capture.paths.lock().unwrap();
    assert_eq!(paths.len(), 2);
    assert!(!paths.iter().any(|path| path == "/v1/messages"));
}

#[tokio::test]
async fn batch_results_without_context_pass_through_unchanged() {
    let capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_batch_ccr_upstream(capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    // No batch create through this proxy, so there is no stored context and
    // the results bytes must pass through untouched.
    let response = reqwest::Client::new()
        .get(format!(
            "http://{proxy}/v1/messages/batches/batch_unknown/results"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let text = response.text().await.unwrap();
    assert!(text.contains("headroom_retrieve"));
    assert!(!capture
        .paths
        .lock()
        .unwrap()
        .iter()
        .any(|p| p == "/v1/messages"));
}

#[tokio::test]
async fn sse_ccr_tool_call_is_recorded_without_mutating_stream() {
    let upstream = spawn_sse_ccr_upstream().await;
    let proxy = spawn_proxy(upstream, upstream).await;
    let before = stats().ccr_stream_tool_calls;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let text = response.text().await.unwrap();
    // Bytes are passed through unchanged; the retrieval tool call is left for
    // the client to resolve (reference streaming behaviour).
    assert!(text.contains("\"name\":\"headroom_retrieve\""));
    assert!(text.contains("message_stop"));
    assert!(stats().ccr_stream_tool_calls > before);
}

#[tokio::test]
async fn anthropic_message_with_cache_control_freezes_whole_message() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let sibling = long_text("same message sibling should remain");

    post_json(
        proxy,
        "/v1/messages",
        json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "cached block", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": sibling}
                    ]
                }
            ]
        }),
    )
    .await;

    let bodies = anthropic_capture.bodies.lock().unwrap();
    assert_eq!(bodies[0]["messages"][0]["content"][1]["text"], sibling);
    assert!(bodies[0].get("tools").is_none());
}

#[tokio::test]
async fn anthropic_schema_property_named_cache_control_does_not_block_auto_cache_control() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/messages",
        json!({
            "tools": [
                {
                    "name": "schema_tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "cache_control": {"type": "string"}
                        }
                    }
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": long_text("anthropic schema property")}]}]
        }),
    )
    .await;

    let bodies = anthropic_capture.bodies.lock().unwrap();
    let tools = bodies[0]["tools"].as_array().unwrap();
    let schema_tool = tools
        .iter()
        .find(|tool| tool["name"] == "schema_tool")
        .unwrap();
    assert!(schema_tool.get("cache_control").is_some());
}

#[tokio::test]
async fn anthropic_existing_tool_cache_control_preserves_tool_order_without_auto_placement() {
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/messages",
        json!({
            "tools": [
                {
                    "name": "z_cached_tool",
                    "cache_control": {"type": "ephemeral"},
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "a_plain_tool",
                    "input_schema": {"type": "object"}
                }
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": long_text("existing tool cache control")}
                ]}]
        }),
    )
    .await;

    let bodies = anthropic_capture.bodies.lock().unwrap();
    let tools = bodies[0]["tools"].as_array().unwrap();
    assert!(bodies[0]["messages"][0]["content"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert_eq!(tools[0]["name"], "z_cached_tool");
    assert_eq!(tools[1]["name"], "a_plain_tool");
    assert!(tools[0].get("cache_control").is_some());
    assert!(tools[1].get("cache_control").is_none());
}

#[tokio::test]
async fn direct_decompress_from_proxy_store() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let (proxy, store) = spawn_proxy_with_store(openai, anthropic).await;
    let content = long_text("retrieve me");

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({"messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": content}]}),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    let marker = bodies[0]["messages"][0]["content"].as_str().unwrap();
    let expanded = hr::decompress_text(marker, &store);

    assert_eq!(expanded.output, long_text("retrieve me"));
    assert_eq!(expanded.hits, 1);
}

#[tokio::test]
async fn websocket_passthrough() {
    let openai = spawn_ws_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let (mut socket, _) = connect_async(format!("ws://{proxy}/ws")).await.unwrap();
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            "hello".into(),
        ))
        .await
        .unwrap();

    let message = socket.next().await.unwrap().unwrap();
    assert_eq!(message.into_text().unwrap(), "echo:hello");
}

#[tokio::test]
async fn codex_responses_websocket_compresses_response_create_frames_and_records_usage() {
    let before = stats();
    let ws_capture = WsCapture::default();
    let openai = spawn_codex_ws_capture_upstream(ws_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let jwt_payload =
        "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF93cyJ9fQ";

    let mut request = format!("ws://{proxy}/v1/responses")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("openai-beta", HeaderValue::from_static("existing_beta"));
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer fake.{jwt_payload}.sig")).unwrap(),
    );

    let (mut socket, _) = connect_async(request).await.unwrap();
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({
                "type": "response.create",
                "response": {
                    "input": [
                        {"type": "function_call_output", "call_id": "call_a",
                         "output": long_text("codex ws first")}
                    ]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    assert_eq!(
        socket.next().await.unwrap().unwrap().into_text().unwrap(),
        json!({
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 21,
                    "output_tokens": 5,
                    "input_tokens_details": {"cached_tokens": 7}
                }
            }
        })
        .to_string()
    );

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({"type": "response.cancel", "response_id": "resp_1"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    let _ = socket.next().await.unwrap().unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({
                "type": "response.create",
                "response": {
                    "input": [
                        {"type": "function_call_output", "call_id": "call_b",
                         "output": long_text("codex ws second")}
                    ]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    let _ = socket.next().await.unwrap().unwrap();

    let frames = ws_capture.frames.lock().unwrap();
    assert_eq!(frames.len(), 3);
    let first: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(first["type"], "response.create");
    assert!(first["response"]["input"][0]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert!(first["response"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| { tool["type"] == "function" && tool["name"] == "headroom_retrieve" }));
    assert_eq!(
        serde_json::from_str::<Value>(&frames[1]).unwrap(),
        json!({"type": "response.cancel", "response_id": "resp_1"})
    );
    let third: Value = serde_json::from_str(&frames[2]).unwrap();
    assert!(third["response"]["input"][0]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));

    let headers = ws_capture.headers.lock().unwrap();
    let headers = &headers[0];
    let beta = headers.get("openai-beta").unwrap().to_str().unwrap();
    assert!(beta.contains("existing_beta"));
    assert!(beta.contains("responses_websockets=2026-02-06"));
    assert_eq!(headers.get("ChatGPT-Account-ID").unwrap(), "acct_ws");

    let after = stats();
    assert!(after.compressed_requests >= before.compressed_requests + 2);
    assert!(after.sse_input_tokens >= before.sse_input_tokens + 21);
    assert!(after.sse_output_tokens >= before.sse_output_tokens + 5);
    assert!(after.sse_cache_read_input_tokens >= before.sse_cache_read_input_tokens + 7);
}

#[tokio::test]
async fn codex_responses_websocket_injects_retrieve_tool_for_marker_only_frames() {
    let ws_capture = WsCapture::default();
    let openai = spawn_codex_ws_capture_upstream(ws_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let (mut socket, _) = connect_async(format!("ws://{proxy}/v1/responses"))
        .await
        .unwrap();
    // The frame carries an existing marker but nothing newly
    // compressible: the metadata-only rewrite (retrieve-tool
    // injection) must still reach upstream.
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({
                "type": "response.create",
                "response": {
                    "input": [
                        {"type": "function_call_output", "call_id": "call_a",
                         "output": "<<ccr:aaaaaaaaaaaaaaaaaaaaaaaa>> summary"}
                    ]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    let _ = socket.next().await.unwrap().unwrap();

    let frames = ws_capture.frames.lock().unwrap();
    let frame: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(
        frame["response"]["input"][0]["output"],
        "<<ccr:aaaaaaaaaaaaaaaaaaaaaaaa>> summary"
    );
    assert!(frame["response"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "headroom_retrieve"));
}

#[tokio::test]
async fn codex_responses_websocket_bypass_leaves_response_create_frame_unchanged() {
    let ws_capture = WsCapture::default();
    let openai = spawn_codex_ws_capture_upstream(ws_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let original_text = long_text("codex ws bypass");

    let mut request = format!("ws://{proxy}/v1/responses")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("x-headroom-bypass", HeaderValue::from_static("true"));

    let (mut socket, _) = connect_async(request).await.unwrap();
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({
                "type": "response.create",
                "response": {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": original_text}]
                        }
                    ]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    let _ = socket.next().await.unwrap().unwrap();

    let frames = ws_capture.frames.lock().unwrap();
    let frame: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(
        frame["response"]["input"][0]["content"][0]["text"],
        original_text
    );
    assert!(frame["response"].get("tools").is_none());

    let headers = ws_capture.headers.lock().unwrap();
    assert!(headers[0].get("x-headroom-bypass").is_none());
}

#[tokio::test]
async fn codex_responses_websocket_falls_back_to_http_streaming_when_ws_upstream_fails() {
    let openai_capture = Capture::default();
    let openai = spawn_codex_http_fallback_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let (mut socket, _) = connect_async(format!("ws://{proxy}/v1/responses"))
        .await
        .unwrap();
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({
                "type": "response.create",
                "response": {
                    "input": [
                        {"type": "function_call_output", "call_id": "call_a",
                         "output": long_text("codex ws fallback")}
                    ]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let message = socket.next().await.unwrap().unwrap().into_text().unwrap();
    assert_eq!(message, json!({"type": "response.completed"}).to_string());

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    assert_eq!(body["stream"], true);
    assert!(body.get("type").is_none());
    assert!(body["input"][0]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
}

#[tokio::test]
async fn stats_endpoint_reports_counters() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({"messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": long_text("count me")}]}),
    )
    .await;

    let response: Value = reqwest::get(format!("http://{proxy}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert!(response["total_requests"].as_u64().unwrap() >= 1);
    assert!(response["compressed_requests"].as_u64().unwrap() >= 1);
    assert!(stats().savings_ratio >= 0.0);
}

#[tokio::test]
async fn health_and_metrics_endpoints_are_intercepted() {
    let openai_capture = Capture::default();
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let health: Value = reqwest::get(format!("http://{proxy}/healthz"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["ok"], true);
    assert_eq!(health["service"], "hr-proxy");
    assert!(openai_capture.bodies.lock().unwrap().is_empty());
    assert!(anthropic_capture.bodies.lock().unwrap().is_empty());

    let upstream: Value = reqwest::get(format!("http://{proxy}/healthz/upstream"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(upstream["ok"], true);

    let metrics = reqwest::get(format!("http://{proxy}/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("hr_total_requests"));
}

#[tokio::test]
async fn health_aliases_are_intercepted() {
    let openai_capture = Capture::default();
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let livez: Value = reqwest::get(format!("http://{proxy}/livez"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(livez["alive"], true);

    let readyz = reqwest::get(format!("http://{proxy}/readyz"))
        .await
        .unwrap();
    assert_eq!(readyz.status(), StatusCode::OK);

    let health: Value = reqwest::get(format!("http://{proxy}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["service"], "hr-proxy");
    assert!(openai_capture.bodies.lock().unwrap().is_empty());
    assert!(anthropic_capture.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn readyz_reports_unavailable_when_an_upstream_is_down() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let down_openai = listener.local_addr().unwrap();
    drop(listener);
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(down_openai, anthropic).await;

    let readyz = reqwest::get(format!("http://{proxy}/readyz"))
        .await
        .unwrap();
    assert_eq!(readyz.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn compress_endpoint_compresses_without_forwarding() {
    let openai_capture = Capture::default();
    let anthropic_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(anthropic_capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response: Value = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/compress"))
        .json(&json!({
            "model": "gpt-test",
            "messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": long_text("compress endpoint content")}]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert!(response["messages"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert_eq!(response["ccr_hashes"].as_array().unwrap().len(), 1);
    assert!(response["tools"].as_array().unwrap().iter().any(|tool| {
        tool["type"] == "function" && tool["function"]["name"] == "headroom_retrieve"
    }));
    assert!(openai_capture.bodies.lock().unwrap().is_empty());
    assert!(anthropic_capture.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn bypass_header_skips_target_compression_and_is_stripped_upstream() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let (proxy, store) = spawn_proxy_with_store(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .header("x-headroom-bypass", "true")
        .header("x-request-id", "req-bypass")
        .json(&json!({
            "messages": [{"role": "user", "content": "do not mutate"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-request-id").unwrap(),
        "req-bypass"
    );

    let bodies = openai_capture.bodies.lock().unwrap();
    assert_eq!(bodies[0]["messages"][0]["content"], "do not mutate");
    let headers = openai_capture.headers.lock().unwrap();
    assert!(headers[0].get("x-headroom-bypass").is_none());
    assert_eq!(store.count().unwrap(), 0);
}

#[tokio::test]
async fn retrieve_endpoints_return_stored_original_and_stats() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let (proxy, _store) = spawn_proxy_with_store(openai, anthropic).await;
    let content = format!("line one\nneedle line {}", "x".repeat(640));

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({"messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": content}]}),
    )
    .await;
    let marker = openai_capture.bodies.lock().unwrap()[0]["messages"][0]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let hash = marker
        .trim_start_matches("<<ccr:")
        .trim_end_matches(">>")
        .to_string();

    let full: Value = reqwest::get(format!("http://{proxy}/v1/retrieve/{hash}"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        full["original_content"],
        format!("line one\nneedle line {}", "x".repeat(640))
    );

    let searched: Value = reqwest::get(format!("http://{proxy}/v1/retrieve/{hash}?query=needle"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(searched["count"], 1);
    assert_eq!(searched["results"][0]["line"], 2);

    let stats: Value = reqwest::get(format!("http://{proxy}/v1/retrieve/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stats["store"]["entries"], 1);
}

#[tokio::test]
async fn retrieve_tool_call_formats_provider_results() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let (proxy, _store) = spawn_proxy_with_store(openai, anthropic).await;
    let content = long_text("tool retrieve me");

    post_json(
        proxy,
        "/v1/chat/completions",
        json!({"messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": content}]}),
    )
    .await;
    let marker = openai_capture.bodies.lock().unwrap()[0]["messages"][0]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let hash = marker.trim_start_matches("<<ccr:").trim_end_matches(">>");

    let response: Value = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/retrieve/tool_call"))
        .json(&json!({
            "provider": "anthropic",
            "tool_call": {
                "id": "toolu_1",
                "name": "headroom_retrieve",
                "input": {"hash": hash}
            }
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(response["success"], true);
    assert_eq!(response["tool_result"]["type"], "tool_result");
    assert_eq!(response["tool_result"]["tool_use_id"], "toolu_1");
    assert!(response["tool_result"]["content"]
        .as_str()
        .unwrap()
        .contains("tool retrieve me"));
}

#[tokio::test]
async fn forwarding_filters_hop_by_hop_and_internal_headers() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/files"))
        .header("connection", "close, x-remove-me")
        .header("x-remove-me", "bad")
        .header("x-headroom-mode", "internal")
        .header("x-request-id", "req-123")
        .json(&json!({"ok": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = openai_capture.headers.lock().unwrap();
    let headers = &headers[0];
    assert!(headers.get("connection").is_none());
    assert!(headers.get("x-remove-me").is_none());
    assert!(headers.get("x-headroom-mode").is_none());
    assert_eq!(headers.get("x-request-id").unwrap(), "req-123");
    assert!(headers
        .get("x-forwarded-for")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("127.0.0.1"));
    assert!(headers.get("x-forwarded-proto").is_some());
    assert!(headers.get("x-forwarded-host").is_some());
}

#[tokio::test]
async fn generated_request_id_is_forwarded_and_returned() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/files"))
        .json(&json!({"ok": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let upstream_headers = openai_capture.headers.lock().unwrap();
    let upstream_request_id = upstream_headers[0]
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(upstream_request_id.starts_with("hr-"));
    assert_eq!(
        response.headers().get("x-request-id").unwrap(),
        upstream_request_id
    );
}

#[tokio::test]
async fn codex_backend_requests_infer_chatgpt_account_id_from_oauth_jwt() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let jwt_payload =
        "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF8xMjMifX0";

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/backend-api/codex/responses"))
        .header("authorization", format!("Bearer fake.{jwt_payload}.sig"))
        .json(&json!({"input": "small codex request"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = openai_capture.headers.lock().unwrap();
    assert_eq!(headers[0].get("ChatGPT-Account-ID").unwrap(), "acct_123");
}

#[tokio::test]
async fn upstream_response_headers_preserve_limits_and_filter_hop_by_hop() {
    let openai = spawn_response_header_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/files"))
        .json(&json!({"ok": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = response.headers();
    assert!(headers.get("connection").is_none());
    assert!(headers.get("x-drop-response").is_none());
    assert_eq!(headers.get("x-ratelimit-remaining-tokens").unwrap(), "123");
    assert_eq!(headers.get("retry-after").unwrap(), "9");
    assert_eq!(
        headers.get("headroom-upstream-request-id").unwrap(),
        "upstream-req-123"
    );
    assert!(headers
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("hr-"));
}

#[tokio::test]
async fn websocket_forwards_request_metadata_without_internal_headers() {
    let ws_capture = Capture::default();
    let openai = spawn_ws_capture_upstream(ws_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let mut request = format!("ws://{proxy}/ws").into_client_request().unwrap();
    request
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_static("req-ws"));
    request
        .headers_mut()
        .insert("x-headroom-mode", HeaderValue::from_static("internal"));
    request
        .headers_mut()
        .insert("x-forwarded-for", HeaderValue::from_static("10.0.0.1"));

    let (mut socket, _) = connect_async(request).await.unwrap();
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            "hello".into(),
        ))
        .await
        .unwrap();

    let message = socket.next().await.unwrap().unwrap();
    assert_eq!(message.into_text().unwrap(), "echo:hello");

    let headers = ws_capture.headers.lock().unwrap();
    let headers = &headers[0];
    assert_eq!(headers.get("x-request-id").unwrap(), "req-ws");
    assert!(headers.get("x-headroom-mode").is_none());
    assert!(headers.get("x-forwarded-proto").is_some());
    assert!(headers.get("x-forwarded-host").is_some());
    assert!(headers
        .get("x-forwarded-for")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("10.0.0.1"));
    assert!(headers
        .get("x-forwarded-for")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("127.0.0.1"));
}

#[tokio::test]
async fn compression_response_headers_are_returned_to_client() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .header("x-request-id", "req-compress")
        .json(&json!({"messages": [{"role": "tool", "tool_call_id": "call_bulk", "content": long_text("return headers")}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-request-id").unwrap(),
        "req-compress"
    );
    assert!(response.headers().get("x-headroom-tokens-before").is_some());
    assert!(response.headers().get("x-headroom-tokens-after").is_some());
    assert!(response.headers().get("x-headroom-tokens-saved").is_some());
    assert_eq!(
        response.headers().get("x-headroom-transforms").unwrap(),
        "ccr_live_zone"
    );
    assert!(response.headers().get("x-headroom-ccr-hashes").is_some());
}

#[tokio::test]
async fn oversized_compression_body_returns_413_before_forwarding() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy_with_limit(openai, anthropic, 16).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "too large"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(openai_capture.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn conversations_api_streams_without_compression() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    post_json(
        proxy,
        "/v1/conversations/conv_123/items",
        json!({"items": [{"role": "user", "content": "conversation body"}]}),
    )
    .await;

    let bodies = openai_capture.bodies.lock().unwrap();
    assert_eq!(
        bodies[0],
        json!({"items": [{"role": "user", "content": "conversation body"}]})
    );
    assert_eq!(
        openai_capture.paths.lock().unwrap()[0],
        "/v1/conversations/conv_123/items"
    );
}

#[tokio::test]
async fn sse_response_is_streamed_without_mutation() {
    let openai = spawn_sse_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/stream"))
        .json(&json!({"stream": true}))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let text = response.text().await.unwrap();
    assert_eq!(text, "data: {\"delta\":\"hello\"}\n\ndata: [DONE]\n\n");
}

#[tokio::test]
async fn sse_response_preserves_split_utf8_bytes() {
    let openai = spawn_split_utf8_sse_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let text = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/stream"))
        .json(&json!({"stream": true}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert_eq!(text, "data: {\"delta\":\"é\"}\n\ndata: [DONE]\n\n");
}

#[tokio::test]
async fn sse_usage_is_recorded_without_mutating_stream_bytes() {
    let before = stats();
    let openai = spawn_sse_usage_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let expected = concat!(
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,",
        "\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
        "data: [DONE]\n\n"
    );

    let text = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "stream usage"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert_eq!(text, expected);
    let after = stats();
    assert!(after.sse_streams > before.sse_streams);
    assert!(after.sse_input_tokens >= before.sse_input_tokens + 10);
    assert!(after.sse_output_tokens >= before.sse_output_tokens + 4);
    assert!(after.sse_cache_read_input_tokens >= before.sse_cache_read_input_tokens + 3);
}

#[tokio::test]
async fn anthropic_confirmed_cache_prefix_freezes_next_request_bytes() {
    let capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_anthropic_cached_usage_upstream(capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let cached_bulk = long_text("cached prefix bulk");
    let live_bulk = long_text("live turn bulk");
    // A large uncompressed assistant turn keeps the per-message token
    // estimates realistic: ~2005 tokens for message 1 vs ~14 for the
    // marker-compressed message 0 and ~1005 for the upstream
    // assistant reply, so 2200 confirmed cached tokens cover exactly
    // the first two messages.
    let assistant_bulk = "y".repeat(7000);

    // Turn 1: the upstream confirms a large cached prefix
    // (cache_read + cache_creation tokens) for this session.
    post_json(
        proxy,
        "/v1/messages",
        json!({
            "model": "claude-test",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tr_1", "content": cached_bulk}
                ]},
                {"role": "assistant", "content": assistant_bulk}
            ]
        }),
    )
    .await;

    // Turn 2 (same session: same model, no system prompt): the
    // confirmed-cached prefix must stay byte-identical; only content
    // beyond the frozen floor compresses.
    let turn_two = json!({
        "model": "claude-test",
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tr_1", "content": cached_bulk}
            ]},
            {"role": "assistant", "content": assistant_bulk},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tr_2", "content": live_bulk}
            ]}
        ]
    });
    post_json(proxy, "/v1/messages", turn_two.clone()).await;

    let bodies = capture.bodies.lock().unwrap();
    // Turn 1 compressed the tool_result (no floor yet).
    assert!(bodies[0]["messages"][0]["content"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    // Turn 2: every byte up to the live-zone span (the whole frozen
    // prefix included) is byte-identical to the client's
    // serialization; only the live tool_result was spliced.
    let raw_turn_two = capture.raw_bodies.lock().unwrap()[1].clone();
    let client_bytes = serde_json::to_vec(&turn_two).unwrap();
    let live_token = serde_json::to_string(&live_bulk).unwrap();
    let client_text = std::str::from_utf8(&client_bytes).unwrap();
    let live_span_start = client_text.find(&live_token).unwrap();
    assert_eq!(
        &raw_turn_two[..live_span_start],
        &client_bytes[..live_span_start],
        "confirmed-cached prefix must stay byte-identical"
    );
    assert!(bodies[1]["messages"][2]["content"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
}

#[tokio::test]
async fn responses_confirmed_cache_prefix_freezes_next_request_items() {
    let capture = Capture::default();
    let openai = spawn_responses_cached_usage_upstream(capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let cached_output = long_text("responses cached prefix");
    let live_output = long_text("responses live output");

    let before = stats().sse_input_tokens;
    // Turn 1: the SSE terminal event confirms a large cached prefix
    // for this (model, no instructions) session.
    let text = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/responses"))
        .json(&json!({
            "model": "gpt-test",
            "input": [
                {"type": "function_call_output", "call_id": "c0", "output": cached_output}
            ]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("response.completed"));
    // The state machine runs off the byte path; wait for it to absorb
    // the terminal usage before issuing turn 2.
    for _ in 0..100 {
        if stats().sse_input_tokens > before {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(stats().sse_input_tokens > before);

    // Turn 2 (same session): the confirmed-cached prefix item must
    // stay untouched; only the new output compresses.
    post_json(
        proxy,
        "/v1/responses",
        json!({
            "model": "gpt-test",
            "input": [
                {"type": "function_call_output", "call_id": "c0", "output": cached_output},
                {"type": "function_call_output", "call_id": "c1", "output": live_output}
            ]
        }),
    )
    .await;

    let bodies = capture.bodies.lock().unwrap();
    assert!(bodies[0]["input"][0]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    assert_eq!(bodies[1]["input"][0]["output"], cached_output);
    assert!(bodies[1]["input"][1]["output"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
}

#[tokio::test]
async fn responses_non_streaming_json_advances_the_session_floor() {
    let capture = Capture::default();
    let openai = spawn_responses_json_cached_usage_upstream(capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;
    let cached_output = long_text("responses json cached prefix");
    let live_output = long_text("responses json live output");

    // Turn 1: a plain JSON (non-SSE) response confirms a large cached
    // prefix; the tracker must advance synchronously on this path.
    post_json(
        proxy,
        "/v1/responses",
        json!({
            "model": "gpt-test",
            "input": [
                {"type": "function_call_output", "call_id": "c0", "output": cached_output}
            ]
        }),
    )
    .await;

    // Turn 2 (same session): the confirmed-cached prefix item stays
    // untouched; only the new output compresses.
    post_json(
        proxy,
        "/v1/responses",
        json!({
            "model": "gpt-test",
            "input": [
                {"type": "function_call_output", "call_id": "c0", "output": cached_output},
                {"type": "function_call_output", "call_id": "c1", "output": live_output}
            ]
        }),
    )
    .await;

    {
        let bodies = capture.bodies.lock().unwrap();
        assert!(bodies[0]["input"][0]["output"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
        assert_eq!(bodies[1]["input"][0]["output"], cached_output);
        assert!(bodies[1]["input"][1]["output"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    // The buffered JSON branch records rate-limit gauges too (the
    // upstream header value is unique to this test).
    let metrics = reqwest::get(format!("http://{proxy}/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics
        .contains("proxy_rate_limit_remaining_input_tokens{provider=\"openai_responses\"} 5150"));
}

#[tokio::test]
async fn anthropic_beta_header_sticks_across_session_requests() {
    let capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{proxy}/v1/messages"))
        .header("anthropic-beta", "interleaved-thinking-2025-05-14")
        .json(&json!({"model": "claude-test", "messages": [{"role": "user", "content": "one"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Later request in the same session without the header still
    // carries the previously-seen beta upstream.
    let response = client
        .post(format!("http://{proxy}/v1/messages"))
        .json(&json!({"model": "claude-test", "messages": [{"role": "user", "content": "two"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = capture.headers.lock().unwrap();
    assert_eq!(
        headers[0].get("anthropic-beta").unwrap(),
        "interleaved-thinking-2025-05-14"
    );
    assert_eq!(
        headers[1].get("anthropic-beta").unwrap(),
        "interleaved-thinking-2025-05-14"
    );
}

#[tokio::test]
async fn openai_beta_header_sticks_across_chat_session_requests() {
    let capture = Capture::default();
    let openai = spawn_upstream(capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{proxy}/v1/chat/completions"))
        .header("openai-beta", "some-chat-beta")
        .json(&json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "one"}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Later request in the same session (same model + system) without
    // the header still carries the previously-seen beta upstream,
    // mirroring `openai.py:1535-1574`.
    let response = client
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "two"}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = capture.headers.lock().unwrap();
    assert_eq!(headers[0].get("openai-beta").unwrap(), "some-chat-beta");
    assert_eq!(headers[1].get("openai-beta").unwrap(), "some-chat-beta");
}

#[tokio::test]
async fn subscription_user_agent_never_receives_cache_mutating_metadata() {
    let openai_capture = Capture::default();
    let openai = spawn_upstream(openai_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .header("user-agent", "antigravity/1.2.3")
        .json(&json!({
            "model": "gpt-test",
            "tools": [
                {"type": "function", "function": {"name": "zebra"}},
                {"type": "function", "function": {"name": "apple"}}
            ],
            "messages": [
                {"role": "tool", "tool_call_id": "call_bulk", "content": long_text("subscription")}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bodies = openai_capture.bodies.lock().unwrap();
    let body = &bodies[0];
    // Live-zone compression still applies on subscription auth...
    assert!(body["messages"][0]["content"]
        .as_str()
        .unwrap()
        .starts_with("<<ccr:"));
    // ...but cache-mutating metadata never does: no synthesized
    // prompt_cache_key, and the tool array order is untouched.
    assert!(body.get("prompt_cache_key").is_none());
    assert_eq!(body["tools"][0]["function"]["name"], "zebra");
    assert_eq!(body["tools"][1]["function"]["name"], "apple");
}

#[tokio::test]
async fn empty_batch_create_is_rejected_with_400() {
    let capture = Capture::default();
    let openai = spawn_upstream(Capture::default()).await;
    let anthropic = spawn_upstream(capture.clone()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let response = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/messages/batches"))
        .json(&json!({"requests": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.unwrap();
    assert_eq!(
        body,
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Missing or empty 'requests' field in batch request"
            }
        })
    );
    assert!(capture.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn codex_responses_websocket_echoes_client_subprotocol() {
    let ws_capture = WsCapture::default();
    let openai = spawn_codex_ws_capture_upstream(ws_capture.clone()).await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let mut request = format!("ws://{proxy}/v1/responses")
        .into_client_request()
        .unwrap();
    request.headers_mut().insert(
        "sec-websocket-protocol",
        HeaderValue::from_static("codex-realtime"),
    );

    let (mut socket, handshake) = connect_async(request).await.unwrap();
    assert_eq!(
        handshake
            .headers()
            .get("sec-websocket-protocol")
            .expect("subprotocol echoed back to the client")
            .to_str()
            .unwrap(),
        "codex-realtime"
    );

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            json!({"type": "session.update"}).to_string().into(),
        ))
        .await
        .unwrap();
    let _ = socket.next().await.unwrap().unwrap();

    // The raw subprotocol header was forwarded upstream end-to-end.
    let headers = ws_capture.headers.lock().unwrap();
    assert_eq!(
        headers[0].get("sec-websocket-protocol").unwrap(),
        "codex-realtime"
    );
}

#[tokio::test]
async fn rate_limit_gauges_service_tier_and_terminal_status_surface_in_metrics() {
    let openai = spawn_responses_sse_with_rate_limit_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let text = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/responses"))
        .json(&json!({"input": [], "stream": true}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("response.completed"));

    // The SSE state machine runs off the byte path; poll briefly for
    // its terminal-event metrics.
    let mut metrics = String::new();
    for _ in 0..50 {
        metrics = reqwest::get(format!("http://{proxy}/metrics"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if metrics.contains("proxy_response_status_count_total{status=\"completed\"}") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        metrics.contains("proxy_rate_limit_remaining_requests{provider=\"openai_responses\"} 42")
    );
    assert!(
        metrics.contains("proxy_rate_limit_remaining_tokens{provider=\"openai_responses\"} 9000")
    );
    assert!(metrics.contains("proxy_service_tier_count_total{tier=\"default\"}"));
    assert!(metrics.contains("proxy_response_status_count_total{status=\"completed\"}"));
    assert!(metrics.contains("hr_sse_inferred_cache_write_tokens"));
}

#[tokio::test]
async fn sse_flood_passes_every_client_byte_even_if_parser_saturates() {
    let (openai, expected) = spawn_sse_flood_upstream().await;
    let anthropic = spawn_upstream(Capture::default()).await;
    let proxy = spawn_proxy(openai, anthropic).await;

    let text = reqwest::Client::new()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "flood"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    // Telemetry chunks may be dropped under parser saturation, but
    // the client byte path must stay loss-free and byte-identical.
    assert_eq!(text, expected);
}

async fn post_json(addr: SocketAddr, path: &str, body: Value) {
    let response = reqwest::Client::new()
        .post(format!("http://{addr}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

fn long_text(label: &str) -> String {
    format!("{label} {}", "0123456789 abcdefghij ".repeat(32))
}

async fn spawn_proxy(openai: SocketAddr, anthropic: SocketAddr) -> SocketAddr {
    spawn_proxy_with_store(openai, anthropic).await.0
}

async fn spawn_proxy_with_limit(
    openai: SocketAddr,
    anthropic: SocketAddr,
    max_body_bytes: usize,
) -> SocketAddr {
    let store = SqliteStore::in_memory().unwrap();
    let state = ProxyState::new(
        Url::parse(&format!("http://{openai}")).unwrap(),
        Url::parse(&format!("http://{anthropic}")).unwrap(),
        store,
    )
    .with_max_body_bytes(max_body_bytes);
    spawn_proxy_state(state).await
}

async fn spawn_proxy_with_compression(
    openai: SocketAddr,
    anthropic: SocketAddr,
    compression_enabled: bool,
) -> SocketAddr {
    let store = SqliteStore::in_memory().unwrap();
    let state = ProxyState::new(
        Url::parse(&format!("http://{openai}")).unwrap(),
        Url::parse(&format!("http://{anthropic}")).unwrap(),
        store,
    )
    .with_compression_enabled(compression_enabled)
    .with_compression_mode(if compression_enabled {
        CompressionMode::Ccr
    } else {
        CompressionMode::Off
    });
    spawn_proxy_state(state).await
}

async fn spawn_proxy_with_store(
    openai: SocketAddr,
    anthropic: SocketAddr,
) -> (SocketAddr, SqliteStore) {
    let store = SqliteStore::in_memory().unwrap();
    let state = ProxyState::new(
        Url::parse(&format!("http://{openai}")).unwrap(),
        Url::parse(&format!("http://{anthropic}")).unwrap(),
        store.clone(),
    );
    let addr = spawn_proxy_state(state).await;
    (addr, store)
}

async fn spawn_proxy_state(state: ProxyState) -> SocketAddr {
    let app = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    addr
}

async fn spawn_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route("/healthz", any(echo_body)).route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request| capture_json(capture.clone(), request)
        }),
    );
    spawn_app(app).await
}

async fn spawn_ws_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        any(|ws: WebSocketUpgrade| async move {
            ws.on_upgrade(|mut socket| async move {
                while let Some(Ok(message)) = socket.next().await {
                    if let Message::Text(text) = message {
                        socket
                            .send(Message::Text(format!("echo:{text}").into()))
                            .await
                            .unwrap();
                    }
                }
            })
        }),
    );
    spawn_app(app).await
}

async fn spawn_ws_capture_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        any({
            let capture = capture.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let capture = capture.clone();
                async move {
                    capture.headers.lock().unwrap().push(headers);
                    ws.on_upgrade(|mut socket| async move {
                        while let Some(Ok(message)) = socket.next().await {
                            if let Message::Text(text) = message {
                                socket
                                    .send(Message::Text(format!("echo:{text}").into()))
                                    .await
                                    .unwrap();
                            }
                        }
                    })
                }
            }
        }),
    );
    spawn_app(app).await
}

async fn spawn_codex_ws_capture_upstream(capture: WsCapture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        any({
            let capture = capture.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let capture = capture.clone();
                async move {
                    capture.headers.lock().unwrap().push(headers);
                    // Accept the subprotocol Codex-style clients offer so
                    // the proxy's upstream handshake succeeds.
                    let ws = ws.protocols(["codex-realtime"]);
                    ws.on_upgrade(move |mut socket| async move {
                        while let Some(Ok(message)) = socket.next().await {
                            if let Message::Text(text) = message {
                                capture.frames.lock().unwrap().push(text.to_string());
                                socket
                                    .send(Message::Text(
                                        json!({
                                            "type": "response.completed",
                                            "response": {
                                                "usage": {
                                                    "input_tokens": 21,
                                                    "output_tokens": 5,
                                                    "input_tokens_details": {"cached_tokens": 7}
                                                }
                                            }
                                        })
                                        .to_string()
                                        .into(),
                                    ))
                                    .await
                                    .unwrap();
                            }
                        }
                    })
                }
            }
        }),
    );
    spawn_app(app).await
}

async fn spawn_codex_http_fallback_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    capture
                        .headers
                        .lock()
                        .unwrap()
                        .push(request.headers().clone());
                    capture
                        .paths
                        .lock()
                        .unwrap()
                        .push(request.uri().path().to_string());
                    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
                    capture.raw_bodies.lock().unwrap().push(body.to_vec());
                    capture
                        .bodies
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&body).unwrap());
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "data: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n",
                        ))
                        .unwrap()
                }
            }
        }),
    );
    spawn_app(app).await
}

async fn spawn_response_header_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header("connection", "close, x-drop-response")
                .header("x-drop-response", "bad")
                .header("x-ratelimit-remaining-tokens", "123")
                .header("retry-after", "9")
                .header("request-id", "upstream-req-123")
                .body(Body::from("{\"ok\":true}"))
                .unwrap()
        }),
    );
    spawn_app(app).await
}

async fn spawn_sse_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(Body::from(
                    "data: {\"delta\":\"hello\"}\n\ndata: [DONE]\n\n",
                ))
                .unwrap()
        }),
    );
    spawn_app(app).await
}

async fn spawn_split_utf8_sse_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            let payload = "data: {\"delta\":\"é\"}\n\ndata: [DONE]\n\n"
                .as_bytes()
                .to_vec();
            let split = payload.iter().position(|byte| *byte == 0xc3).unwrap() + 1;
            let chunks = vec![
                Ok::<Bytes, Infallible>(Bytes::copy_from_slice(&payload[..split])),
                Ok::<Bytes, Infallible>(Bytes::copy_from_slice(&payload[split..])),
            ];
            let stream = futures_util::stream::iter(chunks);
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }),
    );
    spawn_app(app).await
}

async fn spawn_sse_usage_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(Body::from(concat!(
                    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,",
                    "\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
                    "data: [DONE]\n\n"
                )))
                .unwrap()
        }),
    );
    spawn_app(app).await
}

/// Extracts the first `<<ccr:HASH>>` marker hash from a serialized request.
fn marker_hash(value: &Value) -> String {
    let text = value.to_string();
    let start = text.find("<<ccr:").expect("ccr marker present") + "<<ccr:".len();
    let end = text[start..].find(">>").expect("ccr marker terminator") + start;
    text[start..end].to_string()
}

async fn capture_request(capture: &Capture, request: Request<Body>) -> Value {
    let path = request.uri().path().to_string();
    let headers = request.headers().clone();
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&body).unwrap();
    capture.headers.lock().unwrap().push(headers);
    capture.paths.lock().unwrap().push(path);
    capture.raw_bodies.lock().unwrap().push(body.to_vec());
    capture.bodies.lock().unwrap().push(value.clone());
    value
}

/// Anthropic upstream that answers the first `rounds_with_tool_calls`
/// requests with a `headroom_retrieve` tool call (hash taken from the
/// compressed request marker) and afterwards with a final text message.
async fn spawn_anthropic_ccr_upstream(
    capture: Capture,
    rounds_with_tool_calls: usize,
) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    let value = capture_request(&capture, request).await;
                    let call_index = capture.bodies.lock().unwrap().len();
                    if call_index <= rounds_with_tool_calls {
                        let hash = marker_hash(&value);
                        Json(json!({
                            "id": format!("msg_{call_index}"),
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "tool_use",
                                "id": format!("toolu_{call_index}"),
                                "name": "headroom_retrieve",
                                "input": {"hash": hash}
                            }],
                            "stop_reason": "tool_use"
                        }))
                    } else {
                        Json(json!({
                            "id": format!("msg_{call_index}"),
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "final answer"}],
                            "stop_reason": "end_turn"
                        }))
                    }
                }
            }
        }),
    );
    spawn_app(app).await
}

/// Anthropic upstream that returns CCR-only tool calls for the first
/// `headroom_rounds` requests, then a mixed private/client tool-call response.
async fn spawn_anthropic_ccr_then_mixed_upstream(
    capture: Capture,
    headroom_rounds: usize,
) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    let value = capture_request(&capture, request).await;
                    let call_index = capture.bodies.lock().unwrap().len();
                    let hash = marker_hash(&value);
                    if call_index <= headroom_rounds {
                        Json(json!({
                            "id": format!("msg_{call_index}"),
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "tool_use",
                                "id": format!("toolu_{call_index}"),
                                "name": "headroom_retrieve",
                                "input": {"hash": hash}
                            }],
                            "stop_reason": "tool_use"
                        }))
                    } else {
                        Json(json!({
                            "id": "msg_mixed_limit",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_headroom_limit",
                                    "name": "headroom_retrieve",
                                    "input": {"hash": hash}
                                },
                                {
                                    "type": "tool_use",
                                    "id": "toolu_project_limit",
                                    "name": "project_tool",
                                    "input": {"path": "Cargo.toml"}
                                }
                            ],
                            "stop_reason": "tool_use"
                        }))
                    }
                }
            }
        }),
    );
    spawn_app(app).await
}

/// Anthropic upstream that returns a headroom retrieval tool call together
/// with a client-owned tool call. The proxy must not continue this response,
/// because it cannot synthesize the non-headroom tool result.
async fn spawn_anthropic_mixed_tool_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    let value = capture_request(&capture, request).await;
                    let hash = marker_hash(&value);
                    Json(json!({
                        "id": "msg_mixed",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_headroom",
                                "name": "headroom_retrieve",
                                "input": {"hash": hash}
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_project",
                                "name": "project_tool",
                                "input": {"path": "Cargo.toml"}
                            }
                        ],
                        "stop_reason": "tool_use"
                    }))
                }
            }
        }),
    );
    spawn_app(app).await
}

/// OpenAI chat upstream: first request gets a `headroom_retrieve` tool call,
/// subsequent requests get a final text completion.
async fn spawn_openai_ccr_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    let value = capture_request(&capture, request).await;
                    let call_index = capture.bodies.lock().unwrap().len();
                    if call_index == 1 {
                        let hash = marker_hash(&value);
                        let arguments = json!({"hash": hash}).to_string();
                        Json(json!({
                            "id": "chatcmpl_1",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "headroom_retrieve",
                                            "arguments": arguments
                                        }
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }]
                        }))
                    } else {
                        Json(json!({
                            "id": "chatcmpl_2",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": "final answer"},
                                "finish_reason": "stop"
                            }]
                        }))
                    }
                }
            }
        }),
    );
    spawn_app(app).await
}

/// OpenAI chat upstream that returns a single choice mixing a headroom
/// retrieval tool call with a client-owned tool call. Exercises the
/// mixed-tool-call guard inside the continuation loop (not the multi-choice
/// guard).
async fn spawn_openai_single_choice_mixed_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    let value = capture_request(&capture, request).await;
                    let hash = marker_hash(&value);
                    let headroom_arguments = json!({"hash": hash}).to_string();
                    let project_arguments = json!({"path": "Cargo.toml"}).to_string();
                    Json(json!({
                        "id": "chatcmpl_single_mixed",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": null,
                                "tool_calls": [
                                    {
                                        "id": "call_headroom",
                                        "type": "function",
                                        "function": {
                                            "name": "headroom_retrieve",
                                            "arguments": headroom_arguments
                                        }
                                    },
                                    {
                                        "id": "call_project",
                                        "type": "function",
                                        "function": {
                                            "name": "project_tool",
                                            "arguments": project_arguments
                                        }
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls"
                        }]
                    }))
                }
            }
        }),
    );
    spawn_app(app).await
}

/// OpenAI chat upstream that returns a headroom retrieval tool call together
/// with a client-owned tool call. The proxy must not continue this response,
/// and must not expose its private retrieval tool call to the client.
async fn spawn_openai_mixed_tool_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    let value = capture_request(&capture, request).await;
                    let hash = marker_hash(&value);
                    let headroom_arguments = json!({"hash": hash}).to_string();
                    let project_arguments_0 = json!({"path": "Cargo.toml"}).to_string();
                    let project_arguments_1 = json!({"path": "src/proxy.rs"}).to_string();
                    Json(json!({
                        "id": "chatcmpl_mixed",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id": "call_project_0",
                                        "type": "function",
                                        "function": {
                                            "name": "project_tool",
                                            "arguments": project_arguments_0
                                        }
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            },
                            {
                                "index": 1,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [
                                        {
                                            "id": "call_headroom",
                                            "type": "function",
                                            "function": {
                                                "name": "headroom_retrieve",
                                                "arguments": headroom_arguments
                                            }
                                        },
                                        {
                                            "id": "call_project_1",
                                            "type": "function",
                                            "function": {
                                                "name": "project_tool",
                                                "arguments": project_arguments_1
                                            }
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls"
                            }
                        ]
                    }))
                }
            }
        }),
    );
    spawn_app(app).await
}

/// Anthropic upstream covering batch create, batch results (a single result
/// holding a `headroom_retrieve` tool call), and `/v1/messages` continuation.
async fn spawn_anthropic_batch_ccr_upstream(capture: Capture) -> SocketAddr {
    let create_capture = capture.clone();
    let results_capture = capture.clone();
    let messages_capture = capture.clone();
    let app = Router::new()
        .route(
            "/v1/messages/batches",
            post(move |request: Request<Body>| {
                let capture = create_capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    Json(json!({
                        "id": "batch_test",
                        "type": "message_batch",
                        "processing_status": "in_progress"
                    }))
                }
            }),
        )
        .route(
            "/v1/messages/batches/{id}/results",
            axum::routing::get(move || {
                let capture = results_capture.clone();
                async move {
                    let hash = capture
                        .bodies
                        .lock()
                        .unwrap()
                        .first()
                        .map(marker_hash)
                        .unwrap_or_else(|| "ab".repeat(12));
                    let line = json!({
                        "custom_id": "req-1",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "id": "msg_b1",
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "tool_use",
                                    "id": "toolu_b1",
                                    "name": "headroom_retrieve",
                                    "input": {"hash": hash}
                                }],
                                "stop_reason": "tool_use"
                            }
                        }
                    });
                    (
                        [("content-type", "application/x-jsonl")],
                        format!("{line}\n"),
                    )
                }
            }),
        )
        .route(
            "/v1/messages",
            post(move |request: Request<Body>| {
                let capture = messages_capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    Json(json!({
                        "id": "msg_final",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "final answer"}],
                        "stop_reason": "end_turn"
                    }))
                }
            }),
        );
    spawn_app(app).await
}

/// Anthropic batch upstream whose result line contains both the private
/// retrieval tool call and a client-owned tool call.
async fn spawn_anthropic_batch_mixed_tool_upstream(capture: Capture) -> SocketAddr {
    let create_capture = capture.clone();
    let results_capture = capture.clone();
    let messages_capture = capture.clone();
    let app = Router::new()
        .route(
            "/v1/messages/batches",
            post(move |request: Request<Body>| {
                let capture = create_capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    Json(json!({
                        "id": "batch_test",
                        "type": "message_batch",
                        "processing_status": "in_progress"
                    }))
                }
            }),
        )
        .route(
            "/v1/messages/batches/{id}/results",
            axum::routing::get(move |request: Request<Body>| {
                let capture = results_capture.clone();
                async move {
                    capture
                        .headers
                        .lock()
                        .unwrap()
                        .push(request.headers().clone());
                    capture
                        .paths
                        .lock()
                        .unwrap()
                        .push(request.uri().path().to_string());
                    capture.raw_bodies.lock().unwrap().push(Vec::new());

                    let hash = capture
                        .bodies
                        .lock()
                        .unwrap()
                        .first()
                        .map(marker_hash)
                        .unwrap_or_else(|| "ab".repeat(12));
                    let line = json!({
                        "custom_id": "req-1",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "id": "msg_b1",
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": "toolu_headroom_b1",
                                        "name": "headroom_retrieve",
                                        "input": {"hash": hash}
                                    },
                                    {
                                        "type": "tool_use",
                                        "id": "toolu_project_b1",
                                        "name": "project_tool",
                                        "input": {"path": "Cargo.toml"}
                                    }
                                ],
                                "stop_reason": "tool_use"
                            }
                        }
                    });
                    (
                        [("content-type", "application/x-jsonl")],
                        format!("{line}\n"),
                    )
                }
            }),
        )
        .route(
            "/v1/messages",
            post(move |request: Request<Body>| {
                let capture = messages_capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    Json(json!({
                        "id": "msg_final",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "unexpected continuation"}],
                        "stop_reason": "end_turn"
                    }))
                }
            }),
        );
    spawn_app(app).await
}

/// SSE upstream emitting an Anthropic stream that contains a
/// `headroom_retrieve` tool_use block.
async fn spawn_sse_ccr_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            let body = concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n",
                "\n",
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_s1\",\"name\":\"headroom_retrieve\",\"input\":{}}}\n",
                "\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n",
                "\n",
            );
            ([("content-type", "text/event-stream")], body)
        }),
    );
    spawn_app(app).await
}

/// Anthropic upstream whose responses confirm a large cached prefix.
async fn spawn_anthropic_cached_usage_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    Json(json!({
                        "id": "msg_cached",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "z".repeat(3500)}],
                        "stop_reason": "end_turn",
                        "usage": {
                            "input_tokens": 50,
                            "output_tokens": 4,
                            "cache_read_input_tokens": 1500,
                            "cache_creation_input_tokens": 700
                        }
                    }))
                }
            }
        }),
    );
    spawn_app(app).await
}

/// Responses SSE upstream emitting a terminal `response.completed`
/// event with a service tier, plus rate-limit response headers.
async fn spawn_responses_sse_with_rate_limit_upstream() -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            let body = concat!(
                "event: response.in_progress\n",
                "data: {\"type\":\"response.in_progress\",\"response\":{}}\n",
                "\n",
                "event: response.completed\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"service_tier\":\"default\",",
                "\"usage\":{\"input_tokens\":30,\"output_tokens\":6,",
                "\"input_tokens_details\":{\"cached_tokens\":12}}}}\n",
                "\n",
                "data: [DONE]\n\n",
            );
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("x-ratelimit-remaining-requests", "42")
                .header("x-ratelimit-remaining-tokens", "9000")
                .body(Body::from(body))
                .unwrap()
        }),
    );
    spawn_app(app).await
}

/// SSE upstream that floods the connection with far more chunks than
/// the parser queue depth (256), exercising the try_send drop path.
async fn spawn_sse_flood_upstream() -> (SocketAddr, String) {
    let mut expected = String::new();
    for index in 0..2000 {
        expected.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"chunk {index} 🦀\"}}}}],\"usage\":null}}\n\n"
        ));
    }
    expected.push_str("data: [DONE]\n\n");

    let body = expected.clone();
    let app = Router::new().route(
        "/{*path}",
        post(move || {
            let body = body.clone();
            async move {
                let chunks: Vec<Result<Bytes, Infallible>> = body
                    .as_bytes()
                    .chunks(64)
                    .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                    .collect();
                let stream = futures_util::stream::iter(chunks);
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    (spawn_app(app).await, expected)
}

/// Responses SSE upstream that captures request bodies and confirms a
/// large cached prefix on the terminal event.
async fn spawn_responses_cached_usage_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    let body = concat!(
                        "event: response.completed\n",
                        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":",
                        "{\"input_tokens\":3000,\"output_tokens\":5,",
                        "\"input_tokens_details\":{\"cached_tokens\":2900}}}}\n",
                        "\n",
                        "data: [DONE]\n\n",
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(Body::from(body))
                        .unwrap()
                }
            }
        }),
    );
    spawn_app(app).await
}

/// Responses upstream returning a buffered JSON (non-SSE) response
/// that confirms a large cached prefix.
async fn spawn_responses_json_cached_usage_upstream(capture: Capture) -> SocketAddr {
    let app = Router::new().route(
        "/{*path}",
        post({
            let capture = capture.clone();
            move |request: Request<Body>| {
                let capture = capture.clone();
                async move {
                    capture_request(&capture, request).await;
                    let body = json!({
                        "id": "resp_cached",
                        "object": "response",
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 3000,
                            "output_tokens": 5,
                            "input_tokens_details": {"cached_tokens": 2900}
                        }
                    });
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        // A header only this test emits for the
                        // openai_responses provider, so the gauge
                        // assertion cannot race other tests.
                        .header("anthropic-ratelimit-input-tokens-remaining", "5150")
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap()
                }
            }
        }),
    );
    spawn_app(app).await
}

async fn spawn_app(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

async fn echo_body(request: Request<Body>) -> impl IntoResponse {
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    (StatusCode::OK, body)
}

async fn capture_json(capture: Capture, request: Request<Body>) -> impl IntoResponse {
    capture
        .headers
        .lock()
        .unwrap()
        .push(request.headers().clone());
    capture
        .paths
        .lock()
        .unwrap()
        .push(request.uri().path().to_string());
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    capture.raw_bodies.lock().unwrap().push(body.to_vec());
    let value: Value = serde_json::from_slice(&body).unwrap();
    capture.bodies.lock().unwrap().push(value);
    Json(json!({"ok": true}))
}

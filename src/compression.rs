use crate::ccr::{content_hash, marker_for_hash, CcrStore};
use crate::stats;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{debug, trace};

const LIVE_ZONE_MIN_BYTES: usize = 512;

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

struct MutationState<'a> {
    store: &'a dyn CcrStore,
    min_bytes: usize,
    hashes: Vec<String>,
    compressed_blocks: usize,
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
    let bytes_before = body.len();
    let tokens_before = estimate_tokens_bytes(body);

    let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
        return RequestCompression {
            body: body.to_vec(),
            compressed: false,
            skipped_reason: Some("invalid_json".to_string()),
            hashes: Vec::new(),
            bytes_before,
            bytes_after: bytes_before,
            tokens_before,
            tokens_after: tokens_before,
        };
    };

    let mut state = MutationState {
        store,
        min_bytes: LIVE_ZONE_MIN_BYTES,
        hashes: Vec::new(),
        compressed_blocks: 0,
    };
    let marker_was_present = value_contains_ccr_marker(&value);
    let metadata_mutated = normalize_provider_metadata(&mut value, shape, auth_mode);

    let reason = match shape {
        ApiShape::OpenAiChatCompletions => mutate_openai_chat(&mut value, &mut state),
        ApiShape::OpenAiResponses => mutate_openai_responses(&mut value, &mut state),
        ApiShape::AnthropicMessages => mutate_anthropic_messages(&mut value, &mut state),
        ApiShape::AnthropicMessageBatches => mutate_anthropic_batches(&mut value, &mut state),
    };
    let prompt_cache_key_injected = if auth_mode == RequestAuthMode::Payg
        && matches!(
            shape,
            ApiShape::OpenAiChatCompletions | ApiShape::OpenAiResponses
        ) {
        inject_openai_prompt_cache_key(&mut value, shape)
    } else {
        false
    };
    let retrieve_tool_injected = if state.compressed_blocks > 0 || marker_was_present {
        inject_retrieve_tool(&mut value, shape)
    } else {
        false
    };

    if state.compressed_blocks == 0
        && !metadata_mutated
        && !retrieve_tool_injected
        && !prompt_cache_key_injected
    {
        return RequestCompression {
            body: body.to_vec(),
            compressed: false,
            skipped_reason: Some(reason.unwrap_or("no_live_zone_content").to_string()),
            hashes: state.hashes,
            bytes_before,
            bytes_after: bytes_before,
            tokens_before,
            tokens_after: tokens_before,
        };
    }

    match serde_json::to_vec(&value) {
        Ok(mutated) => {
            let bytes_after = mutated.len();
            let tokens_after = estimate_tokens_bytes(&mutated);
            RequestCompression {
                body: mutated,
                compressed: state.compressed_blocks > 0,
                skipped_reason: (state.compressed_blocks == 0).then_some(
                    if metadata_mutated || retrieve_tool_injected || prompt_cache_key_injected {
                        "metadata_injected_only".to_string()
                    } else {
                        "retrieve_tool_injected_only".to_string()
                    },
                ),
                hashes: state.hashes,
                bytes_before,
                bytes_after,
                tokens_before,
                tokens_after,
            }
        }
        Err(err) => RequestCompression {
            body: body.to_vec(),
            compressed: false,
            skipped_reason: Some(format!("serialize_error:{err}")),
            hashes: state.hashes,
            bytes_before,
            bytes_after: bytes_before,
            tokens_before,
            tokens_after: tokens_before,
        },
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

fn mutate_openai_chat<'a>(
    value: &mut Value,
    state: &mut MutationState<'a>,
) -> Option<&'static str> {
    if value
        .get("n")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        return Some("multi_choice_n");
    }

    let Some(messages) = value.get_mut("messages").and_then(Value::as_array_mut) else {
        return Some("missing_messages");
    };

    let retrieve_tool_call_ids = headroom_retrieve_tool_call_ids(messages);
    let latest_user = messages.iter().rposition(|message| {
        message
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role == "user")
    });
    let latest_tool = messages.iter().rposition(|message| {
        if message
            .get("role")
            .and_then(Value::as_str)
            .is_none_or(|role| role != "tool")
        {
            return false;
        }
        let name_is_retrieve = message
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(is_retrieve_tool_name);
        let call_is_retrieve = message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .is_some_and(|id| retrieve_tool_call_ids.iter().any(|known| known == id));
        !(name_is_retrieve || call_is_retrieve)
    });

    if latest_user.is_none() && latest_tool.is_none() {
        return Some("no_live_zone_message");
    }

    if let Some(index) = latest_tool {
        let _ = mutate_message_content(&mut messages[index], state, TextShape::OpenAi);
    }
    if let Some(index) = latest_user {
        let _ = mutate_message_content(&mut messages[index], state, TextShape::OpenAi);
    }
    compressed_or_reason(state, "no_live_zone_text")
}

fn mutate_openai_responses<'a>(
    value: &mut Value,
    state: &mut MutationState<'a>,
) -> Option<&'static str> {
    let Some(input) = value.get_mut("input") else {
        return Some("missing_input");
    };

    match input {
        Value::String(_) => {
            compress_string_value(input, state, "responses.input");
            compressed_or_reason(state, "no_live_zone_text")
        }
        Value::Array(items) => {
            let retrieve_call_ids = responses_retrieve_call_ids(items);

            for item in items.iter_mut() {
                let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
                if is_responses_output_item(kind) {
                    let call_id_is_retrieve = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| retrieve_call_ids.iter().any(|known| known == id));
                    if call_id_is_retrieve {
                        continue;
                    }
                    if let Some(output) = item.get_mut("output") {
                        compress_string_value(output, state, "responses.output");
                    }
                    if let Some(content) = item.get_mut("content") {
                        mutate_content_value(content, state, TextShape::Responses);
                    }
                }
            }

            if state.compressed_blocks == 0 {
                if let Some(item) = items.iter_mut().rev().find(|item| {
                    item.get("role")
                        .and_then(Value::as_str)
                        .is_some_and(|role| role == "user" || role == "tool")
                }) {
                    if let Some(content) = item.get_mut("content") {
                        mutate_content_value(content, state, TextShape::Responses);
                    }
                }
            }

            compressed_or_reason(state, "no_live_zone_text")
        }
        _ => Some("unsupported_input"),
    }
}

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

fn normalize_provider_metadata(
    value: &mut Value,
    shape: ApiShape,
    auth_mode: RequestAuthMode,
) -> bool {
    if auth_mode != RequestAuthMode::Payg {
        return false;
    }

    match shape {
        ApiShape::OpenAiChatCompletions | ApiShape::OpenAiResponses => {
            normalize_tool_array(value, ToolShape::OpenAi)
        }
        ApiShape::AnthropicMessages => normalize_tool_array(value, ToolShape::Anthropic),
        ApiShape::AnthropicMessageBatches => value
            .get_mut("requests")
            .and_then(Value::as_array_mut)
            .map(|requests| {
                let mut changed = false;
                for params in requests
                    .iter_mut()
                    .filter_map(|request| request.get_mut("params"))
                {
                    changed |= normalize_tool_array(params, ToolShape::Anthropic);
                }
                changed
            })
            .unwrap_or(false),
    }
}

#[derive(Debug, Clone, Copy)]
enum ToolShape {
    OpenAi,
    Anthropic,
}

fn normalize_tool_array(value: &mut Value, shape: ToolShape) -> bool {
    let preserve_tool_order = any_tool_has_cache_control(value);
    let Some(tools) = value.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let before = Value::Array(tools.clone());

    for tool in tools.iter_mut() {
        sort_tool_schema_keys(tool, shape);
    }
    if !preserve_tool_order {
        tools.sort_by(|left, right| {
            tool_sort_name(left, shape)
                .cmp(tool_sort_name(right, shape))
                .then_with(|| left.to_string().cmp(&right.to_string()))
        });
    }

    let mut changed = Value::Array(tools.clone()) != before;
    if matches!(shape, ToolShape::Anthropic) {
        changed |= auto_place_anthropic_cache_control(value);
    }
    changed
}

fn sort_tool_schema_keys(tool: &mut Value, shape: ToolShape) {
    match shape {
        ToolShape::OpenAi => {
            if let Some(schema) = tool
                .get_mut("function")
                .and_then(|function| function.get_mut("parameters"))
            {
                sort_object_keys_recursive(schema);
            } else if let Some(schema) = tool.get_mut("parameters") {
                sort_object_keys_recursive(schema);
            }
        }
        ToolShape::Anthropic => {
            if let Some(schema) = tool.get_mut("input_schema") {
                sort_object_keys_recursive(schema);
            }
        }
    }
}

fn sort_object_keys_recursive(value: &mut Value) {
    match value {
        Value::Array(items) => {
            for item in items {
                sort_object_keys_recursive(item);
            }
        }
        Value::Object(map) => {
            let mut entries = std::mem::take(map).into_iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, mut child) in entries {
                sort_object_keys_recursive(&mut child);
                map.insert(key, child);
            }
        }
        _ => {}
    }
}

fn tool_sort_name(value: &Value, shape: ToolShape) -> &str {
    match shape {
        ToolShape::OpenAi => value
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| value.get("name").and_then(Value::as_str))
            .unwrap_or(""),
        ToolShape::Anthropic => value.get("name").and_then(Value::as_str).unwrap_or(""),
    }
}

fn any_tool_has_cache_control(value: &Value) -> bool {
    value
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(|tool| tool.get("cache_control").is_some()))
}

fn inject_retrieve_tool(value: &mut Value, shape: ApiShape) -> bool {
    match shape {
        ApiShape::OpenAiChatCompletions => {
            inject_tool_array(value, openai_chat_retrieve_tool(), |tool| {
                tool.get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(is_retrieve_tool_name)
            })
        }
        ApiShape::OpenAiResponses => {
            inject_tool_array(value, openai_responses_retrieve_tool(), |tool| {
                tool.get("name")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        tool.get("function")
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                    })
                    .is_some_and(is_retrieve_tool_name)
            })
        }
        ApiShape::AnthropicMessages => {
            inject_tool_array(value, anthropic_retrieve_tool(), |tool| {
                tool.get("name")
                    .and_then(Value::as_str)
                    .is_some_and(is_retrieve_tool_name)
            })
        }
        ApiShape::AnthropicMessageBatches => {
            let mut changed = false;
            if let Some(requests) = value.get_mut("requests").and_then(Value::as_array_mut) {
                for request in requests {
                    let Some(params) = request.get_mut("params") else {
                        continue;
                    };
                    let inserted = inject_tool_array(params, anthropic_retrieve_tool(), |tool| {
                        tool.get("name")
                            .and_then(Value::as_str)
                            .is_some_and(is_retrieve_tool_name)
                    });
                    changed |= inserted;
                }
            }
            changed
        }
    }
}

fn inject_openai_prompt_cache_key(value: &mut Value, shape: ApiShape) -> bool {
    let key = derive_openai_prompt_cache_key(value, shape);
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if object
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .is_some_and(|key| !key.is_empty())
    {
        return false;
    }

    object.insert("prompt_cache_key".to_string(), Value::String(key));
    true
}

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

fn openai_system_value(value: &Value, shape: ApiShape) -> Value {
    if shape == ApiShape::OpenAiResponses {
        if let Some(instructions) = value.get("instructions") {
            return instructions.clone();
        }
    }

    let messages = match shape {
        ApiShape::OpenAiChatCompletions => value.get("messages"),
        ApiShape::OpenAiResponses => value.get("input").or_else(|| value.get("messages")),
        _ => None,
    };

    messages
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find(|item| {
                item.get("role")
                    .and_then(Value::as_str)
                    .is_some_and(|role| role == "system")
            })
        })
        .and_then(|item| item.get("content"))
        .cloned()
        .unwrap_or(Value::Null)
}

fn canonical_json_sha256(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"null".to_vec());
    let digest = Sha256::digest(bytes);
    let key = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    key
}

fn inject_tool_array(value: &mut Value, tool: Value, exists: impl Fn(&Value) -> bool) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };

    match object.get_mut("tools") {
        Some(Value::Array(tools)) => {
            if tools.iter().any(exists) {
                false
            } else {
                tools.push(tool);
                true
            }
        }
        Some(_) => false,
        None => {
            object.insert("tools".to_string(), Value::Array(vec![tool]));
            true
        }
    }
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

fn auto_place_anthropic_cache_control(value: &mut Value) -> bool {
    if anthropic_cache_control_already_set(value) {
        return false;
    }

    let Some(tools) = value.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let Some(tool) = tools.last_mut().and_then(Value::as_object_mut) else {
        return false;
    };
    tool.insert(
        "cache_control".to_string(),
        json!({
            "type": "ephemeral"
        }),
    );
    true
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

fn mutate_anthropic_messages<'a>(
    value: &mut Value,
    state: &mut MutationState<'a>,
) -> Option<&'static str> {
    let Some(messages) = value.get_mut("messages").and_then(Value::as_array_mut) else {
        return Some("missing_messages");
    };

    let frozen_count = anthropic_frozen_message_count(messages);
    let Some(index) = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(index, message)| {
            *index >= frozen_count
                && message
                    .get("role")
                    .and_then(Value::as_str)
                    .is_some_and(|role| role == "user")
        })
        .map(|(index, _)| index)
    else {
        return Some("no_live_zone_message");
    };

    mutate_message_content(&mut messages[index], state, TextShape::Anthropic)
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

fn mutate_anthropic_batches<'a>(
    value: &mut Value,
    state: &mut MutationState<'a>,
) -> Option<&'static str> {
    let Some(requests) = value.get_mut("requests").and_then(Value::as_array_mut) else {
        return Some("missing_requests");
    };

    let before = state.compressed_blocks;
    for request in requests {
        if let Some(params) = request.get_mut("params") {
            let _ = mutate_anthropic_messages(params, state);
        }
    }

    if state.compressed_blocks == before {
        Some("no_live_zone_text")
    } else {
        None
    }
}

fn mutate_message_content<'a>(
    message: &mut Value,
    state: &mut MutationState<'a>,
    shape: TextShape,
) -> Option<&'static str> {
    let Some(content) = message.get_mut("content") else {
        return Some("missing_content");
    };

    mutate_content_value(content, state, shape);
    compressed_or_reason(state, "no_live_zone_text")
}

fn compressed_or_reason<'a>(
    state: &MutationState<'a>,
    reason: &'static str,
) -> Option<&'static str> {
    if state.compressed_blocks == 0 {
        Some(reason)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
enum TextShape {
    OpenAi,
    Responses,
    Anthropic,
}

fn mutate_content_value<'a>(content: &mut Value, state: &mut MutationState<'a>, shape: TextShape) {
    match content {
        Value::String(_) => compress_string_value(content, state, "content.string"),
        Value::Array(blocks) => {
            for (index, block) in blocks.iter_mut().enumerate() {
                mutate_content_block(block, state, shape, index);
            }
        }
        _ => debug!(kind = ?shape, "skipped unsupported content shape"),
    }
}

fn mutate_content_block<'a>(
    block: &mut Value,
    state: &mut MutationState<'a>,
    shape: TextShape,
    index: usize,
) {
    if block.get("cache_control").is_some() {
        debug!(index, "skipped cache_control block");
        return;
    }

    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
    trace!(
        index,
        block_type,
        snippet = %snippet(&block.to_string()),
        "evaluating content block"
    );

    match shape {
        TextShape::OpenAi => {
            if block_type == "text" || block_type == "input_text" {
                if let Some(text) = block.get_mut("text") {
                    compress_string_value(text, state, "openai.text");
                }
            }
        }
        TextShape::Responses => match block_type {
            "input_text" | "text" => {
                if let Some(text) = block.get_mut("text") {
                    compress_string_value(text, state, "responses.text");
                }
            }
            "function_call_output" | "local_shell_call_output" | "apply_patch_call_output" => {
                if let Some(output) = block.get_mut("output") {
                    compress_string_value(output, state, "responses.output_block");
                }
            }
            _ => debug!(block_type, "skipped non-live responses block"),
        },
        TextShape::Anthropic => match block_type {
            "text" => {
                if let Some(text) = block.get_mut("text") {
                    compress_string_value(text, state, "anthropic.text");
                }
            }
            "tool_result" => {
                if let Some(tool_content) = block.get_mut("content") {
                    mutate_content_value(tool_content, state, TextShape::Anthropic);
                }
            }
            _ => debug!(block_type, "skipped non-live anthropic block"),
        },
    }
}

fn compress_string_value<'a>(
    value: &mut Value,
    state: &mut MutationState<'a>,
    label: &'static str,
) {
    let Some(text) = value.as_str() else {
        debug!(label, "skipped non-string value");
        return;
    };

    if text.contains("<<ccr:") {
        debug!(label, "skipped existing ccr marker");
        return;
    }

    let result = compress(
        text,
        CompressOptions {
            store: state.store,
            min_bytes: state.min_bytes,
        },
    );

    if result.error.is_some() || result.skipped_reason.is_some() {
        debug!(
            label,
            skipped_reason = ?result.skipped_reason,
            error = ?result.error,
            "compression skipped"
        );
        return;
    }

    let Some(hash) = result.hash.clone() else {
        return;
    };

    debug!(
        label,
        hash,
        original_bytes = result.original_bytes,
        compressed_bytes = result.compressed_bytes,
        original_tokens = result.original_tokens,
        compressed_tokens = result.compressed_tokens,
        "compressed live-zone block"
    );
    trace!(label, snippet = %snippet(text), "compressed snippet");

    *value = Value::String(result.output);
    state.hashes.push(hash);
    state.compressed_blocks += 1;
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

    #[test]
    fn compress_emits_ccr_marker_and_stores_original() {
        crate::stats::reset_for_tests();
        let store = SqliteStore::in_memory().unwrap();

        let result = compress("large live content", CompressOptions::new(&store));

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
    fn openai_chat_mutates_only_latest_user_message() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "messages": [
                {"role": "system", "content": "do not touch"},
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "assistant"},
                {"role": "user", "content": long_text("latest user")}
            ],
            "tools": [{"type": "function", "function": {"name": "safe"}}]
        }))
        .unwrap();

        let result = compress_json_request(&body, ApiShape::OpenAiChatCompletions, &store);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert_eq!(value["messages"][0]["content"], "do not touch");
        assert_eq!(value["messages"][1]["content"], "old user");
        assert_eq!(value["tools"][0]["function"]["name"], "safe");
        assert!(value["messages"][3]["content"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn anthropic_messages_skip_system_and_thinking_blocks() {
        let store = SqliteStore::in_memory().unwrap();
        let body = serde_json::to_vec(&json!({
            "system": "do not touch",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "old"}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "secret", "signature": "sig"}]},
                {"role": "user", "content": [
                    {"type": "thinking", "thinking": "do not touch", "signature": "sig"},
                    {"type": "text", "text": long_text("latest")}
                ]}
            ],
            "tools": [{"name": "tool"}]
        }))
        .unwrap();

        let result = compress_json_request(&body, ApiShape::AnthropicMessages, &store);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert_eq!(value["system"], "do not touch");
        assert_eq!(
            value["messages"][2]["content"][0]["thinking"],
            "do not touch"
        );
        assert!(value["messages"][2]["content"][1]["text"]
            .as_str()
            .unwrap()
            .starts_with("<<ccr:"));
    }

    #[test]
    fn responses_mutates_string_input() {
        let store = SqliteStore::in_memory().unwrap();
        let body =
            serde_json::to_vec(&json!({"input": long_text("latest responses input")})).unwrap();

        let result = compress_json_request(&body, ApiShape::OpenAiResponses, &store);
        let value: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.compressed);
        assert!(value["input"].as_str().unwrap().starts_with("<<ccr:"));
    }

    fn long_text(label: &str) -> String {
        format!("{label} {}", "0123456789 abcdefghij ".repeat(32))
    }

    impl<'a> CompressOptions<'a> {
        fn new(store: &'a dyn CcrStore) -> Self {
            Self {
                store,
                min_bytes: 1,
            }
        }
    }
}

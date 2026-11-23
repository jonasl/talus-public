//! Integration tests for `/v1/chat/completions` `max_tokens` / `max_completion_tokens` defaults,
//! context-budget boundary math (`CTX_SLACK = 8`), and pre-stream validation.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use talus_server::config::ServerConfig;
use talus_server::engine::{ScriptedMockEngine, StdioSubprocessEngine, CTX_SLACK};
use talus_server::json::JsonValue;
use talus_server::server::{ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn post_raw(port: u16, path: &str, body: &str) -> (u16, String, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect to test server");
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    s.flush().unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let (head, payload) = resp.split_once("\r\n\r\n").unwrap_or((&resp, ""));
    let code = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .unwrap_or(0);
    (code, head.to_string(), payload.to_string())
}

fn parse_sse_data(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    for block in raw.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut lines = Vec::new();
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                lines.push(rest.trim().to_string());
            }
        }
        if !lines.is_empty() {
            out.push(lines.join("\n"));
        }
    }
    out
}

fn start_mock(
    max_context: usize,
    fit_max_tokens: bool,
    scripts: Vec<String>,
) -> (talus_server::server::ServerHandle, Arc<ScriptedMockEngine>) {
    let tok = ByteTokenizer::new();
    let engine = Arc::new(ScriptedMockEngine::with_tokenizer(
        tok.clone(),
        scripts,
        max_context,
    ));
    let cfg = ServerConfig {
        port: 0,
        api_key: None,
        model_name: "qwen3.8-27b".to_string(),
        max_context,
        fit_max_tokens,
        ..Default::default()
    };
    let state = Arc::new(ServiceState::new(cfg, engine.clone(), tok));
    let srv = TalusServer::bind(state).unwrap().start_background();
    (srv, engine)
}

#[test]
fn streaming_chat_omitted_max_tokens_and_tools_respects_engine_slack() {
    let ctx = 4096usize;
    let (srv, engine) = start_mock(
        ctx,
        false,
        vec![
            "</think>\n\nHello without max_tokens!".to_string(),
            "</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>".to_string(),
        ],
    );
    let port = srv.port();

    // 1a. Streaming chat with max_tokens omitted
    let req1 =
        r#"{"model":"qwen3.8-27b","stream":true,"messages":[{"role":"user","content":"Say hi"}]}"#;
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", req1);
    assert_eq!(code, 200, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let events = parse_sse_data(&payload);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    assert!(
        !payload.contains("\"server_error\"") && !payload.contains("exceeds the context"),
        "unexpected mid-stream error: {payload}"
    );
    let last_chunk = JsonValue::parse(&events[events.len() - 2]).unwrap();
    let pt1 = last_chunk
        .get("usage")
        .unwrap()
        .get_usize("prompt_tokens")
        .unwrap();
    assert_eq!(pt1 + engine.last_max_new() + CTX_SLACK, ctx);

    // 1b. Streaming chat with tools defined and max_tokens omitted (and null max_completion_tokens)
    let req2 = r#"{"model":"qwen3.8-27b","stream":true,"max_completion_tokens":null,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"messages":[{"role":"user","content":"What is the weather in Paris?"}]}"#;
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", req2);
    assert_eq!(code, 200, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let events2 = parse_sse_data(&payload);
    assert_eq!(events2.last().map(String::as_str), Some("[DONE]"));
    assert!(
        !payload.contains("\"server_error\"") && !payload.contains("exceeds the context"),
        "unexpected mid-stream error: {payload}"
    );
    let last_chunk2 = JsonValue::parse(&events2[events2.len() - 2]).unwrap();
    let pt2 = last_chunk2
        .get("usage")
        .unwrap()
        .get_usize("prompt_tokens")
        .unwrap();
    assert!(pt2 > pt1, "tools should increase prompt token count");
    assert_eq!(pt2 + engine.last_max_new() + CTX_SLACK, ctx);
    assert_eq!(
        last_chunk2.get("choices").unwrap().as_array().unwrap()[0].get_str("finish_reason"),
        Some("tool_calls")
    );
}

#[test]
fn streaming_chat_omitted_max_tokens_with_stdio_engine_262144_context() {
    // Reproduces the reported 262144-context stdio engine check `n + max_new + 8 > max_context`
    let py_engine = r#"
import sys
sys.stdout.write("READY 262144\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if line == "QUIT":
        break
    if line.startswith("GEN "):
        parts = line.split()
        max_new = int(parts[1])
        ids = [int(x) for x in parts[-1].split(",") if x]
        n = len(ids)
        if n + max_new + 8 > 262144:
            sys.stdout.write(f"ERR prompt ({n} tokens) + max_new ({max_new}) exceeds the context (262144)\n")
        else:
            for b in [60, 47, 116, 104, 105, 110, 107, 62, 79, 75]:
                sys.stdout.write(f"T {b}\n")
            sys.stdout.write(f"DONE 10 {n} 1.0 2.0 stop 0 0 0\n")
        sys.stdout.flush()
"#;
    let engine = Arc::new(
        StdioSubprocessEngine::spawn(
            "python3",
            &["-u".to_string(), "-c".to_string(), py_engine.to_string()],
            &[],
            262144,
        )
        .expect("spawn stdio engine"),
    );
    let cfg = ServerConfig {
        port: 0,
        api_key: None,
        model_name: "qwen3.8-flash-next".to_string(),
        max_context: 262144,
        ..Default::default()
    };
    let state = Arc::new(ServiceState::new(cfg, engine, ByteTokenizer::new()));
    let srv = TalusServer::bind(state).unwrap().start_background();

    let req = r#"{"model":"qwen3.8-flash-next","stream":true,"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"messages":[{"role":"user","content":"Hello"}]}"#;
    let (code, head, payload) = post_raw(srv.port(), "/v1/chat/completions", req);
    assert_eq!(code, 200, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    assert!(
        !payload.contains("exceeds the context") && !payload.contains("\"server_error\""),
        "mid-stream error: {payload}"
    );
    let events = parse_sse_data(&payload);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
}

#[test]
fn boundary_conditions_exact_room_and_prompt_plus_max_tokens_equal_context() {
    let ctx = 512usize;
    let (srv, engine) = start_mock(ctx, false, vec!["</think>\n\nOK".to_string(); 4]);
    let port = srv.port();

    // First discover the exact prompt_tokens for a known message
    let probe = r#"{"model":"qwen3.8-27b","stream":false,"max_tokens":1,"messages":[{"role":"user","content":"hi"}]}"#;
    let (code, _, body) = post_raw(port, "/v1/chat/completions", probe);
    assert_eq!(code, 200, "{body}");
    let prompt_tokens = JsonValue::parse(&body)
        .unwrap()
        .get("usage")
        .unwrap()
        .get_usize("prompt_tokens")
        .unwrap();
    let usable_room = ctx - CTX_SLACK - prompt_tokens;
    assert!(usable_room > 10);

    // 2a. Exact usable boundary: prompt_tokens + max_tokens + CTX_SLACK == ctx -> 200 OK
    let exact_ok = format!(
        r#"{{"model":"qwen3.8-27b","stream":true,"max_tokens":{usable_room},"messages":[{{"role":"user","content":"hi"}}]}}"#
    );
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", &exact_ok);
    assert_eq!(code, 200, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    assert_eq!(engine.last_max_new(), usable_room);
    assert!(!payload.contains("\"server_error\""));
    assert_eq!(
        parse_sse_data(&payload).last().map(String::as_str),
        Some("[DONE]")
    );

    // 2b. One over usable boundary (usable_room + 1) -> 400 BEFORE SSE stream starts
    let one_over = format!(
        r#"{{"model":"qwen3.8-27b","stream":true,"max_tokens":{},"messages":[{{"role":"user","content":"hi"}}]}}"#,
        usable_room + 1
    );
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", &one_over);
    assert_eq!(code, 400, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("application/json"));
    assert!(!head.to_ascii_lowercase().contains("text/event-stream"));
    assert!(!payload.contains("[DONE]"));
    let err_json = JsonValue::parse(&payload).unwrap();
    let err = err_json.get("error").unwrap();
    assert_eq!(err.get_str("type"), Some("invalid_request_error"));
    assert_eq!(err.get_str("code"), Some("context_length_exceeded"));
    assert_eq!(err.get_str("param"), Some("max_tokens"));

    // 2c. Exact `prompt_tokens + max_tokens == ctx` (violates CTX_SLACK reservation) -> 400 BEFORE SSE stream
    let raw_boundary = ctx - prompt_tokens;
    let raw_req = format!(
        r#"{{"model":"qwen3.8-27b","stream":true,"max_completion_tokens":{raw_boundary},"messages":[{{"role":"user","content":"hi"}}]}}"#
    );
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", &raw_req);
    assert_eq!(code, 400, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("application/json"));
    assert!(!payload.contains("[DONE]"));
    let err_json = JsonValue::parse(&payload).unwrap();
    let err = err_json.get("error").unwrap();
    assert_eq!(err.get_str("type"), Some("invalid_request_error"));
    assert_eq!(err.get_str("code"), Some("context_length_exceeded"));
    assert_eq!(err.get_str("param"), Some("max_completion_tokens"));

    // 2d. With fit_max_tokens=true, raw_boundary clamps cleanly to usable_room and succeeds
    let (srv_fit, engine_fit) = start_mock(ctx, true, vec!["</think>\n\nClamped".to_string()]);
    let (code, head, payload) = post_raw(srv_fit.port(), "/v1/chat/completions", &raw_req);
    assert_eq!(code, 200, "head={head}\npayload={payload}");
    assert_eq!(engine_fit.last_max_new(), usable_room);
    assert_eq!(
        parse_sse_data(&payload).last().map(String::as_str),
        Some("[DONE]")
    );
}

#[test]
fn over_limit_prompt_returns_400_before_streaming_starts() {
    let ctx = 512usize;
    let (srv, _) = start_mock(ctx, false, vec!["should never generate".to_string(); 4]);
    let port = srv.port();

    // Prompt that exceeds context_limit (>= 512 tokens)
    let long_text = "x".repeat(600);
    let req_over = format!(
        r#"{{"model":"qwen3.8-27b","stream":true,"messages":[{{"role":"user","content":"{long_text}"}}]}}"#
    );
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", &req_over);
    assert_eq!(code, 400, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("application/json"));
    assert!(!head.to_ascii_lowercase().contains("text/event-stream"));
    assert!(!payload.contains("\"role\":\"assistant\""));
    assert!(!payload.contains("[DONE]"));
    let err_json = JsonValue::parse(&payload).unwrap();
    let err = err_json.get("error").unwrap();
    assert_eq!(err.get_str("type"), Some("invalid_request_error"));
    assert_eq!(err.get_str("code"), Some("context_length_exceeded"));

    // Prompt inside `ctx - CTX_SLACK .. ctx` (leaves 0 usable completion tokens) also returns 400 before SSE!
    let (srv2, _) = start_mock(2048, false, vec!["ok".to_string()]);
    let (code, _, probe_body) = post_raw(
        srv2.port(),
        "/v1/chat/completions",
        r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":""}]}"#,
    );
    assert_eq!(code, 200);
    let base_tokens = JsonValue::parse(&probe_body)
        .unwrap()
        .get("usage")
        .unwrap()
        .get_usize("prompt_tokens")
        .unwrap();
    assert!(
        base_tokens + CTX_SLACK < ctx,
        "base_tokens={base_tokens} should be < {}",
        ctx - CTX_SLACK
    );
    let fill = ctx - base_tokens - 4;
    let near_limit = "a".repeat(fill);
    let req_near = format!(
        r#"{{"model":"qwen3.8-27b","stream":true,"messages":[{{"role":"user","content":"{near_limit}"}}]}}"#
    );
    let (code, head, payload) = post_raw(port, "/v1/chat/completions", &req_near);
    assert_eq!(code, 400, "head={head}\npayload={payload}");
    assert!(head.to_ascii_lowercase().contains("application/json"));
    assert!(!head.to_ascii_lowercase().contains("text/event-stream"));
    assert!(!payload.contains("[DONE]"));
    let err_json = JsonValue::parse(&payload).unwrap();
    let err = err_json.get("error").unwrap();
    assert_eq!(err.get_str("type"), Some("invalid_request_error"));
    assert_eq!(err.get_str("code"), Some("context_length_exceeded"));
}

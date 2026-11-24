//! Engine trait, `EngineTimings`, `ScriptedMockEngine`, and `StdioSubprocessEngine`
//! implementing the Strata/Talus stdio engine protocol (`READY`, `GEN`, `PP`, `RESUME`, `T`, `DONE`, `STOP`, `QUIT`).

use crate::batch::{BatchClient, Leg, StdinSink};
use crate::json::JsonValue;
use crate::tokenizer::{ByteTokenizer, ID_IM_END};
use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub enum EngineError {
    InvalidRequest(String),
    EngineDied(String),
    EngineStarting(String),
    NonFinite(String),
    /// The engine does not implement this feature (E1 logprobs / E2 scoring): the server answers 400 / 501.
    Unsupported(String),
    Cancelled,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidRequest(s) => f.write_str(s),
            EngineError::EngineDied(s) => f.write_str(s),
            EngineError::EngineStarting(s) => f.write_str(s),
            EngineError::NonFinite(s) => f.write_str(s),
            EngineError::Unsupported(s) => f.write_str(s),
            EngineError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub repeat_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub penalty_last_n: i64,
    pub seed: Option<i64>,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub cache_prompt: bool,
    pub checkpoint: bool,
    pub pin_prefix_tokens: Option<usize>,
    pub reasoning_budget_tokens: Option<i64>,
    pub reasoning_effort: Option<String>,
    pub enable_thinking: Option<bool>,
    /// E1: opt-in top-k logprobs per generated token (`GEN ... logprobs=K`); None = off (no GEN key at all).
    pub logprobs: Option<usize>,
    /// E6: compact JSON Schema (prepared by `talus_text::json_grammar::prepare`) the engine must constrain the
    /// output to; sent as the GEN key `json_schema=` only to engines that advertise `INFO grammar=json_schema`.
    pub json_schema: Option<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            penalty_last_n: 64,
            seed: None,
            stop: Vec::new(),
            ignore_eos: false,
            cache_prompt: true,
            checkpoint: true,
            pin_prefix_tokens: None,
            reasoning_budget_tokens: None,
            reasoning_effort: None,
            enable_thinking: None,
            logprobs: None,
            json_schema: None,
        }
    }
}

impl SamplingParams {
    /// Format GEN sampling key=value tokens matching `StrataEngine.sampling_keys` in `serve/server.py`.
    pub fn to_gen_keys(&self) -> String {
        let mut keys = String::new();
        if self.temperature > 0.0 {
            keys.push_str(&format!(" temperature={}", self.temperature));
        }
        if self.top_p < 1.0 {
            keys.push_str(&format!(" top_p={}", self.top_p));
        }
        if self.top_k >= 0 {
            let clamped = if (1..=64).contains(&self.top_k) {
                self.top_k
            } else {
                64
            };
            keys.push_str(&format!(" top_k={clamped}"));
        }
        if self.min_p > 0.0 && self.min_p <= 1.0 {
            keys.push_str(&format!(" min_p={}", self.min_p));
        }
        let rp_on = (self.repeat_penalty - 1.0).abs() > f64::EPSILON;
        let pf_on = self.frequency_penalty.abs() > f64::EPSILON;
        let pp_on = self.presence_penalty.abs() > f64::EPSILON;
        if rp_on {
            keys.push_str(&format!(" penalty_repeat={}", self.repeat_penalty));
        }
        if pf_on {
            keys.push_str(&format!(" penalty_freq={}", self.frequency_penalty));
        }
        if pp_on {
            keys.push_str(&format!(" penalty_present={}", self.presence_penalty));
        }
        if rp_on || pf_on || pp_on {
            let pln = if self.penalty_last_n > 0 {
                self.penalty_last_n
            } else {
                64
            };
            keys.push_str(&format!(" penalty_last_n={pln}"));
        }
        if let Some(seed) = self.seed {
            if seed > 0 {
                keys.push_str(&format!(" seed={seed}"));
            }
        }
        if !self.checkpoint {
            keys.push_str(" ckpt=0");
        }
        if self.ignore_eos {
            keys.push_str(" ignore_eos=1");
        }
        if !self.cache_prompt {
            keys.push_str(" cache=0");
        }
        if let Some(pin) = self.pin_prefix_tokens {
            if pin > 0 {
                keys.push_str(&format!(" pin={pin}"));
            }
        }
        if let Some(k) = self.logprobs {
            keys.push_str(&format!(" logprobs={k}"));
        }
        if let Some(ref js) = self.json_schema {
            keys.push_str(" json_schema=");
            keys.push_str(&gen_value_encode(js));
        }
        keys
    }
}

/// A GEN key value carries no space (the line is space separated): `%` and the whitespace bytes are %-encoded.
pub fn gen_value_encode(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for ch in v.chars() {
        match ch {
            '%' => out.push_str("%25"),
            ' ' => out.push_str("%20"),
            '\n' => out.push_str("%0A"),
            '\r' => out.push_str("%0D"),
            '\t' => out.push_str("%09"),
            c => out.push(c),
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerateRequest {
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_tokens: Vec<u32>,
    pub stop_strings: Vec<String>,
    /// vision (vision.rs): the SVE1 embeddings file for this request's pictures, the GENI argument
    /// that replaces the GEN verb; `None` = no pictures. The path must hold no space: the engine's
    /// protocol line is space-split.
    pub embeddings: Option<String>,
    /// the `img_keys=` value (#2228): each picture's key (16 hex), comma list, prompt order. Only
    /// sent with `embeddings`; without it the engine matches pictures by embedding hash instead.
    pub img_keys: Option<String>,
    /// the native tower's `@raw` form: the request's encoded pictures, one after another, written
    /// right behind the GENI line (framed by `lens`); `None` = the SVE-file path or no pictures.
    pub raw_frame: Option<Arc<Vec<u8>>>,
    /// the per-picture byte lengths of `raw_frame` (`img_bytes=`), prompt order.
    pub lens: Vec<u64>,
    /// the native tower's `img_px=`: this server's prediction of each picture's resized size,
    /// which the engine checks against what its tower prepared (drift = ERR, never a wrong
    /// picture).  Optional in the protocol; sent whenever the predictor knows.
    pub img_px: Option<String>,
}

/// E1: one generated token's logprob under the target model at temperature 1 (before sampling filters), and the
/// top-k alternatives in descending order. From the engine's `L <lp> <k> <id>:<lp>,...` line.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

impl TokenLogprob {
    /// Parse the part after `L `: `<lp> <k> <id>:<lp>,...` (`-` or nothing for an empty list).
    pub fn parse_l(rest: &str) -> Option<Self> {
        let mut f = rest.split_whitespace();
        let logprob = parse_lp(f.next()?)?;
        let _k = f.next();
        let top = parse_top(f.next().unwrap_or("-"))?;
        Some(Self { logprob, top })
    }

    pub fn to_line(&self) -> String {
        format!(
            "L {} {} {}",
            self.logprob,
            self.top.len(),
            format_top(&self.top)
        )
    }
}

/// E2: one teacher-forced continuation position (the distribution before that token).
#[derive(Clone, Debug, PartialEq)]
pub struct ScorePosition {
    pub id: u32,
    pub logprob: f32,
    pub rank: u32,
    pub entropy: f32,
    pub top: Vec<(u32, f32)>,
}

impl ScorePosition {
    /// Parse the part after `S `: `<id> <lp> <rank> <entropy> <k> <id>:<lp>,...`.
    pub fn parse_s(rest: &str) -> Option<Self> {
        let mut f = rest.split_whitespace();
        let id = f.next()?.parse().ok()?;
        let logprob = parse_lp(f.next()?)?;
        let rank = f.next()?.parse().ok()?;
        let entropy = parse_lp(f.next()?)?;
        let _k = f.next();
        let top = parse_top(f.next().unwrap_or("-"))?;
        Some(Self {
            id,
            logprob,
            rank,
            entropy,
            top,
        })
    }

    pub fn to_line(&self) -> String {
        format!(
            "S {} {} {} {} {} {}",
            self.id,
            self.logprob,
            self.rank,
            self.entropy,
            self.top.len(),
            format_top(&self.top)
        )
    }
}

/// E2 request as the engine sees it (`SCORE <top_k> <prompt ids>|<continuation ids>`).
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreRequest {
    pub prompt_ids: Vec<u32>,
    pub continuation_ids: Vec<u32>,
    pub top_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoreResult {
    pub positions: Vec<ScorePosition>,
    pub prompt_ms: f64,
    pub total_ms: f64,
}

fn parse_lp(s: &str) -> Option<f32> {
    match s {
        "-inf" => Some(f32::NEG_INFINITY),
        "inf" => Some(f32::INFINITY),
        _ => s.parse::<f32>().ok(),
    }
}

fn parse_top(s: &str) -> Option<Vec<(u32, f32)>> {
    if s == "-" || s.is_empty() {
        return Some(Vec::new());
    }
    s.split(',')
        .filter(|e| !e.is_empty())
        .map(|e| {
            let (id, lp) = e.split_once(':')?;
            Some((id.parse().ok()?, parse_lp(lp)?))
        })
        .collect()
}

fn format_top(top: &[(u32, f32)]) -> String {
    if top.is_empty() {
        return "-".to_string();
    }
    top.iter()
        .map(|(id, lp)| format!("{id}:{lp}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineTimings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub draft_n: Option<usize>,
    pub draft_accepted: Option<usize>,
    pub finish_reason: String,
}

impl Default for EngineTimings {
    fn default() -> Self {
        Self {
            cache_n: 0,
            prompt_n: 0,
            prompt_ms: 1.0,
            predicted_n: 0,
            predicted_ms: 1.0,
            draft_n: None,
            draft_accepted: None,
            finish_reason: "stop".to_string(),
        }
    }
}

impl EngineTimings {
    pub fn round1(v: f64) -> f64 {
        (v * 10.0).round() / 10.0
    }

    pub fn round3(v: f64) -> f64 {
        (v * 1000.0).round() / 1000.0
    }

    /// Format into the `timings` JSON object matching `request_timings` in `serve/server.py`.
    pub fn to_json(&self, prompt_tokens: usize, completion_tokens: usize) -> JsonValue {
        let reused = self.cache_n.min(prompt_tokens);
        let p_n = if self.prompt_n > 0 {
            self.prompt_n
        } else {
            prompt_tokens.saturating_sub(reused)
        };
        let p_ms = Self::round1(self.prompt_ms);
        let d_n = if completion_tokens > 0 {
            completion_tokens
        } else {
            self.predicted_n
        };
        let d_ms = Self::round1(self.predicted_ms);
        let p_per_tok = if p_n > 0 {
            Self::round3(p_ms / p_n as f64)
        } else {
            0.0
        };
        let p_per_sec = if p_n > 0 && p_ms > 0.0 {
            Self::round1(p_n as f64 * 1000.0 / p_ms)
        } else {
            0.0
        };
        let d_per_tok = if d_n > 0 {
            Self::round3(d_ms / d_n as f64)
        } else {
            0.0
        };
        let d_per_sec = if d_n > 0 && d_ms > 0.0 {
            Self::round1(d_n as f64 * 1000.0 / d_ms)
        } else {
            0.0
        };
        let mut obj = JsonValue::obj(vec![
            ("cache_n", JsonValue::from(reused)),
            ("prompt_n", JsonValue::from(p_n)),
            ("prompt_ms", JsonValue::from(p_ms)),
            ("prompt_per_token_ms", JsonValue::from(p_per_tok)),
            ("prompt_per_second", JsonValue::from(p_per_sec)),
            ("predicted_n", JsonValue::from(d_n)),
            ("predicted_ms", JsonValue::from(d_ms)),
            ("predicted_per_token_ms", JsonValue::from(d_per_tok)),
            ("predicted_per_second", JsonValue::from(d_per_sec)),
        ]);
        if let Some(draft_n) = self.draft_n {
            obj.insert("draft_n", JsonValue::from(draft_n));
            obj.insert(
                "draft_n_accepted",
                JsonValue::from(self.draft_accepted.unwrap_or(0)),
            );
        }
        obj
    }
}

/// Crash/restart bookkeeping an engine reports (`/health`, `/metrics`, `/v1/talus/status`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineHealth {
    /// Engine processes that exited on their own (not through unload/shutdown).
    pub crashes: u64,
    /// Automatic restarts after a crash.
    pub restarts: u64,
    /// Set when the crash-loop cap was hit: the engine stays down until `load` (POST /load) or a server restart.
    pub down: Option<String>,
    /// The last crash: exit status, time and the engine's last stderr lines.
    pub last_crash: Option<String>,
    /// Crashed (or on its way out) and not restarted yet: the supervisor restarts it shortly (within the cap).
    pub restart_pending: bool,
    /// An automatic restart is loading the model now.
    pub restarting: bool,
}

/// Core `Engine` trait for `talus-server`.
pub trait Engine: Send + Sync {
    fn max_context(&self) -> usize;

    fn is_loaded(&self) -> bool {
        true
    }

    fn total_slots(&self) -> usize {
        1
    }

    fn model_path(&self) -> Option<&str> {
        None
    }

    fn build_version(&self) -> Option<&str> {
        Some("0.1.44")
    }

    /// Process id of the engine child, if it runs in one (`/v1/talus/status` `engine.rss_bytes`).
    fn pid(&self) -> Option<u32> {
        None
    }

    /// A value from the engine's start-up `INFO key=value` lines.
    fn info_value(&self, _key: &str) -> Option<String> {
        None
    }

    fn load(&self) -> Result<(), EngineError> {
        Ok(())
    }

    /// Free the model's memory; the next request or `load` brings it back. Engines that cannot say so: the server
    /// answers `/unload` with 501 and refuses `idle_unload_seconds`.
    fn unload(&self) -> Result<(), EngineError> {
        Err(EngineError::Unsupported(
            "this engine cannot unload".to_string(),
        ))
    }

    /// `unload` really frees the engine (and `load` / the next request brings it back).
    fn supports_lifecycle(&self) -> bool {
        false
    }

    /// Crash/restart state. Engines without a process report the default (no crashes).
    fn health(&self) -> EngineHealth {
        EngineHealth::default()
    }

    /// Called every moment by the server's supervisor thread: notice a crash and restart right away (within the cap),
    /// so clients polling `/health` recover without sending a request. Blocks while the model loads.
    fn supervise(&self) {}

    /// A request is inside the engine right now (holding it; others queue behind). None = not tracked.
    fn in_flight(&self) -> Option<usize> {
        None
    }

    /// Stop the engine for good and wait until it has exited (its memory is free), before the server replaces
    /// itself with a new run config (talus-tui's apply). The default does nothing.
    fn shutdown(&self) {}

    /// Generate tokens for `req`, invoking `on_progress(read_tokens, total_tokens)` during prompt read
    /// and `on_token(token_id)` for each generated token (returning `false` stops early).
    fn generate(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError>;

    /// E1: the engine emits per-token logprobs when a request sets `sampling.logprobs` (`INFO logprobs=1`).
    fn supports_logprobs(&self) -> bool {
        false
    }

    /// As [`Self::generate`], and `on_token` also gets the token's logprobs when the request asked for them (and the
    /// engine supports them); else None. The default delegates to `generate` (no logprobs).
    fn generate_with_logprobs(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        self.generate(req, cancel, &mut |t| on_token(t, None), on_progress)
    }

    /// E2: teacher-forced scoring (`SCORE`, `INFO score=1`).
    fn supports_score(&self) -> bool {
        false
    }

    fn score(&self, _req: &ScoreRequest) -> Result<ScoreResult, EngineError> {
        Err(EngineError::Unsupported(
            "scoring not supported by this engine".to_string(),
        ))
    }

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32>;

    fn detokenize(&self, ids: &[u32]) -> String;

    fn cancel(&self);

    fn save_slot(&self, _slot_id: usize, _path: &Path) -> Result<(usize, usize, f64), EngineError> {
        Err(EngineError::InvalidRequest(
            "slot save/restore is not supported by this engine".to_string(),
        ))
    }

    fn restore_slot(
        &self,
        _slot_id: usize,
        _path: &Path,
    ) -> Result<(usize, usize, f64), EngineError> {
        Err(EngineError::InvalidRequest(
            "slot save/restore is not supported by this engine".to_string(),
        ))
    }

    fn erase_slot(&self, _slot_id: usize) -> Result<usize, EngineError> {
        Ok(0)
    }

    /// `VGEOM` between requests: the native vision tower's geometry constants
    /// (align, min_tokens, max_tokens) for the server's own resize predictor.  Engines without
    /// the tower answer ERR; engines without the command at all never get here (the server asks
    /// only when the config says the tower is native).
    fn vgeom(&self) -> Result<(i64, i64, i64), EngineError> {
        Err(EngineError::InvalidRequest(
            "this engine does not answer VGEOM".to_string(),
        ))
    }

    /// E6: the engine constrains output to `SamplingParams::json_schema`. Engines that do not say so get a 400 for
    /// `response_format: json_schema` instead of an unconstrained answer.
    fn supports_json_schema(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
struct MockState {
    turn: usize,
    last_prompt: Vec<u32>,
    /// Multi-conversation prefix cache history (snapshots of prompt and prompt+reply).
    cache_snapshots: Vec<Vec<u32>>,
    saved_slots: HashMap<PathBuf, Vec<u32>>,
}

/// Configurable in-process mock engine (`ScriptedMockEngine`) supporting:
/// - Multi-turn scripted token streams (each request gets the next script and the last repeats, as in `serve/server.py`)
/// - Tool-call responses (Qwen XML, JSON, and GLM `<arg_key>` formats)
/// - `<think>...</think>` reasoning responses
/// - Prompt cache hits (`cache_n` via prefix matching across cached conversation snapshots or explicit override)
/// - Echoed prompts (`echo_prompt`)
/// - `ignore_eos` (runs through `<|im_end|>` up to `max_tokens`)
/// - `cache_prompt: false` (forces `cache_n = 0`)
/// - Speculative draft counts (`draft_n`, `draft_accepted`)
/// - Full stdio engine protocol simulation (`run_stdio_loop`: `INFO`, `READY`, `GEN`, `PP`, `RESUME`, `T`, `DONE`, `STOP`, `QUIT`).
pub struct ScriptedMockEngine {
    tok: ByteTokenizer,
    max_context: AtomicUsize,
    loaded: AtomicBool,
    scripts: Mutex<Vec<Vec<u32>>>,
    raw_scripts: Mutex<Vec<String>>,
    echo_prompt: AtomicBool,
    fixed_cache_n: Mutex<Option<usize>>,
    fixed_drafts: Mutex<Option<(usize, usize)>>,
    delay: Duration,
    cancelled: AtomicBool,
    state: Mutex<MockState>,
    /// E1/E2 support advertised (`INFO logprobs=1 score=1`) and served with deterministic mock values.
    logprobs_on: AtomicBool,
    score_on: AtomicBool,
    /// E6 tests: advertise json_schema support, and the last schema a request carried.
    json_schema_ok: AtomicBool,
    last_json_schema: Mutex<Option<String>>,
    /// Batch mode (`--batch` on the mock): the engine-side protocol responder, and the bus the
    /// admission/pump machinery reads - the same client code path the real engine uses.
    batch: Mutex<Option<Arc<MockBatch>>>,
}

/// The mock's deterministic logprobs for generated step `step` (0-based) choosing `tok`: the token is the top-1.
pub fn mock_token_logprob(step: usize, tok: u32, k: usize) -> TokenLogprob {
    let lp = -0.05 * ((step % 7) as f32) - 0.01;
    let mut top = Vec::with_capacity(k);
    for j in 0..k {
        if j == 0 {
            top.push((tok, lp));
        } else {
            top.push((tok.wrapping_add(j as u32), lp - j as f32));
        }
    }
    TokenLogprob { logprob: lp, top }
}

impl Default for ScriptedMockEngine {
    fn default() -> Self {
        Self::new(ByteTokenizer::new())
    }
}

impl ScriptedMockEngine {
    pub fn new(tok: ByteTokenizer) -> Self {
        Self::with_tokenizer(
            tok,
            vec!["Thinking about it.</think>\n\nHello from the mock engine.".to_string()],
            32768,
        )
    }

    pub fn with_scripts(scripts: Vec<String>) -> Self {
        Self::with_tokenizer(ByteTokenizer::new(), scripts, 32768)
    }

    pub fn with_tokenizer(tok: ByteTokenizer, scripts: Vec<String>, max_context: usize) -> Self {
        let scripts = if scripts.is_empty() {
            vec!["Thinking about it.</think>\n\nHello from the mock engine.".to_string()]
        } else {
            scripts
        };
        let end = tok.encode("<|im_end|>", true);
        let encoded = scripts
            .iter()
            .map(|s| {
                let mut ids = tok.encode(s, true);
                ids.extend_from_slice(&end);
                ids
            })
            .collect();
        Self {
            tok,
            max_context: AtomicUsize::new(max_context),
            loaded: AtomicBool::new(true),
            scripts: Mutex::new(encoded),
            raw_scripts: Mutex::new(scripts),
            echo_prompt: AtomicBool::new(false),
            fixed_cache_n: Mutex::new(None),
            fixed_drafts: Mutex::new(None),
            delay: Duration::ZERO,
            cancelled: AtomicBool::new(false),
            state: Mutex::new(MockState {
                turn: 0,
                last_prompt: Vec::new(),
                cache_snapshots: Vec::new(),
                saved_slots: HashMap::new(),
            }),
            logprobs_on: AtomicBool::new(false),
            score_on: AtomicBool::new(false),
            json_schema_ok: AtomicBool::new(false),
            last_json_schema: Mutex::new(None),
            batch: Mutex::new(None),
        }
    }

    /// Advertise and serve E1 logprobs (mock values, see [`mock_token_logprob`]).
    pub fn set_logprobs_support(&self, on: bool) {
        self.logprobs_on.store(on, Ordering::Relaxed);
    }

    /// Advertise and serve E2 scoring (mock values: position i = `mock_token_logprob(i, id, k)`).
    pub fn set_score_support(&self, on: bool) {
        self.score_on.store(on, Ordering::Relaxed);
    }

    pub fn set_json_schema_support(&self, on: bool) {
        self.json_schema_ok.store(on, Ordering::Relaxed);
    }

    /// The `json_schema` of the last generate request (E6 tests).
    pub fn last_json_schema(&self) -> Option<String> {
        self.last_json_schema.lock().unwrap().clone()
    }

    pub fn set_max_context(&self, ctx: usize) {
        self.max_context.store(ctx.max(1), Ordering::Relaxed);
    }

    pub fn set_scripts(&self, scripts: Vec<String>) {
        let end = self.tok.encode("<|im_end|>", true);
        let encoded = scripts
            .iter()
            .map(|s| {
                let mut ids = self.tok.encode(s, true);
                ids.extend_from_slice(&end);
                ids
            })
            .collect();
        *self.scripts.lock().unwrap() = encoded;
        *self.raw_scripts.lock().unwrap() = scripts;
        self.state.lock().unwrap().turn = 0;
    }

    pub fn set_echo_prompt(&self, echo: bool) {
        self.echo_prompt.store(echo, Ordering::Relaxed);
    }

    pub fn set_fixed_cache_n(&self, cache_n: Option<usize>) {
        *self.fixed_cache_n.lock().unwrap() = cache_n;
    }

    pub fn set_fixed_drafts(&self, drafts: Option<(usize, usize)>) {
        *self.fixed_drafts.lock().unwrap() = drafts;
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    pub fn last_prompt(&self) -> Vec<u32> {
        self.state.lock().unwrap().last_prompt.clone()
    }

    fn compute_reused(prompt_ids: &[u32], snapshots: &[Vec<u32>]) -> usize {
        if prompt_ids.len() <= 1 {
            return 0;
        }
        let max_reused = prompt_ids.len() - 1;
        let mut best = 0usize;
        for snap in snapshots {
            let common = prompt_ids
                .iter()
                .zip(snap.iter())
                .take_while(|(a, b)| a == b)
                .count();
            best = best.max(common.min(max_reused));
        }
        best
    }

    fn record_snapshot(state: &mut MockState, seq: Vec<u32>) {
        if seq.is_empty() {
            return;
        }
        if !state.cache_snapshots.iter().any(|s| s == &seq) {
            if state.cache_snapshots.len() >= 32 {
                state.cache_snapshots.remove(0);
            }
            state.cache_snapshots.push(seq);
        }
    }

    /// Run a Strata/Talus stdio engine protocol loop over `reader` and `writer`.
    /// Useful both for testing `StdioSubprocessEngine` and for running `talus-server --mock-stdio-worker`
    /// as the backend engine executable for `serve/server.py --engine strata`!
    pub fn run_stdio_loop<R: BufRead, W: Write>(
        &self,
        mut reader: R,
        mut writer: W,
    ) -> std::io::Result<()> {
        let ctx = self.max_context();
        writeln!(
            writer,
            "INFO version=0.1.44 arena_mib=1024 vram_free_mib=8192 slot_cache=1"
        )?;
        if self.logprobs_on.load(Ordering::Relaxed) || self.score_on.load(Ordering::Relaxed) {
            writeln!(
                writer,
                "INFO logprobs={} score={}",
                u8::from(self.logprobs_on.load(Ordering::Relaxed)),
                u8::from(self.score_on.load(Ordering::Relaxed))
            )?;
        }
        writeln!(writer, "READY {ctx} stop")?;
        writer.flush()?;

        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                line.clear();
                continue;
            }
            if trimmed == "QUIT" {
                break;
            }
            if trimmed == "STOP" {
                self.cancel();
                line.clear();
                continue;
            }
            if trimmed == "CLEAR" {
                let n = self.erase_slot(0).unwrap_or(0);
                writeln!(writer, "CLEARED {n}")?;
                writer.flush()?;
                line.clear();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("SAVE ") {
                let path = Path::new(rest.trim());
                match self.save_slot(0, path) {
                    Ok((n, _, _)) => writeln!(writer, "SAVED {n}")?,
                    Err(e) => writeln!(writer, "ERR {e}")?,
                }
                writer.flush()?;
                line.clear();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("RESTORE ") {
                let path = Path::new(rest.trim());
                match self.restore_slot(0, path) {
                    Ok((n, _, _)) => writeln!(writer, "RESTORED {n}")?,
                    Err(e) => writeln!(writer, "ERR {e}")?,
                }
                writer.flush()?;
                line.clear();
                continue;
            }
            if trimmed.starts_with("SCORE ") {
                match parse_score_cmd(trimmed) {
                    Ok(req) => match self.score(&req) {
                        Ok(res) => {
                            for p in &res.positions {
                                writeln!(writer, "{}", p.to_line())?;
                            }
                            writeln!(
                                writer,
                                "DONE {} {} {:.1} {:.1} stop 0 0 0",
                                0,
                                req.prompt_ids.len() + req.continuation_ids.len(),
                                res.prompt_ms,
                                res.total_ms
                            )?;
                        }
                        Err(e) => writeln!(writer, "ERR {e}")?,
                    },
                    Err(e) => writeln!(writer, "ERR {e}")?,
                }
                writer.flush()?;
                line.clear();
                continue;
            }
            if trimmed.starts_with("VRAM") {
                writeln!(
                    writer,
                    "VRAM free_mib=8192 target_mib=700 released_mib=0 evicted=0"
                )?;
                writer.flush()?;
                line.clear();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("GEN ") {
                // crash tests: TALUS_MOCK_CRASH_FILE=<path>: while that file exists, a GEN removes it and exits 3
                if let Some(f) = std::env::var_os("TALUS_MOCK_CRASH_FILE") {
                    if std::fs::remove_file(&f).is_ok() {
                        eprintln!("mock engine: simulated crash on GEN");
                        // one token first: a crash mid-request, not one before any answer (those are sent again)
                        writeln!(writer, "T 97")?;
                        writer.flush()?;
                        std::process::exit(3);
                    }
                }
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() < 2 {
                    writeln!(writer, "ERR malformed GEN command")?;
                    writer.flush()?;
                    line.clear();
                    continue;
                }
                let max_new_tokens = parts[0].parse::<usize>().unwrap_or(64);
                let mut sampling = SamplingParams::default();
                for &kv in &parts[1..parts.len() - 1] {
                    if let Some((k, v)) = kv.split_once('=') {
                        match k {
                            "temperature" => {
                                if let Ok(val) = v.parse() {
                                    sampling.temperature = val;
                                }
                            }
                            "top_p" => {
                                if let Ok(val) = v.parse() {
                                    sampling.top_p = val;
                                }
                            }
                            "top_k" => {
                                if let Ok(val) = v.parse() {
                                    sampling.top_k = val;
                                }
                            }
                            "min_p" => {
                                if let Ok(val) = v.parse() {
                                    sampling.min_p = val;
                                }
                            }
                            "penalty_repeat" => {
                                if let Ok(val) = v.parse() {
                                    sampling.repeat_penalty = val;
                                }
                            }
                            "penalty_freq" => {
                                if let Ok(val) = v.parse() {
                                    sampling.frequency_penalty = val;
                                }
                            }
                            "penalty_present" => {
                                if let Ok(val) = v.parse() {
                                    sampling.presence_penalty = val;
                                }
                            }
                            "penalty_last_n" => {
                                if let Ok(val) = v.parse() {
                                    sampling.penalty_last_n = val;
                                }
                            }
                            "seed" => sampling.seed = v.parse().ok(),
                            "ignore_eos" => sampling.ignore_eos = v == "1" || v == "true",
                            "cache" => sampling.cache_prompt = !(v == "0" || v == "false"),
                            "ckpt" => sampling.checkpoint = !(v == "0" || v == "false"),
                            "pin" => sampling.pin_prefix_tokens = v.parse().ok(),
                            "logprobs" => sampling.logprobs = v.parse().ok(),
                            _ => {}
                        }
                    }
                }
                let ids_str = parts[parts.len() - 1];
                let prompt_ids: Vec<u32> = ids_str
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<u32>().ok())
                    .collect();
                let req = GenerateRequest {
                    prompt_ids: prompt_ids.clone(),
                    max_new_tokens,
                    sampling,
                    stop_tokens: self.tok.stop_tokens.clone(),
                    stop_strings: Vec::new(),
                    embeddings: None,
                    img_keys: None,
                    raw_frame: None,
                    lens: Vec::new(),
                    img_px: None,
                };
                let cancel = Arc::new(AtomicBool::new(false));
                let mut emitted_tokens = Vec::new();
                let res = self.generate_with_logprobs(
                    &req,
                    cancel,
                    &mut |tok_id, lp| {
                        emitted_tokens.push((tok_id, lp));
                        true
                    },
                    &mut |_read, _total| {},
                );
                match res {
                    Ok(timings) => {
                        if timings.cache_n > 0 {
                            writeln!(writer, "RESUME {}", timings.cache_n)?;
                        }
                        writeln!(
                            writer,
                            "PP {} {} {:.1} 1000.0",
                            timings.prompt_n,
                            prompt_ids.len(),
                            timings.prompt_ms
                        )?;
                        for (tok_id, lp) in emitted_tokens {
                            if let Some(lp) = lp {
                                writeln!(writer, "{}", lp.to_line())?;
                            }
                            writeln!(writer, "T {tok_id}")?;
                        }
                        let draft_acc = timings.draft_accepted.unwrap_or(0);
                        let draft_off = timings.draft_n.unwrap_or(0);
                        writeln!(
                            writer,
                            "DONE {} {} {:.1} {:.1} {} {} {} {} 0 0 0 0 0.0 {}",
                            timings.predicted_n,
                            prompt_ids.len(),
                            timings.prompt_ms,
                            timings.predicted_ms,
                            timings.finish_reason,
                            draft_acc,
                            draft_off,
                            timings.cache_n,
                            timings.prompt_n
                        )?;
                        writer.flush()?;
                    }
                    Err(e) => {
                        writeln!(writer, "ERR {e}")?;
                        writer.flush()?;
                    }
                }
            }
            line.clear();
        }
        Ok(())
    }
}

/// The mock's batch mode: a responder thread that speaks the engine side of the slot protocol
/// (`BGEN` answered on the control lines, the slot's own `BT`/`BDONE` window, `BSTOP` ending a
/// slot at its next line), and the bus the machinery reads - so a mock-hosted request runs
/// through the very admission/pump code the stdio engine uses.
pub(crate) struct MockBatch {
    pub(crate) bus: Arc<EngineBus>,
    feeder: Arc<BusFeeder>,
    tx: mpsc::Sender<String>,
    scripts: Arc<Mutex<std::collections::VecDeque<Vec<u32>>>>,
    pace_ms: Arc<AtomicU64>,
    refusals: Arc<AtomicUsize>,
    stopped: Arc<Mutex<std::collections::HashSet<usize>>>,
    /// Concurrent requests serialize on their admission, exactly as the real server's control
    /// handoff does: one `BGEN` answered on the control lines at a time.
    admit_lock: Arc<Mutex<()>>,
}

impl ScriptedMockEngine {
    /// Put the mock in batch mode with `n` engine slots (idempotent). Scripts set afterwards are
    /// served one per `BGEN`.
    pub fn set_batch_slots(&self, n: usize) {
        let mut slot = self.batch.lock().unwrap();
        if slot.is_some() {
            return;
        }
        let (bus, feeder) = EngineBus::manual(n);
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(MockBatch {
            bus,
            feeder: Arc::new(feeder),
            tx,
            scripts: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            pace_ms: Arc::new(AtomicU64::new(0)),
            refusals: Arc::new(AtomicUsize::new(0)),
            stopped: Arc::new(Mutex::new(std::collections::HashSet::new())),
            admit_lock: Arc::new(Mutex::new(())),
        });
        let responder = shared.clone();
        thread::spawn(move || mock_batch_responder(rx, responder));
        *slot = Some(shared);
    }

    /// The token scripts the batch responder serves, one per `BGEN`.
    pub fn set_batch_scripts(&self, scripts: Vec<Vec<u32>>) {
        let Some(b) = self.batch.lock().unwrap().clone() else {
            return;
        };
        let mut q = b.scripts.lock().unwrap();
        q.clear();
        q.extend(scripts);
    }

    /// Pause between a slot's `BT` lines, so a test can `BSTOP` a running one.
    pub fn set_batch_pace_ms(&self, ms: u64) {
        if let Some(b) = self.batch.lock().unwrap().clone() {
            b.pace_ms.store(ms, Ordering::Relaxed);
        }
    }

    /// The next `n` BGENs are refused (`ERR BGEN:`), exercising the park-and-retry path.
    pub fn seed_batch_refusals(&self, n: usize) {
        if let Some(b) = self.batch.lock().unwrap().clone() {
            b.refusals.store(n, Ordering::Relaxed);
        }
    }

    /// The mock's batch bus, for assertions about slot state.
    #[allow(dead_code)] // the cfg(test) slot tests are its only callers so far
    pub(crate) fn batch_bus(&self) -> Option<Arc<EngineBus>> {
        self.batch.lock().unwrap().as_ref().map(|b| b.bus.clone())
    }

    /// One request through the slot protocol: admit, then read the slot to its `BDONE`. The
    /// stop-token conventions match the solo path: the stop token is suppressed (the engine does
    /// emit it as a `BT`, and the slot ends there - the machinery then waits for the `BDONE`
    /// that follows in the same window, their #1869).
    fn generate_via_batch(
        &self,
        b: Arc<MockBatch>,
        req: &GenerateRequest,
        cancel: &Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        self.load()?;
        let keys = req.sampling.to_gen_keys();
        let csv = req
            .prompt_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let sink = crate::batch::ScriptSink(b.tx.clone());
        let client = crate::batch::BatchClient {
            bus: &b.bus,
            sink: &sink,
        };
        let cmd = |slot: usize| format!("BGEN {slot} {}{} {}", req.max_new_tokens, keys, csv);
        let is_stop = |t: u32| self.tok.is_stop_token(t) || req.stop_tokens.contains(&t);
        let leg = {
            let _admit = b.admit_lock.lock().unwrap();
            client.admit(
                &cmd,
                None,
                &req.prompt_ids,
                cancel,
                &mut |t, lp| {
                    if is_stop(t) && !req.sampling.ignore_eos {
                        false
                    } else {
                        on_token(t, lp)
                    }
                },
                on_progress,
            )?
        };
        if leg.cancelled {
            // The consumer left during admission: the machinery already `BSTOP`ped the slot (if
            // it continued) and handed its lines to the router. Nothing left to read.
            return Ok(EngineTimings {
                cache_n: leg.reused,
                prompt_n: leg.prompt_n,
                prompt_ms: leg.prompt_ms,
                predicted_n: 0,
                predicted_ms: 0.5,
                draft_n: None,
                draft_accepted: None,
                finish_reason: if leg.finish == "stop" {
                    "stop".to_string()
                } else {
                    "cancel".to_string()
                },
            });
        }
        if !leg.cont {
            // The admission consumed the whole leg. Its token rides `on_token` already - either
            // yielded, or suppressed as a stop token, which then does not count as predicted.
            return Ok(EngineTimings {
                cache_n: leg.reused,
                prompt_n: leg.prompt_n,
                prompt_ms: leg.prompt_ms,
                predicted_n: leg
                    .admission
                    .into_iter()
                    .count()
                    .saturating_sub(if leg.finish == "stop" { 1 } else { 0 }),
                predicted_ms: 0.5,
                draft_n: None,
                draft_accepted: None,
                finish_reason: if leg.finish.is_empty() {
                    "length".to_string()
                } else {
                    leg.finish
                },
            });
        }
        let emitted: Vec<u32> = leg.admission.into_iter().collect();
        let res = client.pump(leg.slot, &req.prompt_ids, &emitted, cancel, &mut |t| {
            if is_stop(t) && !req.sampling.ignore_eos {
                false // suppressed, as the solo path suppresses it: the slot ends here
            } else {
                on_token(t, None)
            }
        })?;
        // The engine's `produced` counts the stop token the solo path suppresses (their #1869's
        // slot end is that same token); a cancelled leg absorbs its last tokens unreported.
        let suppressed = match res.finish.as_str() {
            "stop" | "abort" => 1,
            _ => 0,
        };
        Ok(EngineTimings {
            cache_n: leg.reused,
            prompt_n: leg.prompt_n,
            prompt_ms: leg.prompt_ms,
            predicted_n: res.produced.saturating_sub(suppressed),
            predicted_ms: res.decode_ms.max(0.5),
            draft_n: None,
            draft_accepted: None,
            finish_reason: if res.finish == "abort" {
                "stop".to_string()
            } else {
                res.finish
            },
        })
    }
}

/// The mock's engine side of the slot protocol; see [`MockBatch`]. One thread reads commands, and
/// one per admitted slot plays out its `BT` window at the configured pace, ending at a stop token,
/// `max_new`, or a `BSTOP` - and always with its `BDONE`, as the engine does.
fn mock_batch_responder(rx: mpsc::Receiver<String>, b: Arc<MockBatch>) {
    let feeder = b.feeder.clone();
    while let Ok(cmd) = rx.recv() {
        let mut w = cmd.split_whitespace();
        match w.next().unwrap_or("") {
            "BGEN" | "BGENI" => {
                let slot = w.next().and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
                // The sampling keys start with a space, so `max_new` is its own word here.
                let max_new: usize = w.next().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
                if b.refusals.load(Ordering::SeqCst) > 0 {
                    b.refusals.fetch_sub(1, Ordering::SeqCst);
                    feeder.push("ERR BGEN: no such free slot (--batch 2) or a bad max_new");
                    continue;
                }
                let toks = b
                    .scripts
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or_else(|| vec![ID_IM_END]);
                let plen = 8usize;
                feeder.push(format!("PP {plen} {plen} 1.0"));
                feeder.push(format!("T {}", toks.first().copied().unwrap_or(ID_IM_END)));
                let admitted_stop = toks.first() == Some(&ID_IM_END);
                if toks.len() <= 1 || max_new <= 1 {
                    // The prompt's read spent everything: the leg is done on the control lines.
                    let fin = if admitted_stop { "stop" } else { "length" };
                    feeder.push(format!("DONE 1 {plen} 1.0 0.5 {fin} 0 0 0 0 0 0 0 0.0 0 0"));
                    feeder.push(format!("BADM {slot} 0"));
                    continue;
                }
                feeder.push(format!(
                    "DONE 1 {plen} 1.0 0.5 length 0 0 0 0 0 0 0 0.0 0 0"
                ));
                feeder.push(format!("BADM {slot} 1"));
                let (feeder2, pace, stopped) =
                    (feeder.clone(), b.pace_ms.clone(), b.stopped.clone());
                thread::spawn(move || {
                    let mut produced = 1usize; // the engine's BDONE counts the admission's token
                    let mut fin = "length";
                    for &t in &toks[1..] {
                        if produced >= max_new {
                            break;
                        }
                        let ms = pace.load(Ordering::Relaxed);
                        if ms > 0 {
                            thread::sleep(Duration::from_millis(ms));
                        }
                        if stopped.lock().unwrap().remove(&slot) {
                            fin = "cancel";
                            break;
                        }
                        feeder2.push(format!("BT {slot} {t}"));
                        produced += 1;
                        if t == ID_IM_END {
                            fin = "stop";
                            break;
                        }
                    }
                    feeder2.push(format!("BDONE {slot} {produced} {fin} 1.0"));
                    // A BSTOP that landed after the window ends must not poison the next tenant.
                    stopped.lock().unwrap().remove(&slot);
                });
            }
            "BSTOP" => {
                if let Some(s) = w.next().and_then(|x| x.parse::<usize>().ok()) {
                    b.stopped.lock().unwrap().insert(s);
                }
            }
            // STOP (an admission's cancel) and anything else: the mock's legs answer inline, so
            // there is nothing mid-decode to stop on the control lines.
            _ => {}
        }
    }
}

impl Engine for ScriptedMockEngine {
    fn max_context(&self) -> usize {
        self.max_context.load(Ordering::Relaxed)
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Relaxed)
    }

    fn load(&self) -> Result<(), EngineError> {
        self.loaded.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn unload(&self) -> Result<(), EngineError> {
        self.loaded.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn supports_lifecycle(&self) -> bool {
        true
    }

    fn supports_json_schema(&self) -> bool {
        self.json_schema_ok.load(Ordering::Relaxed)
    }

    fn generate(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        self.generate_with_logprobs(req, cancel, &mut |t, _| on_token(t), on_progress)
    }

    fn supports_logprobs(&self) -> bool {
        self.logprobs_on.load(Ordering::Relaxed)
    }

    fn supports_score(&self) -> bool {
        self.score_on.load(Ordering::Relaxed)
    }

    fn score(&self, req: &ScoreRequest) -> Result<ScoreResult, EngineError> {
        if !self.score_on.load(Ordering::Relaxed) {
            return Err(EngineError::Unsupported(
                "scoring not supported by this engine".to_string(),
            ));
        }
        let positions = req
            .continuation_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let t = mock_token_logprob(i, id, req.top_k);
                ScorePosition {
                    id,
                    logprob: t.logprob,
                    rank: 1,
                    entropy: 0.5,
                    top: t.top,
                }
            })
            .collect();
        Ok(ScoreResult {
            positions,
            prompt_ms: 1.0,
            total_ms: 1.0,
        })
    }

    fn generate_with_logprobs(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        if let Some(b) = self.batch.lock().unwrap().clone() {
            // The same eligibility the stdio engine applies: `L` lines carry no slot number,
            // so a request that wants logprobs stays on the control lines.
            if req.sampling.logprobs.is_none() {
                return self.generate_via_batch(b, req, &cancel, on_token, on_progress);
            }
        }
        let lp_k = req
            .sampling
            .logprobs
            .filter(|_| self.logprobs_on.load(Ordering::Relaxed));
        let lp_of = |step: usize, t: u32| lp_k.map(|k| mock_token_logprob(step, t, k));
        self.load()?;
        self.cancelled.store(false, Ordering::Relaxed);
        let prompt_len = req.prompt_ids.len();

        let (script_tokens, reused) = {
            let scripts = self.scripts.lock().unwrap();
            let mut st = self.state.lock().unwrap();
            st.last_prompt = req.prompt_ids.clone();
            *self.last_json_schema.lock().unwrap() = req.sampling.json_schema.clone();

            let fixed = *self.fixed_cache_n.lock().unwrap();
            let reused = if !req.sampling.cache_prompt {
                0
            } else if let Some(c) = fixed {
                c.min(prompt_len.saturating_sub(1))
            } else {
                Self::compute_reused(&req.prompt_ids, &st.cache_snapshots)
            };

            let tokens = if self.echo_prompt.load(Ordering::Relaxed) {
                let mut t = req.prompt_ids.clone();
                t.push(ID_IM_END);
                t
            } else {
                let idx = st.turn.min(scripts.len().saturating_sub(1));
                if scripts.len() > 1 {
                    st.turn += 1;
                }
                scripts.get(idx).cloned().unwrap_or_else(|| vec![ID_IM_END])
            };
            (tokens, reused)
        };

        let prompt_n = prompt_len.saturating_sub(reused);
        let prompt_ms = ((prompt_n as f64) * 0.05).max(1.0);
        on_progress(prompt_len, prompt_len);

        let mut predicted_n = 0usize;
        let mut finish_reason = "length".to_string();
        let mut generated_ids: Vec<u32> = Vec::new();
        let t0 = Instant::now();

        if req.sampling.ignore_eos {
            // Strip stop tokens and repeat filler tokens until `max_new_tokens` is reached.
            let non_eos: Vec<u32> = script_tokens
                .iter()
                .copied()
                .filter(|&id| !self.tok.is_stop_token(id) && !req.stop_tokens.contains(&id))
                .collect();
            let filler = if non_eos.is_empty() {
                vec![u32::from(b'x')]
            } else {
                non_eos
            };
            for i in 0..req.max_new_tokens {
                if cancel.load(Ordering::Relaxed) || self.cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancel".to_string();
                    break;
                }
                if !self.delay.is_zero() {
                    thread::sleep(self.delay);
                }
                let tok_id = filler[i % filler.len()];
                predicted_n += 1;
                generated_ids.push(tok_id);
                if !on_token(tok_id, lp_of(predicted_n - 1, tok_id)) {
                    finish_reason = "stop".to_string();
                    break;
                }
            }
        } else {
            for &tok_id in script_tokens.iter().take(req.max_new_tokens) {
                if cancel.load(Ordering::Relaxed) || self.cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancel".to_string();
                    break;
                }
                if !self.delay.is_zero() {
                    thread::sleep(self.delay);
                }
                if self.tok.is_stop_token(tok_id) || req.stop_tokens.contains(&tok_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                predicted_n += 1;
                generated_ids.push(tok_id);
                if !on_token(tok_id, lp_of(predicted_n - 1, tok_id)) {
                    finish_reason = "stop".to_string();
                    break;
                }
            }
        }

        if req.sampling.checkpoint {
            let mut st = self.state.lock().unwrap();
            // Save system-prompt boundary snapshot if `<|im_end|>\n` appears in prompt_ids
            if let Some(pos) = req.prompt_ids.iter().position(|&id| id == ID_IM_END) {
                let end_idx = (pos + 2).min(req.prompt_ids.len());
                Self::record_snapshot(&mut st, req.prompt_ids[..end_idx].to_vec());
            }
            if let Some(pin) = req.sampling.pin_prefix_tokens {
                if pin > 0 && pin <= req.prompt_ids.len() {
                    Self::record_snapshot(&mut st, req.prompt_ids[..pin].to_vec());
                }
            }
            Self::record_snapshot(&mut st, req.prompt_ids.clone());
            let mut full = req.prompt_ids.clone();
            full.extend_from_slice(&generated_ids);
            Self::record_snapshot(&mut st, full);
        }

        let elapsed_ms = (t0.elapsed().as_secs_f64() * 1000.0).max(1.0);
        let fixed_drafts = *self.fixed_drafts.lock().unwrap();
        let (draft_n, draft_accepted) = if let Some((dn, da)) = fixed_drafts {
            (Some(dn), Some(da))
        } else if predicted_n > 0 {
            let dn = (predicted_n / 2).max(1);
            let da = (dn * 2 / 3).max(1).min(dn);
            (Some(dn), Some(da))
        } else {
            (Some(0), Some(0))
        };

        Ok(EngineTimings {
            cache_n: reused,
            prompt_n,
            prompt_ms,
            predicted_n,
            predicted_ms: elapsed_ms,
            draft_n,
            draft_accepted,
            finish_reason,
        })
    }

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tok.encode(text, parse_special)
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    fn save_slot(&self, _slot_id: usize, path: &Path) -> Result<(usize, usize, f64), EngineError> {
        let mut st = self.state.lock().unwrap();
        let tokens = st.last_prompt.clone();
        let count = tokens.len();
        let content: String = tokens
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, content.as_bytes())
            .map_err(|e| EngineError::InvalidRequest(format!("cannot save slot: {e}")))?;
        let n_written = content.len();
        st.saved_slots.insert(path.to_path_buf(), tokens);
        Ok((count, n_written, 1.0))
    }

    fn restore_slot(
        &self,
        _slot_id: usize,
        path: &Path,
    ) -> Result<(usize, usize, f64), EngineError> {
        let raw = std::fs::read_to_string(path).map_err(|_| {
            EngineError::InvalidRequest(format!(
                "no saved session named {:?}",
                path.file_name().unwrap_or_default()
            ))
        })?;
        let n_read = raw.len();
        let tokens: Vec<u32> = raw
            .trim()
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<u32>().ok())
            .collect();
        let count = tokens.len();
        let mut st = self.state.lock().unwrap();
        st.last_prompt = tokens.clone();
        Self::record_snapshot(&mut st, tokens);
        Ok((count, n_read, 1.0))
    }

    fn erase_slot(&self, _slot_id: usize) -> Result<usize, EngineError> {
        let mut st = self.state.lock().unwrap();
        let n = st.cache_snapshots.len();
        st.cache_snapshots.clear();
        st.last_prompt.clear();
        Ok(n)
    }
}

/// Parse `SCORE <top_k> <prompt ids>|<continuation ids>` (the mock engine's side of E2).
fn parse_score_cmd(line: &str) -> Result<ScoreRequest, String> {
    let rest = line
        .strip_prefix("SCORE ")
        .ok_or_else(|| "malformed SCORE command".to_string())?;
    let (k, ids) = rest
        .trim()
        .split_once(' ')
        .ok_or_else(|| "malformed SCORE command".to_string())?;
    let top_k = k
        .parse::<usize>()
        .map_err(|_| "malformed SCORE top_k".to_string())?;
    let (p, c) = ids
        .split_once('|')
        .ok_or_else(|| "malformed SCORE ids".to_string())?;
    let parse = |s: &str| -> Vec<u32> {
        s.split(',')
            .filter_map(|x| x.trim().parse::<u32>().ok())
            .collect()
    };
    Ok(ScoreRequest {
        prompt_ids: parse(p),
        continuation_ids: parse(c),
        top_k,
    })
}

/// Every engine line after `READY` reaches its consumer through here: a dedicated reader thread
/// drains stdout into `lines` so a request takes engine output off a queue instead of blocking on
/// the pipe itself (the mirror image of the engine's own stdin reader thread). `ended` poisons the
/// bus: `recv` still yields any buffered line and only then reports the close, which is how a
/// request reading a real pipe sees it - remaining output first, EOF last.
/// A line from the engine, or the end of the stream: what the bus receives deliver. The timed
/// variants add `TimedOut` so a caller can poll its own state meanwhile (a client that left, a
/// slot to steal back).
#[derive(Debug)]
pub(crate) enum BusLine {
    Line(String),
    Ended,
    TimedOut,
}

/// One engine slot's admission state as this server sees it - the Rust shape of
/// `serve/server.py`'s `slot_busy`/`slot_claims`/`slot_releasing`/`slot_held` bookkeeping, where a
/// busy flag with no owner is a leak and gets reaped (their issue #1603).
struct SlotState {
    busy: bool,
    /// A request owns this slot, through an admission or a slot pump.
    claimed: bool,
    /// Its consumer left: the router eats the slot's remaining lines and frees the slot at its
    /// BDONE - the inline version of the Python client's background BDONE wait.
    releasing: bool,
    /// What the releasing bookkeeping runs on: the conversation the consumer left (prompt + what
    /// it saw), and the `BT` tokens the engine fed after that. At BDONE the slot holds all of it
    /// but the last token.
    held_stream: Vec<u32>,
    tail: Vec<u32>,
    /// Their #1994: the engine refused a BGEN here; the slot stays out of the picks, busy and
    /// ownerless, until the park ages out.
    park_until: Option<Instant>,
    /// Their #1603: since when this busy flag has had no owner.
    orphan_since: Option<Instant>,
    /// What the slot's conversation holds: the prompt and every token fed, all but the last.
    held: Vec<u32>,
}

struct BusInner {
    ctl: std::collections::VecDeque<String>,
    /// Per-slot queues and state, `nslots` long (empty when the engine runs no batch slots).
    q: Vec<std::collections::VecDeque<String>>,
    slot: Vec<SlotState>,
    nslots: usize,
    ended: bool,
}

/// Every engine line after `READY` reaches its consumer through here: a dedicated reader thread
/// drains stdout so a request takes engine output off a queue instead of blocking on the pipe
/// itself (the mirror image of the engine's own stdin reader thread). The thread also ROUTES: `BT`
/// and `BDONE` lines go to their batch slot's own queue - or, when a slot is releasing, are
/// swallowed right here to finish its conversation bookkeeping - and everything else to the
/// control queue. EOF poisons every queue: each drains first and only then reports the close,
/// which is how a request reading a real pipe sees it - remaining output first, EOF last.
pub(crate) struct EngineBus {
    inner: Mutex<BusInner>,
    signal: std::sync::Condvar,
    /// The stdout read error that ended the stream, if it was not a clean EOF; a receive hands it
    /// to the next consumer, the way an inline read would have raised it.
    read_err: Mutex<Option<String>>,
    orphan_slot_ns: AtomicU64,
    slot_park_ns: AtomicU64,
}

/// Which queue a receive reads: the engine's control lines, or one batch slot's own lines.
#[derive(Clone, Copy)]
enum Sel {
    Ctl,
    Slot(usize),
}

impl EngineBus {
    /// Start the thread that owns the engine's stdout; it feeds the returned bus until EOF.
    /// `nslots` is the engine's `INFO batch_slots`: with none, every line is a control line.
    pub(crate) fn spawn<R: std::io::Read + Send + 'static>(
        reader: BufReader<R>,
        nslots: usize,
    ) -> Arc<EngineBus> {
        let bus = Arc::new(EngineBus::new(nslots));
        let feeding = Arc::clone(&bus);
        let result = thread::Builder::new()
            .name("talus-engine-stdout".to_string())
            .spawn(move || {
                let mut reader = reader;
                let mut line = String::new();
                loop {
                    line.clear();
                    // No bound: the engine prints only while a request runs, and a consumer that
                    // vanished takes the process down with its `quit_and_wait`.
                    match reader.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => feeding.route(std::mem::take(&mut line)),
                        Err(e) => {
                            *feeding.read_err.lock().unwrap() = Some(format!("{e}"));
                            break;
                        }
                    }
                }
                feeding.close();
            });
        if let Err(e) = result {
            *bus.read_err.lock().unwrap() = Some(format!("stdout reader thread: {e}"));
            bus.close();
        }
        bus
    }

    /// A bus fed by hand - what the mock engine's scripted responder and the bus tests use where
    /// a pipe would be.
    pub(crate) fn manual(nslots: usize) -> (Arc<EngineBus>, BusFeeder) {
        let bus = Arc::new(EngineBus::new(nslots));
        let feeder = BusFeeder {
            bus: Arc::clone(&bus),
        };
        (bus, feeder)
    }

    fn new(nslots: usize) -> EngineBus {
        EngineBus {
            inner: Mutex::new(BusInner {
                ctl: std::collections::VecDeque::new(),
                q: (0..nslots).map(|_| Default::default()).collect(),
                slot: (0..nslots)
                    .map(|_| SlotState {
                        busy: false,
                        claimed: false,
                        releasing: false,
                        held_stream: Vec::new(),
                        tail: Vec::new(),
                        park_until: None,
                        orphan_since: None,
                        held: Vec::new(),
                    })
                    .collect(),
                nslots,
                ended: false,
            }),
            signal: std::sync::Condvar::new(),
            read_err: Mutex::new(None),
            orphan_slot_ns: AtomicU64::new(Duration::from_secs(3).as_nanos() as u64),
            slot_park_ns: AtomicU64::new(Duration::from_secs(2).as_nanos() as u64),
        }
    }

    /// Shorten the park and the reap (tests, so neither sleeps seconds; their Python original
    /// paces the same clock).
    #[allow(dead_code)] // the cfg(test) slot tests are its only callers so far
    pub(crate) fn set_slot_knobs(&self, orphan: Duration, park: Duration) {
        self.orphan_slot_ns
            .store(orphan.as_nanos() as u64, Ordering::Relaxed);
        self.slot_park_ns
            .store(park.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Deliver one engine line to its queue, swallowing the releasing slots' own lines.
    fn route(&self, line: String) {
        let mut inner = self.inner.lock().unwrap();
        if inner.nslots > 0 {
            let head = line.split_whitespace().next().unwrap_or("");
            if head == "BT" || head == "BDONE" {
                if let Some(s) = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|x| x.parse::<usize>().ok())
                {
                    if s < inner.nslots {
                        if inner.slot[s].releasing {
                            if head == "BT" {
                                if let Some(t) = line
                                    .split_whitespace()
                                    .nth(2)
                                    .and_then(|x| x.parse::<u32>().ok())
                                {
                                    inner.slot[s].tail.push(t);
                                }
                            } else {
                                let nonfinite = line
                                    .split_whitespace()
                                    .nth(3)
                                    .is_some_and(|f| f == "nonfinite");
                                Self::finish_release(&mut inner, s, nonfinite);
                            }
                        } else {
                            inner.q[s].push_back(line);
                        }
                        self.signal.notify_all();
                        return;
                    }
                }
            }
        }
        inner.ctl.push_back(line);
        self.signal.notify_all();
    }

    /// A releasing slot reached its BDONE: the slot now holds the left conversation and every
    /// token after it, but the last one (the engine's cache keeps a prompt plus all but the final
    /// token); a `nonfinite` BDONE dropped it all (their #879).
    fn finish_release(inner: &mut BusInner, s: usize, nonfinite: bool) {
        let st = &mut inner.slot[s];
        let mut all = std::mem::take(&mut st.held_stream);
        let keep = !all.is_empty() && !nonfinite;
        all.extend(std::mem::take(&mut st.tail));
        if keep && all.len() > 1 {
            all.truncate(all.len() - 1);
        } else {
            all.clear();
        }
        st.held = all;
        st.releasing = false;
        st.busy = false;
        st.claimed = false;
    }

    fn pop_sel(inner: &mut BusInner, sel: Sel) -> Option<String> {
        match sel {
            Sel::Ctl => inner.ctl.pop_front(),
            Sel::Slot(s) if s < inner.nslots => inner.q[s].pop_front(),
            Sel::Slot(_) => None,
        }
    }

    fn recv_from(&self, sel: Sel, timeout: Option<Duration>) -> Result<BusLine, EngineError> {
        let mut guard = self.inner.lock().unwrap();
        let deadline = timeout.map(|d| Instant::now() + d);
        loop {
            if let Some(line) = Self::pop_sel(&mut guard, sel) {
                return Ok(BusLine::Line(line));
            }
            if guard.ended {
                return match self.read_err.lock().unwrap().as_ref() {
                    Some(e) => Err(EngineError::EngineDied(format!(
                        "engine stdout read failed: {e}"
                    ))),
                    None => Ok(BusLine::Ended),
                };
            }
            guard = match deadline {
                Some(dl) => {
                    let now = Instant::now();
                    if now >= dl {
                        return Ok(BusLine::TimedOut);
                    }
                    self.signal.wait_timeout(guard, dl - now).unwrap().0
                }
                None => self.signal.wait(guard).unwrap(),
            };
        }
    }

    /// The next control line, `TimedOut` after `None` of waiting (`None`: wait as long as it takes).
    pub(crate) fn recv_ctl(&self, timeout: Option<Duration>) -> Result<BusLine, EngineError> {
        self.recv_from(Sel::Ctl, timeout)
    }

    /// One slot's next line: `BT`/`BDONE` the router kept for it, waiting up to `timeout`.
    pub(crate) fn recv_slot(&self, slot: usize, timeout: Duration) -> Result<BusLine, EngineError> {
        self.recv_from(Sel::Slot(slot), Some(timeout))
    }

    /// The next engine line, waiting while none has arrived. `Ok(None)` means stdout hit EOF and
    /// the queue drained - the engine is gone or going.
    pub(crate) fn recv(&self) -> Result<Option<String>, EngineError> {
        match self.recv_ctl(None)? {
            BusLine::Line(l) => Ok(Some(l)),
            BusLine::Ended => Ok(None),
            BusLine::TimedOut => unreachable!("an unbounded receive cannot time out"),
        }
    }

    /// Stdout closed: every queue reports `Ended` once it drained.
    fn close(&self) {
        self.inner.lock().unwrap().ended = true;
        self.signal.notify_all();
    }

    /// Empty one slot's queue - required before a `BGEN`, or the new request reads the lines its
    /// predecessor left behind.
    pub(crate) fn drain_slot(&self, slot: usize) {
        let mut guard = self.inner.lock().unwrap();
        if slot < guard.nslots {
            guard.q[slot].clear();
        }
    }

    /// The first free slot, claimed; `None` when all are busy. Frees on its way any parked
    /// refusal that aged out and any busy flag whose owner leaked - their #1603 gives such a flag
    /// `orphan_slot` of looking ownerless first, so a slot set moments before its owner registers
    /// is never taken.
    pub(crate) fn claim_free_slot(&self) -> Option<usize> {
        let mut guard = self.inner.lock().unwrap();
        let now = Instant::now();
        let orphan = Duration::from_nanos(self.orphan_slot_ns.load(Ordering::Relaxed));
        for st in guard.slot.iter_mut() {
            if let Some(d) = st.park_until {
                if now >= d {
                    st.park_until = None;
                    if !st.claimed && !st.releasing {
                        st.busy = false;
                    }
                }
            }
        }
        for (s, st) in guard.slot.iter_mut().enumerate() {
            if !st.busy {
                st.busy = true;
                st.claimed = true;
                st.orphan_since = None;
                return Some(s);
            }
        }
        for (s, st) in guard.slot.iter_mut().enumerate() {
            if st.busy && !st.claimed && !st.releasing && st.park_until.is_none() {
                match st.orphan_since {
                    None => st.orphan_since = Some(now),
                    Some(t) if now.duration_since(t) >= orphan => {
                        eprintln!(
                            "[talus-server] parallel: slot {s} was marked busy with no request \
                             using it; freed (issue #1603)"
                        );
                        st.busy = false;
                        st.orphan_since = None;
                        self.signal.notify_all();
                        return Some(s);
                    }
                    _ => {}
                }
            } else {
                st.orphan_since = None;
            }
        }
        None
    }

    /// Sleep up to `timeout` for a slot to come free (the claim loop's wait; the claim re-runs on
    /// wake and on deadline).
    pub(crate) fn wait_slot(&self, timeout: Duration) {
        let guard = self.inner.lock().unwrap();
        let _ = self.signal.wait_timeout(guard, timeout);
    }

    /// The engine refused a BGEN (their #1994): keep the slot busy with no owner until the park
    /// ages out; the engine's view of it needs the moment to catch up.
    pub(crate) fn unclaim_park(&self, slot: usize) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(st) = guard.slot.get_mut(slot) {
            st.claimed = false;
            st.busy = true;
            st.park_until = Some(
                Instant::now() + Duration::from_nanos(self.slot_park_ns.load(Ordering::Relaxed)),
            );
            st.orphan_since = None;
        }
    }

    /// A slot whose BDONE its own request read: free, holding `held`.
    pub(crate) fn release_done(&self, slot: usize, held: Vec<u32>) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(st) = guard.slot.get_mut(slot) {
            st.busy = false;
            st.claimed = false;
            st.releasing = false;
            st.park_until = None;
            st.orphan_since = None;
            st.held_stream.clear();
            st.tail.clear();
            st.held = held;
        }
        self.signal.notify_all();
    }

    /// A slot's consumer left (its `BSTOP` is the caller's to send): take over the bookkeeping.
    /// Any of its lines still queued are consumed here - with its BDONE among them the slot is
    /// free on the spot (`true`); otherwise the router frees it at the BDONE still to come.
    pub(crate) fn begin_release(&self, slot: usize, stream: Vec<u32>) -> bool {
        let mut guard = self.inner.lock().unwrap();
        if slot >= guard.nslots {
            return true;
        }
        let st = &mut guard.slot[slot];
        st.claimed = false;
        st.releasing = true;
        st.held_stream = stream;
        let mut done = None;
        while let Some(line) = guard.q[slot].pop_front() {
            let head = line.split_whitespace().next().unwrap_or("");
            if head == "BT" {
                if let Some(t) = line
                    .split_whitespace()
                    .nth(2)
                    .and_then(|x| x.parse::<u32>().ok())
                {
                    guard.slot[slot].tail.push(t);
                }
            } else if head == "BDONE" {
                done = Some(
                    line.split_whitespace()
                        .nth(3)
                        .is_some_and(|f| f == "nonfinite"),
                );
            }
        }
        match done {
            Some(nonfinite) => {
                Self::finish_release(&mut guard, slot, nonfinite);
                self.signal.notify_all();
                true
            }
            None => {
                self.signal.notify_all();
                false
            }
        }
    }

    /// What a slot's conversation holds, for the next turn's slot pick.
    #[allow(dead_code)] // the cfg(test) slot tests use it; stage-3's pick will too
    pub(crate) fn slot_held(&self, slot: usize) -> Vec<u32> {
        self.inner
            .lock()
            .unwrap()
            .slot
            .get(slot)
            .map(|st| st.held.clone())
            .unwrap_or_default()
    }
}

/// Hands engine lines to an [`EngineBus`] from outside: the mock engine's scripted responder, and
/// the bus's own tests, in place of a pipe.
pub(crate) struct BusFeeder {
    bus: Arc<EngineBus>,
}

impl BusFeeder {
    pub(crate) fn push(&self, line: impl Into<String>) {
        self.bus.route(line.into());
    }

    /// Stdout closed: every queue reports `Ended` once it drained.
    #[allow(dead_code)] // the cfg(test) bus tests close a hand-fed bus with it
    pub(crate) fn close(&self) {
        self.bus.close();
    }
}

struct SubprocessHandle {
    child: Child,
    stdin: Arc<Mutex<EngineStdin>>,
    /// The engine's post-READY lines, delivered by the thread [`EngineBus::spawn`] started.
    bus: Arc<EngineBus>,
    /// Keeps the engine's keeper thread alive (see [`spawn_kept`]); declared last, so it drops after `child`.
    _keeper: std::sync::mpsc::Sender<()>,
}

/// The engine's stdin; `close` sends EOF (the 27B serve loop has no QUIT command and ends at EOF).
pub(crate) struct EngineStdin(Option<ChildStdin>);

impl EngineStdin {
    fn close(&mut self) {
        self.0 = None;
    }

    fn open(&mut self) -> std::io::Result<&mut ChildStdin> {
        self.0
            .as_mut()
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::BrokenPipe))
    }
}

impl Write for EngineStdin {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.open()?.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.open()?.flush()
    }
}

/// Spawn `cmd` from a dedicated thread that lives as long as the returned sender. The child is started with
/// PR_SET_PDEATHSIG, which fires when the spawning THREAD exits, not the process: a respawn from a short-lived HTTP
/// connection thread got SIGTERM as soon as that request ended. The keeper thread ends when the sender is dropped,
/// which happens only after the child has been stopped and reaped (or when talus-server exits).
fn spawn_kept(mut cmd: Command) -> std::io::Result<(Child, std::sync::mpsc::Sender<()>)> {
    let (child_tx, child_rx) = std::sync::mpsc::channel();
    let (keep_tx, keep_rx) = std::sync::mpsc::channel::<()>();
    thread::Builder::new()
        .name("engine-keeper".to_string())
        .spawn(move || {
            let _ = child_tx.send(cmd.spawn());
            // blocks until the handle (and its sender) is dropped
            let _ = keep_rx.recv();
        })?;
    let child = child_rx
        .recv()
        .map_err(|_| std::io::Error::other("engine keeper thread ended before spawning"))??;
    Ok((child, keep_tx))
}

/// Subprocess engine that spawns an external executable speaking the Strata/Talus stdio protocol:
/// `INFO ...`, `READY <ctx> [stop]`, `GEN <max_new> [keys] <ids>`, `PP`, `RESUME`, `T`, `DONE`, `STOP`, `QUIT`.
pub struct StdioSubprocessEngine {
    exe: PathBuf,
    args: Vec<String>,
    env: Vec<(String, String)>,
    cwd: Option<PathBuf>,
    log: Option<PathBuf>,
    default_ctx: usize,
    tok: ByteTokenizer,
    max_context: usize,
    stop_token: Option<u32>,
    info: HashMap<String, String>,
    /// The running engine process; None after `unload` until the next `load` or request.
    proc: Mutex<Option<SubprocessHandle>>,
    /// Pid of the running process, 0 while unloaded.
    pid: std::sync::atomic::AtomicU32,
    /// The running process's stdin for `STOP` while a request holds `proc`.
    stdin_cancel: Mutex<Option<Arc<Mutex<EngineStdin>>>>,
    /// The engine's last stderr lines (filled by the stderr tee; read when it crashes).
    stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>>,
    crash: Mutex<CrashState>,
    /// Requests holding `proc` (0 or 1): the rest of the server's active requests are queued.
    busy: AtomicUsize,
}

/// Prefix of the error when the command could not be written: the engine never saw the request, so it can be sent again
/// to the restarted engine. (Later failures are not retried: the engine may have started generating.)
pub(crate) const STDIN_WRITE_FAILED: &str = "engine stdin write failed";
/// The pipe took the command and the engine exited at once without a word: it was killed just before, usually (seen on
/// Flash-Next, whose 91 threads take a moment to go). The client has seen nothing of this request, so it is sent once more.
/// A prompt that itself crashes the engine this fast must not reach the cap sooner for it: a death on the resend costs no
/// entry in the restart window (`free_restart`). An exit later than `RESEND_WINDOW` is an ordinary crash, with no resend.
pub(crate) const ENDED_BEFORE_ANSWERING: &str = "the engine process ended before answering";

pub(crate) const RESEND_WINDOW: Duration = Duration::from_secs(2);

fn never_reached_engine(err: Option<&EngineError>) -> bool {
    matches!(err, Some(EngineError::EngineDied(m))
        if m.starts_with(STDIN_WRITE_FAILED) || m == ENDED_BEFORE_ANSWERING)
}

/// Prefix of the error a request gets while the engine is down (the server answers 503 `engine_down`).
pub const ENGINE_DOWN: &str = "the engine is down";

/// At most this many automatic restarts within [`RESTART_WINDOW`]; one more crash and the engine stays down.
pub const MAX_RESTARTS: usize = 3;
pub const RESTART_WINDOW: Duration = Duration::from_secs(600);
/// Least time from a crash to its restart, so a crash loop does not spend the whole cap in an instant.
pub const RESTART_DELAY: Duration = Duration::from_secs(2);
/// Longest wait for a dying engine to finish exiting (its GPU memory is torn down before it can be waited for).
const EXIT_WAIT: Duration = Duration::from_secs(60);
/// Stderr lines kept for a crash report.
const STDERR_TAIL_LINES: usize = 20;

#[derive(Default)]
struct CrashState {
    crashes: u64,
    restarts_total: u64,
    /// When the automatic restarts within the window happened.
    restarts: std::collections::VecDeque<Instant>,
    /// The process crashed and has not been restarted yet.
    pending_restart: bool,
    crashed_at: Option<Instant>,
    /// An automatic restart is starting the engine.
    restarting: bool,
    /// Stopped for good (`shutdown`): nothing restarts it.
    stopped: bool,
    /// The engine died on a request's one resend: that request already paid for a restart, so the next is not counted
    /// against the cap.
    free_restart: bool,
    down: Option<String>,
    last_crash: Option<String>,
}

impl CrashState {
    /// Crash-loop cap: with MAX_RESTARTS automatic restarts already used within RESTART_WINDOW, the engine stays down.
    /// Returns the reason once down.
    fn trip_if_over_cap(&mut self) -> Option<String> {
        if let Some(ref why) = self.down {
            return Some(why.clone());
        }
        if self.free_restart {
            return None;
        }
        let now = Instant::now();
        while self
            .restarts
            .front()
            .is_some_and(|t| now.duration_since(*t) > RESTART_WINDOW)
        {
            self.restarts.pop_front();
        }
        if self.restarts.len() < MAX_RESTARTS {
            return None;
        }
        let why = format!(
            "crashed again after {} restarts in {} min, not restarting (POST /load or a server restart tries again); \
             last crash: {}",
            self.restarts.len(),
            RESTART_WINDOW.as_secs() / 60,
            self.last_crash.as_deref().unwrap_or("?")
        );
        eprintln!("[talus-server] engine down: {why}");
        self.down = Some(why.clone());
        Some(why)
    }
}

/// Holds `busy` up for one request inside the engine.
struct BusyGuard<'a>(&'a AtomicUsize);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What an engine announces before `READY`.
struct Ready {
    max_context: usize,
    stop_token: Option<u32>,
    info: HashMap<String, String>,
}

impl StdioSubprocessEngine {
    pub fn spawn(
        exe: impl AsRef<Path>,
        args: &[String],
        env: &[(String, String)],
        default_ctx: usize,
    ) -> Result<Self, EngineError> {
        Self::spawn_with_cwd(exe, args, env, None, default_ctx)
    }

    pub fn spawn_with_cwd(
        exe: impl AsRef<Path>,
        args: &[String],
        env: &[(String, String)],
        cwd: Option<&Path>,
        default_ctx: usize,
    ) -> Result<Self, EngineError> {
        Self::spawn_with_log(exe, args, env, cwd, None, default_ctx)
    }

    /// As [`Self::spawn_with_cwd`], and with `log` set the engine's stderr is also appended to that file (the run
    /// config's `"log"`, talus#58). The file is opened for append (created if missing) before the engine starts; the
    /// stderr passthrough to talus-server's own stderr stays. A log that cannot be opened is a warning, not an error.
    pub fn spawn_with_log(
        exe: impl AsRef<Path>,
        args: &[String],
        env: &[(String, String)],
        cwd: Option<&Path>,
        log: Option<&Path>,
        default_ctx: usize,
    ) -> Result<Self, EngineError> {
        let exe_path = exe.as_ref().to_path_buf();
        let stderr_tail = Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let (handle, ready) =
            Self::start(&exe_path, args, env, cwd, log, default_ctx, &stderr_tail)?;
        let mut tok = ByteTokenizer::new();
        tok.stop_tokens = ready.stop_token.into_iter().collect();
        let engine = Self {
            exe: exe_path,
            args: args.to_vec(),
            env: env.to_vec(),
            cwd: cwd.map(|p| p.to_path_buf()),
            log: log.map(|p| p.to_path_buf()),
            default_ctx,
            tok,
            max_context: ready.max_context,
            stop_token: ready.stop_token,
            info: ready.info,
            proc: Mutex::new(None),
            pid: std::sync::atomic::AtomicU32::new(0),
            stdin_cancel: Mutex::new(None),
            stderr_tail,
            crash: Mutex::new(CrashState::default()),
            busy: AtomicUsize::new(0),
        };
        engine.install(&mut engine.proc.lock().unwrap(), handle);
        Ok(engine)
    }

    /// Start the engine process and read its `INFO` lines up to `READY`.
    fn start(
        exe_path: &Path,
        args: &[String],
        env: &[(String, String)],
        cwd: Option<&Path>,
        log: Option<&Path>,
        default_ctx: usize,
        stderr_tail: &Arc<Mutex<std::collections::VecDeque<String>>>,
    ) -> Result<(SubprocessHandle, Ready), EngineError> {
        let log_file = log.and_then(|p| match open_engine_log(p) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!(
                    "[talus-server] warning: cannot open engine log {}: {e}; engine stderr goes to stderr only",
                    p.display()
                );
                None
            }
        });
        let mut cmd = Command::new(exe_path);
        cmd.args(args);
        if !args
            .iter()
            .any(|a| a == "--serve" || a == "--mock-stdio-worker")
        {
            cmd.arg("--serve");
        }
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // always piped: the tee keeps the last lines for a crash report (and copies to our stderr and the log)
            .stderr(Stdio::piped());
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            extern "C" {
                fn prctl(option: i32, arg2: usize, arg3: usize, arg4: usize, arg5: usize) -> i32;
            }
            const PR_SET_PDEATHSIG: i32 = 1;
            const SIGTERM: usize = 15;
            unsafe {
                cmd.pre_exec(|| {
                    let _ = prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0);
                    Ok(())
                });
            }
        }

        let (mut child, keeper) = spawn_kept(cmd)
            .map_err(|e| EngineError::EngineDied(format!("failed to spawn {:?}: {e}", exe_path)))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EngineError::EngineDied("missing child stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EngineError::EngineDied("missing child stdout".to_string()))?;
        if let Some(stderr) = child.stderr.take() {
            stderr_tail.lock().unwrap().clear();
            tee_engine_stderr(stderr, log_file, Arc::clone(stderr_tail));
        }
        let mut reader = BufReader::new(stdout);

        let mut max_context = default_ctx.max(1);
        let mut stop_token = None;
        let mut info = HashMap::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| EngineError::EngineDied(format!("reading engine stdout: {e}")))?;
            if n == 0 {
                return Err(EngineError::EngineDied(
                    "engine exited before READY".to_string(),
                ));
            }
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("INFO ") {
                for kv in rest.split_whitespace() {
                    if let Some((k, v)) = kv.split_once('=') {
                        info.insert(k.to_string(), v.to_string());
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("READY") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if let Some(first) = parts.first() {
                    if let Ok(ctx) = first.parse::<usize>() {
                        max_context = ctx;
                    }
                }
                if let Some(second) = parts.get(1) {
                    if let Ok(st) = second.parse::<u32>() {
                        stop_token = Some(st);
                    }
                }
                break;
            } else if let Some(rest) = trimmed.strip_prefix("ERR") {
                return Err(EngineError::EngineDied(rest.trim().to_string()));
            }
        }

        let nslots = info
            .get("batch_slots")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        Ok((
            SubprocessHandle {
                child,
                stdin: Arc::new(Mutex::new(EngineStdin(Some(stdin)))),
                bus: EngineBus::spawn(reader, nslots),
                _keeper: keeper,
            },
            Ready {
                max_context,
                stop_token,
                info,
            },
        ))
    }

    /// Make `handle` the running process.
    fn install(&self, slot: &mut Option<SubprocessHandle>, handle: SubprocessHandle) {
        *self.stdin_cancel.lock().unwrap() = Some(handle.stdin.clone());
        self.pid.store(handle.child.id(), Ordering::SeqCst);
        *slot = Some(handle);
    }

    /// The running process, started again first if it was unloaded. A restart must come back with the context and
    /// stop token it started with: the server sized its requests and tokenizer to them.
    fn running<'a>(
        &self,
        slot: &'a mut Option<SubprocessHandle>,
    ) -> Result<&'a mut SubprocessHandle, EngineError> {
        self.finish_exit_if_dying(slot);
        self.reap_if_crashed(slot);
        if slot.is_none() {
            let delay = {
                let mut c = self.crash.lock().unwrap();
                if let Some(ref why) = c.down {
                    return Err(EngineError::EngineDied(format!("{ENGINE_DOWN}: {why}")));
                }
                if !c.pending_restart {
                    Duration::ZERO
                } else {
                    if let Some(why) = c.trip_if_over_cap() {
                        return Err(EngineError::EngineDied(format!("{ENGINE_DOWN}: {why}")));
                    }
                    c.crashed_at.map_or(Duration::ZERO, |t| {
                        RESTART_DELAY.saturating_sub(t.elapsed())
                    })
                }
            };
            std::thread::sleep(delay);
            let was_restart = {
                let mut c = self.crash.lock().unwrap();
                let was = c.pending_restart;
                if was {
                    if !std::mem::take(&mut c.free_restart) {
                        c.restarts.push_back(Instant::now());
                    }
                    c.restarts_total += 1;
                    c.pending_restart = false;
                    c.restarting = true;
                    eprintln!(
                        "[talus-server] restarting the engine after a crash ({} of {} within {} min)",
                        c.restarts.len(),
                        MAX_RESTARTS,
                        RESTART_WINDOW.as_secs() / 60
                    );
                }
                was
            };
            let started = Self::start(
                &self.exe,
                &self.args,
                &self.env,
                self.cwd.as_deref(),
                self.log.as_deref(),
                self.default_ctx,
                &self.stderr_tail,
            )
            .and_then(|(mut handle, ready)| {
                if ready.max_context != self.max_context || ready.stop_token != self.stop_token {
                    stop_process(&mut handle, Duration::from_secs(15));
                    return Err(EngineError::EngineDied(format!(
                        "the engine came back with READY {} {:?}, it started with READY {} {:?}",
                        ready.max_context, ready.stop_token, self.max_context, self.stop_token
                    )));
                }
                Ok(handle)
            });
            {
                let mut c = self.crash.lock().unwrap();
                c.restarting = false;
                if let (true, Err(e)) = (was_restart, started.as_ref()) {
                    // a restart that does not come up counts against the cap like a crash
                    eprintln!("[talus-server] engine restart failed: {e}");
                    c.last_crash = Some(format!("restart failed: {e}"));
                    c.pending_restart = true;
                    c.crashed_at = Some(Instant::now());
                    let _ = c.trip_if_over_cap();
                }
            }
            let handle = started?;
            self.install(slot, handle);
        }
        Ok(slot.as_mut().expect("installed above"))
    }

    /// The process leader is gone and the rest of the process is being torn down: a killed engine shows this at once,
    /// but can only be waited for when its GPU memory is free, seconds later. Its stdin is already closed.
    fn dying(&self) -> bool {
        let pid = self.pid.load(Ordering::SeqCst);
        pid != 0
            && std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
                stat.rsplit(") ")
                    .next()
                    .is_some_and(|rest| rest.starts_with(['Z', 'X']))
            })
    }

    /// A request that finds the engine dying waits for its exit instead of writing to a dead pipe (it can then restart
    /// it and be served; it used to get a 500).
    fn finish_exit_if_dying(&self, slot: &mut Option<SubprocessHandle>) {
        if self.dying() {
            self.wait_for_exit(slot, EXIT_WAIT);
        }
    }

    fn wait_for_exit(&self, slot: &mut Option<SubprocessHandle>, limit: Duration) {
        let deadline = Instant::now() + limit;
        while let Some(h) = slot.as_mut() {
            if !matches!(h.child.try_wait(), Ok(None)) || Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// The engine process exited on its own: log it (exit status, last stderr lines), forget the handle, and mark it
    /// for a restart on the next request. A live process (or none) is left alone.
    fn reap_if_crashed(&self, slot: &mut Option<SubprocessHandle>) {
        let status = match slot.as_mut().map(|h| h.child.try_wait()) {
            Some(Ok(Some(st))) => st,
            _ => return,
        };
        let pid = slot.as_ref().map(|h| h.child.id()).unwrap_or(0);
        *slot = None;
        *self.stdin_cancel.lock().unwrap() = None;
        self.pid.store(0, Ordering::SeqCst);
        // noted before the pause below, so /health never reads "unloaded" in between
        self.crash.lock().unwrap().pending_restart = true;
        // the stderr tee may still be draining the pipe: give it a moment
        std::thread::sleep(Duration::from_millis(100));
        let tail: Vec<String> = self.stderr_tail.lock().unwrap().iter().cloned().collect();
        let report = format!(
            "engine pid {pid} exited ({status}) at unix {}; last stderr: {}",
            crate::json::unix_timestamp_secs(),
            if tail.is_empty() {
                "(none)".to_string()
            } else {
                tail.join(" | ")
            }
        );
        eprintln!("[talus-server] engine crash: {report}");
        let mut c = self.crash.lock().unwrap();
        c.crashes += 1;
        c.last_crash = Some(report);
        c.pending_restart = true;
        c.crashed_at = Some(Instant::now());
        let _ = c.trip_if_over_cap();
    }

    pub fn stop_token(&self) -> Option<u32> {
        self.stop_token
    }

    pub fn cwd(&self) -> Option<&Path> {
        self.cwd.as_deref()
    }

    pub fn info(&self) -> &HashMap<String, String> {
        &self.info
    }

    /// Batch slots the engine will run (`INFO batch_slots=`, absent when the engine runs one request at a
    /// time): the number of conversations the `BGEN` protocol can hold open at once.
    pub fn batch_slots(&self) -> usize {
        self.info
            .get("batch_slots")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// Whether the engine keeps a finished slot's conversation cached (`INFO slot_cache=`), so a follow-up
    /// request can be re-admitted to it without re-prompting.
    pub fn slot_cache(&self) -> bool {
        self.info
            .get("slot_cache")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Pipelined slot groups the engine resolved (`INFO batch_groups=`, 1 when there is no split):
    /// the slot order that spreads requests across groups, and whether a window spans a slot's group.
    pub fn batch_groups(&self) -> usize {
        self.info
            .get("batch_groups")
            .and_then(|v| v.parse().ok())
            .filter(|g| *g > 0)
            .unwrap_or(1)
    }

    pub fn exe(&self) -> &Path {
        &self.exe
    }

    pub fn args(&self) -> &[String] {
        &self.args
    }

    pub fn env(&self) -> &[(String, String)] {
        &self.env
    }
}

impl StdioSubprocessEngine {
    fn score_on(
        handle: &mut SubprocessHandle,
        req: &ScoreRequest,
    ) -> Result<ScoreResult, EngineError> {
        let csv = |ids: &[u32]| {
            ids.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let cmd = format!(
            "SCORE {} {}|{}\n",
            req.top_k,
            csv(&req.prompt_ids),
            csv(&req.continuation_ids)
        );
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(cmd.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("{STDIN_WRITE_FAILED}: {e}")))?;
        }
        let mut positions = Vec::with_capacity(req.continuation_ids.len());
        loop {
            let Some(line) = handle.bus.recv()? else {
                return Err(EngineError::EngineDied(
                    "the engine process ended unexpectedly".to_string(),
                ));
            };
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("S ") {
                let p = ScorePosition::parse_s(rest).ok_or_else(|| {
                    EngineError::EngineDied(format!("malformed S line: {trimmed}"))
                })?;
                positions.push(p);
            } else if let Some(rest) = trimmed.strip_prefix("DONE ") {
                let f: Vec<&str> = rest.split_whitespace().collect();
                let prompt_ms = f.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                let total_ms = f.get(3).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                if f.get(4) == Some(&"nonfinite") {
                    return Err(EngineError::NonFinite(
                        "the engine encountered non-finite logits".to_string(),
                    ));
                }
                return Ok(ScoreResult {
                    positions,
                    prompt_ms,
                    total_ms,
                });
            } else if let Some(rest) = trimmed.strip_prefix("ERR") {
                return Err(EngineError::InvalidRequest(rest.trim().to_string()));
            }
            // progress lines (PP ...) and anything else are ignored
        }
    }

    /// The body a `GEN` or `BGEN` carries: the sampling keys, the native tower's framing keys
    /// and `@raw` frame, and the prompt's comma list. The engine rewrites a `BGEN` into a `GEN`
    /// with this body untouched, so both requests build it the same way.
    fn gen_body(&self, req: &GenerateRequest) -> (String, Option<Arc<Vec<u8>>>, String) {
        let csv_ids: String = req
            .prompt_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        // pictures: GENI's embeddings file is the first space-split token without an `=`, so it
        // goes after the sampling keys and before the ids (serve/server.py's head construction).
        // The native `@raw` form: the per-picture byte lengths frame the frame behind the line,
        // and the server's size predictions ride for the tower to check - both keys before the
        // emb-path token, which is the engine's own line order.
        let mut keys = req.sampling.to_gen_keys();
        let mut frame = None;
        if let (Some(f), true) = (&req.raw_frame, req.embeddings.as_deref() == Some("@raw")) {
            keys += &format!(
                " img_bytes={}",
                req.lens
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            if let Some(px) = &req.img_px {
                keys += &format!(" img_px={px}");
            }
            frame = Some(f.clone());
        }
        (keys, frame, csv_ids)
    }

    fn generate_on(
        &self,
        handle: &mut SubprocessHandle,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        let (keys, frame, csv_ids) = self.gen_body(req);
        let cmd = match &req.embeddings {
            Some(path) => match &req.img_keys {
                Some(k) => format!(
                    "GENI {}{} img_keys={} {} {}\n",
                    req.max_new_tokens, keys, k, path, csv_ids
                ),
                None => format!("GENI {}{} {} {}\n", req.max_new_tokens, keys, path, csv_ids),
            },
            None => format!("GEN {}{} {}\n", req.max_new_tokens, keys, csv_ids),
        };
        {
            let mut stdin = handle.stdin.lock().unwrap();
            let wrote = match frame {
                // the frame follows the line at once, no newline between: the engine reads
                // exactly the img_bytes lengths, so control lines cannot hide in it
                Some(f) => stdin
                    .write_all(cmd.as_bytes())
                    .and_then(|_| stdin.write_all(&f)),
                None => stdin.write_all(cmd.as_bytes()),
            };
            wrote
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("{STDIN_WRITE_FAILED}: {e}")))?;
        }

        let mut reused = 0usize;
        let mut stop_sent = false;
        let mut pending_lp: Option<TokenLogprob> = None;

        // nothing from the engine for this request yet
        let mut answered = false;
        let written_at = Instant::now();
        loop {
            if cancel.load(Ordering::Relaxed) && !stop_sent {
                if let Ok(mut stdin) = handle.stdin.lock() {
                    let _ = stdin.write_all(b"STOP\n");
                    let _ = stdin.flush();
                }
                stop_sent = true;
            }
            let Some(line) = handle.bus.recv()? else {
                // no word from the engine right after the write: it was on its way out when the command arrived
                let resend = !answered && written_at.elapsed() <= RESEND_WINDOW;
                return Err(EngineError::EngineDied(if resend {
                    ENDED_BEFORE_ANSWERING.to_string()
                } else {
                    "the engine process ended unexpectedly".to_string()
                }));
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            answered = true;
            if let Some(rest) = trimmed.strip_prefix("L ") {
                pending_lp = TokenLogprob::parse_l(rest);
            } else if let Some(rest) = trimmed.strip_prefix("RESUME ") {
                reused = rest.trim().parse::<usize>().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("PP ") {
                let f: Vec<&str> = rest.split_whitespace().collect();
                if f.len() >= 2 {
                    let read = f[0].parse::<usize>().unwrap_or(0);
                    let total = f[1].parse::<usize>().unwrap_or(0);
                    on_progress(reused + read, total);
                }
            } else if let Some(rest) = trimmed.strip_prefix("T ") {
                let lp = pending_lp.take();
                if let Ok(tok_id) = rest.trim().parse::<u32>() {
                    if (self.tok.is_stop_token(tok_id) || req.stop_tokens.contains(&tok_id))
                        && !req.sampling.ignore_eos
                    {
                        if !stop_sent {
                            if let Ok(mut stdin) = handle.stdin.lock() {
                                let _ = stdin.write_all(b"STOP\n");
                                let _ = stdin.flush();
                            }
                            stop_sent = true;
                        }
                    } else if !stop_sent && !on_token(tok_id, lp) {
                        if let Ok(mut stdin) = handle.stdin.lock() {
                            let _ = stdin.write_all(b"STOP\n");
                            let _ = stdin.flush();
                        }
                        stop_sent = true;
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("DONE ") {
                let f: Vec<&str> = rest.split_whitespace().collect();
                if f.len() < 5 {
                    return Err(EngineError::EngineDied(format!(
                        "malformed DONE line: {trimmed}"
                    )));
                }
                let predicted_n = f[0].parse::<usize>().unwrap_or(0);
                let prompt_tokens = f[1].parse::<usize>().unwrap_or(req.prompt_ids.len());
                let prompt_ms = f[2].parse::<f64>().unwrap_or(0.0);
                let predicted_ms = f[3].parse::<f64>().unwrap_or(0.0);
                let finish = f[4].to_string();
                if finish == "nonfinite" {
                    return Err(EngineError::NonFinite(
                        "the engine encountered non-finite logits".to_string(),
                    ));
                }
                let (draft_accepted, draft_n, done_reused) = if f.len() >= 8 {
                    (
                        Some(f[5].parse::<usize>().unwrap_or(0)),
                        Some(f[6].parse::<usize>().unwrap_or(0)),
                        f[7].parse::<usize>().unwrap_or(reused),
                    )
                } else {
                    (None, None, reused)
                };
                let cache_n = done_reused.min(prompt_tokens);
                let prompt_n = prompt_tokens.saturating_sub(cache_n);
                return Ok(EngineTimings {
                    cache_n,
                    prompt_n,
                    prompt_ms,
                    predicted_n,
                    predicted_ms,
                    draft_n,
                    draft_accepted,
                    finish_reason: if stop_sent && finish == "cancel" {
                        if cancel.load(Ordering::Relaxed) {
                            "cancel".to_string()
                        } else {
                            "stop".to_string()
                        }
                    } else {
                        finish
                    },
                });
            } else if let Some(rest) = trimmed.strip_prefix("ERR") {
                return Err(EngineError::InvalidRequest(rest.trim().to_string()));
            }
        }
    }

    /// Whether a request can run on a batch slot: the engine has slots, and its features ride
    /// the slot wires. `L` (logprob) lines carry no slot number, so a request that wants them
    /// keeps the control lines to itself - the solo path, correct beside decoding slots because
    /// it holds `proc` for its whole exchange, which blocks every admission behind it.
    fn slot_eligible(&self, req: &GenerateRequest) -> bool {
        self.batch_slots() > 0 && req.sampling.logprobs.is_none()
    }

    /// The handshake of one slot leg: claim a slot, drain, `BGEN`/`BGENI`, read the answer on
    /// the control lines. The caller holds `proc` across it - the handshake IS the engine's
    /// control-line handoff - and by the time this returns the lines are free again, the slot
    /// (if it continued) decoding on its own.
    fn admit_slot(
        &self,
        bus: &Arc<EngineBus>,
        sink: &StdinSink,
        req: &GenerateRequest,
        cancel: &Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Leg, EngineError> {
        let (keys, frame, csv) = self.gen_body(req);
        let cmd = |slot: usize| match (&req.embeddings, &req.img_keys) {
            (Some(path), Some(k)) => format!(
                "BGENI {slot} {}{keys} img_keys={k} {path} {csv}",
                req.max_new_tokens
            ),
            (Some(path), None) => {
                format!("BGENI {slot} {}{keys} {path} {csv}", req.max_new_tokens)
            }
            _ => format!("BGEN {slot} {}{keys} {csv}", req.max_new_tokens),
        };
        let client = BatchClient { bus, sink };
        let is_stop = |t: u32| self.tok.is_stop_token(t) || req.stop_tokens.contains(&t);
        client.admit(
            &cmd,
            frame.as_deref().map(|v| v.as_slice()),
            &req.prompt_ids,
            cancel,
            &mut |t, lp| {
                if is_stop(t) && !req.sampling.ignore_eos {
                    false // the solo path suppresses the stop token too, and never yields it
                } else {
                    on_token(t, lp)
                }
            },
            on_progress,
        )
    }

    /// Read the slot's leg to its `BDONE` and assemble the figures. The pump holds neither
    /// `proc` nor the control lines: the next request admits while this one streams.
    fn pump_slot(
        &self,
        bus: &Arc<EngineBus>,
        sink: &StdinSink,
        leg: Leg,
        req: &GenerateRequest,
        cancel: &Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
    ) -> Result<EngineTimings, EngineError> {
        let client = BatchClient { bus, sink };
        let emitted: Vec<u32> = leg.admission.into_iter().collect();
        let is_stop = |t: u32| self.tok.is_stop_token(t) || req.stop_tokens.contains(&t);
        let res = client.pump(leg.slot, &req.prompt_ids, &emitted, cancel, &mut |t| {
            if is_stop(t) && !req.sampling.ignore_eos {
                false
            } else {
                on_token(t, None)
            }
        })?;
        self.slot_timings(&leg, res.produced, res.decode_ms, &res.finish, cancel)
    }

    /// A slot leg's figures in the solo path's `EngineTimings` shape: the admission's `DONE`
    /// carries the prompt side (and its own decode time), the `BDONE` the rest. The engine's
    /// `produced` counts the stop token as the solo `DONE` does, and `BDONE` carries no draft
    /// counters, so a slot leg reports none.
    fn slot_timings(
        &self,
        leg: &Leg,
        produced: usize,
        decode_ms: f64,
        finish: &str,
        cancel: &Arc<AtomicBool>,
    ) -> Result<EngineTimings, EngineError> {
        if finish == "nonfinite" {
            return Err(EngineError::NonFinite(
                "the engine encountered non-finite logits".to_string(),
            ));
        }
        let cache_n = leg.done_reused.unwrap_or(leg.reused).min(leg.prompt_n);
        Ok(EngineTimings {
            cache_n,
            prompt_n: leg.prompt_n.saturating_sub(cache_n),
            prompt_ms: leg.prompt_ms,
            predicted_n: produced,
            predicted_ms: leg.predicted_ms + decode_ms,
            draft_n: None,
            draft_accepted: None,
            finish_reason: if finish == "cancel" || finish == "abort" {
                // The solo distinction: a request the client left reports `cancel`; one that
                // ended on its stop token reports `stop`.
                if cancel.load(Ordering::Relaxed) {
                    "cancel".to_string()
                } else {
                    "stop".to_string()
                }
            } else {
                finish.to_string()
            },
        })
    }

    /// The resend's outcome: if the engine died again, that death costs no entry in the restart window.
    fn after_resend(&self, slot: &mut Option<SubprocessHandle>, err: Option<&EngineError>) {
        if matches!(err, Some(EngineError::EngineDied(_))) {
            self.crash.lock().unwrap().free_restart = true;
        }
        self.settle_after_error(slot, err);
        if slot.is_some() {
            // the process is alive (a bad reply, not a death): no later crash inherits the pass
            self.crash.lock().unwrap().free_restart = false;
        }
    }

    /// A request failed with `EngineDied`: the engine usually just exited, but its stdout closes a moment before it
    /// can be waited for. Wait briefly, so the crash is recorded now and the next request restarts it.
    fn settle_after_error(&self, slot: &mut Option<SubprocessHandle>, err: Option<&EngineError>) {
        if !matches!(err, Some(EngineError::EngineDied(_))) {
            return;
        }
        let limit = if self.dying() || never_reached_engine(err) {
            EXIT_WAIT
        } else {
            Duration::from_secs(2)
        };
        self.wait_for_exit(slot, limit);
        self.reap_if_crashed(slot);
    }
}

impl Engine for StdioSubprocessEngine {
    fn max_context(&self) -> usize {
        self.max_context
    }

    fn shutdown(&self) {
        // engines that ignore QUIT while idle (GLM was seen to) are killed after a short grace; the GPU driver frees
        // their memory either way
        self.quit_and_wait(Duration::from_secs(15));
    }

    fn build_version(&self) -> Option<&str> {
        self.info.get("version").map(|s| s.as_str())
    }

    fn pid(&self) -> Option<u32> {
        Some(self.pid.load(Ordering::SeqCst)).filter(|&p| p != 0)
    }

    fn is_loaded(&self) -> bool {
        self.pid.load(Ordering::SeqCst) != 0
    }

    /// Start the engine if it is not running. Also the manual override after the crash-loop cap: it clears the
    /// "down" state and the restart window.
    fn load(&self) -> Result<(), EngineError> {
        let mut slot = self.proc.lock().unwrap();
        {
            let mut c = self.crash.lock().unwrap();
            if c.down.take().is_some() {
                eprintln!(
                    "[talus-server] engine was down after a crash loop; POST /load starts it again"
                );
                c.restarts.clear();
                c.pending_restart = false;
            }
        }
        self.running(&mut slot).map(|_| ())
    }

    fn supervise(&self) {
        // a request or an unload holds the lock: it deals with a crash itself
        let Ok(mut slot) = self.proc.try_lock() else {
            return;
        };
        self.reap_if_crashed(&mut slot);
        let due = {
            let c = self.crash.lock().unwrap();
            c.pending_restart
                && c.down.is_none()
                && !c.stopped
                && c.crashed_at.is_none_or(|t| t.elapsed() >= RESTART_DELAY)
        };
        if due {
            if let Err(e) = self.running(&mut slot) {
                eprintln!("[talus-server] engine not restarted: {e}");
            }
        }
    }

    fn health(&self) -> EngineHealth {
        // notice a crash while idle (the lock is free then); a busy engine is alive
        if let Ok(mut slot) = self.proc.try_lock() {
            self.reap_if_crashed(&mut slot);
        }
        let dying = self.dying();
        let c = self.crash.lock().unwrap();
        EngineHealth {
            crashes: c.crashes,
            restarts: c.restarts_total,
            down: c.down.clone(),
            last_crash: c.last_crash.clone(),
            restart_pending: (c.pending_restart || dying) && c.down.is_none(),
            restarting: c.restarting,
        }
    }

    fn in_flight(&self) -> Option<usize> {
        Some(self.busy.load(Ordering::SeqCst))
    }

    /// Stop the engine process (QUIT, then a kill after a grace period) and wait for it, so its GPU memory is free.
    /// Waits for a running request (it holds the process lock). The next request or `load` starts it again.
    fn unload(&self) -> Result<(), EngineError> {
        let mut slot = self.proc.lock().unwrap();
        self.reap_if_crashed(&mut slot);
        // a deliberate unload: no automatic restart is owed
        {
            let mut c = self.crash.lock().unwrap();
            c.pending_restart = false;
            c.free_restart = false;
        }
        if let Some(mut handle) = slot.take() {
            *self.stdin_cancel.lock().unwrap() = None;
            self.pid.store(0, Ordering::SeqCst);
            stop_process(&mut handle, Duration::from_secs(15));
        }
        Ok(())
    }

    fn supports_lifecycle(&self) -> bool {
        true
    }

    fn info_value(&self, key: &str) -> Option<String> {
        self.info.get(key).cloned()
    }

    fn supports_json_schema(&self) -> bool {
        self.info
            .get("grammar")
            .is_some_and(|v| v.split(',').any(|g| g == "json_schema"))
    }

    fn generate(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        self.generate_with_logprobs(req, cancel, &mut |t, _| on_token(t), on_progress)
    }

    fn supports_logprobs(&self) -> bool {
        self.info.get("logprobs").map(|v| v == "1").unwrap_or(false)
    }

    fn supports_score(&self) -> bool {
        self.info.get("score").map(|v| v == "1").unwrap_or(false)
    }

    fn score(&self, req: &ScoreRequest) -> Result<ScoreResult, EngineError> {
        if !self.supports_score() {
            return Err(EngineError::Unsupported(
                "scoring not supported by this engine".to_string(),
            ));
        }
        let mut slot = self.proc.lock().unwrap();
        self.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = BusyGuard(&self.busy);
        let handle = self.running(&mut slot)?;
        let mut res = Self::score_on(handle, req);
        self.settle_after_error(&mut slot, res.as_ref().err());
        if never_reached_engine(res.as_ref().err()) {
            let handle = self.running(&mut slot)?;
            res = Self::score_on(handle, req);
            self.after_resend(&mut slot, res.as_ref().err());
        }
        res
    }

    fn generate_with_logprobs(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        let mut slot = self.proc.lock().unwrap();
        self.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = BusyGuard(&self.busy);
        if !self.slot_eligible(req) {
            let handle = self.running(&mut slot)?;
            let mut res = self.generate_on(handle, req, cancel.clone(), on_token, on_progress);
            self.settle_after_error(&mut slot, res.as_ref().err());
            if never_reached_engine(res.as_ref().err()) {
                let handle = self.running(&mut slot)?;
                res = self.generate_on(handle, req, cancel, on_token, on_progress);
                self.after_resend(&mut slot, res.as_ref().err());
            }
            return res;
        }
        // The handshake runs with `proc` held (the control lines are the engine's handoff); the
        // pump then runs without it, which is the whole point of the slot: the next request
        // admits while this one decodes.
        let mut bus;
        let mut sink;
        {
            let handle = self.running(&mut slot)?;
            bus = handle.bus.clone();
            sink = StdinSink(handle.stdin.clone());
        }
        let mut res = self.admit_slot(&bus, &sink, req, &cancel, on_token, on_progress);
        if never_reached_engine(res.as_ref().err()) {
            self.settle_after_error(&mut slot, res.as_ref().err());
            let handle = self.running(&mut slot)?;
            bus = handle.bus.clone();
            sink = StdinSink(handle.stdin.clone());
            res = self.admit_slot(&bus, &sink, req, &cancel, on_token, on_progress);
            self.after_resend(&mut slot, res.as_ref().err());
        }
        let leg = match res {
            Ok(leg) => leg,
            Err(e) => {
                self.settle_after_error(&mut slot, Some(&e));
                return Err(e);
            }
        };
        if leg.cancelled {
            let produced = leg.admission.into_iter().count();
            return self.slot_timings(
                &leg,
                produced,
                0.0,
                if cancel.load(Ordering::Relaxed) {
                    "cancel"
                } else {
                    "stop"
                },
                &cancel,
            );
        }
        if !leg.cont {
            let produced = leg.admission.into_iter().count();
            let finish = leg.finish.clone();
            return self.slot_timings(&leg, produced, 0.0, &finish, &cancel);
        }
        // Let the engine go: the pump reads only the slot's routed lines.
        drop(slot);
        let res = self.pump_slot(&bus, &sink, leg, req, &cancel, on_token);
        if matches!(res.as_ref().err(), Some(EngineError::EngineDied(_))) {
            let mut slot = self.proc.lock().unwrap();
            self.settle_after_error(&mut slot, res.as_ref().err());
        }
        res
    }

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tok.encode(text, parse_special)
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    fn cancel(&self) {
        let stdin = self.stdin_cancel.lock().ok().and_then(|s| s.clone());
        if let Some(stdin) = stdin {
            if let Ok(mut stdin) = stdin.lock() {
                let _ = stdin.write_all(b"STOP\n");
                let _ = stdin.flush();
            }
        }
    }

    fn save_slot(&self, _slot_id: usize, path: &Path) -> Result<(usize, usize, f64), EngineError> {
        let mut slot = self.proc.lock().unwrap();
        self.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = BusyGuard(&self.busy);
        let handle = self.running(&mut slot)?;
        let cmd = format!("SAVE {}\n", path.display());
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(cmd.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("SAVE write failed: {e}")))?;
        }
        let (n, bytes, ms) = read_session_reply_on(&mut || handle.bus.recv(), "SAVE", "SAVED")?;
        let written = bytes.unwrap_or_else(|| {
            std::fs::metadata(path)
                .map(|m| m.len() as usize)
                .unwrap_or(0)
        });
        Ok((n, written, ms.unwrap_or(1.0)))
    }

    fn restore_slot(
        &self,
        _slot_id: usize,
        path: &Path,
    ) -> Result<(usize, usize, f64), EngineError> {
        let mut slot = self.proc.lock().unwrap();
        self.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = BusyGuard(&self.busy);
        let handle = self.running(&mut slot)?;
        let cmd = format!("RESTORE {}\n", path.display());
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(cmd.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("RESTORE write failed: {e}")))?;
        }
        let (n, bytes, ms) =
            read_session_reply_on(&mut || handle.bus.recv(), "RESTORE", "RESTORED")?;
        let read_bytes = bytes.unwrap_or_else(|| {
            std::fs::metadata(path)
                .map(|m| m.len() as usize)
                .unwrap_or(0)
        });
        Ok((n, read_bytes, ms.unwrap_or(1.0)))
    }

    fn erase_slot(&self, _slot_id: usize) -> Result<usize, EngineError> {
        let mut slot = self.proc.lock().unwrap();
        self.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = BusyGuard(&self.busy);
        let handle = self.running(&mut slot)?;
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(b"CLEAR\n")
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("CLEAR write failed: {e}")))?;
        }
        // an unanswered CLEAR (EOF, or a word we do not know) reads as zero cleared, as the inline read did
        match handle.bus.recv()? {
            Some(line) => {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("CLEARED") {
                    Ok(rest.trim().parse::<usize>().unwrap_or(0))
                } else {
                    Ok(0)
                }
            }
            None => Ok(0),
        }
    }

    fn vgeom(&self) -> Result<(i64, i64, i64), EngineError> {
        let mut slot = self.proc.lock().unwrap();
        self.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = BusyGuard(&self.busy);
        let handle = self.running(&mut slot)?;
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(b"VGEOM\n")
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("VGEOM write failed: {e}")))?;
        }
        // progress lines can sit between the command and the answer; only VGEOM/ERR settle it
        loop {
            let Some(line) = handle.bus.recv()? else {
                return Err(EngineError::EngineDied(
                    "engine exited during VGEOM".to_string(),
                ));
            };
            let trimmed = line.trim();
            let mut f = trimmed.split_whitespace();
            match f.next().unwrap_or("") {
                "VGEOM" => {
                    let (Some(align), Some(min_t), Some(max_t)) = (
                        f.next().and_then(|x| x.parse().ok()),
                        f.next().and_then(|x| x.parse().ok()),
                        f.next().and_then(|x| x.parse().ok()),
                    ) else {
                        return Err(EngineError::InvalidRequest(format!(
                            "bad VGEOM answer: {trimmed}"
                        )));
                    };
                    return Ok((align, min_t, max_t));
                }
                "ERR" => return Err(EngineError::InvalidRequest(trimmed[3..].trim().to_string())),
                _ => continue,
            }
        }
    }
}

/// Read the engine's reply to `SAVE` / `RESTORE` (`ok` = `SAVED` / `RESTORED`, followed by the token count and
/// optionally the bytes and milliseconds). Engines with a session manager (Laguna, Flash-Next) print progress lines
/// first (`SESSION done total`, `SWAIT ...`) and answer failures with `SERR kind 0 why`; others answer `ERR why`.
/// `recv` yields engine lines, `Ok(None)` at their end.
fn read_session_reply_on(
    recv: &mut dyn FnMut() -> Result<Option<String>, EngineError>,
    cmd: &str,
    ok: &str,
) -> Result<(usize, Option<usize>, Option<f64>), EngineError> {
    loop {
        let Some(line) = recv()? else {
            return Err(EngineError::EngineDied(format!(
                "engine exited during {cmd}"
            )));
        };
        let trimmed = line.trim();
        let word = trimmed.split_whitespace().next().unwrap_or("");
        if word.is_empty() || word == "SESSION" || word == "SWAIT" {
            continue;
        }
        let rest = trimmed[word.len()..].trim();
        if word == ok {
            let mut f = rest.split_whitespace();
            let n = f.next().and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
            let bytes = f.next().and_then(|x| x.parse::<usize>().ok());
            let ms = f.next().and_then(|x| x.parse::<f64>().ok());
            return Ok((n, bytes, ms));
        }
        if word == "ERR" {
            return Err(EngineError::InvalidRequest(rest.to_string()));
        }
        if word == "SERR" {
            // `SERR <kind> <tokens> <why>`
            let mut f = rest.splitn(3, ' ');
            let kind = f.next().unwrap_or("io");
            let _ = f.next();
            let why = f.next().unwrap_or("").trim();
            return Err(EngineError::InvalidRequest(format!("{kind}: {why}")));
        }
        return Err(EngineError::EngineDied(format!(
            "unexpected {cmd} reply: {trimmed}"
        )));
    }
}

impl StdioSubprocessEngine {
    /// Send QUIT, wait up to `grace` for the engine to exit, then kill it. Idempotent.
    pub fn quit_and_wait(&self, grace: Duration) {
        if let Ok(mut slot) = self.proc.lock() {
            {
                let mut c = self.crash.lock().unwrap();
                c.stopped = true;
                c.pending_restart = false;
            }
            if let Some(mut handle) = slot.take() {
                self.pid.store(0, Ordering::SeqCst);
                stop_process(&mut handle, grace);
            }
        }
    }
}

/// Send QUIT, wait up to `grace` for the process to exit, then kill it and reap it.
fn stop_process(handle: &mut SubprocessHandle, grace: Duration) {
    if let Ok(mut stdin) = handle.stdin.lock() {
        let _ = stdin.write_all(b"QUIT\n");
        let _ = stdin.flush();
        // and EOF: the 27B serve loop has no QUIT command (it waited out the whole grace period, then the kill)
        stdin.close();
    }
    let deadline = Instant::now() + grace;
    loop {
        match handle.child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) if Instant::now() >= deadline => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    let _ = handle.child.kill();
    let _ = handle.child.wait();
}

/// Open the engine log for append, creating it (and its parent directory) if missing.
fn open_engine_log(path: &Path) -> std::io::Result<std::fs::File> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
}

/// Copy the engine's stderr to talus-server's stderr and to `log` until the engine closes it (exits). Write errors
/// on either side are ignored so the engine never blocks on a full stderr pipe.
fn tee_engine_stderr(
    mut src: std::process::ChildStderr,
    mut log: Option<std::fs::File>,
    tail: Arc<Mutex<std::collections::VecDeque<String>>>,
) {
    let spawned = thread::Builder::new()
        .name("engine-stderr".to_string())
        .spawn(move || {
            use std::io::Read;
            let mut buf = [0u8; 8192];
            let mut line: Vec<u8> = Vec::new();
            loop {
                match src.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if let Some(f) = log.as_mut() {
                            let _ = f.write_all(&buf[..n]);
                        }
                        let mut err = std::io::stderr().lock();
                        let _ = err.write_all(&buf[..n]);
                        let _ = err.flush();
                        drop(err);
                        for &b in &buf[..n] {
                            if b == b'\n' {
                                push_tail(&tail, &line);
                                line.clear();
                            } else if line.len() < 512 {
                                line.push(b);
                            }
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
            if !line.is_empty() {
                push_tail(&tail, &line);
            }
        });
    if let Err(e) = spawned {
        eprintln!("[talus-server] warning: cannot start the engine log thread: {e}");
    }
}

fn push_tail(tail: &Mutex<std::collections::VecDeque<String>>, line: &[u8]) {
    let text = String::from_utf8_lossy(line).trim_end().to_string();
    if text.is_empty() {
        return;
    }
    let mut t = tail.lock().unwrap();
    if t.len() == STDERR_TAIL_LINES {
        t.pop_front();
    }
    t.push_back(text);
}

impl Drop for StdioSubprocessEngine {
    fn drop(&mut self) {
        if let Some(handle) = self.proc.get_mut().ok().and_then(Option::as_mut) {
            if let Ok(mut stdin) = handle.stdin.lock() {
                let _ = stdin.write_all(b"QUIT\n");
                let _ = stdin.flush();
            }
            let _ = handle.child.kill();
            let _ = handle.child.wait();
        }
    }
}

#[cfg(test)]
mod engine_bus_tests {
    use super::{EngineBus, EngineError};
    use std::io::BufReader;

    #[test]
    fn lines_arrive_in_order_and_eof_repeats() {
        let data = b"T 1\nT 2\nDONE 1 1 0 0 stop\n".to_vec();
        let bus = EngineBus::spawn(BufReader::new(std::io::Cursor::new(data)), 0);
        assert_eq!(bus.recv().unwrap().as_deref(), Some("T 1\n"));
        assert_eq!(bus.recv().unwrap().as_deref(), Some("T 2\n"));
        assert_eq!(bus.recv().unwrap().as_deref(), Some("DONE 1 1 0 0 stop\n"));
        assert_eq!(bus.recv().unwrap(), None);
        assert_eq!(bus.recv().unwrap(), None);
    }

    /// Hands out its bytes once, then fails every read: the engine pipe dying mid-stream.
    struct BreaksAfterFirstRead {
        data: Vec<u8>,
        delivered: bool,
    }

    impl std::io::Read for BreaksAfterFirstRead {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.delivered {
                return Err(std::io::Error::other("simulated pipe break"));
            }
            self.delivered = true;
            let n = buf.len().min(self.data.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            Ok(n)
        }
    }

    #[test]
    /// Output the reader thread had already buffered is delivered before the failure it read after:
    /// a consumer sees the engine's last words, then the error, never the other way around.
    fn buffered_lines_precede_a_later_read_error() {
        let data = b"T 1\nT 2\n".to_vec();
        let inner = BreaksAfterFirstRead {
            data,
            delivered: false,
        };
        let bus = EngineBus::spawn(BufReader::new(inner), 0);
        assert_eq!(bus.recv().unwrap().as_deref(), Some("T 1\n"));
        assert_eq!(bus.recv().unwrap().as_deref(), Some("T 2\n"));
        assert!(matches!(bus.recv(), Err(EngineError::EngineDied(_))));
    }
}

#[cfg(test)]
mod session_reply_tests {
    use super::{read_session_reply_on, EngineError};

    /// A `recv` source scripted from engine output: one line per call, then EOF.
    fn line_source(script: &str) -> impl FnMut() -> Result<Option<String>, EngineError> {
        let mut lines = script.lines().map(str::to_string).collect::<Vec<_>>();
        lines.reverse();
        move || {
            let line = lines.pop();
            Ok(line)
        }
    }

    #[test]
    fn plain_reply() {
        assert_eq!(
            read_session_reply_on(&mut line_source("SAVED 120\n"), "SAVE", "SAVED").unwrap(),
            (120, None, None)
        );
    }

    #[test]
    fn progress_lines_are_skipped_and_fields_parsed() {
        // Laguna: SESSION progress, then `SAVED <tokens> <bytes> <ms>`; Flash-Next adds SWAIT.
        let mut recv = line_source(
            "SESSION 1 4\nSWAIT commit\nSESSION 4 4\n\nRESTORED 9024 441253888 113.3\n",
        );
        let (n, b, ms) = read_session_reply_on(&mut recv, "RESTORE", "RESTORED").unwrap();
        assert_eq!((n, b), (9024, Some(441253888)));
        assert!((ms.unwrap() - 113.3).abs() < 1e-9);
    }

    #[test]
    fn serr_and_err_are_request_errors() {
        let mut recv = line_source("SESSION 0 4\nSERR invalid 0 nothing to save\n");
        match read_session_reply_on(&mut recv, "SAVE", "SAVED") {
            Err(EngineError::InvalidRequest(m)) => assert_eq!(m, "invalid: nothing to save"),
            other => panic!("{other:?}"),
        }
        let mut recv = line_source("ERR no such file\n");
        match read_session_reply_on(&mut recv, "RESTORE", "RESTORED") {
            Err(EngineError::InvalidRequest(m)) => assert_eq!(m, "no such file"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn eof_and_garbage_are_engine_errors() {
        assert!(matches!(
            read_session_reply_on(&mut line_source("SESSION 1 2\n"), "SAVE", "SAVED"),
            Err(EngineError::EngineDied(_))
        ));
        assert!(matches!(
            read_session_reply_on(&mut line_source("T 5\n"), "SAVE", "SAVED"),
            Err(EngineError::EngineDied(_))
        ));
    }
}

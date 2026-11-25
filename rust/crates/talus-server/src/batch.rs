//! The batch-slot client: a request admitted into one of the engine's `--batch` slots rather
//! than owning its control lines. This is the Rust shape of `serve/server.py`'s
//! `generate_batched`: `BGEN` into a free slot, the admission answered on the control lines
//! (`DONE` + `BADM`), the run read from the slot's own `BT`/`BDONE` lines off the bus.
//!
//! Two invariants from the engine's line discipline ([`crate::engine::EngineBus`] routes its
//! output): the slot queue is drained before a `BGEN` is sent, and a `BSTOP` has no ack - the
//! slot is free only at its `BDONE`, so a consumer that leaves marks the slot releasing and the
//! router finishes the bookkeeping when the lines arrive.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::{
    BusLine, EngineBus, EngineError, EngineStdin, TokenLogprob, ENDED_BEFORE_ANSWERING,
    RESEND_WINDOW, STDIN_WRITE_FAILED,
};

/// Something engine commands can be written to: the real engine's stdin, or a mock's script
/// channel. A frame is the `@raw` picture bytes: they follow the line at once, no newline
/// between, exactly as the solo path writes them.
pub(crate) trait BatchSink: Send + Sync {
    fn send(&self, line: &str, frame: Option<&[u8]>) -> Result<(), EngineError>;
}

/// The real engine's stdin.
#[allow(dead_code)] // the solo/slot choice that uses it lands with the admission wiring
pub(crate) struct StdinSink(pub(crate) Arc<Mutex<EngineStdin>>);

impl BatchSink for StdinSink {
    fn send(&self, line: &str, frame: Option<&[u8]>) -> Result<(), EngineError> {
        let mut stdin = self.0.lock().unwrap();
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| match frame {
                Some(f) => stdin.write_all(f),
                None => Ok(()),
            })
            .and_then(|_| stdin.flush())
            .map_err(|e| EngineError::EngineDied(format!("{STDIN_WRITE_FAILED}: {e}")))
    }
}

/// A mock engine's command channel: one command per message, no frames.
pub(crate) struct ScriptSink(pub(crate) mpsc::Sender<String>);

impl BatchSink for ScriptSink {
    fn send(&self, line: &str, _frame: Option<&[u8]>) -> Result<(), EngineError> {
        self.0.send(line.to_string()).map_err(|_| {
            EngineError::EngineDied("the mock engine's command channel closed".to_string())
        })
    }
}

/// What an admission produced: the engine read the prompt and answered `BADM`.
#[derive(Debug, Default)]
pub(crate) struct Leg {
    pub(crate) slot: usize,
    /// `false`: everything finished in the admission (`max_new` was spent on the token the
    /// prompt's read produced) - the slot is done, no slot lines follow.
    pub(crate) cont: bool,
    /// The admission's own token. The slot's run starts here, and the slot's `BDONE` counts it.
    pub(crate) admission: Option<u32>,
    pub(crate) admission_lp: Option<TokenLogprob>,
    pub(crate) reused: usize,
    pub(crate) prompt_n: usize,
    pub(crate) prompt_ms: f64,
    pub(crate) predicted_ms: f64,
    /// The `DONE` line's own reuse count (its eighth field), which the engine's conversation
    /// cache reports for the leg; `None` on an engine that does not send it, where `reused`
    /// (the `RESUME` line) is the figure - the same fallback the solo path applies.
    pub(crate) done_reused: Option<usize>,
    /// The finish word of the admission's `DONE` line (`stop`/`length`/`cancel`/`nonfinite`/empty).
    pub(crate) finish: String,
    /// The request was cancelled before its admission ended: the `STOP` went out, and a slot
    /// that continued was `BSTOP`ped and released here.
    pub(crate) cancelled: bool,
}

/// A slot leg's end, from its `BDONE` - or, when the consumer left, the last numbers it has.
#[derive(Debug)]
pub(crate) struct Pumped {
    pub(crate) produced: usize,
    pub(crate) finish: String,
    pub(crate) decode_ms: f64,
}

/// How often one request tries another slot after the engine refused a BGEN (server.py's #1994).
const BGEN_REFUSALS_MAX: usize = 6;
/// How long the control queue may stay quiet before a cancelled admission's `STOP` goes out
/// (server.py's 0.5 s poll).
const CTL_POLL: Duration = Duration::from_millis(500);
/// The slot queue's heartbeat cadence while pumping, when to check for a cancel (server.py's 10 s).
const SLOT_TICK: Duration = Duration::from_secs(10);
/// The wait for a leaving slot's `BDONE` right behind its stop token (server.py's #1869).
const BDONE_AFTER_EOS: Duration = Duration::from_secs(2);
/// A slot that never frees is a hung engine, but the claim loop reaps leaked busy flags on its
/// own way (server.py's 1 s re-check).
const CLAIM_WAIT: Duration = Duration::from_secs(1);

/// The admission/pump state machine over one engine's bus. The caller owns the engine's control
/// lines: it holds them across `admit` (which sends on them and reads them), and gives them up
/// before `pump` - a slot decodes without them, and a waiting request that kept them would
/// deadlock the admission of the one slot it waits for (server.py's ENGINE_REVIEW finding 1).
pub(crate) struct BatchClient<'a> {
    pub(crate) bus: &'a EngineBus,
    pub(crate) sink: &'a dyn BatchSink,
    /// The engine's pipelined slot groups (`INFO batch_groups=`, 1 without): drives the slot
    /// choice's group economics, as upstream's picker weighs them.
    pub(crate) groups: usize,
}

impl<'a> BatchClient<'a> {
    /// Send a `BGEN`/`BGENI` into a free slot and read its admission: the engine runs the prompt
    /// read as a one-token `GEN` on the control lines, answering `PP` progress, the admission's
    /// `T` token, its `DONE` figures, and `BADM` - `cont` says whether the slot decodes on. An
    /// `ERR BGEN:` is the engine's momentary view, not the request's fault: the slot is parked
    /// and another tried, up to `BGEN_REFUSALS_MAX` times before the request fails. `cmd(slot)`
    /// builds the whole command line for the slot picked; `prompt` is the conversation's head,
    /// kept for a leaving slot's bookkeeping.
    pub(crate) fn admit(
        &self,
        cmd: &dyn Fn(usize) -> String,
        frame: Option<&[u8]>,
        prompt: &[u32],
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32, Option<TokenLogprob>) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Leg, EngineError> {
        let mut refused = 0usize;
        'outer: loop {
            let slot = self.pick_slot(prompt, cancel)?;
            // Required before the send: otherwise the new request reads what its predecessor
            // left in the queue.
            self.bus.drain_slot(slot);
            let line = cmd(slot);
            self.sink.send(&line, frame)?;
            let written_at = Instant::now();
            let mut answered = false;
            let mut leg = Leg {
                slot,
                ..Default::default()
            };
            let mut pending_lp: Option<TokenLogprob> = None;
            let mut stop_sent = false;
            let mut aborted = false; // on_token false: STOP goes out, the leg is read to its end
            loop {
                match self.bus.recv_ctl(Some(CTL_POLL))? {
                    BusLine::TimedOut => {
                        if (cancel.load(Ordering::Relaxed) || aborted) && !stop_sent {
                            self.sink.send("STOP", None)?;
                            stop_sent = true;
                        }
                    }
                    BusLine::Ended => {
                        // The solo path's rule, kept: a death right behind the write means the
                        // engine never saw the request, and the caller sends it once more.
                        let resend = !answered && written_at.elapsed() <= RESEND_WINDOW;
                        return Err(EngineError::EngineDied(if resend {
                            ENDED_BEFORE_ANSWERING.to_string()
                        } else {
                            "the engine process ended unexpectedly".to_string()
                        }));
                    }
                    BusLine::Line(raw) => {
                        let trimmed = raw.trim();
                        answered = true;
                        if let Some(rest) = trimmed.strip_prefix("T ") {
                            if let Ok(tok) = rest.trim().parse::<u32>() {
                                let lp = pending_lp.take();
                                leg.admission = Some(tok);
                                leg.admission_lp = lp.clone();
                                if !on_token(tok, lp) {
                                    aborted = true;
                                }
                            }
                        } else if let Some(rest) = trimmed.strip_prefix("L ") {
                            pending_lp = TokenLogprob::parse_l(rest);
                        } else if let Some(rest) = trimmed.strip_prefix("RESUME ") {
                            leg.reused = rest.trim().parse::<usize>().unwrap_or(0);
                        } else if let Some(rest) = trimmed.strip_prefix("PP ") {
                            let f: Vec<&str> = rest.split_whitespace().collect();
                            if let (Some(read), Some(total)) = (
                                f.first().and_then(|x| x.parse::<usize>().ok()),
                                f.get(1).and_then(|x| x.parse::<usize>().ok()),
                            ) {
                                on_progress(leg.reused + read, total);
                            }
                        } else if let Some(rest) = trimmed.strip_prefix("DONE ") {
                            let f: Vec<&str> = rest.split_whitespace().collect();
                            leg.prompt_n =
                                f.get(1).and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
                            leg.prompt_ms =
                                f.get(2).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
                            leg.predicted_ms =
                                f.get(3).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
                            leg.finish = f.get(4).unwrap_or(&"").to_string();
                            leg.done_reused = f.get(7).and_then(|x| x.parse::<usize>().ok());
                        } else if let Some(rest) = trimmed.strip_prefix("BADM ") {
                            let f: Vec<&str> = rest.split_whitespace().collect();
                            leg.cont = f.get(1).is_some_and(|x| *x == "1");
                            if (cancel.load(Ordering::Relaxed) || aborted) && leg.cont {
                                // Their finally: a continuing slot its engine just agreed to,
                                // for a request nobody waits for - BSTOP it and let the router
                                // finish what the lines say.
                                self.sink.send(&format!("BSTOP {slot}"), None)?;
                                let mut stream = prompt.to_vec();
                                stream.extend(leg.admission.iter().copied());
                                self.bus.begin_release(slot, stream);
                                leg.cancelled = true;
                            }
                            if !leg.cont {
                                // The whole leg rode the admission: no `BDONE` will come to free
                                // the slot, so the hold is settled here - prompt plus the
                                // admission's token, but the last one.
                                self.bus.release_done(
                                    slot,
                                    slot_holds(prompt, leg.admission.as_slice(), &leg.finish),
                                );
                            }
                            return Ok(leg);
                        } else if let Some(rest) = trimmed.strip_prefix("ERR") {
                            let why = rest.trim();
                            if why.starts_with("BGEN:")
                                && refused < BGEN_REFUSALS_MAX
                                && !cancel.load(Ordering::Relaxed)
                                && !aborted
                            {
                                self.bus.unclaim_park(slot);
                                refused += 1;
                                continue 'outer;
                            }
                            // The engine refused it and is idle: no DONE follows, and no slot
                            // was taken (their #1059).
                            self.bus.release_done(slot, Vec::new());
                            return Err(EngineError::InvalidRequest(why.to_string()));
                        }
                        // INFO/WARN lines interleave; none of them answers an admission.
                    }
                }
            }
        }
    }

    /// Read a slot's own lines to its `BDONE`. `emitted` is what the engine has so far - the
    /// admission's token first, then the `BT` tokens; the slot's `BDONE` counts all of it. A
    /// set `cancel` sends `BSTOP` once and absorbs the rest until the `BDONE` (there is no ack,
    /// and the slot is free only at its `BDONE`); an `on_token` false stops the yielding here
    /// and reads for the `BDONE` up to `BDONE_AFTER_EOS` - the engine prints it in the same
    /// window as the stop token (their #1869) - falling back to releasing-inline when it does
    /// not come.
    pub(crate) fn pump(
        &self,
        slot: usize,
        prompt: &[u32],
        emitted: &[u32],
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Pumped, EngineError> {
        let mut stop_sent = false;
        let mut emitted: Vec<u32> = emitted.to_vec();
        loop {
            match self.bus.recv_slot(slot, SLOT_TICK)? {
                BusLine::TimedOut => {
                    if cancel.load(Ordering::Relaxed) && !stop_sent {
                        self.sink.send(&format!("BSTOP {slot}"), None)?;
                        stop_sent = true;
                    }
                }
                BusLine::Ended => return Err(engine_gone()),
                BusLine::Line(raw) => {
                    let trimmed = raw.trim();
                    if let Some(rest) = trimmed.strip_prefix("BT ") {
                        let mut f = rest.split_whitespace();
                        let (Some(_), Some(tok)) = (
                            f.next().and_then(|x| x.parse::<usize>().ok()),
                            f.next().and_then(|x| x.parse::<u32>().ok()),
                        ) else {
                            continue;
                        };
                        emitted.push(tok);
                        if cancel.load(Ordering::Relaxed) {
                            if !stop_sent {
                                self.sink.send(&format!("BSTOP {slot}"), None)?;
                                stop_sent = true;
                            }
                            continue; // counted, not yielded: absorb until the BDONE
                        }
                        if !on_token(tok) {
                            if !stop_sent {
                                self.sink.send(&format!("BSTOP {slot}"), None)?;
                            }
                            return self.leave_slot(slot, prompt, &emitted, "abort");
                        }
                    } else if let Some(rest) = trimmed.strip_prefix("BDONE ") {
                        // `BDONE <slot> <produced> <finish> <decode_ms>`
                        let f: Vec<&str> = rest.split_whitespace().collect();
                        let produced = f.get(1).and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
                        let finish = f.get(2).unwrap_or(&"").to_string();
                        let decode_ms = f.get(3).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
                        self.bus
                            .release_done(slot, slot_holds(prompt, &emitted, &finish));
                        return Ok(Pumped {
                            produced,
                            finish,
                            decode_ms,
                        });
                    }
                    // INFO/WARN interleaving is not a slot's answer either.
                }
            }
        }
    }

    /// A consumer left mid-slot (its `BSTOP` was the caller's to send): read briefly for the
    /// `BDONE` that usually follows in the same window, take its figures, and free the slot with
    /// the conversation it holds; otherwise hand the slot to the router, which frees it at its
    /// own BDONE exactly as it swallows any other releasing slot's lines.
    fn leave_slot(
        &self,
        slot: usize,
        prompt: &[u32],
        emitted: &[u32],
        why: &str,
    ) -> Result<Pumped, EngineError> {
        let mut tail = emitted.to_vec();
        let deadline = Instant::now() + BDONE_AFTER_EOS;
        loop {
            let left = deadline
                .saturating_duration_since(Instant::now())
                .max(Duration::from_millis(1));
            match self.bus.recv_slot(slot, left)? {
                BusLine::TimedOut => break,
                BusLine::Ended => break, // the engine is gone; a restart rebuilds the slots anyway
                BusLine::Line(raw) => {
                    let trimmed = raw.trim();
                    if let Some(rest) = trimmed.strip_prefix("BT ") {
                        let mut f = rest.split_whitespace();
                        let _ = f.next();
                        if let Some(t) = f.next().and_then(|x| x.parse::<u32>().ok()) {
                            tail.push(t);
                        }
                        continue;
                    }
                    if let Some(rest) = trimmed.strip_prefix("BDONE ") {
                        let f: Vec<&str> = rest.split_whitespace().collect();
                        let produced = f.get(1).and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
                        let finish = f.get(2).copied().unwrap_or("").to_string();
                        let decode_ms = f.get(3).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
                        self.bus
                            .release_done(slot, slot_holds(prompt, &tail, &finish));
                        return Ok(Pumped {
                            produced,
                            finish,
                            decode_ms,
                        });
                    }
                }
            }
        }
        let mut stream = prompt.to_vec();
        stream.extend(tail.iter().copied());
        self.bus.begin_release(slot, stream);
        Ok(Pumped {
            produced: emitted.len(),
            finish: why.to_string(),
            decode_ms: 0.0,
        })
    }

    /// The first free slot, waiting for one. Their #1603: a busy flag with no owner is a leak,
    /// and the claim loop is where such a flag gets reaped - a frozen flag otherwise waits
    /// forever on a slot nothing will free.
    fn pick_slot(&self, prompt: &[u32], cancel: &AtomicBool) -> Result<usize, EngineError> {
        loop {
            if let Some(s) = self.bus.claim_slot_with(|free, busy, total| {
                let gs = if self.groups > 1 && total.is_multiple_of(self.groups) {
                    total / self.groups
                } else {
                    0
                };
                pick_slot_index(free, busy, total, self.groups, gs, prompt)
            }) {
                return Ok(s);
            }
            if cancel.load(Ordering::Relaxed) {
                return Err(EngineError::Cancelled);
            }
            self.bus.wait_slot(CLAIM_WAIT);
        }
    }
}

/// The port of `serve/server.py`'s slot choice: `free` is the free slots (index, held
/// conversation, last-use stamp) in index order, `busy` the busy indices, `total` the slot
/// count. `slot_groups`/`slot_gs` carry upstream's `slot_groups`/`slot_gs`: the group count,
/// and the pipeline stride when the engine actually steps a group's slots in one window -
/// `slot_gs == 0` means groups the engine does not pipeline (what `talus-server` always passes
/// today, since it learns groups only from `INFO batch_groups=`, which reports pipelined ones).
/// It is pure so the group-mode rules - which a non-grouped engine never exercises - stay
/// unit-testable.
///
/// Three regimes, as upstream. Pipelined groups: the conversation a pipelined slot holds is
/// not a conversation cache, so prefix scoring gives way to window economics - fill a hole
/// below a busy slot of the same group (it costs no extra row), then spread over the groups in
/// stage-first order. Non-pipelined groups keep prefix scoring but demand a prefix worth
/// unbalancing the groups for (512 tokens), then spread by fewest-busy-group. No groups at
/// all: any held prefix wins, else the free slot that holds nothing, oldest use first - so the
/// conversations other slots hold stay for their next turns.
pub(crate) fn pick_slot_index(
    free: &[(usize, &[u32], u64)],
    busy: &[usize],
    total: usize,
    slot_groups: usize,
    slot_gs: usize,
    prompt: &[u32],
) -> Option<usize> {
    if free.is_empty() {
        return None;
    }
    // A group count that does not divide the slot count is no grouping at all, as upstream's
    // start-up check demands; `stride` then degenerates and the branch flag says non-pipelined.
    let groups = if slot_groups > 1 && total.is_multiple_of(slot_groups) {
        slot_groups
    } else {
        1
    };
    let stride = total / groups;
    let pipelined = groups > 1 && slot_gs > 0;
    // The order that spreads across the pipeline's groups first: all of stage 0, then stage 1:
    // 0, gs, 2gs.., then 1, gs+1, ... For one group this is just the natural order.
    let order: Vec<usize> = if groups > 1 {
        (0..stride)
            .flat_map(|t| (0..groups).map(move |g| g * stride + t))
            .collect()
    } else {
        (0..total).collect()
    };
    let in_order = |b: usize| order.iter().position(|x| *x == b).unwrap_or(usize::MAX);
    let free: Vec<(usize, &[u32], u64)> = {
        let mut f: Vec<(usize, &[u32], u64)> = free.to_vec();
        f.sort_by_key(|(b, _, _)| in_order(*b));
        f
    };
    if pipelined {
        // A pipelined group's window spans its slots up to its last busy one, so a free slot
        // below a busy one of its group costs no extra row.
        let fills_hole = |b: usize| {
            let g0 = b - b % stride;
            (b + 1..g0 + stride).any(|x| busy.contains(&x))
        };
        return free
            .iter()
            .min_by_key(|(b, _, _)| (usize::from(!fills_hole(*b)), in_order(*b)))
            .map(|(b, _, _)| *b);
    }
    let held_prefix = |h: &[u32]| -> usize {
        if !h.is_empty() && h.len() < prompt.len() && prompt[..h.len()] == *h {
            h.len()
        } else {
            0
        }
    };
    let best = free
        .iter()
        .max_by_key(|(_, held, _)| held_prefix(held))
        .copied()
        .expect("free is not empty");
    if held_prefix(best.1) >= if groups > 1 { 512 } else { 1 } {
        return Some(best.0);
    }
    let busy_in = |b: usize| -> usize {
        if groups <= 1 {
            return 0;
        }
        let g = b / stride;
        busy.iter().filter(|x| **x / stride == g).count()
    };
    free.iter()
        .min_by_key(|(b, held, used)| (busy_in(*b), usize::from(!held.is_empty()), *used))
        .map(|(b, _, _)| *b)
}

/// What a slot holds after feeding `emitted` on top of `prompt`: all of it but the last token
/// (the engine's cache keeps a prompt plus all but the final token) - and nothing at all on a
/// `nonfinite` BDONE, which dropped the slot's state (their #879).
fn slot_holds(prompt: &[u32], emitted: &[u32], finish: &str) -> Vec<u32> {
    if prompt.is_empty() || finish == "nonfinite" {
        return Vec::new();
    }
    let mut all = prompt.to_vec();
    all.extend(emitted.iter().copied());
    all.pop();
    all
}

fn engine_gone() -> EngineError {
    EngineError::EngineDied("the engine process ended unexpectedly".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineTimings, GenerateRequest, ScriptedMockEngine};
    use crate::tokenizer::{ByteTokenizer, ID_IM_END};

    fn request(max_new: usize) -> GenerateRequest {
        GenerateRequest {
            prompt_ids: (100..116u32).collect(),
            max_new_tokens: max_new,
            sampling: crate::engine::SamplingParams::default(),
            stop_tokens: Vec::new(),
            stop_strings: Vec::new(),
            embeddings: None,
            img_keys: None,
            raw_frame: None,
            lens: Vec::new(),
            img_px: None,
        }
    }

    fn mock(n: usize) -> ScriptedMockEngine {
        let m = ScriptedMockEngine::new(ByteTokenizer::default());
        m.set_batch_slots(n);
        m
    }

    fn run(m: &ScriptedMockEngine, max_new: usize) -> (Vec<u32>, EngineTimings) {
        run_req(m, request(max_new))
    }

    fn run_req(m: &ScriptedMockEngine, req: GenerateRequest) -> (Vec<u32>, EngineTimings) {
        let mut toks = Vec::new();
        let timings = m
            .generate(
                &req,
                Arc::new(AtomicBool::new(false)),
                &mut |t| {
                    toks.push(t);
                    true
                },
                &mut |_, _| {},
            )
            .expect("the batch request completes");
        (toks, timings)
    }

    /// A plain claim of any free slot: the picker with no preference, for tests that only
    /// want to know a slot was released.
    fn take_free(bus: &crate::engine::EngineBus) -> Option<usize> {
        bus.claim_slot_with(|free, _, _| free.first().map(|(s, _, _)| *s))
    }

    fn held(m: &ScriptedMockEngine, slot: usize) -> Vec<u32> {
        m.batch_bus().unwrap().slot_held(slot)
    }

    #[test]
    fn admit_happy_path_runs_the_slot_to_its_bdone() {
        let m = mock(2);
        m.set_batch_scripts(vec![vec![10, 11, 12, 13]]);
        let (toks, t) = run(&m, 4);
        assert_eq!(toks, vec![10, 11, 12, 13]);
        assert_eq!(t.finish_reason, "length");
        assert_eq!(t.predicted_n, 4);
        // The slot keeps the conversation but the last token - the engine's cache rule.
        assert_eq!(held(&m, 0).len(), request(4).prompt_ids.len() + 3);
        // Free again for the next request.
        assert_eq!(take_free(&m.batch_bus().unwrap()), Some(0));
    }

    #[test]
    fn slot_stop_token_is_suppressed_and_the_slot_holds_it_not() {
        let m = mock(2);
        m.set_batch_scripts(vec![vec![10, 11, ID_IM_END]]);
        let (toks, t) = run(&m, 8);
        assert_eq!(toks, vec![10, 11]); // the stop token rides `BDONE`, never the stream
        assert_eq!(t.finish_reason, "stop");
        assert_eq!(t.predicted_n, 2);
        assert_eq!(held(&m, 0).len(), request(8).prompt_ids.len() + 2);
        assert_eq!(take_free(&m.batch_bus().unwrap()), Some(0));
    }

    #[test]
    fn cont_zero_admission_frees_the_slot_itself() {
        let m = mock(2);
        m.set_batch_scripts(vec![vec![ID_IM_END]]);
        let (toks, t) = run(&m, 4);
        assert!(toks.is_empty()); // a stop admission is suppressed on the way in
        assert_eq!(t.finish_reason, "stop");
        assert_eq!(t.predicted_n, 0);
        // prompt + one token, less the last: the prompt is what the slot still holds.
        assert_eq!(held(&m, 0), request(4).prompt_ids);
        assert_eq!(take_free(&m.batch_bus().unwrap()), Some(0));
    }

    #[test]
    fn bgen_refusal_parks_the_slot_and_retries_another() {
        let m = mock(2);
        m.batch_bus().unwrap().set_slot_knobs(
            std::time::Duration::from_secs(3),
            std::time::Duration::from_millis(20),
        );
        m.seed_batch_refusals(1);
        m.set_batch_scripts(vec![vec![10, 11, ID_IM_END]]);
        let (toks, t) = run(&m, 8);
        assert_eq!(toks, vec![10, 11]);
        assert_eq!(t.finish_reason, "stop");
        // The refusal parked slot 0; the retry ran on slot 1, and slot 0 holds nothing.
        assert_eq!(held(&m, 1).len(), request(8).prompt_ids.len() + 2);
        assert!(held(&m, 0).is_empty());
    }

    #[test]
    fn consumer_abort_hands_the_slot_to_the_router_and_reports_cancel() {
        let m = mock(2);
        m.set_batch_pace_ms(30);
        m.set_batch_scripts(vec![vec![10, 11, 12, 13, 14, 15, 16]]);
        let mut toks = Vec::new();
        let t = m
            .generate(
                &request(20),
                Arc::new(AtomicBool::new(false)),
                &mut |t| {
                    toks.push(t);
                    t != 11 // yield 10 and 11, then leave - the pump `BSTOP`s and lets go
                },
                &mut |_, _| {},
            )
            .expect("the aborted request still ends in a BDONE");
        assert_eq!(toks, vec![10, 11]);
        assert_eq!(t.finish_reason, "cancel");
        // Freed without ever reading the slot's remaining lines: the router took them.
        assert_eq!(take_free(&m.batch_bus().unwrap()), Some(0));
    }

    #[test]
    fn router_finishes_a_releasing_slot_at_its_bdone() {
        let (bus, feeder) = EngineBus::manual(1);
        assert_eq!(take_free(&bus), Some(0));
        // The consumer left with 1,2,3 seen; the engine is still feeding the slot.
        assert!(!bus.begin_release(0, vec![1, 2, 3]));
        feeder.push("BT 0 4");
        assert!(
            bus.slot_held(0).is_empty(),
            "held only settles at the BDONE"
        );
        feeder.push("BDONE 0 4 stop 1.0");
        // 1,2,3 carried over, 4 swallowed from the wire, less the last one.
        assert_eq!(bus.slot_held(0), vec![1, 2, 3]);
        assert_eq!(take_free(&bus), Some(0));
    }

    #[test]
    fn router_drops_everything_on_a_nonfinite_bdone() {
        let (bus, feeder) = EngineBus::manual(1);
        assert_eq!(take_free(&bus), Some(0));
        assert!(!bus.begin_release(0, vec![1, 2]));
        feeder.push("BT 0 3");
        feeder.push("BDONE 0 3 nonfinite 1.0");
        assert!(bus.slot_held(0).is_empty()); // their #879: the slot's state is gone
        assert_eq!(take_free(&bus), Some(0));
    }

    #[test]
    fn logprobs_requests_stay_off_the_slots() {
        // The mock gates like the stdio engine: `L` lines carry no slot number, so a request
        // that wants them never admits - and a batch script left waiting proves no slot ran.
        let m = mock(2);
        m.set_batch_scripts(vec![vec![10, 11, ID_IM_END]]);
        let mut req = request(8);
        req.sampling.logprobs = Some(3);
        let (toks, _) = run_req(&m, req);
        assert!(!toks.is_empty(), "the solo mock path answered it");
        assert!(held(&m, 0).is_empty() && held(&m, 1).is_empty());
    }

    #[test]
    fn eof_poisons_after_the_queues_drain() {
        let (bus, feeder) = EngineBus::manual(1);
        feeder.push("DONE 1 4 2.0 1.0 stop 0 0 0 0 0 0 0 0.0 0 0");
        feeder.close();
        assert!(matches!(
            bus.recv_ctl(Some(std::time::Duration::from_millis(50))),
            Ok(BusLine::Line(_))
        ));
        assert!(matches!(
            bus.recv_ctl(Some(std::time::Duration::from_millis(50))),
            Ok(BusLine::Ended)
        ));
    }

    #[test]
    fn the_slot_choice_takes_the_held_conversation_prefix() {
        let empty = Vec::<u32>::new();
        let ha = vec![7u32, 8, 9];
        let free = vec![(0usize, empty.as_slice(), 1u64), (1, ha.as_slice(), 2)];
        // A held start of the prompt wins, whatever else the slots hold.
        assert_eq!(
            pick_slot_index(&free, &[], 2, 1, 0, &[7, 8, 9, 10]),
            Some(1)
        );
        // A hold equal to the prompt is not a prefix (the engine reads nothing then - their
        // rule, kept): the empty slot gets the request instead.
        assert_eq!(pick_slot_index(&free, &[], 2, 1, 0, &[7, 8, 9]), Some(0));
        // No prefix: prefer the slot that holds nothing, then the one used longest ago.
        let h5 = vec![5u32];
        let h9 = vec![9u32];
        let free = vec![
            (0usize, h5.as_slice(), 9u64),
            (1, h9.as_slice(), 3),
            (2, empty.as_slice(), 7),
        ];
        assert_eq!(pick_slot_index(&free, &[], 3, 1, 0, &[7, 7]), Some(2));
        let free = vec![(0usize, h5.as_slice(), 9u64), (1, h9.as_slice(), 3)];
        assert_eq!(pick_slot_index(&free, &[], 2, 1, 0, &[7, 7]), Some(1));
        assert_eq!(pick_slot_index(&[], &[], 2, 1, 0, &[7, 7]), None);
    }

    #[test]
    fn the_slot_choice_follows_the_pipeline_group_rules() {
        let empty = Vec::<u32>::new();
        fn slot(idx: usize, held: &[u32]) -> (usize, &[u32], u64) {
            (idx, held, 0)
        }
        // Pipelined groups (total 4, groups 2): a free slot under a busy one of its group
        // joins the busy slot's window for free - the hole beats the spread order.
        let f = vec![slot(0, &empty), slot(1, &empty), slot(2, &empty)];
        assert_eq!(pick_slot_index(&f, &[3], 4, 2, 2, &[1]), Some(2));
        // No holes: spread stage-first (order 0, 2, 1, 3).
        let f = vec![slot(1, &empty), slot(2, &empty), slot(3, &empty)];
        assert_eq!(pick_slot_index(&f, &[0], 4, 2, 2, &[1]), Some(2));
        // Pipelined holds are no conversation cache: a perfect prefix loses to group economics.
        let ha = vec![1u32, 2];
        let f = vec![slot(1, &ha), slot(2, &empty)];
        assert_eq!(pick_slot_index(&f, &[0, 3], 4, 2, 2, &[1, 2, 3]), Some(2));
        // Non-pipelined groups demand a 512-token prefix before unbalancing the groups: 4 lose.
        let f = vec![slot(0, &ha), slot(2, &empty)];
        assert_eq!(pick_slot_index(&f, &[1, 3], 4, 2, 0, &[1, 2, 3]), Some(2));
        // ... and take the prefix when it is worth it.
        let long = (0..512u32).collect::<Vec<_>>();
        let prompt = (0..600u32).collect::<Vec<_>>();
        let f = vec![(1usize, long.as_slice(), 0u64), (2, empty.as_slice(), 1)];
        assert_eq!(pick_slot_index(&f, &[0, 3], 4, 2, 0, &prompt), Some(1));
        // A group count that does not divide the slots is no grouping at all.
        let f = vec![slot(0, &ha), slot(1, &empty)];
        assert_eq!(pick_slot_index(&f, &[], 2, 3, 1, &[1, 2, 3]), Some(0));
    }

    #[test]
    fn the_next_turn_lands_on_the_slot_that_holds_the_conversation() {
        let m = mock(2);
        m.set_batch_scripts(vec![vec![10, 11], vec![20, 21], vec![30, 31]]);
        // Two conversations, first turns: the second goes to the slot still empty, not the
        // one holding the other conversation.
        let (a1, _) = run(&m, 2); // slot 0: prompt 100..116 plus 10, 11 (minus the last held)
        let mut req_b = request(2);
        req_b.prompt_ids = (200u32..216).collect();
        run_req(&m, req_b.clone()); // slot 1: it was the one holding nothing
        let held_0 = held(&m, 0);
        let held_1 = held(&m, 1);
        // Conversation A's next turn: its prompt must admit on slot 0, the slot holding it.
        let mut turn2 = request(2);
        turn2.prompt_ids = (100u32..116)
            .chain(a1.iter().copied())
            .chain([999])
            .collect();
        let (_, _) = run_req(&m, turn2);
        assert!(
            held(&m, 0).starts_with(&held_0[..5]) && held(&m, 0).len() > held_0.len(),
            "A's slot absorbed the turn: held grew from {held_0:?}"
        );
        assert_eq!(
            held(&m, 1),
            held_1,
            "the other slot's conversation stayed put"
        );
        assert_eq!(
            held(&m, 0).last().copied(),
            Some(30),
            "the turn's own leg rode A's slot"
        );
    }

    // The solo-if-idle policy (`parallel_solo`): a request that finds the engine to itself
    // runs on the control lines, and an arrival's very queueing evicts it into a slot. These
    // drive the mock's mirrored policy; the wire behaviour they pin is the real engine's.

    #[test]
    fn parallel_solo_runs_an_undisturbed_leg_on_the_control_lines() {
        let m = mock(2);
        m.set_parallel_solo(true);
        m.set_batch_scripts(vec![vec![200, 201, ID_IM_END]]);
        let (toks, t) = run(&m, 10);
        // The leg ended on its stop token with no eviction and no second leg: nothing rode a
        // slot, and the finish is the leg's own. The stop token itself is suppressed, as any
        // leg that ends on one (the batch window does the same).
        assert_eq!(toks, vec![200, 201]);
        assert_eq!(t.finish_reason, "stop");
    }

    #[test]
    fn an_arrival_evicts_a_solo_leg_and_it_continues_in_a_slot() {
        let m = std::sync::Arc::new(mock(2));
        m.set_parallel_solo(true);
        // A paced leg gives the test a real mid-stream window; the scripts are long and the
        // budget generous so the arrival lands (and the promotion leg still ends on
        // `length`) even when a full test suite keeps the box busy.
        m.set_batch_pace_ms(20);
        // Script order on the mock's wire: A's `GEN` leg, A's promoted `BGEN`, B's `BGEN`.
        let a1: Vec<u32> = (200..230).collect();
        let a2: Vec<u32> = (230..240).chain(std::iter::once(ID_IM_END)).collect();
        m.set_batch_scripts(vec![a1.clone(), a2.clone(), vec![300, 301, 302, ID_IM_END]]);
        let seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let a = {
            let m = m.clone();
            let seen = seen.clone();
            std::thread::spawn(move || {
                let mut toks = Vec::new();
                let t = m
                    .generate(
                        &request(28),
                        Arc::new(AtomicBool::new(false)),
                        &mut |x| {
                            toks.push(x);
                            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            true
                        },
                        &mut |_, _| {},
                    )
                    .expect("the evicted request completes");
                (toks, t)
            })
        };
        // Wait for A genuinely mid-read - its second token is on the wire - rather than
        // trusting a sleep: B's queueing alone is the eviction trigger.
        let parked = std::time::Instant::now();
        while seen.load(std::sync::atomic::Ordering::SeqCst) < 2
            && parked.elapsed() < std::time::Duration::from_millis(300)
        {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let b_start = std::time::Instant::now();
        let (b_toks, _) = run(&m, 4);
        let b_elapsed = b_start.elapsed();
        let (a_toks, a) = a.join().expect("A's thread lands");

        // The promotion rode A's grown prompt (ids plus what the solo leg emitted), so no
        // token doubled: the stream is A's script from its start through the emitted point,
        // then the slot window's - contiguous, since the promotion continued the exchange.
        // Where the legs met is timing's choice, not the test's; the scripts are disjoint
        // ranges, so the consumer's stream shows the split exactly.
        let cut = match a_toks.iter().position(|&t| t == 230) {
            Some(c) => c,
            None => panic!("no 230; stream={a_toks:?} finish={}", a.finish_reason),
        };
        assert_eq!(&a_toks[..cut], &a1[..cut], "the solo leg's emitted tokens");
        assert_eq!(
            &a_toks[cut..],
            &a2[..a2.len() - 1],
            "the promoted leg continues the script"
        );
        // The promoted leg's window ends on a stop token, and its budget covers it: the
        // exchange's finish is that stop - never the hand-over's `cancel`, never a stray.
        assert_eq!(a.finish_reason, "stop");
        assert_eq!(b_toks, vec![300, 301, 302]); // the window's stop token is suppressed
                                                 // Had A not given way, B's admission could not have been answered before A's whole
                                                 // leg drained (its `proc`-equivalent rides the promotion).
        assert!(
            b_elapsed < std::time::Duration::from_millis(500),
            "the solo leg did not give way: B waited {b_elapsed:?}"
        );
    }
}

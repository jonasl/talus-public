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
    BusLine, EngineBus, EngineError, EngineStdin, TokenLogprob, STDIN_WRITE_FAILED,
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
            let slot = self.pick_slot(cancel)?;
            // Required before the send: otherwise the new request reads what its predecessor
            // left in the queue.
            self.bus.drain_slot(slot);
            let line = cmd(slot);
            self.sink.send(&line, frame)?;
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
                    BusLine::Ended => return Err(engine_gone()),
                    BusLine::Line(raw) => {
                        let trimmed = raw.trim();
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
                            leg.finish = f.get(4).unwrap_or(&"").to_string();
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
    fn pick_slot(&self, cancel: &AtomicBool) -> Result<usize, EngineError> {
        loop {
            if let Some(s) = self.bus.claim_free_slot() {
                return Ok(s);
            }
            if cancel.load(Ordering::Relaxed) {
                return Err(EngineError::Cancelled);
            }
            self.bus.wait_slot(CLAIM_WAIT);
        }
    }
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
        let mut toks = Vec::new();
        let timings = m
            .generate(
                &request(max_new),
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
        assert_eq!(m.batch_bus().unwrap().claim_free_slot(), Some(0));
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
        assert_eq!(m.batch_bus().unwrap().claim_free_slot(), Some(0));
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
        assert_eq!(m.batch_bus().unwrap().claim_free_slot(), Some(0));
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
        assert_eq!(m.batch_bus().unwrap().claim_free_slot(), Some(0));
    }

    #[test]
    fn router_finishes_a_releasing_slot_at_its_bdone() {
        let (bus, feeder) = EngineBus::manual(1);
        assert_eq!(bus.claim_free_slot(), Some(0));
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
        assert_eq!(bus.claim_free_slot(), Some(0));
    }

    #[test]
    fn router_drops_everything_on_a_nonfinite_bdone() {
        let (bus, feeder) = EngineBus::manual(1);
        assert_eq!(bus.claim_free_slot(), Some(0));
        assert!(!bus.begin_release(0, vec![1, 2]));
        feeder.push("BT 0 3");
        feeder.push("BDONE 0 3 nonfinite 1.0");
        assert!(bus.slot_held(0).is_empty()); // their #879: the slot's state is gone
        assert_eq!(bus.claim_free_slot(), Some(0));
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
}

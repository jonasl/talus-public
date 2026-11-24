#!/usr/bin/env python3
"""vision_bench - client-side interference and cost measurements against a live talus-server.

Built for the option-C acceptance list (docs/C-VISION-TOWER.md section 8): with the vision
tower resident in the engine, re-measure what option B measured - text-only TTFT through a
picture burst, the two-client prefill interference, and the repeat-image cost now that no
row cache exists on either side.  The numbers double as the "before" baseline for slot
serving: a single-session engine serializes the second client, and the harness is what
shows the size of that serialization.

Stdlib only; stream (SSE) requests with stream_options.include_usage, because the final
usage chunk carries the server-measured `timings` (prefill ms, decode rate) - sharper
than any client-side estimate.  The model is a reasoning variant: its deltas arrive as
`reasoning_content` before `content`, so TTFT is the first non-empty delta of either.

  python3 tools/vision_bench.py smoke
  python3 tools/vision_bench.py suite --reps 40 --json-out artifacts/vb_pre.json
  python3 tools/vision_bench.py soak --minutes 20 --engine-pid 12345
"""

import argparse
import base64
import http.client
import json
import socket
import statistics
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone

# ~120 words of neutral prose; --reps scales it.  40 reps lands near 7000 prompt tokens,
# long enough that prefill (not decode) dominates and interference is measurable.
PARAGRAPH = (
    "The harbour light came up green and the ferries resumed their patient crossing, "
    "while on the seawall a row of gulls renegotiated a bag of chips with the same "
    "seriousness the engineers once brought to the bridge. Wind off the channel carried "
    "diesel and cold salt, and the town pulled its collars up around itself, grateful "
    "for the grey. Somewhere behind the fishmongers a radio played a tune from a decade "
    "nobody remembers fondly, and the market stalls folded their awnings one by one, "
    "each click of the mechanism a small period at the end of a wet, ordinary afternoon."
)

SHORT_MSG = "Name three kinds of fish."


def long_prompt(reps, case=None):
    # a case tag PREPENDED because prefix caches match the common head: an appended tag
    # changes nothing the cache cares about
    head = f"[case {case}] " if case is not None else ""
    return head + "\n\n".join([PARAGRAPH] * reps) + "\n\nAnswer in one short sentence: how many paragraphs were above?"


def image_msg(text, image_path, case=None):
    with open(image_path, "rb") as f:
        url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return [
        {"type": "text", "text": (f"[case {case}] " if case is not None else "") + text},
        {"type": "image_url", "image_url": {"url": url}},
    ]


def new_conn(base, timeout):
    u = urllib.parse.urlparse(base)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    c = cls(host, port, timeout=timeout)
    return c


def chat(base, body, timeout=1800):
    """One streaming chat request.  Returns metrics: ttft (first non-empty delta),
    chunk count, and the final usage+timings block the server attaches."""
    out = {
        "t0": time.monotonic(),
        "ttft_s": None,
        "total_s": None,
        "first_chunk_s": None,
        "err": None,
        "usage": None,
        "timings": None,
    }
    u = urllib.parse.urlparse(base)
    path = u.path.rstrip("/") + "/v1/chat/completions"
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    try:
        c = new_conn(base, timeout)
        c.request("POST", path, payload, headers)
        r = c.getresponse()
        if r.status != 200:
            out["err"] = f"HTTP {r.status}: {r.read()[:200].decode(errors='replace')}"
            out["total_s"] = time.monotonic() - out["t0"]
            return out
        buf = ""
        n = 0
        while True:
            try:
                chunk = r.read(4096)
            except socket.timeout:
                out["err"] = "socket timeout mid-stream"
                break
            if not chunk:
                break
            if out["first_chunk_s"] is None:
                out["first_chunk_s"] = time.monotonic() - out["t0"]
            buf += chunk.decode(errors="replace")
            while "\n\n" in buf:
                ev, buf = buf.split("\n\n", 1)
                for line in ev.split("\n"):
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        continue
                    j = json.loads(data)
                    if "usage" in j:
                        out["usage"] = j["usage"]
                        out["timings"] = j.get("timings")
                        continue
                    for ch in j.get("choices") or []:
                        d = ch.get("delta") or {}
                        piece = d.get("content") or d.get("reasoning_content") or ""
                        if piece:
                            if out["ttft_s"] is None:
                                out["ttft_s"] = time.monotonic() - out["t0"]
                            n += 1
        out["total_s"] = time.monotonic() - out["t0"]
        out["completion_chunks"] = n
    except Exception as e:  # a failed request is a measurement too, not a harness crash
        out["err"] = f"{type(e).__name__}: {e}"
        out["total_s"] = time.monotonic() - out["t0"]
    return out


def base_body(model, content, max_tokens):
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0}


def fmt(m, label):
    u, t = m.get("usage") or {}, m.get("timings") or {}
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    pms, pps = t.get("prompt_ms"), t.get("predicted_per_second")
    ttft = "ERR " + str(m["err"]) if m["err"] else (
        f"ttft {m['ttft_s'] * 1000:7.0f} ms" if m["ttft_s"] else "ttft      -  ")
    return (f"{label:<26} {ttft}  total {m['total_s']:7.2f} s  "
            f"prompt {pt or 0:6d} tok  prefill {pms or 0:8.0f} ms ({t.get('prompt_per_second') or 0:6.0f} tok/s)  "
            f"decode {pps or 0:5.1f} tok/s  gen {ct or 0:4d} tok")


def run_suite(a):
    """The five measurements, sequential so their queueing is deterministic, with the
    solo baselines taken first so pair/pair-image have a same-run comparison."""
    results = {}

    def one(name, content, max_tokens):
        m = chat(a.base, base_body(a.model, content, max_tokens))
        results[name] = m
        print(fmt(m, name))
        time.sleep(a.gap)
        return m

    solo_short = one("solo-short-text", SHORT_MSG, 24)
    solo_long = one("solo-long-text", long_prompt(a.reps, case="solo"), a.max_tokens)
    solo_img = (one("solo-image", image_msg(a.img_text, a.image, case="solo"), a.max_tokens)
                if a.image else None)

    # two clients: A long, B short and staggered.  On a single-session engine B's own
    # TTFT is the serialization cost; with slots it should collapse toward A's own TTFT.
    # A carries a case tag per pair variant so its prefill is genuinely cold (a warm A
    # would have no work left to interfere with).  B stays identical to the solo short
    # request on purpose: its TTFT then measures pure queueing, not its own prefill.
    def pair(image, case):
        a_metrics = {}

        def A():
            # tag once: for the picture form the tag rides on the text part of the message
            content = (image_msg(long_prompt(a.reps), a.image, case=case) if image
                       else long_prompt(a.reps, case=case))
            a_metrics["m"] = chat(a.base, base_body(a.model, content, a.max_tokens))

        th = threading.Thread(target=A)
        th.start()
        time.sleep(a.delay)
        b = chat(a.base, base_body(a.model, SHORT_MSG, 24))
        th.join()
        return a_metrics["m"], b

    for tag, image, case in (("pair-text", False, "pair-t"), ("pair-image", True, "pair-i")):
        if tag == "pair-image" and not a.image:
            continue
        A, B = pair(image, case)
        results[tag] = {"A": A, "B": B}
        print(fmt(A, tag + " [A long]"))
        print(fmt(B, tag + " [B short]"))
        if tag == "pair-text":
            results[tag]["B_serialization_ms"] = B["ttft_s"] * 1000 if B["ttft_s"] else None
        time.sleep(a.gap)

    # repeat-image: identical request twice (KV prefix may hit), then the same image with
    # changed text (prefix misses before the picture, so the tower re-encodes: this is
    # the row-cache-free cost acceptance asks to measure).  Rounds 1-2 share a case tag;
    # round 3 must not, or the changed text would still ride the shared prefix.
    if a.image:
        r1 = one("repeat-image-1st", image_msg(a.img_text, a.image, case="rep"), a.max_tokens)
        r2 = one("repeat-image-same", image_msg(a.img_text, a.image, case="rep"), a.max_tokens)
        r3 = one("repeat-image-vartext", image_msg(a.img_text + " Describe colors only.", a.image, case="rep-v"), a.max_tokens)
        if r1["ttft_s"] and r3["ttft_s"]:
            results["repeat-image-reencode-ms"] = (r3["ttft_s"] - r1["ttft_s"]) * 1000
        results["repeat-image"] = {"same_prefix_ttft_delta_ms":
                                   (r2["ttft_s"] - r1["ttft_s"]) * 1000 if r1["ttft_s"] and r2["ttft_s"] else None}
    return results


def mem_sample(engine_pid):
    """Engine RSS (VmRSS) and the box's MemAvailable - the soak's memory watch."""
    s = {}
    if engine_pid:
        try:
            for line in open(f"/proc/{engine_pid}/status"):
                if line.startswith("VmRSS:"):
                    s["engine_rss_mib"] = int(line.split()[1]) // 1024
                    break
        except OSError:
            pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                s["mem_available_mib"] = int(line.split()[1]) // 1024
                break
    except OSError:
        pass
    return s


def run_soak(a):
    """A long mixed request stream with memory sampling - the kva soak shape.  Run
    against a gate engine (coexistence: thin profile + STRATA_ARENA_MMAP=1), not the
    box's only production engine, unless that is explicitly the point."""
    out_path = a.json_out + ".soak.jsonl"
    t_end = time.monotonic() + a.minutes * 60
    workload = [
        ("short", lambda i: base_body(a.model, SHORT_MSG, 24)),
        # long and image rounds carry a per-iteration case tag so the conversation cache
        # cannot serve them warm: the soak wants steady cold-prefill stress, not replays.
        # image-repeat is deliberately untagged - it is the cache-warm round.
        ("long", lambda i: base_body(a.model, long_prompt(min(a.reps, 10), case=f"soak{i}"), 24)),
        ("image", lambda i: base_body(a.model, image_msg(f"{a.img_text} (take {i})", a.image, case=i), 24)),
        ("image-repeat", lambda i: base_body(a.model, image_msg(a.img_text, a.image, case="soak-rep"), 24)),
    ]
    workload = [w for w in workload if a.image or w[0] not in ("image", "image-repeat")]
    stats = {}
    last_sample = 0.0
    n = 0
    with open(out_path, "a") as f:
        while time.monotonic() < t_end:
            name, mk = workload[n % len(workload)]
            m = chat(a.base, mk(n), timeout=600)
            if not m["err"]:
                stats.setdefault(name, []).append(m["ttft_s"])
            rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": name,
                   "ttft_s": m["ttft_s"], "total_s": m["total_s"], "err": m["err"],
                   "usage": m["usage"], "timings": m["timings"], **mem_sample(a.engine_pid)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if time.monotonic() - last_sample > 30 or n % 10 == 0:
                print(f"[{n:4d}] {name:12} ttft {m['ttft_s'] * 1000 if m['ttft_s'] else -1:7.0f} ms  "
                      f"rss {rec.get('engine_rss_mib', '?')} MiB  avail {rec.get('mem_available_mib', '?')} MiB")
            n += 1
        print("\nsoak ttft medians (ms):")
        for name, v in sorted(stats.items()):
            print(f"  {name:14} n={len(v):4d} median {statistics.median(v) * 1000:8.0f}  "
                  f"p95 {sorted(v)[int(len(v) * 0.95)] * 1000 if len(v) > 10 else float('nan'):8.0f}")
    print(f"raw: {out_path}")
    return {"soak_requests": n, "medians_ms": {k: statistics.median(v) * 1000 for k, v in stats.items()}}


def main():
    # flags live on a parent so they work before or after the subcommand
    f = argparse.ArgumentParser(add_help=False)
    f.add_argument("--base", default="http://127.0.0.1:8081")
    f.add_argument("--model", default="qwen3.8-flash-next-uncensored")
    f.add_argument("--image", default="/tmp/opencode/vision-test-3.png")
    f.add_argument("--img-text", default="Describe this image in one short sentence.")
    f.add_argument("--reps", type=int, default=40, help="paragraph repeats for the long prompt")
    f.add_argument("--max-tokens", type=int, default=48, help="generation budget for measured requests")
    f.add_argument("--delay", type=float, default=0.5, help="pair: B's start delay after A (s)")
    f.add_argument("--gap", type=float, default=1.0, help="pause between suite requests (s)")
    f.add_argument("--json-out", default=None)
    f.add_argument("--engine-pid", type=int, default=None, help="engine child pid, for the soak's RSS sampling")
    f.add_argument("--minutes", type=float, default=20, help="soak duration")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], parents=[f])
    # command as a plain positional: argparse would fight over flag values between a
    # top-level parser and a subparser that share a parent, and only one order works
    p.add_argument("cmd", choices=["smoke", "suite", "soak"],
                   help="smoke: stream parsing and image intake still work; suite: the five "
                        "acceptance measurements; soak: long mixed stream + memory samples")
    a = p.parse_args()
    if a.image and not __import__("os").path.exists(a.image):
        a.image = None
        print(f"note: image {a.image!r} not found, picture scenarios skipped", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if a.json_out is None:
        a.json_out = f"artifacts/vision_bench_{a.cmd}_{stamp}.json"

    if a.cmd == "smoke":
        r = {"text": chat(a.base, base_body(a.model, SHORT_MSG, 16)),
             "image": chat(a.base, base_body(a.model, image_msg(a.img_text, a.image), 16)) if a.image else None}
    elif a.cmd == "suite":
        r = run_suite(a)
    else:
        r = run_soak(a)

    import os
    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    with open(a.json_out, "w") as f:
        json.dump({"started": stamp, "args": {k: str(v) for k, v in vars(a).items()}, "results": r}, f, indent=1)
    print(f"\njson: {a.json_out}")
    bad = json.dumps(r).count('"err": "')
    print(f"requests with errors: {bad}")
    sys.exit(1 if bad and a.cmd != "suite" else 0)


if __name__ == "__main__":
    main()

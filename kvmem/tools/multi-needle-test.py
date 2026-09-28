#!/usr/bin/env python3
"""
KVMem MULTI-needle retrieval test - the "can it weave a net?" test.

needle-test.py proves KVMem can find ONE fact anywhere in 200K+ of history.
That is the easy case: the probe question names the department, so retrieval
has a strong signal to match against.

This test asks the harder question: when a single question needs SEVERAL facts
scattered across the context, how many does retrieval actually bring back?

Why this matters: KVMem keeps a bounded GPU window (--kvmem-budget). Finding
one needle needs the right block to rank #1. Gathering N needles needs all N
blocks to rank in the top-N. Those are very different requirements, and this
is the failure mode that would bite a long-horizon agent.

Phases
  1. fill     - plant N distinct needles across the context
  2. singles  - ask about each needle alone      (baseline, should be ~100%)
  3. groups   - ask for K needles in ONE question, K = 2, 4, 8 ...
                score = how many of the K came back

The gap between `singles` and `groups` is the real measure of the retrieval
ceiling.

Usage:
    python multi-needle-test.py --port 18200 --chunks 12 --chunk-tokens 8000
    python multi-needle-test.py --port 18200 --chunks 4 --chunk-tokens 2000  # quick
"""
import argparse, json, random, re, string, sys, time, urllib.error, urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()

# Unique per needle so every probe has exactly one correct answer.
DEPARTMENTS = [
    "Procurement", "Logistics", "Compliance", "Facilities", "Payroll",
    "Security", "Archiving", "Telemetry", "Actuarial", "Benefits",
    "Catering", "Dispatch", "Estimating", "Fulfilment", "Geospatial",
    "Housekeeping", "Inventory", "Janitorial", "Kiosks", "Licensing",
    "Maintenance", "Navigation", "Onboarding", "Packaging", "Quarantine",
    "Receiving", "Scheduling", "Training", "Underwriting", "Vending",
    "Warehousing", "Yard", "Zoological",
]

CODE_RE = re.compile(r"\b[A-Z]{4}-\d{4}\b")


def make_filler(n, rng):
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 24))) + "."
        for _ in range(n)
    )


def make_needle(idx, rng, dept):
    code = "".join(rng.choice(string.ascii_uppercase) for _ in range(4)) + "-" + \
           "".join(rng.choice(string.digits) for _ in range(4))
    return {
        "code": code, "dept": dept,
        "text": (f"Administrative note {idx}: the authorization code for the "
                 f"{dept} department archive is {code}. "
                 f"Record this for the audit trail."),
    }


def chat(port, messages, max_tokens, timeout):
    body = json.dumps({
        "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "reasoning_budget_tokens": 0,      # keep thinking out of the way
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return {
        "content": (d["choices"][0]["message"].get("content") or "").strip(),
        "usage": d.get("usage", {}), "elapsed": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--chunk-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--groups", type=str, default="2,4,8",
                    help="comma-separated group sizes to probe")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.chunks > len(DEPARTMENTS):
        print(f"ERROR: --chunks must be <= {len(DEPARTMENTS)}", file=sys.stderr)
        return 1

    depts = rng.sample(DEPARTMENTS, args.chunks)
    needles = [make_needle(i + 1, rng, depts[i]) for i in range(args.chunks)]

    messages, pt = [], 0
    print(f"=== Phase 1: filling context, {args.chunks} chunks ===", flush=True)
    for i, nd in enumerate(needles):
        sents = max(4, args.chunk_tokens // 15)
        body = f"{make_filler(sents // 2, rng)}\n\n{nd['text']}\n\n{make_filler(sents - sents // 2, rng)}"
        messages.append({"role": "user", "content": f"Tool result {i+1} follows.\n\n{body}"})
        messages.append({"role": "assistant", "content": "Acknowledged."})
        try:
            r = chat(args.port, messages, 8, args.timeout)
        except urllib.error.HTTPError as e:
            print(f"  chunk {i+1}: context ceiling hit ({e})", flush=True)
            needles, messages = needles[:i], messages[:-2]
            break
        except Exception as e:
            print(f"  chunk {i+1}: FAILED {type(e).__name__}: {e}", flush=True)
            return 1
        pt = r["usage"].get("prompt_tokens", 0)
        print(f"  chunk {i+1:>3}/{args.chunks}  prompt={pt:>7}  {r['elapsed']:.1f}s", flush=True)

    n = len(needles)
    print(f"\n=== Context: {pt} tokens, {n} needles planted ===\n", flush=True)
    base = messages

    # ---- Phase 2: each needle alone (baseline) --------------------------
    print("=== Phase 2: single-needle baseline ===", flush=True)
    sample = [0, n // 3, 2 * n // 3, n - 1]
    sample = sorted(set(i for i in sample if 0 <= i < n))
    singles_ok = 0
    for i in sample:
        nd = needles[i]
        q = (f"What is the authorization code for the {nd['dept']} department "
             f"archive? Reply with only the code.")
        try:
            r = chat(args.port, base + [{"role": "user", "content": q}], 32, args.timeout)
        except Exception as e:
            print(f"  [{nd['dept']}] FAILED {e}", flush=True); continue
        ok = nd["code"] in r["content"].upper()
        singles_ok += ok
        print(f"  [{nd['dept']:12}] expect {nd['code']}  got {r['content'][:20]!r}  "
              f"{'OK' if ok else 'MISS'}  {r['elapsed']:.1f}s", flush=True)
    print(f"  -> singles {singles_ok}/{len(sample)}\n", flush=True)

    # ---- Phase 3: K needles in ONE question ------------------------------
    print("=== Phase 3: multi-needle groups (the real test) ===", flush=True)
    results = {}
    for K in [int(x) for x in args.groups.split(",")]:
        if K > n:
            print(f"  K={K}: skipped (only {n} needles)", flush=True); continue
        # Probe 3 different groups of size K to avoid a single lucky/unlucky draw.
        tot_found = tot_asked = 0
        for trial in range(3):
            idxs = rng.sample(range(n), K)
            picked = [needles[i] for i in idxs]
            names = ", ".join(p["dept"] for p in picked)
            q = (f"List the authorization codes for these {K} departments, one per "
                 f"line as 'Department: CODE'. Departments: {names}")
            try:
                r = chat(args.port, base + [{"role": "user", "content": q}], 512, args.timeout)
            except Exception as e:
                print(f"  K={K} trial{trial+1}: FAILED {e}", flush=True); continue
            got = set(CODE_RE.findall(r["content"].upper()))
            found = sum(1 for p in picked if p["code"] in got)
            tot_found += found; tot_asked += K
            print(f"  K={K} trial{trial+1}: {found}/{K}  ({r['elapsed']:.1f}s)", flush=True)
        if tot_asked:
            rate = 100.0 * tot_found / tot_asked
            results[K] = rate
            print(f"  -> K={K}: {tot_found}/{tot_asked} = {rate:.0f}%\n", flush=True)

    print("=== SUMMARY ===", flush=True)
    print(f"  single-needle : {singles_ok}/{len(sample)}"
          f" = {100.0*singles_ok/max(1,len(sample)):.0f}%", flush=True)
    for K, rate in sorted(results.items()):
        print(f"  group of {K:<2}   : {rate:.0f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
KVMem long-context needle test.

Fills the context with many chunks (mimicking tool-result rounds), planting a
unique "needle" fact at each depth. Then asks about needles from different
depths and checks whether retrieval actually surfaced them.

This is the test that matters for KVMem: its whole value proposition is
retrieval quality at long context. Short-prompt speed tests prove nothing.

Usage:
    python needle-test.py --port 18200 --chunks 30 --chunk-tokens 8000
    python needle-test.py --port 18200 --chunks 6 --chunk-tokens 2000   # quick
"""
import argparse, json, random, string, sys, time, urllib.request, urllib.error

# --------------------------------------------------------------------------
# Needle generation: unique, unambiguous, easy to grade.
# --------------------------------------------------------------------------

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()

def make_filler(n_sentences, rng):
    out = []
    for _ in range(n_sentences):
        n = rng.randint(12, 24)
        out.append(" ".join(rng.choice(WORDS) for _ in range(n)) + ".")
    return " ".join(out)

# Departments must be UNIQUE across needles. With a random pick from a small
# pool, two needles share a department and the probe question ("what is the
# code for the X department archive?") becomes genuinely ambiguous - the model
# then answers with the other valid code and gets scored as a miss.
# That is a test-design bug, not a retrieval failure. Learned the hard way.
DEPARTMENTS = [
    "Procurement", "Logistics", "Compliance", "Facilities", "Payroll",
    "Security", "Archiving", "Telemetry", "Actuarial", "Benefits",
    "Catering", "Dispatch", "Estimating", "Fulfilment", "Geospatial",
    "Housekeeping", "Inventory", "Janitorial", "Kiosks", "Licensing",
    "Maintenance", "Navigation", "Onboarding", "Packaging", "Quarantine",
    "Receiving", "Scheduling", "Training", "Underwriting", "Vending",
    "Warehousing", "Yard", "Zoological",
]

def make_needle(idx, rng, dept=None):
    """A unique fact plus the question that probes it."""
    code = "".join(rng.choice(string.ascii_uppercase) for _ in range(4)) + "-" + \
           "".join(rng.choice(string.digits) for _ in range(4))
    if dept is None:
        dept = rng.choice(DEPARTMENTS)
    return {
        "code": code,
        "dept": dept,
        # The needle text deliberately sits inside otherwise-unremarkable prose.
        "text": (f"Administrative note {idx}: the authorization code for the "
                 f"{dept} department archive is {code}. "
                 f"Record this for the audit trail."),
        "question": (f"What is the authorization code for the {dept} "
                     f"department archive? Reply with only the code."),
    }

# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def chat(port, messages, max_tokens, timeout):
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_budget_tokens": 0,   # keep thinking out of the way for grading
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "usage": d.get("usage", {}),
        "elapsed": dt,
        "finish": d["choices"][0].get("finish_reason"),
    }

def approx_tokens(s):
    """Rough English estimate; the server's own usage numbers are authoritative."""
    return max(1, len(s) // 4)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--chunk-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Plant one needle per chunk so we get recall data across every depth.
    # Departments are drawn without replacement so every probe question has
    # exactly one correct answer.
    if args.chunks > len(DEPARTMENTS):
        print(f"ERROR: --chunks must be <= {len(DEPARTMENTS)} so that every "
              f"needle has a unique department (otherwise the probe question "
              f"is ambiguous).", file=sys.stderr)
        return 1
    depts = rng.sample(DEPARTMENTS, args.chunks)
    needles = [make_needle(i + 1, rng, depts[i]) for i in range(args.chunks)]

    messages = []
    pt = 0
    print(f"=== Phase 1: filling context with {args.chunks} chunks "
          f"(~{args.chunk_tokens} tokens each) ===", flush=True)

    for i, nd in enumerate(needles):
        # Filler of roughly chunk_tokens, with the needle buried in the middle.
        sents = max(4, args.chunk_tokens // 15)
        first = make_filler(sents // 2, rng)
        second = make_filler(sents - sents // 2, rng)
        chunk = f"{first}\n\n{nd['text']}\n\n{second}"

        messages.append({"role": "user",
                         "content": f"Tool result {i+1} follows.\n\n{chunk}"})
        messages.append({"role": "assistant", "content": "Acknowledged."})

        try:
            r = chat(args.port, messages, max_tokens=8, timeout=args.timeout)
        except urllib.error.HTTPError as e:
            # Context ceiling reached. Stop filling and grade what we have -
            # the recall test is the point, not hitting an exact token count.
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            print(f"  chunk {i+1}: context ceiling hit "
                  f"({type(e).__name__}: {e}) {detail}", flush=True)
            needles = needles[:i]
            messages = messages[:-2]          # drop the pair that failed
            break
        except Exception as e:
            print(f"  chunk {i+1}: FAILED - {type(e).__name__}: {e}", flush=True)
            return 1

        pt = r["usage"].get("prompt_tokens", 0)
        hit = r["usage"].get("prompt_cache_hit_tokens", 0)
        miss = r["usage"].get("prompt_cache_miss_tokens", 0)
        print(f"  chunk {i+1:>3}/{args.chunks}  prompt={pt:>7}  "
              f"cache_hit={hit:>7}  miss={miss:>6}  {r['elapsed']:.1f}s", flush=True)

    final_pt = pt
    n = len(needles)
    print(f"\n=== Context filled: final prompt_tokens = {final_pt} "
          f"({n} chunks delivered) ===\n", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: probe needles from across the whole depth range.
    # ------------------------------------------------------------------
    probes = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    probes = sorted(set(p for p in probes if 0 <= p < n))

    print("=== Phase 2: needle recall ===", flush=True)
    correct = 0
    for p in probes:
        nd = needles[p]
        msgs = messages + [{"role": "user", "content": nd["question"]}]
        try:
            r = chat(args.port, msgs, max_tokens=64, timeout=args.timeout)
        except Exception as e:
            print(f"  depth {p+1:>3}: FAILED - {type(e).__name__}: {e}", flush=True)
            continue
        ans = r["content"].upper()
        ok = nd["code"] in ans
        correct += ok
        depth_pct = 100.0 * (p + 1) / args.chunks
        print(f"  depth {p+1:>3}/{args.chunks} ({depth_pct:5.1f}%)  "
              f"expect={nd['code']}  got={r['content'][:40]!r}  "
              f"{'OK' if ok else 'MISS'}  {r['elapsed']:.1f}s  "
              f"finish={r['finish']}", flush=True)

    print(f"\n=== Recall: {correct}/{len(probes)} ===", flush=True)
    return 0 if correct == len(probes) else 2

if __name__ == "__main__":
    sys.exit(main())

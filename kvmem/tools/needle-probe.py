#!/usr/bin/env python3
"""
Follow-up probe: is the recall failure retrieval, or model confusion?

Regenerates the same seeded context as needle-test.py, then asks about the
needles that failed, using several phrasings of increasing distinctiveness.
If a more distinctive phrasing succeeds, the blocks ARE in the window and the
problem is the model choosing among near-identical candidates - not retrieval.

Reuses needle-test.py's generators so the content is byte-identical.
"""
import sys, urllib.request, urllib.error, json, time
sys.path.insert(0, '.')
from importlib import import_module
import random
nt = import_module('needle-test')

PORT = 18200
CHUNKS = 17          # what actually fit last run
CHUNK_TOKENS = 8000

def chat(messages, max_tokens=64, timeout=900):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "reasoning_budget_tokens": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"].get("content", "").strip(), d.get("usage", {}), time.time()-t0

rng = random.Random(1234)
needles = [nt.make_needle(i+1, rng) for i in range(CHUNKS)]

messages = []
print("=== rebuilding context ===", flush=True)
for i, nd in enumerate(needles):
    sents = max(4, CHUNK_TOKENS // 15)
    chunk = (f"{nt.make_filler(sents//2, rng)}\n\n{nd['text']}\n\n"
             f"{nt.make_filler(sents - sents//2, rng)}")
    messages.append({"role": "user", "content": f"Tool result {i+1} follows.\n\n{chunk}"})
    messages.append({"role": "assistant", "content": "Acknowledged."})
    out, u, dt = chat(messages, max_tokens=8)
    print(f"  {i+1}/{CHUNKS}  prompt={u.get('prompt_tokens')}  "
          f"hit={u.get('prompt_cache_hit_tokens')}  {dt:.1f}s", flush=True)

# The two probes that failed, plus one that passed as a control.
TARGETS = [(12, "FAILED"), (16, "FAILED"), (8, "control-pass")]

for idx, tag in TARGETS:
    nd = needles[idx]
    print(f"\n===== depth {idx+1}/17  [{tag}]  expect {nd['code']} "
          f"({nd['dept']}) =====", flush=True)

    variants = [
        ("original     ", nd["question"]),
        ("quote-needle ", f"Find this exact sentence in the history and report its code: "
                          f"\"Administrative note {idx+1}: the authorization code for the "
                          f"{nd['dept']} department archive is ....\" What is the code?"),
        ("dept-only    ", f"Search the history for the word \"{nd['dept']}\" and tell me "
                          f"the code associated with it. Reply with only the code."),
        ("note-number  ", f"What is the authorization code in 'Administrative note {idx+1}'? "
                          f"Reply with only the code."),
    ]
    for name, q in variants:
        try:
            out, u, dt = chat(messages + [{"role": "user", "content": q}])
        except Exception as e:
            print(f"  {name} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        ok = nd["code"] in out.upper()
        print(f"  {name} got={out[:36]!r:40} {'OK ' if ok else 'MISS'}  "
              f"hit={u.get('prompt_cache_hit_tokens')}  {dt:.1f}s", flush=True)

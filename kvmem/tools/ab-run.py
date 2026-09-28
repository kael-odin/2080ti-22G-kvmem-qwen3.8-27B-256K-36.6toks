#!/usr/bin/env python3
"""
KVMem A/B harness: one command = start server with given flags, run a fixed
workload, report timings, stop server.

Why a harness: the earlier speed comparisons were taken from runs at different
times with different workloads, which is not evidence. This runs a FIXED
workload (same chunk count, same chunk size, same seed) so two invocations
differ only by the flags passed in.

Usage:
    python ab-run.py --label "ub512"  --chunks 8
    python ab-run.py --label "ub1024" --chunks 8 --extra "-ub 1024"

Reports one JSON line at the end so runs can be diffed mechanically.
"""
import argparse, json, os, random, socket, string, subprocess, sys, time, urllib.request

ROOT = r"F:\AI-Models"
BIN = os.path.join(ROOT, r"kvmem-llama.cpp\build-cu13286-sm75\bin")
EXE = os.path.join(BIN, "llama-kvmem-server.exe")
MODEL = os.path.join(ROOT, r"models\qwen3.8-27B\gsq-rco\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
MMPROJ = os.path.join(ROOT, r"models\qwen3.8-27B\gsq-rco\mmproj-Qwen3.8-27B-BF16.gguf")
PORT = 18200

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()


def port_open(port):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port)); s.close(); return True
    except Exception:
        return False


def kill_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-kvmem-server.exe"],
                   capture_output=True)


def wait_ready(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_open(PORT):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5):
                    time.sleep(5)   # let warmup settle
                    return True
            except Exception:
                pass
        time.sleep(3)
    return False


def chat(messages, max_tokens, timeout=900):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "reasoning_budget_tokens": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return time.time() - t0, d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--chunk-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--extra", default="", help="extra server flags")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base_flags = ("-c 262144 -n 20480 --kvmem-budget 36864 --kvmem-gen-reserve 61440 "
                  "--kv-dtype q8_0 --spec-type draft-mtp --spec-draft-n-max 3 "
                  "--kvmem-mtp-state replay --enable-thinking --reasoning-budget 32768 "
                  "--no-ui --host 127.0.0.1 --port 18200").split()
    extra = args.extra.split() if args.extra.strip() else []
    cmd = [EXE, "-m", MODEL, "--mmproj", MMPROJ, "--mmproj-offload",
           "--image-max-tokens", "512"] + base_flags + extra

    print(f"[{args.label}] starting server ...", flush=True)
    kill_server(); time.sleep(4)
    logp = os.path.join(ROOT, "logs", "kvmem-tests", f"ab-{args.label}.log")
    lf = open(logp, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=BIN, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_ready():
        print(f"[{args.label}] ERROR: server did not become ready", flush=True)
        try: proc.kill()
        except Exception: pass
        return 1

    def filler(n):
        return " ".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 24))) + "."
                        for _ in range(n))

    messages, times, prompt_tok = [], [], 0
    print(f"[{args.label}] running {args.chunks} chunks ...", flush=True)
    try:
        for i in range(args.chunks):
            sents = max(4, args.chunk_tokens // 15)
            body = filler(sents)
            messages.append({"role": "user", "content": f"Tool result {i+1} follows.\n\n{body}"})
            messages.append({"role": "assistant", "content": "Acknowledged."})
            dt, usage = chat(messages, 8)
            prompt_tok = usage.get("prompt_tokens", 0)
            times.append(dt)
            print(f"  [{args.label}] chunk {i+1}/{args.chunks}  "
                  f"prompt={prompt_tok:>7}  {dt:6.1f}s", flush=True)
    except Exception as e:
        print(f"[{args.label}] workload failed: {type(e).__name__}: {e}", flush=True)
        try: proc.kill()
        except Exception: pass
        return 1

    # steady-state = last 4 chunks (avoids the retrieval-activation transient)
    steady = times[-4:] if len(times) >= 4 else times
    per_chunk = sum(steady) / len(steady)
    result = {
        "label": args.label, "extra": args.extra, "chunks": args.chunks,
        "final_prompt_tokens": prompt_tok,
        "all_times": [round(t, 1) for t in times],
        "steady_avg_s": round(per_chunk, 2),
        "steady_tok_per_s": round(16000 / per_chunk, 1) if per_chunk else 0,
    }
    print("AB_RESULT " + json.dumps(result), flush=True)

    try: proc.kill()
    except Exception: pass
    lf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

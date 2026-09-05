# 🚀 KV-Streaming for RTX 2080 Ti (sm_75): Run 260K Context on a 22 GB Card

**跑超大上下文，不需要超大显存。** This repo makes [adaptive KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) work on **RTX 2080 Ti / Turing (sm_75)** — a 2018, 22 GB GPU — so you can serve a **260,000-token context** with a **27B model** that would otherwise need a 40GB+ card.

> **正常推理要给 KV 缓存让路, 显存没了上下文就断了。KV 流式把"记忆(缓存)"按需搬到内存, 显存只留当前窗口。** That's the whole trick.

---

## ✨ Why this matters (and why it's not a free lunch)

**The problem everyone with a small GPU hits:** your model fits, but the **KV cache** for a long document doesn't. A 27B model + 260K tokens of KV needs **~29 GB** — impossible on 22 GB.

**The fix this repo ships:** adaptive KV streaming keeps the **backing store in system RAM** and only keeps a small staging pool on the GPU. You get a **~4.7× longer context** on the *same* card.

| | Fixed KV (the old way) | **This repo: KV streamed** |
|---|---|---|
| Max context on 22 GB | ~53K tokens ❌ | **~262K tokens ✅** |
| Decode speed | fast | **30–42 tg/s when resident; ~9.6 tg/s after the 135K resident boundary** |
| VRAM used for KV | all of it | resident pool plus streamed host-backed pages |
| Docs of LL seats (RAM) | — | uses system RAM instead |

**80C video card → 260K context, in practice, with correct answers.** Verified on a real needle-in-a-haystack at **260,096 tokens**, exact answer recovered.

---

## 2. Is this valuable? (honest scorecard)

| Claim | True? |
|---|---|
| "I invented long context" | ❌ No — upstream did. |
| "I make 25K tokens usable again on a 2018 card" | ✅ **Yes, specific & verifiable** |
| "sm_75 support that upstream lacks" | ✅ **Yes** (2 real crashes fixed) |
| "A 22 GB card now rivals a 40 GB card for context" | ✅ For context **only**, not raw speed |

**What's genuinely novel here:** a **drop-in fix + measurement report** for running long-context KV streaming on the *oldest widely-owned sm_75 hardware*, with the numbers to prove it. If you own a 2080 Ti / 2070 / 2060 (or any Turing card) and thought "I can't do long context," — this repo is your answer.

---

## 3 Files & quick start (Windows)

**Requirements:** Windows 10/11, Visual Studio 2022 Build Tools (MSVC), CUDA 12.4, Git, Python 3.11.

```bat
:: 1. Clone & prepare the source (checks out adaptive-KV + applies sm_75 patches)
scripts\prepare-source.bat

:: 2. Build for sm_75 (explicitly enables quantized Flash Attention)
scripts\build-sm75.bat

:: 3. Package the server + DLLs into bin\
scripts\package-sm75.bat
```

Then run a single case with the validator:

```bat
python tools\run_case.py ^
  --server bin\llama-server.exe ^
  --model  D:\models\Qwen3.8-27B.gguf ^
  --mmproj D:\models\mmproj.gguf ^
  --ctx 262144 --stage-mib 1024 ^
  --cache-k q4_0 --cache-v q4_0 --mtp --load-only --label load-262k
```

> For a **b10816/upstream-fresh** build (recommended for new users), use
> `prepare_source-b10816.bat` + `build-b10816-sm75.bat`. It merges the fix onto 366-later upstream.

---

## 4 Measured results (RTX 2080 Ti 22 GB, Qwen3.8-27B)

**Headline 260K needle-in-a-haystack — exact answer recovered:**

| Metric | Value |
|---|---|
| Context | **260,096 tokens** |
| Prompt speed | 197.03 tok/s |
| Generation | 3.96 tg/s (streamed) |
| MTP draft | 12/12 accepted |
| Min free VRAM | 1,139 MiB |

**Fixed vs streamed KV — the decode trade-off (Turbo IQ4_XS):**

| Context | KV mode | Prefill t/s | Decode t/s | Min free VRAM |
|---|---|---|---:|---:|
| 64K | fixed | 516.64 | 33.35 | 3.54 GiB |
| 128K | fixed | 308.64 | 25.99 | 1.11 GiB |
| 170K | fixed | 377.65 | 25.50 | **69 MiB — unsafe** |
| 262K | adaptive 1,024 MiB | 205.76 | **3.08** | 2.30 GiB |

**Resident-boundary check (UD-IQ4_XS, q5_1 KV, 3,072 MiB pool, 2026-09-06):** a 135,087-token prompt crossed the resident partition (trace: resident pages 511→510, ring slots 16→32), recovered the marker exactly, and decoded at 9.64 t/s (prefill 226.4 t/s). Inside the resident pool (≤ ~130K tokens) decode stays at 30–42 t/s. Details and caveats in the [ops report](docs/ops-stability-optimization-report.md).

**The takeaway:** when fixed KV fits, it's faster for decode. KV streaming is for when it doesn't — and it's the *only* way to hit 262K on 22 GB.

*(More configs, including GSQ-RCO and a Q8_0/Q8_0 comparison, in* [`results/benchmark-summary.csv`](results/benchmark-summary.csv) *and* [`docs/report-zh.md`](docs/report-zh.md)*.)*

---

## 5 What's special about the sm_75 port (the actual work)

The upstream KV branch was built/tested on an RTX 5070 Ti (Ampere, sm_80). On Turing (sm_75) it **crashed** at ~58K tokens:

```text
flash_attn_ext_f16 has no device code compatible with CUDA arch 750
CUDA error: invalid argument
```

Two **Turing-specific** fixes are patched in:

1. **Large streamed prefill** uses the sm_75-capable 32-column `<256,256,4,8>` partial MMA (upstream used a 64-column form sm_75 lacks).
2. **Small MTP verification batches** (≤32 queries) use native vector path — fixes a logits glitch that accepted EOS too early, producing "ORCHID" at the 260K extreme.

Non-Turing hardware keeps the original unchanged path.

**Merge onto latest upstream `b10816` (2026-09-05):** a true three-way integration (366 commits later), preserving upstream's new sparse-attention path. See [`patches/0002-merge-adaptive-kv-into-b10816-sm75.patch`](patches/0002-merge-adaptive-kv-into-b10816-sm75.patch) + [`docs/b10816-merge-notes-zh.md`](docs/b10816-merge-notes-zh.md).

---

## 6 Honest boundaries & safety

- This is a **capacity** feature, not a free speedup: streaming has a decode cost when KV actively crosses GPU↔RAM. Fixed KV is faster for decode when it fits.
- A *load-only* pass proves *allocation*, not *long-context correctness*.
- Tested on **one** GPU (2080 Ti), one model line (Qwen3.8-27B). Not proven across every model / KV type / multi-slot.
- Don't run `run_case.py` alongside a production `llama-server` on the same GPU.
- The validator binds only `127.0.0.1`, tracks its own child PID, and writes one result dir per run.
- No model weights, CUDA redistributables, or private keys are shipped here.

---

## 7 License & thanks

MIT (see [LICENSE](LICENSE)). The adaptive KV implementation originates from **RaymondHuang210129's** [llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming), subject to upstream llama.cpp's MIT. GSQ/RCO models have their own licenses; **no weights are redistributed**.

**中文版:** 见 [README_zh-CN.md](README_zh-CN.md).

---

*RTX 2080 Ti · sm_75 · adaptive KV streaming · 22GB VRAM · 260K context · Qwen 3.8 · speculative MTP decode · long-context inference · llama.cpp Turing port*
<div align="center">
  <img src="assets/readme/hero.svg" alt="RTX 2080 Ti 22G long-context lab: KVMem 262K ctx at 36.6 tok/s + KV streaming archive" width="100%">
</div>

# RTX 2080 Ti 22G (sm_75) Long-Context Lab

> **A 2018 22 GB card serving 260K-token context at 36.6 tok/s decode.**
> This repo ports [KVMem](https://github.com/kvmem/kvmem-llama.cpp) (retrieval-based long context) to Turing / sm_75 / Windows with every number measured and reproducible — **the best performance-per-cost long-context setup we know of for a single 2080 Ti 22G card**.

**中文主文档:** [README.md](README.md) — 本文件为英文摘要，完整文档以中文为准。

## Measured results (RTX 2080 Ti 22G, reproducible)

**Headline · KVMem** (Qwen3.8-27B-GSQ-RCO-IQ3_S + MTP + vision, 262K context, q8_0 KV):

| Metric | This repo (2080 Ti 22G) | Upstream author (RTX 5060 Ti 16G) |
|---|---|---|
| **decode** | **36.6 tok/s** | 31.7 tok/s |
| MTP acceptance | 58.6% / 61.6% (n_max=3 + ReplaySSM) | — |
| Context | 262,144 tokens | 256K |
| Vision | mmproj BF16 on GPU, all checks pass | ✅ |
| VRAM | 16,552 / 22,528 MiB (**~6 GB headroom**) | full |
| Quality | arithmetic/code/Chinese/logic all correct, no IQ3 garble | — |

**Archive · KV streaming** (lossless full-history attention): exact answer recovered at a **260,096-token** needle-in-a-haystack — see [kv-streaming/](kv-streaming/README_EN.md).

Upstream's rc3 prebuilt has only been physically tested on an RTX 5060 Ti (sm_120); **the published Turing numbers here remain one of a kind**. Decode is bandwidth-bound (2080 Ti 616 GB/s vs 5060 Ti 448 GB/s), and the bandwidth advantage converts directly into speed.

## Two routes, when to use which

| | **KVMem (headline · retrieval)** | KV streaming (archive · lossless) |
|---|---|---|
| Mechanism | completed KV blocks stored in **host RAM**; each step **retrieves** relevant blocks into a bounded GPU working set; attention sees only the working set | KV backing store in **system RAM**, small resident pool on GPU, attention sees the **entire history** |
| Decode at 260K | **constant ~36.6 tok/s** | 30–42 tok/s inside the resident pool, **~9.6 tok/s past 135K** |
| Attention fidelity | retrieval-approximate (near-lossless on benchmarks) | **lossless** |
| Best at | everyday long agent conversations, multi-turn tool work | exact full-document synthesis |

## Quick start (Windows)

```bat
cd kvmem
prepare-source.bat                        :: fetch upstream source (baseline 13b7a15) + apply patches
cd kvmem-src && build-windows-sm75.bat    :: build for sm_75
python ..\kvmem\tools\needle-test.py --port 18200 --chunks 30 --chunk-tokens 8000
```

No-build option: the official [v0.16.0-rc3 Windows prebuilt](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) covers RTX 20-series compile targets. Requirements: VS 2022/18 MSVC, CUDA Toolkit **13.2 Update 2**.

## Repo layout

```text
├─ README.md            ← Chinese main doc (canonical)
├─ README_EN.md         ← this file
├─ kvmem/               ★ headline: KVMem port (build script, patches, tools, reports)
└─ kv-streaming/        ◇ archive: adaptive KV streaming (lossless route)
```

## Honest boundaries

- KVMem is **retrieval-approximate**: full-document synthesis may miss blocks — switch to streaming for those tasks.
- One generation cannot exceed `--kvmem-gen-reserve` (16,384 tokens on the IQ3 recipe, including thinking); still unfixed upstream.
- Measured on **one GPU, one model line**. Feedback from other Turing cards welcome.
- No model weights, CUDA redistributables, or keys are shipped.

## License & credits

MIT (see [LICENSE](LICENSE)). KVMem originates from [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp); adaptive KV streaming from [RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming). This repo ships port patches, build scripts and measurements only — no weights.

---

*RTX 2080 Ti · sm_75 · Turing · KVMem · retrieval long-context · KV streaming · 22GB VRAM · 262K context · local LLM · Qwen3.8 · MTP speculative decoding · llama.cpp*

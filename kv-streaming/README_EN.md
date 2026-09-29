# 🚀 KV-Streaming for RTX 2080 Ti (sm_75): 260K Context on a 22 GB Card (archive · lossless route)

> This folder archives the complete **adaptive KV streaming** route: sm_75 patches, build scripts, measurements and the troubleshooting guide. It complements the **KVMem headline** in the root README — use this route when a task needs **lossless full-history attention** (exact full-document synthesis).

This makes [adaptive KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) work on **RTX 2080 Ti / Turing (sm_75)** — a 2018, 22 GB GPU — so you can serve a **260,000-token context** with a **27B model** that would otherwise need a 40GB+ card.

> **The trick:** when VRAM can't hold the KV cache, the context breaks. KV streaming moves the "memory" (KV cache) to system RAM on demand — the GPU keeps a small resident pool and streams the rest. Context ceiling goes from ~53K to 262K.

**中文完整文档:** [README.md](README.md) — canonical.

## Measured headline

- Exact answer recovered at a **260,096-token** needle-in-a-haystack (prefill 197 tok/s, decode 3.96 tok/s streamed, MTP draft 12/12).
- Decode: **30–42 tok/s inside the resident pool; ~9.6 tok/s past the 135K resident boundary**.
- sm_75 support upstream lacks: 2 real Turing crashes fixed (32-column partial MMA for streamed prefill; native vector path for small MTP batches).

## Quick start (Windows)

```bat
scripts\prepare-source.bat      :: clone + apply sm_75 patches
scripts\build-sm75.bat          :: build (quantized Flash Attention on)
scripts\package-sm75.bat        :: package server + DLLs
```

Full trade-off tables, the `--fit off` headline pitfall, the q5_1 `GGML_CUDA_FA_ALL_QUANTS=ON` requirement and the ops report are in the [Chinese README](README.md).

## License

MIT (see [../LICENSE](../LICENSE)). Upstream: [llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming). No model weights redistributed.

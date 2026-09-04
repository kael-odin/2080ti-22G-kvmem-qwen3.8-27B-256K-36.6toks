# kv-streaming-2080ti-22G

RTX 2080 Ti (sm_75) compatibility patch, reproducible Windows build, and measured Qwen3.8-27B long-context profiles for [adaptive KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming).

## What This Fixes

The upstream experimental branch was validated primarily on an RTX 5070 Ti. On an RTX 2080 Ti, a multi-page streamed prefill reached about 58K tokens and failed with:

```text
flash_attn_ext_f16 has no device code compatible with CUDA arch 750
CUDA error: invalid argument
```

The patch in `patches/` adds two Turing-specific dispatch rules:

1. Large streamed prefill uses the 32-column `<256, 256, 4, 8>` partial MMA specialization supported by sm_75 instead of the unsupported 64-column specialization.
2. Small batches of at most 32 queries, including MTP verification batches, use the native vector partial path. This avoids a long-context logits mismatch that previously accepted EOS too early.

The non-Turing path remains unchanged.

## Verified Result

A real 260,096-token needle-in-a-haystack request completed on one RTX 2080 Ti 22 GB with:

- Qwen3.8-27B IQ4_XS community fine-tune
- Q4_0/Q4_0 target KV in pinned host memory
- 1,024 MiB adaptive resident/staging pool
- MTP draft-2
- vision projector loaded
- exact answer recovered
- 197.03 prompt tokens/s
- 3.96 generated tokens/s
- 12/12 MTP draft tokens accepted
- 1,139 MiB minimum free VRAM

This proves compatibility and full-context correctness for the tested case. It does not imply production-grade coverage across models, GPUs, KV types, or multi-slot serving.

## Better 22 GB Profiles

Adaptive streaming is a capacity feature, not a free speedup. If fixed KV fits, fixed KV is much faster for decode.

| Profile | Model | Context | KV | Streaming | b/ub | Measured result |
|---|---|---:|---|---|---|---|
| High-quality | GSQ-RCO IQ3_S MTP | 170K | Q8_0/Q8_0 | off | 512/128 | Loads with MTP+vision and 1.18 GiB free VRAM |
| Maximum context | GSQ-RCO IQ3_S MTP | 262K | Q4_0/Q4_0 | off | 512/128 | 260K NIAH exact; 182.20 pp t/s; 16.75 tg t/s; 0.88 GiB minimum free VRAM |
| Preserve fine-tune | Turbo IQ4_XS MTP | 262K | Q4_0/Q4_0 | 1,024 MiB | 512/512 | 260K NIAH exact; 197.03 pp t/s; 3.96 tg t/s; 1.11 GiB minimum free VRAM |

The GSQ model is smaller and therefore can keep a 262K Q4_0/Q4_0 cache on the GPU. It decodes about 4.2x faster than the streamed Turbo profile at the end of a 260K prompt, but it is based on the original Qwen3.8-27B rather than the Turbo community fine-tune.

At 64K on the GSQ model, fixed Q8_0/Q8_0 was the best tested quality/speed point:

| KV | Mode | Prefill t/s | Decode t/s | Exact NIAH |
|---|---|---:|---:|---|
| Q4_0/Q4_0 | fixed | 393.29 | 25.31 | yes |
| Q8_0/Q4_0 | fixed | 382.50 | 25.79 | yes |
| Q8_0/Q8_0 | fixed | 392.60 | 26.70 | yes |
| Q8_0/Q4_0 | adaptive 1,024 MiB | 402.70 | 12.88 | yes |
| Q8_0/Q8_0 | adaptive 1,024 MiB | 398.44 | 10.78 | yes |

Q8_0/Q4_0 did not reproduce the severe prefill slowdown seen in another model/build, but it had no speed advantage over Q8_0/Q8_0 here. Use Q8_0/Q8_0 while it fits, Q4_0/Q4_0 for 262K fixed KV, and adaptive streaming only when model identity or capacity requires it.

## Build

Requirements:

- Windows 10/11 x64
- Visual Studio 2022 Build Tools with MSVC
- CUDA Toolkit 12.4
- Git
- Python 3.11 and `pip install -r requirements.txt` for the validation tool

Run from a normal command prompt:

```bat
scripts\prepare-source.bat
scripts\build-sm75.bat
scripts\package-sm75.bat
```

The source script checks out adaptive-KV commit `d873e5d` and applies the sm_75 patch. The build explicitly sets `CMAKE_CUDA_ARCHITECTURES=75` and enables all quantized Flash Attention instances.

## Validate

The test controller refuses to run while another `llama-server` or watchdog is active. It binds only to `127.0.0.1:18080-18089`, records the child PID, terminates only that process tree, and writes one immutable result directory per run.

```bat
python tools\run_case.py ^
  --server bin\llama-server-kv.exe ^
  --model D:\models\Qwen3.8-27B.gguf ^
  --mmproj D:\models\mmproj.gguf ^
  --ctx 262144 ^
  --stage-mib 1024 ^
  --cache-k q4_0 ^
  --cache-v q4_0 ^
  --mtp ^
  --load-only ^
  --label load-262k
```

For a real request, pass an OpenAI `messages` JSON array with `--messages` and repeat `--expect` for required answer fragments. Keep output and request timeouts bounded.

## Model Comparison

A matched 96-chunk WikiText-2 run on the same sm_75 binary and Q8_0/Q8_0 KV produced:

| Model | PPL |
|---|---:|
| TurboFCFusion IQ4_XS MTP | 6.0503 +/- 0.0637 |
| GSQ-RCO IQ3_S MTP | 6.5148 +/- 0.0711 |

This is an end-model comparison, not a pure quantization comparison: the Turbo file is a community fine-tune while GSQ-RCO quantizes the original Qwen3.8-27B base. The GSQ model card separately reports 99.8% recovery of its own BF16 base across AIME25, GPQA-Diamond, and LiveCodeBench v6. Both facts matter.

See `results/benchmark-summary.csv` and `docs/report-zh.md` for details and caveats.

## Safety and Scope

- The scripts do not modify an existing llama.cpp installation.
- Do not run the validation tool alongside a production llama-server on the same GPU.
- A load-only pass proves allocation and startup, not long-context correctness.
- The 262K GSQ fixed-KV request reached only 897 MiB free VRAM at peak. Treat it as usable but close to the limit; 250K is the safer everyday capacity.
- Raw model files, CUDA redistributables, private prompts, API keys, and machine-specific logs are intentionally excluded.

## Upstream and License

This repository contains a small patch and test/build tooling. The adaptive implementation itself comes from RaymondHuang210129's llama.cpp fork and remains subject to its upstream llama.cpp MIT license. GSQ/RCO models and tooling have their own upstream licenses. No model weights are redistributed here.

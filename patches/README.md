# Patches

## 0001-cuda-fix-adaptive-kv-streaming-on-sm75.patch

Minimal 17-line fix against the adaptive-KV fork at commit `d873e5d` (llama.cpp b10450 base).

What it changes in `ggml/src/ggml-cuda/fattn.cu`:

1. Large streamed prefill (more than 32 queries) on Turing dispatches to the 32-column `<256, 256, 4, 8>` partial MMA specialization, which has real sm_75 device code, instead of the 64-column specialization that compiles to `NO_DEVICE_CODE` on sm_75.
2. Batches of at most 32 queries - including MTP verification batches - route to the native vector partial path, avoiding a logits mismatch that previously caused a premature EOS acceptance at extreme context.

Non-Turing GPUs keep the original 64-column path.

## 0002-merge-adaptive-kv-into-b10816-sm75.patch

Complete net diff (54 files, ~10.6k insertions) from clean upstream llama.cpp **b10816** (`427291b5b`) to the merged tree (`3373a00fa`) that carries:

- the full adaptive KV streaming implementation (60 commits from RaymondHuang210129's fork, squash-merged),
- the sm_75 dispatch fix above,
- three merge-corrective fixes required by upstream evolution between b10450 and b10816 (see `docs/b10816-merge-notes-zh.md`).

Verified to apply cleanly with `git apply` on a pristine b10816 checkout.

## Reproducing the b10816 build

```bat
git clone https://github.com/ggml-org/llama.cpp src-b10816
cd src-b10816
git checkout 427291b5b
git apply path\to\0002-merge-adaptive-kv-into-b10816-sm75.patch
cmake -S . -B build -G "Ninja Multi-Config" ^
  -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DGGML_CUDA_CUB_3DOT2=OFF -DGGML_CUDA_NCCL=OFF ^
  -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF ^
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON
cmake --build build --config Release --target llama-server --parallel 6
```

Requires MSVC 19.44 (VS 2022 Build Tools) and CUDA Toolkit 12.4 on Windows.

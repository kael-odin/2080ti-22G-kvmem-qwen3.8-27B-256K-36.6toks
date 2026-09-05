# b10816 合并说明（中文）

日期：2026-09-05 ｜ 合并人：kael-odin ｜ 状态：代码合并完成，二进制已产出，8K 冒烟通过，长上下文重验待做

## 一句话

把 adaptive-KV 流式实现（含 sm_75 修复）从旧基线搬到了上游 llama.cpp **b10816**（`427291b5b`）
上，产出可复现补丁 `patches/0002-merge-adaptive-kv-into-b10816-sm75.patch`，本地已编出
`build 10878` 二进制并通过 8K 冒烟。README 里的大表数字仍来自旧基线（`d873e5d`）构建，
新基线的 260K 真实验收还没做，不能直接套用。

## 基线

- 上游：`ggml-org/llama.cpp @ 427291b5b`（b10816，`llama.cpp-src` 本地同版本）。
- 被合并方：`RaymondHuang210129/llama.cpp-adaptive-kv-streaming @ d873e5d`
 （`feature/adaptive-kv-stream` 分支尖，约基于 b10450）。
- 上游在这之间演进了约 366 个提交，其中包含新的稀疏注意力（sparse attention）路径，
  正好落在同一套 Flash Attention MMA 内核里，所以这是一次真正的三路合并，不是摘樱桃。

合并提交（本地 `src-b10816` 仓库 `b10816-kvstream` 分支）：

- `b38934d8b` Merge adaptive-KV fork into b10816 with sm75 support（54 文件，约 10.6k 新增）
- `3373a00fa` Fix fork tensor naming and b10816 indexer call for merged tree（2 文件修正）

## 冲突点与解法

### 1. `launch_fattn` 签名冲突（`ggml/src/ggml-cuda/fattn-common.cuh`）

- 上游新增了 `use_sparse` 参数；fork 新增了 `partial_dst / partial_meta` 输出。
- 解法：两个参数都要，新签名是
  `launch_fattn(..., use_sparse, ..., partial_dst = nullptr, partial_meta = nullptr)`。
- `output_partial`（是否输出部分结果）与 `stream_k` 互斥，已加断言。
  输出 partial 时强制单 block per tile，避免 fixup 竞争，保证逐位精确。

### 2. MMA 内核模板正交化（`ggml/src/ggml-cuda/fattn-mma-f16.cuh`）

- 上游模板已有 `use_sparse`；fork 需要 `output_partial`。
- 解法：模板同时携带两者，正交组合：
  `flash_attn_ext_f16_process_tile<..., use_sparse, ..., output_partial>`。
- partial 输出时跳过归一化除法，改为写 `meta_j[0]/meta_j[1]` 到 fixup 行，
  由上层 softmax 做精确合并。
- 上游 b10816 的稀疏分发逻辑原样保留。Qwen3.8 的 DKQ=256 不走稀疏路径，
  所以本次验证不受稀疏分支影响，但代码层面两者共存。

### 3. Turing sm_75 修复保留（`ggml/src/ggml-cuda/fattn.cu`）

0001 补丁的两条 Turing 分流规则在合并后原样保留：

- 大型流式 prefill（Q > 32）：用 sm_75 有设备代码的 32 列
  `<256, 256, 4, 8>` partial MMA，而不是 64 列 `<256, 256, 8, 8>`。
- 小 batch（Q ≤ 32，含 MTP 验证 batch）：走 native vector partial，
  避开此前长上下文误收 EOS 的 logits 偏差。
- 非 Turing 卡保持原来的 64 列路径不变。

### 4. 上游演进导致的两次修正（`3373a00fa`）

- `src/llama-kv-cache.cpp`：fork 里残留的 `name_tag` 变量在 b10816 已不存在，
  改为直接命名 `cache_k_l%d / cache_v_l%d`。
- `src/llama-memory-hybrid-idx.cpp`：上游把字符串字面量调用改成了按值传参，
  indexer KV 缓存现在传 stage size `0`，明确常驻 GPU、不参与流式。

## 复现构建

```bat
scripts\prepare-source-b10816.bat
scripts\build-b10816-sm75.bat
```

要求：Windows 10/11 x64，VS 2022 Build Tools（MSVC 19.44），CUDA Toolkit 12.4，
`CMAKE_CUDA_ARCHITECTURES=75`，`GGML_CUDA_FA_ALL_QUANTS=ON`。

补丁已验证可在干净的 b10816 检出上 `git apply` 干净应用。

## 验证状态

| 项目 | 状态 |
|---|---|
| 8K 冒烟（MTP draft-2，2/2 接受，`--kv-stream-stage-mib` 可见） | 通过（build 10878） |
| 官方 b10816 20K 前缀基线（K4V4，`validation-runs-prod`） | 通过，可作对照 |
| 64K / 128K / 260K 真实 NIAH（新基线） | 未做，这是下一步 |
| PPL（WikiText-2，96 chunk） | 未做，建议与旧表对照 |

在长上下文重验完成前，对外只宣称“兼容性冒烟通过”，性能数字仍归属旧基线构建。

## 文件对照

- 本仓库补丁：`patches/0002-merge-adaptive-kv-into-b10816-sm75.patch`
- 本地合并树：`kv-streaming/src-b10816`（分支 `b10816-kvstream`）
- 本地构建：`kv-streaming/build-b10816`，产出二进制 `kv-streaming/bin-b10816`
- 旧基线（对照用，勿删）：`kv-streaming/src`（`d873e5d` + fattn.cu 未提交修改）、
  `kv-streaming/bin`（build 10511）

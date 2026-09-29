# 踩坑手册（Troubleshooting）—— RTX 2080 Ti 22G · Qwen3.8 · KV 流式

> 按踩坑严重度排序。每条都是实际踩过、有日志/数据支撑的坑。
> 配套深度报告：[ops-stability-optimization-report.md](ops-stability-optimization-report.md)

---

## ⚠️ 头号坑：`--fit` 默认开启会静默把层降级到 CPU，与 KV 流式互斥

**症状**：启动即失败，报错极具误导性：

```
E llama_init_from_model: failed to initialize the context:
  block KV streaming requires the CUDA backend
```

看起来像"CUDA 没装好/驱动问题"，实际与 CUDA 安装毫无关系。

**真因**：llama.cpp 的 `--fit` 默认 on（"adjust unset arguments to fit in device memory"）。
当显存预算超限时（典型：262144 上下文 + draft-mtp，draft f16 KV 额外 ~1 GiB），
fit 会**静默把部分 transformer 层降级到 CPU**（via tensor_buft_overrides / 削减 n_gpu_layers）。
KV 流式要求**所有注意力层在同一个 CUDA 设备上**，遇到任何 CPU 层立即抛出上述错误。

**插桩证据**（`patches/0003` + `LLAMA_KV_STREAM_DBG=1`）：

```
kvdbg: layer il=3 has_kv=1 offload=1 dev=CPU     ← 262144 + draft-mtp 时
kvdbg: NULL-proc trigger il=3 dev_name=CPU reg=CPU
kvdbg: layer il=3 has_kv=1 offload=1 dev=CUDA0   ← 180224 时（对照组）
```

**修复**：加 `--fit off`。实测 262144 + draft-mtp + q5_1 KV + 视觉全通过：
显存 21.8/22.5 GiB，MTP acceptance 0.61，短程 decode 37.5 t/s，
135K（跨常驻边界）12.4-13.1 t/s，260K 级 NIAH 答案正确。

**通用排查口诀**：任何"重启后 CUDA 消失/引擎变 CPU"类报错，先查
`tasklist | findstr llama-server`（孤儿进程占显存）和是否被 fit 降层，
余量 < 2 GiB 时不要叠加视觉/超长上下文。

---

## 2. 上下文阈值：draft-mtp 在 2080 Ti 22G 上的实际边界

`--fit off` 后，q5_1 KV + draft-mtp + 视觉的组合实测边界：

| 上下文 | 结果 |
|---|---|
| 131072 / 163840 / 174080 / **180224** | ✅ MTP 激活 |
| 196608 / 229376 / 262144 | ❌ 显存预算超限 → fit 降层 → 头号坑报错 |

每 1K 上下文成本 ≈ 32 KiB（主 KV q5_1 28 KiB/token + draft f16 4 KiB/token，
仅 16 个全注意力层——Qwen3.5 混合架构的架构红利）。
262144 + draft-mtp 需要约 23.9 GiB 预算，物理不可行；**262K 与 MTP 二选一**：
- 262K：去掉 spec 参数（无 MTP，流式 decode ~10-17 t/s）
- MTP：ctx ≤ 180224（推荐 174080/180224）

## 3. 误导性报错合集

| 报错 | 真因 |
|---|---|
| `block KV streaming requires the CUDA backend` | fit 降层到 CPU（见头号坑），不是驱动问题 |
| `block KV streaming requires exactly one sequence (-np 1)` | Parallel Slots 未显式设 1（unsloth 表单留 auto 会触发） |
| `block KV streaming currently supports only the target context, not MTP/draft contexts` | 上游原版行为：上游 fork 本身不支持 MTP+流式（MTP 支持为本仓库作者的独立修复，见 patches/0002） |
| 引擎"变成 CPU"、prefill 掉一个数量级、CPU 九核跑满 | **标准构建没有 q5_1 KV 的 GPU 注意力内核**，静默回退 CPU。需要带 `GGML_CUDA_FA_ALL_QUANTS=ON` 的构建（本仓库 patches/0002 已含），或改用 q4_0 |

## 4. unsloth desktop 启动器相关（自定义引擎路径）

1. **Speculative Decoding 下拉框对自定义引擎基本无用**：选 MTP 实际只发 `--spec-default`（或 `--spec-type none`），且会把 Extra Arguments 里的 `--spec-type` / `--spec-draft-n-max` **静默过滤**（UI 提示语声称 "extras win"，实测不传）。**MTP 参数必须依赖引擎侧自动注入或直接用脚本启动。**
2. **Sampling 面板的采样参数只影响自带聊天界面**；第三方 agent 走 API 时以 agent 请求参数为准（请求级 > 启动级），Extra 里的采样参数同样只是兜底默认值，可省略。
3. **"vision projector in system memory" 警告**：对自定义引擎路径的保守提示（桌面端无法确认 mmproj 的 GPU 放置）。实测视觉全 GPU 正常工作，仅可能略慢。
4. 停止模型时注意**孤儿进程**：若经由包装脚本/垫片启动，停止操作可能只杀包装层。启动前先 `tasklist | findstr llama-server` 确认清场，否则残留进程占显存会导致新实例 CUDA 初始化失败。

## 5. 历史踩坑（Turing sm_75 / 旧基线时代）

1. **sm_75 无 64 列 partial MMA 设备代码**：上游 64 列 `<256,256,8,8>` 在 Turing 编译为 NO_DEVICE_CODE，长 prefill 到 ~57K 崩溃。修复：32 列 `<256,256,4,8>` 分流（patches/0001）。
2. **小批量（Q ≤ 32，含 MTP 验证 batch）logits 偏差**：曾导致 260K 极限测试误收 EOS（只输出 "ORCHID" 前缀）。修复：Q ≤ 32 走 native vector partial（patches/0001）。
3. **Windows WDDM write-combined KV 存储**：fork 的 write-combined host cache 在 Windows 上有问题，需要 `05ba2068a` 式修复（避免 write-combined）。
4. **草稿 KV 量化是净亏损**：draft f16 KV 1024 MiB → Q4 288 MiB，但计算缓冲 324 → 1360 MiB，净显存反而更差。draft KV 保持 f16。
5. **`--cache-ram 0` 不是 KV 流式**：它是 prompt-cache 的内存预算。KV 流式只认 `--kv-stream-stage-mib`。
6. **批处理脚本必须 CRLF + 谨慎用程序化改写**：python 文本模式改写会把 CRLF 变 LF，cmd 解析直接乱码；改完必须二进制校验行尾。
7. **测速必须预热多轮**：首条请求含 CUDA 图捕获，单轮数据严重低估；历史上有过 14↔43 t/s 乱跳的教训（170K 近满显存的软件层波动）。

## 6. 二进制/源码防丢失

- 本仓库 `patches/0002` 是**完整源码记录**（506,746 字节 diff = b10816 → 合并树，
  已校验与本地合并树逐字节一致）。按 patches/README.md 的配方可完整重建二进制。
- 2026-09-07 教训：本地旧基线 `src/`（d873e5d）与 `bin/`（build 10511，曾验证 260K+MTP）
  被当作垃圾删除，导致 262K+MTP 一度"无解"。**补丁即源码，勿删 patches/ 目录。**
- 新增 `patches/0003-debug-instrumentation-kvstream-init.patch`：本文档头号坑的
  插桩（`LLAMA_KV_STREAM_DBG=1` 逐层打印层/设备/触发点），排查同类问题直接复用。

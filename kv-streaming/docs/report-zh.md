# RTX 2080 Ti 22G 适配与实测报告

测试日期：2026-09-04

## 结论

本项目已经把 adaptive KV streaming 适配到 RTX 2080 Ti（sm_75），并修复了两个 Turing 专属问题：

1. 长 prompt 到约 58K token 后调用未生成 sm_75 设备代码的 64 列 partial MMA kernel，导致 CUDA `invalid argument` 和进程退出。
2. 第一版崩溃修复后，小型 MTP 验证 batch 仍走 Turing partial MMA，在 260K 极限上下文错误接受 EOS，只输出 `ORCHID`。

最终补丁采用分流策略：

- Turing 大型 prefill：使用 sm_75 支持的 32 列 `<256,256,4,8>` partial MMA。
- Turing 小型 batch（`Q <= 32`，包括 MTP 验证）：使用 native vector partial。
- Ampere 及更新架构：保持作者原来的 64 列路径。

修复后的 260,096-token NIAH 在 target K4V4 adaptive KV、MTP draft-2、视觉 projector 常驻时精确通过，MTP 12/12 接受。

## 测试硬件与软件

- GPU：NVIDIA GeForce RTX 2080 Ti 22,528 MiB，compute capability 7.5
- CPU：Intel Core i5-12400F，6C/12T
- RAM：32 GiB
- OS：Windows 11 / WDDM
- CUDA：12.4.99
- 编译器：MSVC 19.44 / Visual Studio 2022 Build Tools
- adaptive KV 基线：`d873e5db9a698a8063347e586ed047242d51fdce`
- CUDA 架构：`CMAKE_CUDA_ARCHITECTURES=75`

## 模型

### TurboFCFusion IQ4_XS MTP

- 社区多阶段融合模型
- 文件大小：15,309,039,136 bytes
- 权重类型：IQ4_XS，约 4.25 bpw
- MTP：包含 1 个 next-token prediction layer
- 视觉：匹配的 F16 projector

### GSQ-RCO IQ3_S MTP

- 原版 Qwen3.8-27B 的 GSQ + RCO 非均匀逐 tensor 混合精度量化
- 文件大小：12,120,016,960 bytes
- 实际类型：IQ3_S，约 3.4375 bpw
- SHA-256：`58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`
- MTP：包含 1 个 next-token prediction layer
- 官方 BF16 projector SHA-256：`13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16`

GSQ 模型卡的 “task-lossless” 是相对它自己的原版 Qwen3.8-27B BF16：AIME25 和 LiveCodeBench v6 相同，GPQA-Diamond 低 0.51 分，三项平均保留 99.8%。它不是 TurboFCFusion 的重新量化，因此不能把二者差异解释为单纯 IQ3_S 对 IQ4_XS。

## MTP 兼容性

### 修复前

64K adaptive K4V4 在约 57,875 token 报错：

```text
flash_attn_ext_f16 has no device code compatible with CUDA arch 750
CUDA error: invalid argument
```

### 修复后

| 模型 | Context | KV | Mode | Prefill | Decode | MTP | NIAH |
|---|---:|---|---|---:|---:|---:|---|
| Turbo | 64K | K4V4 | adaptive 1536 | 504.87 | 33.33 | 12/12 | exact |
| Turbo | 128K | K4V4 | adaptive 1536 | 366.00 | 10.33 | 12/12 | exact |
| Turbo | 260K prompt | K4V4 | adaptive 1024 | 197.03 | 3.96 | 12/12 | exact |

所以修复后无需在 262K 禁用 MTP。草稿 context 仍保持普通 GPU KV，只有 target KV 流式化。这是合理结构，因为 MTP 每轮访问草稿 KV，把它也搬到主存会进一步拖慢投机解码。

## KV 精度、上下文与速度

### GSQ 64K 固定 KV

| KV | Prefill tok/s | Decode tok/s | 最低空闲显存 | NIAH |
|---|---:|---:|---:|---|
| K4V4 | 393.29 | 25.31 | 6,051 MiB | exact |
| K8V4 | 382.50 | 25.79 | 5,373 MiB | exact |
| K8V8 | 392.60 | 26.70 | 4,984 MiB | exact |

这组实测中 K8V4 的 prefill 只比 K4V4 慢约 2.7%，没有复现另一模型/构建中的严重慢路径。不过 K8V4 也没有速度优势；在显存允许时应直接选 K8V8。

### GSQ 64K adaptive KV

| KV | Prefill tok/s | Decode tok/s | NIAH |
|---|---:|---:|---|
| K8V4 adaptive 1024 | 402.70 | 12.88 | exact |
| K8V8 adaptive 1024 | 398.44 | 10.78 | exact |

adaptive prefill 保持得很好，但 decode 约下降一半。因此原则是：**固定 KV 能放下时不要开启 streaming。**

### 固定 K8V8 上下文边界

以下均加载 MTP 和视觉 projector：

| Context | ubatch | 最低空闲显存 | 结论 |
|---:|---:|---:|---|
| 128K | 128 | 2,996 MiB | 充足 |
| 160K | 256 | 1,223 MiB | 推荐均衡档 |
| 170K | 128 | 1,212 MiB | 高质量最大档 |
| 170K | 256 | 876 MiB | 临界，不推荐 |
| 192K | 128 | 196 MiB | 淘汰 |

### 262K

| 模型 | KV | Mode | ubatch | 260K Prefill | Decode | 最低空闲显存 | NIAH |
|---|---|---|---:|---:|---:|---:|---|
| GSQ IQ3_S | K4V4 | fixed | 128 | 182.20 | 16.75 | 897 MiB | exact |
| Turbo IQ4_XS | K4V4 | adaptive 1024 | 512 | 197.03 | 3.96 | 1,139 MiB | exact |

GSQ 固定 KV 的极限 decode 约为 Turbo adaptive 的 4.2 倍。它把更小权重节省出的显存直接留给完整 GPU KV，是 22G 卡上更有效的 262K 方案。

但 897 MiB 低于 1 GiB 推荐余量，所以 262K 属于可用临界档。日常建议 250K；需要严格 262K 时关闭其他 GPU 应用。

## 草稿 KV 量化实验

将 262K MTP 草稿 KV 从 F16/F16 改为 Q4_0/Q4_0：

- 草稿 KV：1,024 MiB -> 288 MiB
- 草稿计算 buffer：324 MiB -> 1,360 MiB
- 全卡最低空闲显存：566 MiB -> 346 MiB

所以量化草稿 KV 在该构建上净效果更差，不采用。保留 F16 草稿 KV。

## WikiText-2 PPL

同一 sm_75 引擎、96 chunks、context 1024、K8V8：

| 模型 | PPL |
|---|---:|
| TurboFCFusion IQ4_XS | 6.0503 +/- 0.0637 |
| GSQ-RCO IQ3_S | 6.5148 +/- 0.0711 |

Turbo 的 PPL 更低。由于两者底模不同，此结果只能说明当前两个最终 GGUF 的整体语言建模差异，不能证明 GSQ 量化算法劣于 IQ4_XS。

## 推荐配置

1. 日常现有工作流：继续使用已经稳定的 Turbo 170K K4V4 生产文件，不改动。
2. 高精度 KV 档：GSQ 160K K8V8，`b512/ub256`。
3. 高精度最大档：GSQ 170K K8V8，`b512/ub128`。
4. 超长快速档：GSQ 250K K4V4 fixed，`b512/ub128`；严格需要时可升到 262K。
5. 保留社区微调行为的超长档：Turbo 262K K4V4 adaptive，stage 1024 MiB，MTP draft-2。
6. 不推荐 K8V4 作为常规档：当前 GSQ 上它没有明显速度优势；只在后续长程质量实验表明 K8V8 放不下而 K4V4 质量不足时再考虑。

## 后续质量评测

PPL 和 NIAH 已完成。模型替换前还应运行统一的生成评测：

- AIME25：数学推理
- GPQA-Diamond：高难科学问答
- LiveCodeBench v6：代码生成
- 本地 coding-agent 真实任务集
- 视觉确定性测试与 OCR/截图任务
- 风格、拒答率、思考长度和去审查行为

公平比较必须使用同一 prompt、采样参数、上下文、MTP 设置和 scorer，并明确两者是不同底模，不只是不同比特量化。

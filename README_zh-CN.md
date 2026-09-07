# 🚀 KV 流式 for RTX 2080 Ti (sm_75)：22G 显存跑 26 万上下文

**跑超大上下文，不需要超大显存。** 本仓库让 [adaptive KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) 在 **RTX 2080 Ti（Turing / sm_75，2018 年的老卡，22G 显存）** 上真正可用——用 27B 模型跑到 **26 万 token 上下文**，而这原本需要 40G 以上显存。

> **一句话原理：** 推理时显存不够装 KV 缓存，上下文就断了。KV 流式把"记忆(KV 缓存)"**按需搬到系统内存(RAM)**，显存保留一块常驻池、其余分页从 RAM 流式拉取。上下文上限从 ~5 万拉到 26 万。

---

## ✨ 为什么值得关注（以及它不是免费午餐）

**每一个小显存玩家的痛：** 模型塞得下，但**长文档的 KV 缓存塞不下**。27B 模型 + 26万 token 的 KV 需要 **~29G**，22G 卡根本装不下。

**这个仓库的解法：** adaptive KV streaming——KV 的"后备存储"放在**系统内存**，显存只留一小块 staging 池。你在**同一张卡**上把上下文拉到 **~4.7 倍**。

| | 固定 KV（旧办法） | **本仓库：KV 流式** |
|---|---|---|
| 22G 卡最大上下文 | ~5.3 万 token ❌ | **~26.2 万 token ✅** |
| 生成速度 | 快 | **常驻区内 30–42 tok/s；越过 135K 常驻边界后约 9.6 tok/s** |
| KV 占显存 | 全部 | 只占当前窗口 |
| 代价 | — | 改用系统内存 |

**实测：22G 老卡，26 万 token，长文测试精确回答。卡在 260,096 token 的 needle-in-a-haystack 上精确通过。**

---

## 2. 这值不值？（说实话）

| 说法 | 真实吗 |
|---|---|
| "我发明了长上下文" | ❌ 不，上游就有 |
| "让 2018 老卡重新能跑长文" | ✅ **是，独有且可验证** |
| "sm_75 支持是上游没有的" | ✅ **是（修了 2 个真崩溃）** |
| "22G 卡在上下文上能比 40G 卡" | ✅ 仅指**上下文长度**，不包括速度 |

**真正新的是：** 一套 **开箱即用的补丁 + 实测报告**，让 KV 流式在**最老的 sm_75 Turing 卡**上跑起来，并用数据证明。如果你有 2080 Ti / 2070 / 2060 这些 Turing 卡，觉得"长上下文和我无关"——这个仓库就是答案。

---

## 3. 文件与快速开始（Windows）

**环境：** Windows 10/11、Visual Studio 2022 Build Tools(MSVC)、CUDA 12.4、Git、Python 3.11。

```bat
:: 1. 拉取源码 + 打 sm_75 补丁
scripts\prepare-source.bat

:: 2. 为 sm_75 编译（显式启用量化 Flash Attention）
scripts\build-sm75.bat

:: 3. 打包服务器 + DLL 到 bin\
scripts\package-sm75.bat
```

跑一个验证用例：

```bat
python tools\run_case.py ^
  --server bin\llama-server.exe ^
  --model  D:\models\Qwen3.8-27B.gguf ^
  --mmproj D:\models\mmproj.gguf ^
  --ctx 262144 --stage-mib 1024 ^
  --cache-k q4_0 --cache-v q4_0 --mtp --load-only --label load-262k
```

> 建议新用户直接用 **b10816 最新上游** 版本：`prepare_source-b10816.bat` + `build-b10816-sm75.bat`。它把修复合并到上游 366 个 commit 之后的版本上。

---

## 4 实测结果（RTX 2080 Ti 22G，Qwen3.8-27B）

**核心记录：260K needle-in-a-haystack——精确命中：**

| 指标 | 数值 |
|---|---|
| 上下文 | **260,096 tokens** |
| 预铺速度 | 197.03 tok/s |
| 生成速度 | 3.96 tg/s（流式）|
| MTP 草稿 | 12/12 接受 |
| 最低余显存 | 1,139 MiB |

**固定 vs 流式 KV —— 解码取舍（Turbo IQ4_XS）：**

| 上下文 | KV 模式 | 预铺 t/s | 解码 t/s | 最低余显存 |
|---|---|---:|---:|---:|
| 64K | 固定 | 516.64 | 33.35 | 3.54 GiB |
| 128K | 固定 | 308.64 | 25.99 | 1.11 GiB |
| 170K | 固定 | 377.65 | 25.50 | **69 MiB——危险** |
| 262K | 流式(1,024 MiB池) | 205.76 | **3.08** | 2.30 GiB |

**一句话：** 固定 KV 装得下时，解码更快；**装不下时才用流式**——而要在 22G 卡上跑到 262K，**流式是唯一办法**。

**常驻边界实测（UD-IQ4_XS，q5_1 KV，3,072 MiB 池，2026-09-06）：** 一个 135,087-token 的请求跨过常驻分区（trace：resident pages 511→510，ring slots 16→32），标记精确找回，decode 9.64 t/s（prefill 226.4 t/s）。在常驻池内（≤ ~13 万 token）decode 保持 30–42 t/s。细节与限制见[运维报告](docs/ops-stability-optimization-report.md)。

**2026-09-07 更新——q5_1 KV 必须带 `GGML_CUDA_FA_ALL_QUANTS=ON`（官方标准构建会 CPU 回退）：** 用这一个 CMake 开关重建官方 b10816（源码零改动）后，fixed q5_1 获得 GPU 注意力内核，**135K 全场最快**：decode 23.8 t/s（流式 9.2–10.2、fixed q4_0 18.6），prefill 339.8（峰值 412）。日常配置验收（TURBO-Fable，-c 174080，q5_1，ub256）：decode 24.8–26.2、标记精确、1024px 视觉正常、显存峰值 22.13G 无 OOM。结论：**≤ ~18 万上下文用 fixed q5_1（带该开关）；流式退居 >18 万档（262K）**——流式 9.5 t/s 的代价来自逐层部分归并/尾页机制，而非注意力本身。详见[运维报告](docs/ops-stability-optimization-report.md)第十节。

*(其他配置（含 GSQ-RCO、Q8_0/Q8_0 对照）见 [`results/benchmark-summary.csv`](results/benchmark-summary.csv) 和 [`docs/report-zh.md`](docs/report-zh.md)。)*

---

## 5. sm_75 移植到底做了什么（真正的活）

上游 KV 分支在 **RTX 5070 Ti（Ampere/sm_80）** 上测的。到 **Turing（sm_75）** 在 ~58K token 就崩：

```text
flash_attn_ext_f16 has no device code compatible with CUDA arch 750
CUDA error: invalid argument
```

两个**只针对 Turing** 的修复：

1. **超大流式预铺** 用 sm_75 支持的 32 列 `<256,256,4,8>` partial MMA（上游用的 64 列 sm_75 不支持）。
2. **小型 MTP 验证 batch（≤32 queries）** 走 native vector 路径——修掉一个在 260K 极限下误判 EOS、只输出 "ORCHID" 的 logits 问题。

非 Turing 卡保持上游原路径不变。

**合并到最新上游 b10816（2026-09-05）：** 真正的三方合并（晚了 366 个 commit），保留上游新加的 sparse-attention 路径。见 [`补丁/0002-merge-adaptive-kv-into-b10816-sm75.patch`](patches/0002-merge-adaptive-kv-into-b10816-sm75.patch) + [`docs/b10816-merge-notes-zh.md`](docs/b10816-merge-notes-zh.md)。

---

## 6 诚实的边界与安全

- 这是**容量**特性，不是免费提速：流式在 KV 频繁跨 GPU↔RAM 时有解码代价。能塞下时固定 KV 更快。
- *load-only* 只证明**能分配**，不证明**长上下文正确**。
- 只在**一张卡**（2080 Ti）、一个模型线（Qwen3.8-27B）上测过，不代表覆盖所有模型/量化/多槽。
- 别在 `run_case.py` 跑验证时,同一 GPU 上再起生产 llama-server。
- 验证工具只监听 `127.0.0.1`，跟踪自己的子进程，每次运行写独立结果目录。
- 仓库不包含模型权重/私有 key/CUDA 运行时。

---

## 7 许可与致谢

MIT（见 [LICENSE](LICENSE)）。adaptive KV 实现源自 **RaymondHuang210129** 的 [llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)，遵循上游 llama.cpp 的 MIT。GSQ/RCO 模型有各自许可；**本仓库不分发模型权重**。

**English:** see [README.md](README.md).

---

*RTX 2080 Ti · sm_75 · adaptive KV streaming · 22GB 显存 · 26万上下文 · Qwen 3.8 · MTP 投机解码 · 长文本推理 · llama.cpp Turing 移植*
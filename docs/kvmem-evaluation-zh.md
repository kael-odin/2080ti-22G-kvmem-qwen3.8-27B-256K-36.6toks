# KVMem 方案评估 —— 与现有 kv-streaming 的对比

> 评估日期：2026-09-18
> 评估对象：<https://github.com/kvmem/kvmem-llama.cpp>（本地已克隆到 `F:\AI-Models\kvmem-llama.cpp`）
> 结论一句话：**技术上完全能实现目标生产配置，且是"不掉速"的唯一现实路径；但本机（2080 Ti / sm_75）不在作者测试矩阵内，需自行验证。**

---

## 一、先说清楚：kvmem 和 kv-streaming 不是同类方案

两者都能在 22G 卡上跑 26 万 token，但**实现路线的哲学完全不同**。

| | 你的 kv-streaming | KVMem |
|---|---|---|
| 注意力覆盖范围 | **全部历史**（无损） | **仅检索窗口**（近似） |
| KV 放哪 | 后备存储在 RAM，显存留常驻池 + 传输环 | 已完成块存 RAM，显存放检索工作集 |
| 长上下文时注意力计算量 | 随上下文**线性增长** | **恒定**（= budget + gen_reserve） |
| 长上下文时 PCIe 流量 | 随上下文增长 | 每轮只传检索命中的块 |
| 核心机制 | 按 256-token 页异步预取 | 用当前 query 检索相关块 |
| 减速 | **结构性，不可消除** | 不存在（代价转移到了检索质量上） |

**一句话概括**：kv-streaming 是"把全部记忆搬过来看"，KVMem 是"只把相关的记忆搬过来看"。
**减速不是被优化掉的，是被换掉了。**

> 有意思的是，kvmem 的 README **点名对比了 Raymond Huang 的 adaptive KV-cache streaming**——也就是你这条路线。他们承认你的方法保真，但明确写道："更长上下文会同时增加注意力开销和 PCIe 流量，最终拖慢 decode"。他们就是冲着这个痛点做的。

### 代价：它不是无损长上下文

KVMem 是**检索近似**，不是全量注意力。论文自报数据：

| Benchmark | KVMem（32K 窗口） | 全量 256K | 差异 |
|---|---|---|---|
| LongMemEval-S | 85.6% | 86.6% | **-1.0** |
| AgentLongBench | 60.9% | 59.5% | **+1.4** |

"near-lossless"是在这两个 benchmark 上的结论，**不是通用保证**。

**适用边界**：
- ✅ **甜点区**：工具型 agent（多轮读写文件、累积上下文）——query 明确，检索命中率高
- ⚠️ **风险区**：需要全局综合的任务（"总结全文主题"、"对比散落在 200K 各处的 5 个事实"）——检索可能漏块

---

## 二、目标生产配置 vs kvmem IQ3 recipe

你要的配置和 kvmem 的 **IQ3 recipe** 几乎一字不差：

| 你要的 | kvmem IQ3 recipe | 状态 |
|---|---|---|
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | `--default-model` **就是它** | ✅ |
| 开 MTP | `--spec-type draft-mtp`，MTP3 + ReplaySSM，服务器默认开 | ✅ |
| 视觉 | `--mmproj` + `--mmproj-offload`（GPU 视觉） | ✅ |
| 开满上下文 | `-c 262144`（实测跑到 262058 / 262144 = 99.97%） | ✅ |
| **KV cache 均 8 位** | `--kv-dtype q8_0` ← **就是默认值** | ✅ |
| 全程不明显掉速 | 256K 满上下文 decode **31.74 t/s** | ✅ |

启动脚本原文（`scripts/start-iq3.sh`）：

```bash
--recipe iq3 \
--default-model .../Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
--default-mmproj .../mmproj-Q8_0.gguf \
--default-vision-device gpu \
--kv q8_0 --budget 36864 --reserve 16384
```

服务器实际收到的参数（`scripts/start-server.py` 组装）：

```
-c 262144 -n 16384
--kvmem-budget 36864 --kvmem-gen-reserve 16384
--kv-dtype q8_0 --spec-type draft-mtp
--enable-thinking --reasoning-budget 4096
```

### 实测数据（RTX 5060 Ti 16G，256K 多轮工具测试）

33 个请求，每轮 +8K token，最终占满 262058 / 262144：

| 指标 | 数值 |
|---|---|
| 全任务聚合 prefill（首遍） | 437.13 token/s |
| 全任务聚合 prefill（有效，含缓存管理） | 242.06 token/s |
| 工具轮次聚合 decode | **31.74 token/s** |
| 最终代码 decode（512 token） | 30.53 token/s |
| MTP 草稿接受率 | 64.70% |
| 整卡显存峰值 | 15.6 GiB |
| 运行期 RAM 峰值（RSS） | 13.1 GiB |

**对比你现在的 262K 方案**（kv-streaming，135K 时 9.2–13 t/s）：**约 2.5–3 倍**。

---

## 三、必须知道的四条限制

1. **单次生成有硬上限** `--kvmem-gen-reserve`（IQ3 = 16384 token，**含思考**）。
   超了会直接崩：`no free GPU slot for block N` / `llama_decode(gen) failed rc=1`。
   你现在是 `-n 81920`。README 建议在 system prompt 注入：
   *"Keep each turn's output, including thinking, within 16384 tokens"*。
   作者文档里专门论证了为什么不能简单偷检索预算（会丢掉检索到的事实），后续计划加 ring buffer 解决。

2. **NVMe 层未实现**。仓库里有 `nvme_kv_tier.hpp` 源码，但 README 明写 "NVMe offload is not implemented"。
   → **host RAM 是硬上限**。你 32G RAM，262K 实测峰值 13.1G RSS，够用。

3. **是 pre-release**。当前 v0.16.0-rc1，仓库 2026-09-14 建立（很新），无 LICENSE 文件，
   测试矩阵只有一台机器。文档自述："IQ3 首次运行在第 10 个工具请求发生 CUDA `unknown error`"。

4. **API 无鉴权无 TLS**。README 明说 "No auth or TLS. Bind 127.0.0.1"。对外暴露需自行加反代。

---

## 四、硬件适配：风险与好消息

### ✅ 好消息：sm_75 没被 CUDA 13 砍掉

NVIDIA 官方明确：CUDA 13.0 移除的是 **CC 7.5 之前**的离线编译支持，
即 Maxwell (5.x)、Pascal (6.x)、Volta (7.0/7.2)。

> "the support for offline compilation for architectures prior to compute capability (CC) 7.5
> will be removed in the next major CUDA Toolkit release, CUDA 13.0"

**Turing 的 CC 7.5 正好是那条线本身**，是 CUDA 13.x 支持的**最低架构**，没有被移除。
→ 理论上可以用 CUDA 13.2 编译 sm_75 目标。

（参考：<https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/>）

### ⚠️ 风险一：nvcc 版本必须精确到 Update 2

CUDA 13.2 **基础版**的 nvcc 是 **13.2.51**（官方 release notes 组件表）。
而 kvmem README 点名警告：

> "A Windows build made with nvcc **13.2.51** produced **garbage output** from Qwen3.8-27B IQ3_S
> even with KVMem and MTP disabled; rebuilding unchanged source with **13.2.86** restored correct output."

**必须装 CUDA 13.2 Update 2（nvcc 13.2.86）或更新**，不能只装 13.2。

而且作者强调：**"编译成功 + 健康检查通过 + 小 Q8 模型测试通过"都不能证明 IQ3 推理正确**。
→ 验证时必须**人眼读输出有没有乱码**。

### ⚠️ 风险二：sm_75 本身无人验证

全仓库 grep `sm_75` / `Turing` / `2080` / `compute capab` → **零命中**。
作者原话："other GPU targets have not been tested here"。

而且你在 sm_75 上**已经踩过一次这类坑**：
`kv-streaming\public-repo\patches\0001-cuda-fix-adaptive-kv-streaming-on-sm75.patch`
就是在修 sm_75 特有的崩溃。同类风险在这里完全可能存在。

### 硬件对照表

| | 作者测试环境 | 本机 |
|---|---|---|
| GPU | RTX 5060 Ti 16G（Blackwell, **sm_120a**） | 2080 Ti 22G（Turing, **sm_75**） |
| CUDA | **13.2 Update 2（nvcc 13.2.86）** | 12.8.93（需新增 13.2） |
| 驱动 | — | 616.92（CUDA UMD 13.4，**满足** 13.x 要求的 ≥580） |
| 系统 | Ubuntu 22.04 WSL2 / native Windows | Windows 11 |
| 构建 arch | 默认 `CMAKE_CUDA_ARCHITECTURES=120a-real` | **必须改成 `75`** |

**次要问题**：启动脚本是 bash 且依赖 `ss`(iproute2)，Windows 上需走 WSL 或直接调 `start-server.py`。

### 💡 你的 22G 是优势

作者被 16G 逼着把 budget 压到 36K、reserve 压到 16K。你多 6G，且 **IQ3 权重只有 11.3G**
（比你现在用的 IQ4_XS 14.26G **轻约 3G**）。

省下的显存可以直接堆到 `--kvmem-budget` 和 `--kvmem-gen-reserve` 上——
**"单次生成不能超 16K"那条限制，在你卡上基本可以调没**。
这可能是 22G 卡真正吃到红利的地方。

---

## 五、视觉组件选型（F16 vs BF16）

### 结论：跟 kvmem 配方走 —— 用 `gsq-rco\mmproj-Qwen3.8-27B-BF16.gguf`

### 先纠正一个隐含前提

**F16 和 BF16 都是 16 位**，体积、显存占用、PCIe 流量**完全一样**。本机两份实测：

| 文件 | 大小 | 来源 |
|---|---|---|
| `models\qwen3.8-27B\turbofable\mmproj-F16.gguf` | 884.64 MB | DavidAU |
| `models\qwen3.8-27B\gsq-rco\mmproj-Qwen3.8-27B-BF16.gguf` | 888.01 MB | unsloth |

差 3.4 MB 是元数据/对齐，可忽略。

### 对速度的影响：基本为零

同样字节数、同样带宽、同样走 tensor core。**真正影响视觉速度的是另外三件事**：

| 因素 | 影响量级 | 说明 |
|---|---|---|
| **放 GPU 还是 CPU** | **53 倍** ⚠️ | 实测首次图片编码：GPU **0.41 秒** vs CPU **21.79 秒** |
| `--image-max-tokens` | 线性 | 512 vs 1024，直接决定视觉 token 数 |
| 图片分辨率 | 线性 | 决定视觉行数 |

**"放 GPU 还是 CPU"是唯一真正重要的变量。** CPU 那 21.79 秒是 IQ4 配方——正是为给 16G 省显存才把视觉头丢 CPU 的。你 22G 不用受这个罪，IQ3 配方本来就是 GPU 视觉。

### F16 与 BF16 的技术差异

| | 指数位 | 尾数位 | 特点 |
|---|---|---|---|
| F16 | 5 | 10 | 精度高，动态范围窄 |
| BF16 | 8 | 7 | 动态范围宽，精度低 |

视觉投影头是**权重加载**，不涉及动态范围溢出问题，**两者质量差异实践中不可感知**。

### 所以选型依据是"来源匹配"，不是精度

- kvmem 官方 IQ3 配方用的是 **unsloth 的 `mmproj-BF16.gguf`**，并且**进一步量化成 Q8_0**：
  ```bash
  build/bin/llama-quantize --max-buffer-size 256 \
    mmproj-BF16.gguf mmproj-Q8_0.gguf Q8_0
  ```
  888 MB → 约 450 MB（27 个 `ffn_down` 张量自动回落 F16）。
- 你现在脚本指向的是 turbofable 的 F16（DavidAU 版）。虽然 `models\模型文件分类说明.txt`
  写了"mmproj 只匹配架构不绑定微调版"所以能跑，但 **kvmem 的验证是在 unsloth BF16 那条路径上做的**。

**建议**：用 `gsq-rco\mmproj-Qwen3.8-27B-BF16.gguf`，并考虑按官方配方量化成 Q8_0 + `--mmproj-offload`。
既省一半显存，又和作者测过的路径一致。

---

## 六、本机 CUDA 现状盘点

见 `docs\CUDA版本管理.md`。要点：

- **驱动**：616.92（CUDA UMD **13.4**）—— 全局唯一，定义上限
- **Toolkit**：只有 **v12.8**（nvcc 12.8.93）；PATH 里残留的 v12.4 目录**已不存在**
- **运行时**：三套并存互不干扰 —— llama.cpp 用 CUDA 12.x、ComfyUI 用 CUDA 13.0、系统 Python 用 12.8

**对 kvmem 而言**：需**新增** CUDA 13.2 Update 2（与 v12.8 **并存**，不要卸载 12.8），驱动不用动。

---

## 七、建议的下一步

先做**低成本的探路验证**，别一上来就调性能。

### 阶段一：能不能跑（目标：排除乱码 / 崩溃）

1. 安装 **CUDA 13.2 Update 2**（nvcc 13.2.86），与现有 v12.8 并存
2. `git submodule update --init` + `scripts/apply-patches.sh`
3. **`CMAKE_CUDA_ARCHITECTURES=75`** 重新配置并编译（不是默认的 `120a-real`）
4. 用 IQ3_S 模型跑健康检查 —— **务必人眼读输出**
   （作者明确警告：health check 通过 ≠ IQ3 推理正确）

### 阶段二：能不能用（目标：找到本机最优 budget / reserve）

5. 跑 `scripts/multimodal_canary.py` 复现任务一（图文）和任务二（256K 工具）
6. 逐步调大 `--kvmem-gen-reserve`（22G 的余量优势在这里兑现）
7. 用 `kv-streaming\tools\` 里现成的 needle/marker 验证法交叉验证召回精度

### 阶段三：能不能替代

8. 与现有 kv-streaming 做 A/B：
   - KVMem：快、近似
   - kv-streaming：慢、无损
9. 按任务类型分流（工具型 agent → KVMem；全局综合 → 流式）

### 回滚保障

现有 faq5 生产链路（`scripts\start\start-qwen38-27B.bat`）**完全不受影响**，
kvmem 是独立目录 + 独立端口（18200），可并行验证。

---

## 附：关键事实速查

| 项 | 值 |
|---|---|
| 仓库 | <https://github.com/kvmem/kvmem-llama.cpp> |
| 本地路径 | `F:\AI-Models\kvmem-llama.cpp` |
| 版本 | v0.16.0-rc1（pre-release） |
| 上游 pin | ggml-org/llama.cpp @ `b81c99b` |
| 论文 | <https://arxiv.org/abs/2609.04852> |
| 默认端口 | 18200 |
| 服务二进制 | `build/bin/llama-kvmem-server` |
| 核心 KV 类型 | `--kv-dtype q8_0`（IQ3 配方默认） |
| 单次生成上限 | `--kvmem-gen-reserve`（IQ3 = 16384，含思考） |

(结束)

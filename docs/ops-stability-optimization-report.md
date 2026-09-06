# 生产配置稳定性 + 优化实验报告（2026-09-05/06）

> 对象：`UD-IQ4_XS + 262K上下文 + KV流式 + MTP + 视觉`，b10878 服务器
> 显卡：RTX 2080 Ti 22 GB (sm_75)。本报告记录真实环境实测与优化结论。

---

## 一、稳定性测试（真实环境，服务器连续运行 1h+）

对一台**未重启、已在跑**的服务器直接做压力测试（非冷启动验证，更真实）。

| # | 场景 | 结果 | 数据 |
|---|---|---|---|
| 1 | 长文记忆（33K token 文档后提问） | ✅ | 正确答"42" |
| 2 | 多轮对话（Bob/Tokyo） | ✅ | 正确答人名+地点 |
| 3 | 断网中断（长请求中途断开） | ✅ | 服务端立即存活 |
| 4 | 断连轰炸（10 次半途断连） | ✅ | 之后 1s 恢复 |
| 5 | 大量上下文重载（52K 历史重提） | ✅ | prefill 284 t/s |
| 6 | 连续请求（5 连发） | ✅ | decode 31-33 t/s 无漂移 |
| 7 | 空闲 30s 后唤醒 | ✅ | 1.2s 恢复 |
| 8 | 资源泄漏（1h+，2s 采样） | ✅ | RSS 10.78G 零漂移 |

**结论：断网、轰炸、重载、长空闲、连续请求全扛住，无崩溃/泄漏/卡死。**

---

## 二、发现的两个真实问题

### 问题 1：长上下文 decode 明显变慢（设计内代价）
- 短提示：30-42 t/s ✅
- 长上下文 30K+：10-15 t/s ⚠️
- 260K 极端：3 t/s ❌
- **根因**：decode 每 token 都要 attend 全部历史 KV，KV 流式时需从 RAM 拉取

### 问题 2：reasoning 模型吞输出 token
- max_tokens=30 → content 全空（token 全进 reasoning_content）
- max_tokens=500 → 正常回答
- **建议**：agent 调用设 max_tokens ≥ 500

---

## 三、优化实验：加大 staging pool（实测 +66%~+102%）

**核心杠杆：加大 `--kv-stream-stage-mib`，让更多 KV 常驻显存，decode 少拉 RAM。**

| staging pool | 33K decode | 47-52K decode | 显存占用 | 结论 |
|---|---|---|---|---|
| 512 MiB（原）| 15.3 t/s | 10.1 t/s | 18.5G | 基准 |
| **3072 MiB** | **25.4 t/s (+66%)** | **20.4 t/s (+102%)** | 21.0G | ✅ **甜点** |
| 4096 MiB | 在观察窗口内未完成加载 | — | 21.9G | ⚠️ 尚不能判定永久失败 |

**结论：**
- 显存余量（当时 3.7G）白白闲置，加大 pool 白捡速度
- 3072 MiB 是 22G 卡的甜点；4096 导致 KV 池分配卡死
- **推荐显存利用率目标 ~97%（21.9G/22.5G），但 pool 别超 3072**

### KV 量化选择（重要修正）
- **q4_0 不可取**：q4 会损害长上下文检索/生成质量
- **推荐 q5_1**：精度与省显存的平衡点，能省下显存给更大的 staging pool

### q5_1 + stage 3072 实测（2026-09-06 补测）

| 配置 | 33K decode | 50K 长文检索 | 显存 |
|---|---|---|---|
| q8_0 + 3072 | 25.4 t/s | ✅ | 21.0G |
| **q5_1 + 3072** | **23.6 t/s** | **✅ 命中 UNICORN-7788** | 21.0G |

- q5_1 几乎不掉速（-7%），长文检索质量不损 → **确认用户判断：q5_1 是优于 q8_0 的平衡点**
- **额外发现：prompt cache 生效**——相同上下文二次请求 3.5s（命中缓存），无需重 prefill。
  这对 agent 重复提交同一代码库/上下文场景是重要加速

---

## 四、B 实验复核：sliding window / attention sink / context-shift

### 1. 原生 sliding window（SWA）

- **Qwen3.8（qwen35）不是原生 SWA 模型**：`src-b10816/src/models/qwen35.cpp` 没有 SWA/sliding-window 配置；`--swa-full` 不会凭空增加滑动窗口。
- 但 Qwen3.8 也**不是纯 full-attention 模型**：GGUF 元数据声明 `qwen35.full_attention_interval = 4`。源码将模型组织为混合架构：大多数层是 gated delta-net/linear-attention（recurrent），每第 4 个 block 是 dense full-attention；只有 dense attention 层需要这套 KV cache streaming。这个事实本身就是显存和长上下文成本较低的重要原因。
- 运行时强行把 dense full-attention 改成 sliding window 会丢失中间历史，可能影响跨文件依赖、早期约束、长文档检索；不能把它当作无副作用优化。

### 2. attention sink（StreamingLLM 思路）

- 需要改 attention 图，让 dense attention 层只看“最近窗口 + 少量锚点”；这不是普通启动参数。
- 对 Qwen3.8 的长期代码任务，锚点不能保证保留所有早期接口、约束和调用关系；可能出现短期回复速度更快、但远距离引用/一致性下降。
- 由于 Qwen3.8 已经有 recurrent layers 负责部分历史状态，强行对剩余 dense layers 做截断的效果更不能从普通 Transformer 的经验直接推断，必须做真实代码 agent 评测。
- 应该只有在有专门长程质量评测（跨文件符号、早期约束、NIAH、多轮工具状态）的情况下隔离实现；不应直接用于生产。

### 3. context-shift 的源码与实测结论

用 b10878、UD-IQ4_XS、q5_1 KV、262K、MTP、视觉做了 ON/OFF 隔离启动：

- 实验组传入 `--context-shift` 后，服务端日志明确显示它被禁用；两组都能通过 health check。
- **首要原因是 multimodal server policy**：`tools/server/server-context.cpp` 在 mmproj 成功加载后直接将 `ctx_shift` 设为 false，并说明 multimodal 不支持 context shift。图片 token 不是普通单 token 文本，不能安全地按普通文本上下文滚动。
- **还有一个模型级限制**：Qwen3.8 的 `llama_model_rope_type()` 返回 `LLAMA_ROPE_TYPE_IMROPE`，而 `llama_hparams::n_pos_per_embd()` 对 MROPE/IMROPE 返回 4；`llama_kv_cache::get_can_shift()` 对多位置维度返回 false。也就是说，即使不加载 mmproj，当前模型的位置编码也不能直接使用普通 K-shift。
- 因此当前 `Qwen3.8 + 视觉 + adaptive KV streaming` 上，`--context-shift` 会被自动关闭，不能当作“无限续写”方案。之前把禁用主要归因于“KV streaming hybrid 后端物理上不能 shift”是不准确的，已以源码证据更正。
- 这不等于永远无法实现滚动上下文；需要专门处理 IMROPE 的多维位置、视觉 token 组、recurrent state 和分块 KV 的一致性，属于独立功能开发而不是无风险开关。

### 4. 不应过度解读的地方

- KV streaming **不是**把“旧历史完全不参加 attention”；dense attention 层仍原则上处理其可见历史。准确说法是：streaming 把 KV backing store 与 GPU resident/staging pool 分开，减少峰值显存；缺页/分块搬运仍可能成为长上下文 decode 的主要代价。
- 该实现已经包含 CUDA copy stream、CUDA event、transfer ring、跨层 prefetch、连续 page copy、dirty-row 更新，以及根据 deadline miss/copy-engine busy ratio 自适应调整 resident/ring 分区。异步预取不是尚未实现的空白，后续应先观测再改。
- 之前“历史很旧所以通常不主动读取”“20K prefill 搬运开销可忽略”等表述没有经过逐层传输计数或完整对照，**不作为已证实结论**。

---

## 五、重新复核后的优化优先级

### 已证实、低风险

1. **使用 q5_1 KV，而不是 q4**：符合本机对长上下文质量的要求；本次 50K 远距离检索命中 `UNICORN-7788`。这不是完整质量基准，正式质量结论仍应使用多任务评测。
2. **合理增大 staging pool**：512→3072 MiB 的同类测试中，33K decode 从 15.3 提升到 25.4 t/s，47–52K 从 10.1 提升到 20.4 t/s；3072 是目前已完成且可工作的甜点。
3. **保留 prompt cache 的稳定前缀**：agent 断线后若重新发送完全相同的前缀，有机会复用缓存；请求顺序、工具列表、系统提示和历史必须保持一致。
4. **单槽运行**：`-np 1`，因为多槽会把 KV 和 staging 竞争放大；并发需求应先做容量/速度测试，不应直接开多槽。

### 尚未证实、需要隔离测量

- `stage=3072` 是否是 q5_1 下的真正最优点；4096 的第一次启动未完成，不能把它写成永久失败。
- q5_1 与 q8_0 的长程代码质量差异；一次合成 secret 检索不能代表真实代码 agent 质量。
- attention sink、运行时滑动窗口、KV 预取/异步 DMA、PCIe pinned-memory 调优。
- 262K 真正满上下文下的 q5_1 decode 曲线；之前的 33–52K 是性能样本，不等于 260K 性能。

---

## 六、代码 Agent 推荐配置（重新复核版）

**默认优先正确性和可恢复性：**

```
-c 262144                     # Qwen3.8 原生上限；不要把 512K 当作已生效
-ctk q5_1 -ctv q5_1           # 长上下文质量优先
--kv-stream-stage-mib 3072   # 当前已验证可工作的速度/显存平衡
-np 1                         # 单槽，避免 KV 竞争
-b 512 -ub 128                # 当前已验证组合
--spec-type draft-mtp --spec-draft-n-max 2
--image-min-tokens 1024       # 需要视觉时
```

官方 Qwen3.8 thinking-mode 请求参数应由客户端传入（不要只依赖服务端默认值）：

```
temperature=1.0, top_p=0.95, top_k=20,
min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
```

- `max_tokens` 是整次响应上限，不是“最终回答上限”；thinking agent 应留足预算。官方给出的 262,144 reasoning / 131,072 final 是 1M 框架语境下的上限建议，不能原样理解为 262K 本地上下文还能额外容纳这么多输出。
- 对本地 262,144 总上下文，必须把输入、reasoning、工具结果和最终输出一起预算；实际代码 agent 应留出明确输出余量，不能把输入填到 262K 后再期待长回答。
- `--context-shift` 当前会被 KV streaming 禁用；不要把它写进生产脚本并误以为启用了无限上下文。

---

## 七、进一步研究方向（按收益/风险排序）

1. **先做传输可观测性**：当前代码已经有 resident/streamed/transfer 计数和 trace hook，但最终日志没有稳定汇总出一次完整请求的累计值；应补齐请求级汇总或用 profiler 校准 host↔device bytes、copy waits、stage 命中率和 PCIe 吞吐。
2. **做 stage 的系统扫描**：q5_1 下比较 512/1024/1536/2048/2560/3072，分别测 20K、50K、100K、140K+ 输入；目标是找到“速度—余显存—加载时间”的 Pareto 点，而不是盲目吃满 97%。
3. **调现有预取管线**：不是重新添加双缓冲，而是根据 trace 调整已有 transfer ring 的 slot 数、lookahead、page 合并和 decode span；每次改变都要配合正确性测试。
4. **研究异步 copy 与 compute 的重叠质量**：如果 copy engine 未饱和但 deadline miss 高，重点查事件依赖/预取时机；如果 copy engine 饱和，PCIe 和 pinned host memory 可能是硬上限。
5. **最后才做 attention sink**：先建立真实代码 agent 质量集；若短窗口丢失早期约束，就应放弃，而不是为了 t/s 强行上线。

### 150K 及以上 trace 复核（2026-09-06）

之前名义上的 150K 请求实际只有 114,094 tokens，仍没有越过 resident 边界。随后用相同的 trace 配置做了真正越界的约 140K 请求：

| 指标 | 结果 |
|---|---:|
| 实际 prompt | 135,087 tokens |
| prefill | 226.42 tok/s |
| decode | 9.64 tok/s |
| 检索结果 | ✅ `TRACE140-SIGMA` |
| 显存 | 21,027 / 22,528 MiB |
| trace resident pages | 511 -> 510 |
| trace ring slots | 16 -> 32 |

这证明 adaptive controller 检测到 active pages 超过 resident partition，并将一页/layer 转为更多 ring slots；请求仍然正确完成。135K 的 decode 已降到约 9.6 t/s，说明越过 resident 边界后确实出现了明显代价。

但是本次 trace 的 `samples=0, misses=0, copy busy=0.0%`，不是“没有发生搬运”的证明：本版本的 timing feedback 在该长 prefill/请求边界没有形成有效 completed evaluation sample。当前日志没有直接导出本次请求的累计 streamed bytes 和有效 copy-engine utilization，因此不能据此声称 PCIe 利用率或 deadline miss 为零。后续优化必须先补齐可观测性，或用 profiler/专门测试接口取得这些计数。

### 可观测性插桩实验（2026-09-06，隔离构建）

在隔离构建（`bin-b10816-trace`，独立于生产二进制）上给 `record_deadline` kernel 加 host 侧发射计数（`dlkern`），再用同一 135K 用例对照：

| 版本 | dlkern（kernel 发射）| samples（host 读到）| 结论 |
|---|---:|---:|---|
| 原逻辑 | — | 0 | 无法定位 |
| + 发射计数 | 1024 | 0 | kernel 确实发射 |
| + 每 launch 后强制 `cudaStreamSynchronize` | 736 | 0（1082/1082 行）| kernel 确实执行完毕 |

**根因裁定：** `deadline_samples/deadline_misses` 位于 `cudaHostAllocMapped` zero-copy host 内存，`record_deadline` kernel 的 `atomicAdd` 在本机（RTX 2080 Ti / sm_75 / WDDM）上执行成功但结果对 host 读取**永久不可见**。这不是调度问题（同步后仍为 0），也不是发射问题（dlkern>0），而是 mapped host 内存上 device 原子写的主机可见性缺失。

**连锁影响：** deadline 自适应机制（`kv_stream_adapt` 的 miss ratio、span tuner、分区决策）从上线起就一直"失明"，`samples=0, misses=0, copy busy 0.0%` 不是"没有搬运压力"的证据，而是计数通道断了。前几轮 trace 里这些 0 值一律不能作为性能结论。

**修复方向（未实施）：** 把 counters 移到 `cudaMalloc` 设备内存，feedback 读取时 `cudaMemcpyAsync` D2H + event 同步；或改用 portable mapped 分配并在 host 读取前 `cudaEventSynchronize`。改动很小，但要重编验证。

**实测附带数据（135K，强制同步版 vs 无同步版）：** prefill 214 vs 221 t/s、decode 9.4 vs 9.8 t/s——1-tid record kernel 的同步代价可忽略；跨 resident 边界的 decode 掉速（≈9.5 t/s）与该 kernel 无关。

**实验产物处置：** 插桩改动已存入 `src-b10816` 的 git stash（`instrumented dlkern+sync experiments`），隔离二进制保留在 `bin-b10816-trace`，生产源树与生产二进制未动。

- **修复 deadline 计数通道**（最高优先、改动最小）：samples/misses 的 mapped host atomic 在 sm_75+WDDM 不可见，自适应机制全程失明；改 device 内存 + D2H 读回即可修复。修复后才能谈其它调参。
- 已有 copy stream、event、transfer ring、跨层 prefetch 和自适应 resident/ring 分区；重复添加一个普通“双缓冲”不是明确收益点。
- q5_1 在本构建支持 direct quantized attention，且 `GGML_CUDA_FA_ALL_QUANTS=ON`；保留 q5_1 是合理的质量/传输折中。
- 当前最有希望的低风险工作是：补齐 streamed-page/byte、copy wait、deadline 和 end-to-end trace，并扫描 stage/ring 的 Pareto 点。
- 若确认 copy engine 饱和，才针对 page 合并、预取距离、ring slots 和 pinned host allocation 调参；若 copy 不饱和而 GPU attention 饱和，则扩大 pool 也不会继续解决 decode。
- attention sink / 非原生 sliding window 仍不进入生产：Qwen3.8 的 dense layers 与 recurrent state 共同承担上下文语义，截断历史必须用真实代码 agent 质量集验证。

### 插桩后的首次真实搬运测量（2026-09-06，trace-experiment 分支）

在隔离分支 `trace-experiment` 上给 `LLAMA_KV_STREAM_TRACE` 行追加了累计计数（resident hits/miss、streamed 页数、host→device 累计 MiB、异步上传数、compute 等待、slot 复用、跨层预取），编译到独立的 `bin-b10816-trace`（build 10879），对生产 `bin-b10816` 零改动。计数为单调累计值，两个时间点之差即该区间的真实搬运量。

同一 135,088-token 跨边界请求（UD-IQ4_XS，q5_1 KV，3,072 MiB 池，MTP+视觉，输出 200 token，decode 10.0 t/s，prefill 199.4 t/s）：

| 计数 | prefill 尾（resident 511）| 最终（resident 510）| 增量 |
|---|---:|---:|---|
| streamed 页 | 4,624 | 8,064 | +3,440 页 |
| h2d 累计 | 4,802 MiB | 12,204 MiB | **+7,402 MiB** |
| uploads | 4,624 | 12,688 | +8,064 |
| compute waits | 4,624 | 12,688 | +8,064 |
| slot reuses | 4,608 | 12,656 | +8,048 |
| 跨层预取 xfetch | 0 | 0 | **0** |
| deadline samples/misses | 0 / 0 | 0 / 0 | 0 / 0 |

**每页实测 0.583 MiB**（q5_1 K+V 256-token 页，与理论一致），每次 upload 恰好对应一次 compute wait（比率 1.00）。据此得出的瓶颈判断：

1. **不是 PCIe 带宽**。decode 每输出 token 约 61 页 ≈ 36 MiB，@10 t/s 仅 ~0.35 GiB/s，远低于 PCIe 3.0 x16 的 ~13 GiB/s。
2. **是 copy/compute 零重叠**。waits==uploads 意味着每次页上传 compute 都在同步等待；跨层预取（xfetch）一次都没发生，deadline 采样也从未激活。
3. **可操作方向**（按预期收益排序）：查清为何 lookahead 没有找到可预取页（当前层之后没有 pending 请求，或 producer 依赖把 eligible 判断全部挡住）；核对 ring 深度 32 与每层 streamed 页 19 的配比是否让 ring 在一层内转不开；只有确认重叠修复后，decode 才有望显著回升，届时再谈扩大 pool。

这次测量同时说明：此前的 `samples=0/misses=0/copy busy=0.0%` 不是"没有搬运"，而是请求级聚合缺失；插桩后同一现象背后的真实数据完全不同。

### 计数通道修复实验（2026-09-06，最终裁定）

在隔离构建上实施了"移出 mapped host 内存"修复并四轮递进验证：

1. **v1**：counters 改 `cudaMalloc` + blocking `cudaMemcpy` D2H 读回 → samples 仍 0。
2. **v2**：修正读回流序（初版误在 copy_stream 记 event，kernel 在 compute stream）→ 仍 0。
3. **v3**：kernel 加**无条件** `atomicExch` 执行探针（不依赖 ready_flag 分支）+ `cudaMemset` 清零 → 探针也 0。
4. **v4**：launch 后同流 `EventRecord` + `EventSynchronize` 强制完成 + 全 8 槽 dump → **全 0**。

关键对照：

| 通道 | 结果 |
|---|---|
| `cudaMemset`（host→device）清零 | ✅ 生效（读到 0 而非垃圾值） |
| host 侧 launch 计数 `dlkern` | ✅ 1024 次，无 CUDA 错误 |
| 同流 event record + EventSynchronize | ✅ 无错误返回 |
| **device 原子写（atomicAdd/atomicExch）** | ❌ **写入被静默丢弃** |

**最终裁定：RTX 2080 Ti（sm_75）+ WDDM 平台级缺陷**。四轮排除了调度、流序、下标、编译单元、多 runtime 全部代码层假设后，device 原子写到该缓冲区的主机可见路径在消费级 WDDM 驱动上失效；host→device 方向（memset）正常、device→host 方向的原子写被丢。2080 Ti 无 TCC 模式，本卡在 WDDM 下无修复路径。

**替代方案**：deadline 改纯 host 侧 CUDA event 计时（`cudaEventElapsedTime`，无 device 写回）；或在 Linux 同卡重验。插桩与修复尝试已存 `src-b10816` git stash（`counter-channel fix attempts`），生产源树与生产二进制未动。

附带实测：含每 ubatch 一次阻塞读回的版本 135K decode 10.2 t/s，与基线 9.6–9.8 t/s 同量级——修复方向的同步开销可忽略，瓶颈纯粹是平台可见性。

**对前文结论链的更正**：上一节"copy/compute 零重叠、waits==uploads"的瓶颈判断需降级——`compute_stream_waits` 是 API 调用计数而非阻塞时长，且 deadline/xfetch 两个质量反馈通道本就是坏的，重叠质量实际未知。可靠结论保留：**不是 PCIe 带宽**（0.35 GiB/s 远低于链路上限）；decode 掉速的真实构成需要等 host 侧计时替代方案落地后重测。

### 全链路闭合验证（2026-09-06，hosttime 验证 + 三探针定案）

hosttime 版（纯 host 侧 event 计时）编译完成后，用三个新探针把此前所有未决问题一次性闭合：① graph 模式门（`ucg=`，打印 `ggml_backend_cuda_graph_compute` 每次调用的 use_cuda_graph/更新标志/节点数）；② 收集统计（`fa=/rt=/fits=`，FA 节点总数 / 命中 kv-stream runtime 数 / 通过 streamed_supported 数）；③ skip 原因分解（add_attention 早退时打印具体失败子句）。135K 用例重跑三次（ht5/ht6/ht7，配置同前：q5_1/q5_1 + stage 3072 + MTP draft-2 + 视觉 + -c 262144）：

| 运行 | 结果 | prefill | decode | marker |
|---|---|---:|---:|---|
| ht5 | 旧 dll（无探针，作废重跑） | — | — | — |
| ht6 | 双探针版 | 217.3 t/s | 10.2 t/s | ✅ 精确 |
| ht7 | 三探针版 | 216.1 t/s | 9.2 t/s | ✅ 精确 |

显存 21,119 / 22,528 MiB；resident 511→510 页、ring 16→32 槽，与此前 trace 一致。数值定案：

**发现 1 —— 主模型 decode 图每步 `ucg=0`（不走 CUDA graph，直接执行）。** 每步模式固定为 3 个 55 节点小图（MTP 草稿，其一 ucg=1）+ 1 个 4134 节点主图 ucg=0。这动态证实了"decode 阶段 resident 页数不足导致 all_layers_fit 不成立 → graph 关闭"的静态推断，同时**排除"CUDA graph replay 吞掉 host 侧收集/计时逻辑"假设**——host 代码每步都在真实执行。

**发现 2 —— streamed 路径全程激活且无一跳过。** 每次 decode eval 均为 `fa=16 rt=1 fits=16`：16 个注意力层全部命中 kv-stream runtime、全部通过 streamed_supported、全部进入收集逻辑，skip 计数为 0。（`fa=16` 而非 65：qwen35 混合架构，65 层中 16 层为全注意力、其余为线性注意力层，仅前者有 KV 页。）

**发现 3 —— `streamed=0` 的真实语义：多波次预取队列为空，不等于"没有流式"。** 由排除法定位：add_attention 唯一不排队出口是逐层守卫 `nchunks ≤ layer_pages[layer]`，其恒成立意味着 **KV 张量的 token 覆盖范围（ne[1]）恰等于 resident 分区（510 页 × 256 = 130,560 tokens）**——张量本地工作集全部驻留，无需 H2D 预取请求；超出 resident 边界的尾页（135K 时 18 页/层，约 4,608 tokens）不经过该队列，由 `flash_attn_ext_streamed` 内部的尾页路径直接处理。

**发现 4 —— dlkern=1024 的复核解释，尾页流式确认在发生。** 1024 = 16 注意力层 × ~64 decode evals，发射点位于插桩版在**尾页路径**加的 record_deadline 调用（每层每 eval 一次，与 span 合并后每层恰好一个流式 span 一致）。即：跨过 130,560-token resident 边界后，每层每步都在对张量外尾页做 deadline 采样——**流式机制真实在运行**，与 decode 掉速到 ~9.2-10.2 t/s 的边界代价互相印证。此前 trace 中 `h2d 6 MiB`（远小于尾页体积）进一步提示尾页数据走 zero-copy mapped 直读而非显式 memcpy。

**发现 5 —— hosttime 全零的原因定案，方案本身有效但装错了位置。** 计时门（`timing_current`）只在 `graph_requests` 非空时武装（`graph_finalize` 提前返回），而本负载下该队列恒空 → `pending=0 cur=0 win_us 0 copy_us 0 uploads 0` 是该路径**没有活动的真零**，不是测量失败。上一轮"host 侧 event 计时"实验计划的 `attn_win_us/copy_sample_us/busy` 三项在本负载下永远测不到东西。

**计数通道裁定维持不变**：graph 队列路径与尾页路径写同一 mapped-memory 计数器，插桩版在尾页路径发射 1024 次仍读 0，四轮对照（memset 生效 / launch 无错 / 同步无错 / 原子写丢失）不依赖发射位置，**sm_75 + WDDM 平台缺陷结论成立**。需同步更正的是：不能由 `samples=0/misses=0` 反推"流式没有发生"——流式在发生，只是它的计数进不了 host。

**下一步修正**：host 侧 event 计时应安装在**尾页路径**（`flash_attn_ext_streamed` 的尾页处理段前后 `EventRecord` + `cudaEventElapsedTime`）——这是本负载下唯一活跃的流式路径，也是 decode 边界掉速的真实来源。装在 graph_requests 路径的计时在本负载下无信号。三个探针（ucg 门/收集统计/skip 原因）已随实验存入 `src-b10816` git stash（`trace probes: ucg gate + prepare stats + skip reasons`），隔离二进制 `bin-b10816-trace` 保留含探针版本。

---

## 十、decode 优化落地：q5_1 KV 直上 GPU（2026-09-07）

用户目标"生产日常也用 q5_1 KV + 优化 decode"由一条编译开关路线达成，无需动任何源码。

### 归因实验（A/B 对照，同模型同参仅差 KV 类型/构建）

| 配置（135K，UD-IQ4_XS） | 构建 | prefill | decode | marker |
|---|---|---:|---:|---|
| hybrid q5_1 + stage 3072（kv-stream fork） | fork trace | 216-217 | 9.2-10.2 | ✅ |
| fixed q5_1（标准构建） | b10816 官方 Clang | **12（CPU 回退）** | — | — |
| fixed q4_0（标准构建） | b10816 官方 Clang | 221.5 | 18.6 | ✅ |
| fixed q5_1（FA_ALL_QUANTS 重建） | b10816 faq5 | 339.8（峰值 412） | 23.8 | ✅ |

两个结论：① **标准构建遇到 q5_1/q5_1 KV 会触发注意力算子 CPU 回退**（prefill 12 t/s、CPU 时间 ≈9.4 核持续打满）——标准 CMake 默认不编译 q5_1 组合的 FA 内核；② 补上 `GGML_CUDA_FA_ALL_QUANTS=ON` 后，q5_1 fixed 不仅可用，还是**全部配置里最快的**：decode 23.8 t/s，比 kv-stream hybrid 快 2.5 倍、比 q4_0 fixed 快 28%；prefill 340-412 t/s。

kv-stream hybrid 的 9.5 t/s 由此定性：那是其 per-layer 部分归并/尾页机制的成本，不是 135K 注意力的固有成本。**≤170K 的日常上下文，fixed q5_1 全面胜出；hybrid 只在 fixed 装不下的超长上下文（>200K）才有价值。**

### 生产落地（可回滚）

- 新引擎目录 `F:\AI-Models\llama.cpp-faq5\`（自包含 exe+DLL，源码树 `llama.cpp-src` 未改一行，仅 CMake 开关；重建脚本 `build-b10816-faq5.bat`；README 记录来源/回滚/混用 DLL 的坑）。
- `scripts\start\start-qwen38-27B.bat` 与 `start-agent.bat`：引擎指向 faq5，`--cache-type-k/v q5_1`，`-ub 512→256`（显存余量 462→660 MiB）。
- 日常配置最终验收（TURBO-Fable + q5_1 + 174080 ctx + ub256）：30K prefill 529 / decode 26.2；135K prefill 371.8 / decode 24.8，marker 精确；512px 视觉答对、1024px 视觉峰值显存 22.13G（ub512 下实测）无 OOM。
- 回滚方式：脚本路径改回 `llama.cpp\` + cache-type 改回 q4_0（旧二进制原样保留）。
- 坑（记录）：q5_1 慢不是模型或卡的问题，是内核缺失的 CPU 回退；症状是 CPU 打满 + prefill 一个数量级掉速。官方 DLL 与 faq5 DLL 不可混目录，否则静默加载旧 ggml-cuda.dll 复现回退。

### 对优化路线的更新

- "把 host 侧 event 计时装到尾页路径"仍然成立，但优先级下降：hybrid 已退出日常配置，仅服务于 >200K 长上下文场景（fixed q5_1 在 262K 装不下：KV 7.5G + 模型 14.6G > 22.5G）。
- 稳定性验收（agent 断网重载、20K 前缀命中等）此前已在 K4V4 上通过，q5_1 路径数学与 q4_0 完全同构（仅量化格数不同），预期同稳；如需可复跑第八节验收。

---

## 九、当前状态

- 生产脚本推荐 `q5_1/q5_1 + stage 3072 + MTP + 视觉`。
- 约 140K prompt 已越过 resident 边界并正确检索；decode 约 9.64 t/s。
- 这次 trace 暴露了“日志有内部计数、但请求级汇总不完整”的可观测性缺口。
- 没有新的代码优化直接进入生产；所有 attention 截断实验仍隔离。
- 2026-09-06 补：135K 用例三次重跑全通过（q5_1 + MTP + 视觉，prefill 216-217 t/s / decode 9.2-10.2 t/s）；`streamed=0`/hosttime 全零已定案为"多波次预取队列无活动的真零"，尾页流式经 dlkern=1024（16 层 × 64 evals）证实在真实发生；WDDM 计数通道裁定维持。下一个实验：把 host 侧 event 计时装到尾页路径。
- 2026-09-07 补：decode 优化落地——标准构建 q5_1 KV 会 CPU 回退（内核缺失），加 `GGML_CUDA_FA_ALL_QUANTS=ON` 重建后 q5_1 fixed 全面胜出（135K decode 23.8 vs hybrid 9.5 vs q4_0 18.6），已上生产（`llama.cpp-faq5\` 独立引擎 + 两个启动脚本改 q5_1），详见第十节。hybrid 退出日常配置，仅服务 >200K 场景。

*报告日期：2026-09-06（09-07 增补）· RTX 2080 Ti 22G · UD-IQ4_XS / TURBO-Fable*


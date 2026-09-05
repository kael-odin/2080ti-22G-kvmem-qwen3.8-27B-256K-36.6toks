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

### 源码复核后的优化判断（2026-09-06）

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

---

## 九、当前状态

- 生产脚本推荐 `q5_1/q5_1 + stage 3072 + MTP + 视觉`。
- 约 140K prompt 已越过 resident 边界并正确检索；decode 约 9.64 t/s。
- 这次 trace 暴露了“日志有内部计数、但请求级汇总不完整”的可观测性缺口。
- 没有新的代码优化直接进入生产；所有 attention 截断实验仍隔离。

*报告日期：2026-09-06 · RTX 2080 Ti 22G · UD-IQ4_XS*


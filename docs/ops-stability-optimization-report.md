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
| 4096 MiB | 加载极慢/卡分配 | — | 21.9G | ❌ 过犹不及 |

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

- **Qwen3.8（qwen35）不是原生 SWA 模型**：`src-b10816/src/models/qwen35.cpp` 没有 SWA/sliding-window 配置；模型本身按 full attention 设计。
- `--swa-full` 只控制模型已经声明的 SWA 层，对 Qwen3.8 不会凭空增加滑动窗口。
- 运行时强行把 full attention 改成 sliding window 会丢失中间历史，可能影响跨文件依赖、早期约束、长文档检索；不能把它当作无副作用优化。

### 2. attention sink（StreamingLLM 思路）

- 需要改 attention 图，让每层只看“最近窗口 + 少量锚点”；这不是普通启动参数。
- 对 Qwen3.8 的长期代码任务，锚点不能保证保留所有早期接口、约束和调用关系；可能出现短期回复速度更快、但远距离引用/一致性下降。
- 应该只有在有专门长程质量评测（跨文件符号、早期约束、NIAH、多轮工具状态）的情况下隔离实现；不应直接用于生产。

### 3. 原生 context-shift 与 KV streaming 的实测结论

用 b10878、UD-IQ4_XS、q5_1 KV、262K、MTP、视觉做了 ON/OFF 隔离启动：

- ON 组传入 `--context-shift` 后，启动日志明确显示：`KV cache shifting is not supported for this context, disabling KV cache shifting`。
- OFF 组正常启动；两组都能通过 health check。
- 因此当前 hybrid + adaptive KV streaming 上，`--context-shift` 会被自动关闭，不能当作“无限续写”方案。
- 这只是当前实现的兼容性结论，不代表经过专门设计后永远无法支持；若要实现，需要为分块 KV 的位置重映射、GPU staging 和 host backing store 设计一致的 shift/淘汰协议。

### 4. 不应过度解读的地方

- KV streaming **不是**把“旧历史完全不参加 attention”；Qwen3.8 仍是 full attention，生成一个 token 原则上仍要处理全历史。准确说法是：streaming 把 KV backing store 与 GPU resident/staging pool 分开，减少峰值显存；缺页/分块搬运仍可能成为长上下文 decode 的主要代价。
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

1. **先做传输可观测性**：记录每次 decode 的 active tokens、stream span 数、host↔device bytes、stage 命中率和 PCIe 吞吐；没有这些计数，任何“加速了搬运”的说法都只是猜测。
2. **做 stage 的系统扫描**：q5_1 下比较 512/1024/1536/2048/2560/3072，分别测 20K、50K、100K、200K 输入；目标是找到“速度—余显存—加载时间”的 Pareto 点，而不是盲目吃满 97%。
3. **研究异步预取/双缓冲**：在 attention 计算当前 chunk 时，异步预取下一个 KV chunk；需要保证 stream、event、生命周期和 pinned host buffer 正确，属于最有价值的代码优化方向。
4. **研究 chunk 布局与访问顺序**：减少碎片、让 host backing store 连续、按 attention 扫描顺序预取，可能比单纯扩大 stage 更有效。
5. **最后才做 attention sink**：先建立真实代码 agent 质量集；若短窗口丢失早期约束，就应放弃，而不是为了 t/s 强行上线。

---

## 八、遗留 / 待办

---

*报告日期：2026-09-05/06 · RTX 2080 Ti 22G · UD-IQ4_XS*
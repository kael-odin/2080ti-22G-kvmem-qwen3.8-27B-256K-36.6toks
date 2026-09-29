# KVMem Windows / sm_75 移植与验证报告

> 日期：2026-09-18
> 结论：**成功**。kvmem 在 2080 Ti（Turing / sm_75）上编译、加载、推理全部通过，
> 目标生产配置（IQ3_S + MTP + 视觉 + 262K + q8_0 KV）跑通，decode **36.6 tok/s**。

---

## 一、最终成果

| 需求 | 状态 | 实测 |
|---|---|---|
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | ✅ | CUDA0 权重缓冲 11159.69 MiB |
| 开 MTP | ✅ | 接受率 58.6% / 61.6%，`n_max=3` + ReplaySSM |
| 视觉 | ✅ | BF16 mmproj **在 GPU**，形状/颜色/位置识别全对 |
| 开满上下文 | ✅ | `n_ctx=262144` |
| KV 均 8 位 | ✅ | `type_k=q8_0 type_v=q8_0` |
| 不明显掉速 | ✅ | **decode 36.6 tok/s** |

**显存占用：16552 / 22528 MiB（约剩 6 GB 余量）**

### 与作者平台对比

| | 显存带宽 | decode |
|---|---|---|
| 作者 RTX 5060 Ti 16G | 448 GB/s | 31.74 t/s |
| **本机 2080 Ti 22G** | **616 GB/s** | **36.6 t/s** |

decode 受显存带宽瓶颈约束，2080 Ti 的带宽优势直接体现。
对比本机现有 kv-streaming（135K 时 9.6–13 t/s），**快约 3–4 倍**。

### 质量验证（人眼验货）

| 测试 | 结果 |
|---|---|
| 算术 | `1234+5678=6912` → ×2 = **13824** ✓ |
| 代码 | 正确的 Python 字符串反转函数 ✓ |
| 中文 | `我是通义千问，由阿里巴巴集团通义实验室独立研发的大语言模型。` ✓ 无乱码 |
| 逻辑 | 传递性推理（Bloops/Razzies/Lazzies）正确 ✓ |
| 创作 | 俳句 5-7-5 音节正确 + `17×23=391` ✓ |
| 视觉 | 左上红圆 / 右下蓝方，位置与颜色全对 ✓ |

作者警告的「IQ3_S 输出乱码」**未出现**。模型能正确自报身份，说明权重读取与处理链路正确。

---

## 二、构建环境（全部就位）

| 组件 | 版本 | 位置 |
|---|---|---|
| CUDA Toolkit | **13.2 Update 2 (nvcc 13.2.86)** | `kvmem-llama.cpp\.setup\cuda-13.2-86\` |
| Visual Studio | VS 18 Community，MSVC 14.51.36231 | 系统安装 |
| CMake | 4.2.3-msvc3（VS18 自带） | `Common7\IDE\CommonExtensions\...` |
| Ninja | 1.13.0 | `C:\Python311\Scripts\ninja.exe` |
| 目标架构 | **sm_75** | 显式指定（默认是 120a-real） |

### ★ CUDA 13.2.86 的安装方式：redist 组件（无需管理员）

**不需要** 3–4 GB 的完整安装包，也**不需要**管理员权限，更**不影响**系统的 CUDA 12.8。

NVIDIA 在 <https://developer.download.nvidia.com/compute/cuda/redist/> 提供独立组件，
清单文件 `redistrib_13.2.2.json` 即 CUDA 13.2 Update 2。本次下载：

| 组件 | 版本 | 大小 |
|---|---|---|
| `cuda_nvcc` | 13.2.86 | 29.6 MB |
| `cuda_cudart` | 13.2.86 | 3.0 MB |
| `cuda_cccl` | 13.2.86 | 3.4 MB |
| `cuda_crt` | 13.2.86 | 0.1 MB |
| `libnvvm` | 13.2.86 | 52.7 MB |
| `libcublas` | 13.4.1.3 | 370.1 MB |

解压后**合并到一个根目录**即可作为工具链使用（各组件都展开为 `bin/ lib/ include/ nvvm/`）。

> ⚠️ **坑**：`cuda_nvcc` 组件里**没有 `cicc.exe` 和 `libdevice.10.bc`**，
> 它们在独立的 `libnvvm` 组件里。少了会报 ptxas 阶段「系统找不到指定的路径」。

> ⚠️ **坑**：CUDA 运行时 DLL（`cublas64_13.dll` / `cublasLt64_13.dll` / `cudart64_13.dll`）
> 在组件的 `bin/x64/` 子目录下，**必须复制到 exe 同目录**才能运行。

### 版本要求（关键）

- **必须 13.2 Update 2（nvcc 13.2.86）**。CUDA 13.2 **基础版**的 nvcc 是 13.2.51，
  作者实测会让 IQ3_S 输出乱码。
- 核对用 `nvcc --version`，**不是** `nvidia-smi`（后者显示的是驱动能力上限）。
- 升级工具包后**必须用全新的 build 目录**重编（换 DLL 无效，kernel 已编进二进制）。

### 为什么不能用系统已有的 CUDA 12.8

```
nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
--microsoft_version=1951
```

CUDA 12.8 的 nvcc **不认识 MSVC 14.51（VS 18）**。这是硬性版本代沟，非配置问题。

> ⚠️ **连带影响**：`llama.cpp-src\build-b10816-faq5.bat` 等构建脚本已失效——它们指向
> 已被删除的 **CUDA v12.4**。
>
> **修正（2026-09-18 实测）**：脚本里的 `VS2022 BuildTools` **是存在的**（3.4 GB，MSVC 14.44），
> 唯一问题是 nvcc 路径。实测 **VS2022 BuildTools(MSVC 14.44) + CUDA 12.8 + sm_75 完全可用**，
> 所以修复只需把 nvcc 从 `v12.4` 改成 `v12.8`，**不必改用 CUDA 13.2**。
>
> 真正的版本代沟是：CUDA 12.8 + **VS18 的 MSVC 14.51** → `cudafe++ ACCESS_VIOLATION`。
> 即老 CUDA 配新 VS 不行，配老 VS 可以。
>
> **教训：升级 VS 大版本会让 CUDA 构建脚本静默失效**（本机因为同时装了旧 BuildTools 才没绝迹）。

---

## 三、Windows 移植改动清单（12 项）

全部改动都是**本机新增**，非上游代码，集中且可复用。核心是新增
`kvmem/include/kvmem/kvmem_win_compat.h`，其余为外科手术式小改。

| # | 问题 | 修法 | 性质 |
|---|---|---|---|
| 1 | 无 `<unistd.h>` | 兼容头 | 编译 |
| 2 | `<windows.h>` 的 `min`/`max` 宏破坏 `numeric_limits<T>::max()` | `NOMINMAX` + `#undef` | 编译 |
| 3 | `off_t` 是 32 位，>2 GB 文件截断 | 宏扩宽至 64 位 | **正确性** |
| 4 | 无 `ssize_t` | `typedef SSIZE_T` | 编译 |
| 5 | **MSVC 默认文本模式会 CRLF 改写，静默损坏 KV 二进制数据** | 强制 `_O_BINARY` | **正确性** ⚠️ |
| 6 | 无 `pread`/`pwrite` | `ReadFile`/`WriteFile` + `OVERLAPPED`（真位置 I/O） | **正确性** |
| 7 | 无 `mkdir` | `_mkdir` | 编译 |
| 8 | 无 `fdatasync` | `_commit` | 编译 |
| 9 | 无 `clock_gettime`/`CLOCK_MONOTONIC` | `QueryPerformanceCounter` | 编译 |
| 10 | **kvmem 建成共享库但零导出 → 不生成 .lib → `llama.dll` 永远链接失败** | Windows 上改静态库 | **构建结构** |
| 11 | 无 `setenv` | `_putenv_s` | 编译 |
| 12 | 上游 llama.cpp 的 UTF-8 源码被 GBK 代码页误读 | 编译加 `/utf-8` | 编译 |

### 两条最重要的坑

**第 5 条 —— 静默数据损坏。** 如果只求「编译通过」就收工，程序能跑、能启动、不报错，
但写进 KV 存储的每个字节都可能被 CRT 的换行转换悄悄改掉。症状是「乱码或答非所问」，
而**永远查不到原因**，因为一切看起来正常。

**第 10 条 —— Linux 与 Windows 的语义差异。** Linux 上共享库默认导出全部符号；
Windows 上零 `__declspec(dllexport)` 就是零导出，连导入库都不生成。
这在 Linux 上完全不会暴露。

### 改动文件清单

```
新增  kvmem/include/kvmem/kvmem_win_compat.h      兼容层本体
改    kvmem/include/kvmem/nvme_kv_tier.hpp        include 段加 _WIN32 分支
改    kvmem/CMakeLists.txt                        Windows 上 kvmem 改静态库
改    kvmem/tests/nvme_kv_tier_test.cpp           include 段
改    kvmem/tests/kvmem_runtime_test.cpp          include 段
改    tools/llama-kvmem-cli.cpp                   setenv → _putenv_s
新增  build-windows-sm75.bat                      Windows 构建脚本
```

### 遗留问题

- `kvmem-mtp-kv-test.exe` 链接失败：它引用了 llama.cpp **内部符号**
  （`llama_batch_allocr`、`llama_kv_cache::type_k` 等），这些不在公开 API 中，
  Windows 上不从 `llama.dll` 导出。**不影响服务器**，但会让每次构建报错。
  → 待办：Windows 上排除该测试目标。
- `build-cu13286-sm75\bin\kvmem.dll` 是早期共享库构建的残留（现已改静态），可删。

---

## 四、运行方式

```bat
:: 构建（首次或改动后）
kvmem-llama.cpp\build-windows-sm75.bat

:: 只配置不编译（快速检查工具链）
kvmem-llama.cpp\build-windows-sm75.bat configure

:: 只编某个目标（迭代用）
kvmem-llama.cpp\build-windows-sm75.bat llama-kvmem-server

:: 清空 build 目录
kvmem-llama.cpp\build-windows-sm75.bat clean
```

产物：`kvmem-llama.cpp\build-cu13286-sm75\bin\llama-kvmem-server.exe`

### 生产配置（已验证可用）

```bat
cd kvmem-llama.cpp\build-cu13286-sm75\bin
llama-kvmem-server.exe ^
  -m   "F:/AI-Models/models/qwen3.8-27B/gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" ^
  --mmproj "F:/AI-Models/models/qwen3.8-27B/gsq-rco/mmproj-Qwen3.8-27B-BF16.gguf" ^
  --mmproj-offload --image-max-tokens 512 ^
  -c 262144 -n 16384 ^
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 ^
  --kv-dtype q8_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --kvmem-mtp-state replay ^
  --enable-thinking --reasoning-budget 4096 ^
  --no-ui --port 18200 --host 127.0.0.1
```

API：`POST http://127.0.0.1:18200/v1/chat/completions`（OpenAI 兼容，**无鉴权无 TLS，绑 127.0.0.1**）

### 22G 的余量优势（下一步优化方向）

作者是按 16G 卡调的（`budget 36864` / `gen_reserve 16384`）。本机显存只用到 16.5G / 22G，
**剩约 6G**，可以把这两项往上调——尤其 `--kvmem-gen-reserve`，
它直接决定「单次生成不能超过多少 token（含思考）」这个硬限制。

---

## 五、验证状态

| 项目 | 状态 |
|---|---|
| 编译 | ✅ 通过（仅一个无关测试目标失败） |
| 加载 | ✅ IQ3_S + mmproj + MTP + 262K + q8_0 |
| 推理正确性 | ✅ 算术/代码/中文/逻辑/创作/视觉 六项全对 |
| MTP | ✅ 接受率 58.6–61.6% |
| 速度 | ✅ decode 36.6 tok/s |
| 视觉 | ✅ GPU 编码，识别正确 |
| **长上下文压力** | ✅ 已做，见下节 |
| **稳定性** | ⚠️ 早期有一次未复现的进程退出（服务 3 个请求后，exit 127）；此后 30+ 次请求稳定 |

---

## 六、长上下文压力测试（2026-09-18）

测试脚本：`kvmem-llama.cpp\needle-test.py`（本机新增，可重跑）

方法：用 17 轮"工具结果"把上下文填到 **261,445 token**（几乎正好填满 262144），
每轮埋一根唯一的"针"（某部门的授权码），然后从不同深度抽问。

### 结果 1：不掉速 —— **证实**

| 上下文 | 每轮耗时 |
|---|---|
| 77K | 48.7 s |
| 138K | 49.9 s |
| 200K | 50.1 s |
| 261K | **52.0 s** |

**上下文增长 3.4 倍，每轮耗时只涨 7%。** 这是 KVMem 相对 KV 流式的核心优势所在。

- 超出 `-c` 时服务器返回干净的 `prompt + max_tokens exceeds n_ctx`（HTTP 400），**不崩溃**
- 前缀缓存全程有效（`cache_hit` 始终接近上一轮 prompt 长度）
- 显存稳定在 17.5 GB，不随上下文增长

### 结果 2：长上下文召回 —— **首轮测试有设计缺陷，结论作废**

首轮测出"3/5"，但**这是测试脚本的 bug，不是 kvmem 的问题**，记录在此以免重蹈覆辙。

**现象**：两次"答错"的答案，恰好是**另一个同部门针的号码**：

| 题 | 期望 | 实际答的 | 真相 |
|---|---|---|---|
| 深度 13 (Payroll) | `NNYD-8176` | `VJOB-6402` | **两者都是 Payroll** |
| 深度 17 (Facilities) | `CSIC-6316` | `IRCU-4754` | **两者都是 Facilities** |

**根因**：针的部门名从 8 个里随机取，17 根针必然重名。
于是"Payroll 部门的授权码是多少"这个问题**本身就有两个正确答案**——
模型答另一个是**完全正确的行为**，却被脚本判为 MISS。

**决定性证据**——换用无歧义的问法重问：

| 目标 | 原问法 | 引用原句 | 只给部门名 | 按编号问 |
|---|---|---|---|---|
| 深度 13 (Payroll) | ❌ | ✅ | ❌ | ✅ |
| 深度 17 (Facilities) | ❌ | ✅ | ✅ | ✅ |
| 深度 9 对照 | ✅ | ✅ | ✅ | ✅ |

**只要问题无歧义，答案就正确** → 说明**信息在窗口内且可检索**，不是检索失败。

**修正**：`needle-test.py` 已改为**部门不重复抽样**（`rng.sample`），
并加了 `--chunks` 上限校验。修正后的数据见下。

### 结果 2b：修正测试后的召回 —— **5/5 全部命中** ✅

同样是 261,447 token 上下文、17 轮填充，部门唯一后抽问 5 个深度：

| 深度 | 期望 | 实际 | |
|---|---|---|---|
| 5.9% | UTPT-7212 | UTPT-7212 | ✅ |
| 29.4% | HUGO-8081 | HUGO-8081 | ✅ |
| 52.9% | BNIV-0242 | BNIV-0242 | ✅ |
| 76.5% | DYFN-6181 | DYFN-6181 | ✅ |
| 100.0% | ZWKJ-7194 | ZWKJ-7194 | ✅ |

**26 万 token 上下文中，从最开头到最末尾，5/5 精确召回。**

**结论修正**：kvmem 在长上下文下的检索是**可靠的**。
先前的"3/5"完全是我的测试歧义造成的。

> **但请记住适用边界**：本测试的针是**唯一且无歧义**的句子。
> 如果真实场景里存在多条高度相似、问法无法区分的信息，仍可能选错——
> 这是所有检索式方案的固有特性。需要绝对精确时用 faq5 / kv-streaming。

> 这个坑值得记住：**针测试必须保证每根针的提问方式唯一指向一个答案**，
> 否则你测的是"模型能否猜中出题人的心思"，不是检索质量。

### 结果 3：加大检索窗口**没有**改善召回

把 `--kvmem-budget` 从 36864 翻倍到 65536 后重测：

- 召回数字**完全没变**（连答错的都一样）
- 代价：每轮耗时从 49.9 s 涨到 **63.1 s（慢 26%）**

**结论：窗口翻倍对召回没有可观测收益，但明确拖慢速度。**
建议先用作者的 36864 配方。

### 对生产的含义

| 场景 | 建议 |
|---|---|
| 要快、长上下文、信息条目可区分 | **kvmem（18200），36.6 t/s，26 万召回 5/5** |
| ≤18 万且要求全量注意力无损 | faq5（8080/8081/8083），23–26 t/s |
| >18 万且要求全量注意力无损 | kv-streaming（18082），~10 t/s |
| 上下文里有多条高度相似、问法分不开的信息 | 用 faq5 / kv-streaming（全量注意力） |

**`--kvmem-budget` 是"速度 ↔ 召回"旋钮，但不是越大越好**——本次实测翻倍后召回没变、速度慢了 26%。
建议先用 36864（作者配方）。

### 下一步建议

1. **长上下文压力测试**：填充至 256K，验证检索召回精度与掉速情况。
   可复用 `kv-streaming\tools\` 里的 needle/marker 验证法，
   或用 kvmem 自带的 `scripts/multimodal_canary.py`。
2. **稳定性观察**：留意那次未复现的退出是否再现。
3. **调参**：利用 6G 余量提升 `--kvmem-gen-reserve`。
4. **与 kv-streaming 做 A/B**：KVMem 快但近似，kv-streaming 慢但无损，按任务分流。

(结束)

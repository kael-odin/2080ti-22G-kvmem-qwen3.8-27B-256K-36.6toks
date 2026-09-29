# KVMem on RTX 2080 Ti 22G（sm_75 / Windows）——移植与复现手册

上游项目：<https://github.com/kvmem/kvmem-llama.cpp>。本仓基线锁定 **13b7a15**（v0.16.0-rc1 + 4 commits，llama.cpp 子模块 pin `b81c99b`）——**即 36.6 tok/s 实测所用的确切源码树**，`prepare-source.bat` 按提交号精确复现，不受上游发版漂移影响。

- 原理与取舍（KVMem vs KV 流式）：[《KVMem 方案评估》](docs/kvmem-evaluation-zh.md)
- 实测数据与踩坑全记录：[《KVMem Windows / sm_75 移植与验证》](docs/kvmem-windows-port-zh.md)

## 本目录内容

| 文件 | 说明 |
|---|---|
| `prepare-source.bat` | **一键完整源码**：克隆上游 → 签出基线 13b7a15 → 子模块 pin b81c99b → 套用本仓补丁 → 安装构建脚本 |
| `build-windows-sm75.bat` | Windows + VS + CUDA 13.2.86 的配置/构建（显式 `CMAKE_CUDA_ARCHITECTURES=75` + `GGML_CUDA_FA_ALL_QUANTS=ON`，上游默认 120a-real 在本机不可用） |
| `include/kvmem_win_compat.h` | 本机新增的 Windows 兼容头（父仓补丁引用它） |
| `patches/0001-kvmem-parent-sm75-windows.patch` | kvmem-llama.cpp 父仓的全部本地改动（CMake、NVMe tier、测试、CLI） |
| `patches/llama.cpp-b81c99b-sm75-tree-snapshot.patch` | llama.cpp 子模块的**树快照**（= 上游 `llama-kvmem-current.patch` 回放 + sm_75 增量） |
| `tools/needle-test.py` 等 5 个脚本 | 对着 `llama-kvmem-server`（默认端口 18200）做单/多 needle、decode、A-B 测试 |
| `docs/` | 两份完整报告（评估 + 移植验证） |

## 复现步骤（Windows，三步）

```bat
cd kvmem
prepare-source.bat                    :: 1. 完整源码就绪（kvmem-src\）
cd kvmem-src
build-windows-sm75.bat                :: 2. 编译（CUDA 13.2 Update 2 必需，免管理员 redist 安装法见移植报告）
:: 3. 用启动配方起 llama-kvmem-server，再跑验证：
python ..\tools\needle-test.py --port 18200 --chunks 30 --chunk-tokens 8000
```

<details>
<summary>手动分步（等价于 prepare-source.bat 做的事）</summary>

```bat
git clone https://github.com/kvmem/kvmem-llama.cpp kvmem-llama.cpp
cd kvmem-llama.cpp
git checkout 13b7a155c6aaf038a02380b16bd8b8ae7eb760e7
git submodule update --init llama.cpp
cd llama.cpp && git checkout b81c99b47 && cd ..

:: 子模块树，二选一：
::  A（快照，一步到位）  git apply --whitespace=nowarn <本仓>\kvmem\patches\llama.cpp-b81c99b-sm75-tree-snapshot.patch
::  B（上游路径）        scripts\apply-patches.sh ，然后自行补 sm_75 差异（见移植报告）

git apply --whitespace=nowarn <本仓>\kvmem\patches\0001-kvmem-parent-sm75-windows.patch
copy <本仓>\kvmem\include\kvmem_win_compat.h kvmem\include\kvmem\
copy <本仓>\kvmem\build-windows-sm75.bat .
build-windows-sm75.bat
```

</details>

> 上游 **v0.16.0-rc3 起提供覆盖 RTX 20 编译目标的 Windows 预编译 ZIP**（官方仅实测 5060 Ti，Turing 待社区验证——本仓数据即 Turing 实测）；只想免编译试玩可选 **prism.3** 实验版的 Bonsai 2 27B 预编译包。

启动配方、显存预算与质量验货清单见[移植报告](docs/kvmem-windows-port-zh.md)。

## 本机实测（2080 Ti 22G，Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp）

| 项 | 结果 |
|---|---|
| decode | **36.6 tok/s**（上游作者 RTX 5060 Ti：31.7 tok/s） |
| MTP 接受率 | 58.6% / 61.6%（n_max=3 + ReplaySSM） |
| 上下文 / KV | `n_ctx=262144`，type_k = type_v = q8_0 |
| 视觉 | BF16 mmproj 在 GPU，位置/颜色/形状识别全对 |
| 显存 | 16552 / 22528 MiB（余 ~6 GB） |
| 质量 | 算术/代码/中文/逻辑/创作/视觉全部人工验货通过，无 IQ3 乱码 |

## 已知边界

- 检索式近似：全局综合类任务（全文总结、跨段事实对比）可能漏块——这类任务请用[本仓备档的 KV 流式方案](../kv-streaming/README.md)。
- 单次生成不得超过 `--kvmem-gen-reserve`（IQ3 配方 16384 token，含思考）；上游 v0.17.0 正在改进。
- 实测仅覆盖一张卡、一个模型线；其他 Turing 卡跑通后欢迎反馈。

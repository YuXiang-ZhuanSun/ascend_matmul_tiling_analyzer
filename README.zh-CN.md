# MatMul Tiling Analyzer

[![CI](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](./pyproject.toml)
[![Source-Faithful](https://img.shields.io/badge/strategy-source--faithful-0A7E8C.svg)](./docs/source_branch_mapping.md)

[项目首页](./README.md) | [English README](./README.en.md) | [源码分支映射](./docs/source_branch_mapping.md) | [策略手册](./docs/ascend950_tiling_strategy.md) | [输出示例](./docs/example_outputs.md)

![MatMul Tiling Analyzer Banner](./docs/assets/banner.svg)

**像读源码一样读 tiling，像内核工程师一样做性能诊断。**

在 `mat_mul_v3` 的调优实践里，最昂贵的往往不是计算本身，而是不可见的决策路径：
这条 shape 命中了哪个分支？分核是否均衡？尾块是否正在悄悄吞噬吞吐？

`MatMul Tiling Analyzer` 将这些不确定性转化为可验证证据：
按源码语义重放策略选择，解码 `tiling_key` 与 `tiling_data`，并展开到每个核心的任务负载视图。

## About

`MatMul Tiling Analyzer` 是面向 **Ascend950 (DAV_3510) / MatMulV3** 的 tiling 分析与诊断工具。
它不替代 runtime，不做启发式猜测，而是把 host 侧分支、kernel 派发与 scheduler 行为连接为一条可追踪链路，服务于调优定位、回归评审与跨团队协作。

## 为什么重要

- 在进入漫长 profiling 循环前，先确认策略分支是否正确。
- 在“可以跑”之外，量化 inter-core / intra-core 切分质量。
- 用每核任务与尾块信息，快速暴露负载不均与利用率损失。
- 产出可复现、可审阅、可自动化处理的报告资产。

## 你可以验证什么

- **策略选择**：`k_equal_zero`、`to_mul`、`basic_streamk`、`basic_aswt`
- **分支细节**：`streamk_sk` / `streamk_dpsk`、`basic_aswt_a_full_load` / `basic_aswt_b_full_load`
- **键值语义**：`tiling_key` 字段级解码（`api_level`、`model`、`full_load`、`l0c2out` 等）
- **调度结果**：inter-core 切分、intra-core 分块、每核任务列表
- **导出产物**：单 case 文本/JSON，批量 `summary.csv` / `summary.json`

## 策略覆盖（当前）

| Strategy | 分析器中的典型触发条件 | Kernel implementation |
| --- | --- | --- |
| `k_equal_zero` | `k == 0` 且无 bias | `MatMulInputKEqZeroClearOutput` |
| `to_mul` | 强制 group acc + fp32 + (`m==1` 或 `n==1`) + `k>=512` | `MatMulToMulActKernel` |
| `basic_streamk` | `x1_format == ND` 且 Stream-K 能力检查通过 | `MatMulStreamKActKernel` |
| `basic_aswt` | 默认回退路径（含 full-load / fixpipe 变体） | `MatMulActKernel` / `MatMulFixpipeOptiActKernel` |

## 工作流程

```mermaid
flowchart LR
    A["Case Input (CLI/CSV)"] --> B["Strategy Selection"]
    B --> C["Tiling Key Decode"]
    C --> D["Tiling Data Reconstruction"]
    D --> E["Scheduler-Level Per-Core Simulation"]
    E --> F["Text / JSON Reports + Batch Summaries"]
```

## 快速开始

```powershell
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
```

单 case 分析：

```powershell
python cli.py --m 2048 --k 4096 --n 256 --dtype bfloat16
```

也可使用安装后的命令：

```powershell
matmul-tiling-analyzer --m 2048 --k 4096 --n 256 --dtype bfloat16
```

运行测试：

```powershell
python -m pytest
```

## 输出产物

当指定 `--output-dir` 时，输出目录结构如下：

```text
results/<run_name>/
  summary.csv
  summary.json
  cases/
    <testcase>.json
    <testcase>.txt
```

## 范围与非目标

- 当前重点：`mat_mul_v3` tiling 分析
- 当前文档/样例覆盖硬件：`Ascend950 (DAV_3510)`
- 项目定位：分析与诊断工具，不是执行时替代实现

## 文档导航

- [项目首页](./README.md)
- [English README](./README.en.md)
- [源码分支映射](./docs/source_branch_mapping.md)
- [Ascend950 Tiling Strategy](./docs/ascend950_tiling_strategy.md)
- [输出示例](./docs/example_outputs.md)
- [Release Notes v0.1.0](./docs/release_notes_v0.1.0.md)
- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [更新日志](./CHANGELOG.md)

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。

# MatMul Tiling Analyzer

[![CI](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](./pyproject.toml)
[![源码同构](https://img.shields.io/badge/strategy-源码同构-0A7E8C.svg)](./docs/source_branch_mapping.md)

[项目首页](./README.md) | [English README](./README.en.md)

![MatMul Tiling Analyzer Banner](./docs/assets/banner.png)

**像读源码一样读 tiling，像内核工程师一样做性能诊断。**

MatMul 调优里最贵的往往不是算力，而是不确定性：这个 shape 到底命中了哪条分支？分核是否均衡？芯片是否真的被喂满？  
`MatMul Tiling Analyzer` 把这些问题变成可核对证据。它重放真实 `mat_mul_v3` 的 tiling 决策，并展开成每个核心的负载视图。

## 为什么做这个项目

- 在昂贵 profiling 之前，先发现分支与调度层面的错误。
- 判断核间/核内分块是否合理，避免“看起来能跑、实际上吃不满”。
- 用每核负载与尾块信息定位潜在性能瓶颈。
- 为性能评审、回归排查提供可读、可复现、可沟通的分析产物。

## 可信度来自哪里

这个项目的核心不是“经验猜测”，而是“源码同构”。

- 用 Python 重写 Ascend C `mat_mul_v3` 的 tiling 策略。
- 与官方算子仓保持分支级映射：[`ops-nn/mat_mul_v3`](https://gitcode.com/cann/ops-nn/tree/master/matmul/mat_mul_v3)
- 把 host 侧分支选择、`tiling_key` / `tiling_data`、kernel dispatch、schedule 分解串成一条可追溯链路。

## 你能得到什么

单个 case 输出：

- 命中的策略与源码分支
- `tiling_key` 关键字段解码
- 相关 `tiling_data` 字段
- 核间分核与核内分块结果
- 每核工作负载摘要与任务布局

批量输出：

- `summary.json`
- `summary.csv`
- `cases/<testcase>.json`
- `cases/<testcase>.txt`

## 快速开始

```powershell
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
```

启动本地面板：

```powershell
matmul-tiling-panel
```

面板会在浏览器中运行，支持 Windows、macOS 和 Linux。你可以手工输入 shape、上传 CSV 用例、查看 selected branch、解码 `tiling_key`，并下钻查看 32 个核心的详细负载。

单 case 分析：

```powershell
python cli.py --m 2048 --k 4096 --n 256 --dtype bfloat16
```

运行测试：

```powershell
python -m pytest
```

## 项目范围

- 当前重点：`mat_mul_v3` 算子 tiling 分析
- 当前文档与示例覆盖硬件：`Ascend950 (DAV_3510)`
- 项目定位：分析诊断工具，不是运行时替代品

## 文档导航

- [项目首页](./README.md)
- [English README](./README.en.md)
- [源码分支映射表](./docs/source_branch_mapping.md)
- [Ascend950 Tiling Strategy](./docs/ascend950_tiling_strategy.md)
- [本地面板](./docs/panel.md)
- [示例输出](./docs/example_outputs.md)
- [Release Notes v0.1.0](./docs/release_notes_v0.1.0.md)
- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [更新日志](./CHANGELOG.md)

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。

# MatMul Tiling Analyzer

[English README](./README.en.md) | [Project Home](./README.md)

## 项目简介

`MatMul Tiling Analyzer` 是一个面向 `Ascend950 (DAV_3510)` 的 `mat_mul_v3` tiling 分析项目。
项目的目标不是凭经验“猜”某个用例会如何分块，而是尽量沿着源码中的真实决策路径，重放 host 侧的 tiling 分支选择，提取 `tiling_key` / `tiling_data`，并把 kernel 侧的分核、分块和每核负载整理成可读报告。

当前重点覆盖：

- `op_host` 中 Ascend950 相关的 tiling 策略选择逻辑
- `mat_mul_v3_apt.cpp` 中的 kernel dispatch
- `block_scheduler_*` 中的分核和核内任务展开
- 紧凑 CSV 与扩展 fuzz CSV 两类输入格式

## 这个项目能做什么

- 判断一个 `mat_mul_v3` 用例会落到哪条 tiling 分支
- 输出 `tiling_key`、关键字段拆解和核心 `tiling_data`
- 展示核间分工、核内分块和每个核心的预期任务
- 把“分析器分支”和“源码分支”建立稳定映射，方便补齐和维护
- 生成适合调试、归档和发布的批量结果文件

## 快速开始

```powershell
cd .\matmul_tiling_analyzer
python -m pip install -e .
python .\cli.py --input=.\cases\quickstart_cases.csv --output-dir=.\results\quickstart
```

分析单个 case：

```powershell
python .\cli.py --m 2048 --k 4096 --n 256 --dtype bfloat16
```

运行测试：

```powershell
python -m pytest
```

## 命令行用法

批量分析 CSV：

```powershell
python .\cli.py --input=.\cases\user_provided_extended_cases.csv --output-dir=.\results\user_provided_extended
```

输出 JSON：

```powershell
python .\cli.py --input=.\cases\quickstart_cases.csv --format=json
```

分析单个转置 case：

```powershell
python .\cli.py --m 100 --k 1920 --n 512 --dtype float16 --transpose-x1
```

## 当前覆盖的主分支

- `k_equal_zero`
- `to_mul`
- `basic_streamk`
- `basic_aswt`
- `basic_aswt_a_full_load`
- `basic_aswt_b_full_load`
- `basic_aswt_fixpipe`

更详细的源码映射见 [docs/source_branch_mapping.md](./docs/source_branch_mapping.md)。

## 支持的输入格式

分析器目前支持两类输入：

- 紧凑 schema：`testcase_name, op_name, stc_inputs, ...`
- 扩展 schema：`testcase_name, network_name, op_name, stc_inputs, stc_ori_inputs, ...`

对于扩展 schema，工具会：

- 从 compilation/runtime 参数中读取 transpose 信息
- 在 transpose 生效后推导 `(m, k, n)`
- 从静态输入元组判断 bias 是否存在
- 将原始 CSV 行保存在结果中，便于追溯

## 输出结果

当指定 `--output-dir` 时，工具会生成：

- `summary.json`
- `summary.csv`
- `cases/<testcase>.json`
- `cases/<testcase>.txt`

示例可见 [docs/example_outputs.md](./docs/example_outputs.md)。

## 目录结构

```text
matmul_tiling_analyzer/
  cli.py
  analyze_cases.py
  pyproject.toml
  LICENSE
  CHANGELOG.md
  CONTRIBUTING.md
  README.md
  README.zh-CN.md
  README.en.md
  cases/
  docs/
  matmul_tiling_analyzer/
  tests/
  results/
```

## 文档导航

- [项目首页](./README.md)
- [English README](./README.en.md)
- [Ascend950 Tiling Strategy](./docs/ascend950_tiling_strategy.md)
- [Source Branch Mapping](./docs/source_branch_mapping.md)
- [Example Outputs](./docs/example_outputs.md)
- [Changelog](./CHANGELOG.md)
- [Contributing](./CONTRIBUTING.md)

## 开发约定

- 优先补齐源码中的真实分支，再考虑额外抽象
- 新增分支时，先更新 `source_mapping.py` 和映射文档
- 每个新增分支都补至少一个代表用例和一个边界用例
- 文档里明确区分“源码事实”和“分析器推断”

## License

项目当前采用 [MIT License](./LICENSE)。

# MatMul Tiling Analyzer

[![CI](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](./pyproject.toml)
[![Source-Faithful](https://img.shields.io/badge/strategy-source--faithful-0A7E8C.svg)](./docs/source_branch_mapping.md)

[Project Home](./README.md) | [中文 README](./README.zh-CN.md)

![MatMul Tiling Analyzer Banner](./docs/assets/banner.png)

**Read tiling like source code. Diagnose performance like a kernel engineer.**

In matmul optimization, the expensive part is often not math but uncertainty: did this shape hit the right branch, is core splitting balanced, and is hardware truly saturated.  
`MatMul Tiling Analyzer` turns that uncertainty into evidence by replaying real `mat_mul_v3` tiling logic and expanding it into per-core workload views.

## Why This Project Exists

- Catch branch and scheduling issues before long profiling loops.
- Validate inter-core and intra-core partitioning quality for each shape.
- Expose utilization risks and bottlenecks with concrete per-core load data.
- Produce clear artifacts for regression reviews and cross-team discussions.

## Why You Can Trust It

The core principle is **source alignment**, not heuristic approximation.

- Strategy logic is rewritten in Python from Ascend C `mat_mul_v3`.
- Branch-by-branch mapping is maintained against the official operator source: [`ops-nn/mat_mul_v3`](https://gitcode.com/cann/ops-nn/tree/master/matmul/mat_mul_v3)
- Host-side branch selection, `tiling_key` / `tiling_data`, kernel dispatch, and schedule decomposition are connected in one traceable analysis path.

## What You Get

Per case:

- selected strategy and source-mapped branch
- decoded `tiling_key` and key fields
- relevant `tiling_data` payload
- inter-core split and intra-core block plan
- per-core workload summary and kernel task layout

Batch outputs:

- `summary.json`
- `summary.csv`
- `cases/<testcase>.json`
- `cases/<testcase>.txt`

## Quick Start

```powershell
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
```

Single case:

```powershell
python cli.py --m 2048 --k 4096 --n 256 --dtype bfloat16
```

Run tests:

```powershell
python -m pytest
```

## Scope

- Current focus: `mat_mul_v3` tiling analysis
- Current hardware scope in docs/examples: `Ascend950 (DAV_3510)`
- Positioning: analysis/diagnosis tool, not a runtime replacement

## Documentation

- [Project Home](./README.md)
- [README.zh-CN.md](./README.zh-CN.md)
- [Source Branch Mapping](./docs/source_branch_mapping.md)
- [Ascend950 Tiling Strategy](./docs/ascend950_tiling_strategy.md)
- [Example Outputs](./docs/example_outputs.md)
- [Release Notes v0.1.0](./docs/release_notes_v0.1.0.md)
- [Contributing](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Changelog](./CHANGELOG.md)

## License

This project is licensed under [MIT](./LICENSE).

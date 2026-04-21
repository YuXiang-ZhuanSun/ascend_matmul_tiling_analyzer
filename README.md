# MatMul Tiling Analyzer

[![CI](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](./pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Ascend950-D46B08.svg)](./docs/ascend950_tiling_strategy.md)

**Tags:** `ascend950` `matmul` `tiling` `operator-analysis` `kernel-scheduling` `dav_3510`

[中文 README](./README.zh-CN.md) | [English README](./README.en.md)

![MatMul Tiling Analyzer Banner](./docs/assets/banner.svg)

`MatMul Tiling Analyzer` is a Python project for analyzing `mat_mul_v3` tiling behavior on `Ascend950 (DAV_3510)`.
It replays host-side strategy selection, reconstructs `tiling_key` and `tiling_data`, maps the selected path to `mat_mul_v3_apt.cpp`, and expands the expected workload on every core.

`MatMul Tiling Analyzer` 是一个面向 `Ascend950 (DAV_3510)` 的 `mat_mul_v3` tiling 分析项目。
它会复现 host 侧策略选择过程，重建 `tiling_key` / `tiling_data`，映射到 `mat_mul_v3_apt.cpp` 的 kernel 分发逻辑，并展开每个核心上的预期负载。

## Why This Project

- Analyze real `mat_mul_v3` tiling branches instead of using black-box guesses
- Keep analyzer behavior traceable to concrete source files and strategy branches
- Make inter-core and intra-core blocking visible in a report-friendly format
- Support both regression CSV and fuzz CSV case formats

## Features

- Ascend950-focused tiling analysis for `mat_mul_v3`
- Reconstruction of `tiling_key`, `tiling_key_fields`, and critical `tiling_data`
- Mapping from host branch to kernel dispatch and scheduler behavior
- Batch export to `summary.json`, `summary.csv`, per-case `.json`, and readable `.txt`
- Per-core workload presentation with active-core summaries
- Test coverage for the CSV parser

## Quick Start

```powershell
cd .\matmul_tiling_analyzer
python -m pip install -e .
python .\cli.py --input=.\cases\quickstart_cases.csv --output-dir=.\results\quickstart
```

Analyze a single case:

```powershell
python .\cli.py --m 2048 --k 4096 --n 256 --dtype bfloat16
```

Run tests:

```powershell
python -m pytest
```

## Command-Line Usage

Batch analysis:

```powershell
python .\cli.py --input=.\cases\user_provided_extended_cases.csv --output-dir=.\results\user_provided_extended
```

JSON output:

```powershell
python .\cli.py --input=.\cases\quickstart_cases.csv --format=json
```

Single-case analysis:

```powershell
python .\cli.py --m 100 --k 1920 --n 512 --dtype float16 --transpose-x1
```

## Repository Layout

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

## Documentation

- [中文 README](./README.zh-CN.md)
- [English README](./README.en.md)
- [Ascend950 Tiling Strategy](./docs/ascend950_tiling_strategy.md)
- [Source Branch Mapping](./docs/source_branch_mapping.md)
- [Example Outputs](./docs/example_outputs.md)
- [Release Notes v0.1.0](./docs/release_notes_v0.1.0.md)
- [Changelog](./CHANGELOG.md)
- [Contributing](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)

## Supported Inputs

The analyzer supports:

- compact CSV schemas such as `testcase_name, op_name, stc_inputs, ...`
- extended fuzz CSV schemas such as `testcase_name, network_name, op_name, stc_inputs, stc_ori_inputs, ...`

For extended rows, the parser will:

- read transpose flags from compilation/runtime attributes
- derive `(m, k, n)` after transpose is applied
- detect bias presence from the static input tuple
- preserve the raw CSV row for traceability

## Output Artifacts

When `--output-dir` is provided, the project writes:

- `summary.json`
- `summary.csv`
- `cases/<testcase>.json`
- `cases/<testcase>.txt`

Examples are documented in [docs/example_outputs.md](./docs/example_outputs.md).

## Project Status

The current implementation focuses on the major Ascend950 branches already modeled in the repository:

- `k_equal_zero`
- `to_mul`
- `basic_streamk`
- `basic_aswt`
- `basic_aswt_a_full_load`
- `basic_aswt_b_full_load`
- `basic_aswt_fixpipe`

Future work should continue to expand source coverage branch by branch and keep the source mapping document up to date.

## License

This project is released under the [MIT License](./LICENSE).

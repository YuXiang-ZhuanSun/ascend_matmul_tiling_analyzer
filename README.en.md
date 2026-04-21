# MatMul Tiling Analyzer

[![CI](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](./pyproject.toml)

**Tags:** `ascend950` `matmul` `tiling` `operator-analysis` `kernel-scheduling`

[中文 README](./README.zh-CN.md) | [Project Home](./README.md)

## Overview

`MatMul Tiling Analyzer` is a Python project for analyzing `mat_mul_v3` tiling behavior on `Ascend950 (DAV_3510)`.
The goal is to follow the real source-code decision path as closely as possible: replay host-side strategy selection, reconstruct `tiling_key` and `tiling_data`, map the final choice to `mat_mul_v3_apt.cpp`, and present inter-core plus intra-core workload in a readable form.

The current focus is:

- Ascend950 tiling selection in `op_host`
- kernel dispatch in `mat_mul_v3_apt.cpp`
- scheduler expansion from `block_scheduler_*`
- support for both compact regression CSV and extended fuzz CSV inputs

## What The Project Provides

- Predict which tiling branch a `mat_mul_v3` case will take
- Decode `tiling_key` and expose the important fields
- Reconstruct the most relevant tiling fields from `tiling_data`
- Show inter-core split, intra-core tile shape, and per-core work
- Keep analyzer branches traceable back to concrete source branches
- Generate artifacts suitable for debugging, regression tracking, and publication

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

Analyze a CSV and save artifacts:

```powershell
python .\cli.py --input=.\cases\user_provided_extended_cases.csv --output-dir=.\results\user_provided_extended
```

Emit JSON to stdout:

```powershell
python .\cli.py --input=.\cases\quickstart_cases.csv --format=json
```

Analyze a transpose case:

```powershell
python .\cli.py --m 100 --k 1920 --n 512 --dtype float16 --transpose-x1
```

## Covered Strategy Families

- `k_equal_zero`
- `to_mul`
- `basic_streamk`
- `basic_aswt`
- `basic_aswt_a_full_load`
- `basic_aswt_b_full_load`
- `basic_aswt_fixpipe`

See [docs/source_branch_mapping.md](./docs/source_branch_mapping.md) for the maintainable branch mapping.

## Supported CSV Schemas

The analyzer supports:

- compact schemas such as `testcase_name, op_name, stc_inputs, ...`
- extended fuzz schemas such as `testcase_name, network_name, op_name, stc_inputs, stc_ori_inputs, ...`

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

- [Project Home](./README.md)
- [中文 README](./README.zh-CN.md)
- [Ascend950 Tiling Strategy](./docs/ascend950_tiling_strategy.md)
- [Source Branch Mapping](./docs/source_branch_mapping.md)
- [Example Outputs](./docs/example_outputs.md)
- [Release Notes v0.1.0](./docs/release_notes_v0.1.0.md)
- [Changelog](./CHANGELOG.md)
- [Contributing](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)

## Development Notes

- Prefer matching a real source branch before introducing extra abstraction
- Register new strategy-source relations in `source_mapping.py` first
- Add one representative case and one edge case for every new branch
- Keep a clear distinction between source-grounded facts and analyzer inference

## License

This project currently uses the [MIT License](./LICENSE).

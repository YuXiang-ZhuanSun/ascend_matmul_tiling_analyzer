# Release Notes: v0.1.0

## Overview

`v0.1.0` is the first public projectized release of `MatMul Tiling Analyzer`.
This version turns the internal prototype into a standalone GitHub-ready repository with a cleaner structure, Python CLI entrypoints, documentation, tests, and reusable output artifacts.

## Highlights

- standalone project layout for `matmul_tiling_analyzer`
- direct CLI usage with `python cli.py --input=...`
- support for compact regression CSV and extended fuzz CSV
- Ascend950-focused modeling of major `mat_mul_v3` tiling branches
- readable per-case workload reports plus structured JSON exports
- source-to-analyzer branch mapping documentation

## Included Strategy Coverage

- `k_equal_zero`
- `to_mul`
- `basic_streamk`
- `basic_aswt`
- `basic_aswt_a_full_load`
- `basic_aswt_b_full_load`
- `basic_aswt_fixpipe`

## Project Files Added In This Release

- `README.md`, `README.zh-CN.md`, `README.en.md`
- `LICENSE`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `.github/workflows/ci.yml`
- GitHub issue and PR templates
- `docs/source_branch_mapping.md`
- `docs/example_outputs.md`
- `docs/ascend950_tiling_strategy.md`

## Validation

- `python -m pytest`
- `python .\cli.py --input=.\cases\quickstart_cases.csv --output-dir=.\results\quickstart --format=text`

## Known Limitations

- the analyzer currently focuses on major Ascend950 branches rather than full source coverage
- some behaviors are still modeled by analysis logic rather than exact source-level replay for every sub-branch
- GitHub release packaging is not yet automated

# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and the project is currently pre-`1.0.0`.

## [0.3.0] - 2026-04-23

### Added

- cross-platform local browser panel via `matmul-tiling-panel`
- manual shape analysis from the panel
- CSV testcase upload from the panel
- 32-core workload heatmap and per-core task drill-down
- panel smoke tests and CI coverage
- release workflow for building wheel and source distribution artifacts

## [0.2.0] - 2026-04-21

### Added

- standard project metadata with `pyproject.toml`, `LICENSE`, and `CHANGELOG.md`
- bilingual repository documentation via `README.md`, `README.zh-CN.md`, and `README.en.md`
- maintainable source-branch mapping document
- example output document for CLI and batch analysis artifacts
- parser smoke tests for compact and extended CSV schemas
- batch artifact export to `summary.json`, `summary.csv`, per-case `.json`, and per-case `.txt`

### Changed

- reorganized the repository into clearer top-level areas: `cases`, `docs`, `matmul_tiling_analyzer`, `tests`
- rewrote the README into a GitHub-friendly project homepage
- improved text report formatting to better emphasize strategy, tiling fields, and per-core load summary
- expanded CSV parsing to support both compact regression rows and extended fuzz rows

### Removed

- legacy `sample_cases.csv` in favor of curated case sets under `cases/`
- transient cache files from the tracked project tree

## [0.1.0] - 2026-04-21

### Added

- initial Ascend950-focused analyzer for `k_equal_zero`, `to_mul`, `basic_streamk`, and `basic_aswt`
- host-side `tiling_key` and `tiling_data` reconstruction
- kernel dispatch mapping for `mat_mul_v3_apt.cpp`
- scheduler-driven per-core workload expansion

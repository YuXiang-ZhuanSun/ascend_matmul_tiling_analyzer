# Physical Core Task Schema

The replay writes each case into one folder: `results/<run>/cases/<case_id>/`.

Each case folder contains:

- `summary.md`: human-readable shape, tiling-key, task count, and dispatch summary.
- `case.json`: input, source trace, tiling-key fields, dispatch, checks, and file references.
- `cores.json`: physical-core task details.

`cores.json` contains:

- `task_grid`: counts for M tiles, N tiles, K segments, and total task units.
- `task_units`: every logical `(m_tile, n_tile, k_segment)` unit with exact M/N/K ranges and assigned AIC.
- `physical_cores`: one entry for every `AIC_x`; stream-k cases also include paired `AIV_x` companion lanes.
- `checks`: assignment and lane-count checks.

Task size is derived from `bin_tiling_data` when present:

- `singleCoreM`: tuple index 4
- `singleCoreN`: tuple index 5
- `singleCoreK`: tuple index 10
- `baseM/baseN/baseK`: tuple indices 7/8/9

For non-stream kernels, AIC cores own the cube work. For stream-k kernels, AIC cores own K-segment compute units and two AIV lanes are recorded as reduction/fixpipe companions for each paired AIC, following the `aivNum == aicNum * 2` constraint in `matmul_v3_basic_streamk_tiling.cpp`.

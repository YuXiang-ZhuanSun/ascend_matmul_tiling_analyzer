# Source Branch Mapping

This document is the maintainable bridge between the real operator source tree and the analyzer implementation.
When a new tiling branch is added to the analyzer, update this file together with `matmul_tiling_analyzer/source_mapping.py`.

## Source Of Truth

- Runtime mapping used by the analyzer:
  - [source_mapping.py](C:/Users/Xiang/Downloads/ops-nn-master-matmul-mat_mul_v3/ops-nn-master-matmul-mat_mul_v3/matmul/mat_mul_v3/matmul_tiling_analyzer/matmul_tiling_analyzer/source_mapping.py)
- Main strategy selection logic:
  - `op_host/op_tiling/arch35/matmul_v3_tiling_advanced.cpp`
  - `op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h`
- Kernel dispatch entry:
  - `op_kernel/mat_mul_v3_apt.cpp`

## Mapping Table

| Analyzer strategy | Analyzer branch | Host-side source branch | Kernel implementation | Scheduler / execution model | Notes |
| --- | --- | --- | --- | --- | --- |
| `k_equal_zero` | `k_equal_zero_clear_output` | `matmul_v3_k_equal_zero_tiling.cpp` | `MatMulInputKEqZeroClearOutput` | AIV clear-output linear split | Triggered when `k == 0` and no bias is present. |
| `to_mul` | `basic_to_mul` | `matmul_v3_to_mul_tiling.cpp` | `MatMulToMulActKernel` | `BlockSchedulerMulBuiltIn` | Used for forced group accumulation on fp32 with vector-style flattening. |
| `basic_streamk` | `streamk_sk` | `matmul_v3_basic_streamk_tiling.cpp` via `CheckStreamKSKTiling` | `MatMulStreamKActKernel` | `BlockSchedulerStreamKBuiltIn` | Stream-K path where the final schedule is composed of SK rounds. |
| `basic_streamk` | `streamk_dpsk` | `matmul_v3_basic_streamk_tiling.cpp` via `CheckStreamKDPSKTiling` | `MatMulStreamKActKernel` | `BlockSchedulerStreamKBuiltIn` | Mixed DP + SK style scheduling. |
| `basic_aswt` | `basic_aswt` | `matmul_v3_basic_aswt_tiling.cpp` | `MatMulActKernel` | `BlockSchedulerAswtBuiltIn` | Default on-the-fly fallback for many regular cases. |
| `basic_aswt` | `basic_aswt_a_full_load` | `matmul_v3_basic_aswt_tiling.cpp` full-load branch | `MatMulActKernel` | `BlockSchedulerAswtBuiltIn` | Analyzer models the A-full-load decision under on-the-fly dispatch. |
| `basic_aswt` | `basic_aswt_b_full_load` | `matmul_v3_basic_aswt_tiling.cpp` full-load branch | `MatMulActKernel` | `BlockSchedulerAswtBuiltIn` | Analyzer models the B-full-load decision under on-the-fly dispatch. |
| `basic_aswt` | `basic_aswt_fixpipe` | `matmul_v3_basic_aswt_tiling.cpp` + helper logic | `MatMulFixpipeOptiActKernel` | `BlockSchedulerAswtBuiltIn` | Covers fixpipe-style `l0c2out` dispatch variants. |

## Priority Order

For the currently modeled Ascend950 path, the host-side strategy order is:

1. `k_equal_zero`
2. `to_mul`
3. `basic_streamk`
4. `basic_aswt`

The first capable strategy wins.

## Maintenance Rules

When you add a new branch:

1. Update `source_mapping.py` with the branch metadata used at runtime.
2. Update this document with the exact source file and kernel dispatch relation.
3. Add at least one case in `cases/` that lands on the new branch.
4. Regenerate example outputs if the user-visible report shape changes.

## Gaps To Track

The analyzer currently focuses on the major Ascend950 branches visible in the existing project code.
Any uncovered source branch should be listed here before implementation so the gap stays explicit instead of silent.

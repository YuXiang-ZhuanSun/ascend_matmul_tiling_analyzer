# Ascend950 MatMulV3 Tiling Strategy

## Overview

This document summarizes the Ascend950-specific `MatMulV3` tiling path that the analyzer currently models.
The focus is the `arch35 / DAV_3510` path, including:

- host-side strategy selection
- `tiling_key` construction
- `tiling_data` generation
- kernel dispatch in `mat_mul_v3_apt.cpp`
- scheduler-level per-core workload decomposition

## Host-Side Priority Order

The strategy order comes from:

- `op_host/op_tiling/arch35/matmul_v3_tiling_advanced.cpp`
- `op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h`

For Ascend950 the priority order is:

1. `MATMUL_INPUT_K_EQUAL_ZERO`
2. `TO_MUL`
3. `BASIC_STREAM_K`
4. `BASIC_ASWT`

The first capable strategy wins.

## Tiling Key Fields

The analyzer decodes the same logical fields used by the operator:

- `api_level`
- `transpose_nibble`
- `batch_model`
- `model`
- `full_load`
- `l0c2out`

Relevant sources:

- `op_host/op_tiling/arch35/matmul_v3_tiling_key.cpp`
- `op_host/op_tiling/arch35/matmul_v3_tiling_key.h`
- `op_kernel/arch35/mat_mul_tiling_key.h`
- `op_kernel/arch35/mat_mul_v3_tiling_key_public.h`

## Mode Summary

### K_EQUAL_ZERO

Sources:

- `matmul_v3_k_equal_zero_tiling.cpp`
- `mat_mul_v3_apt.cpp`
- `mat_mul_input_k_eq_zero_clear_output.h`

Trigger:

- `hasBias == false`
- `k == 0`

Behavior:

- no cube compute
- output is cleared by AIV cores
- `usedCoreNum = aivNum`

### TO_MUL

Sources:

- `matmul_v3_to_mul_tiling.cpp`
- `block_scheduler_mul.h`
- `mat_mul_to_mul_cmct.h`

Trigger:

- force group accumulation for fp32
- `m == 1 || n == 1`
- fp32 input
- `k >= 512`

Behavior:

- flatten the `MN` work into vector tiles
- key fields center around `baseMN`, `baseK`, `tailMN`, `tailK`, `loopK`

### BASIC_STREAM_K

Sources:

- `matmul_v3_basic_streamk_tiling.cpp`
- `block_scheduler_streamk.h`

Trigger highlights:

- `x1_format == ND`
- non self-slice stream-k restrictions are satisfied
- `aivNum == 2 * aicNum`
- `CheckStreamKSKTiling` or `CheckStreamKDPSKTiling` passes

Behavior:

- `baseM/baseN` define the MN tile
- `skSingleCoreK` defines the per-core split-K segment
- the last scheduling round becomes Stream-K work

### BASIC_ASWT

Sources:

- `matmul_v3_basic_aswt_tiling.cpp`
- `matmul_v3_asw_tiling.cpp`
- `matmul_v3_tiling_helper.cpp`
- `block_scheduler_aswt.h`

This is the default fallback path and also the most complex one.
The analyzer currently distinguishes:

- default `basic_aswt`
- `basic_aswt_a_full_load`
- `basic_aswt_b_full_load`
- fixpipe-flavored `l0c2out` variants

## Kernel Dispatch Mapping

Kernel dispatch happens in:

- `op_kernel/mat_mul_v3_apt.cpp`

Current mapping covered by the analyzer:

- `K_EQUAL_ZERO -> MatMulInputKEqZeroClearOutput`
- `TO_MUL -> MatMulToMulActKernel`
- `STREAM_K -> MatMulStreamKActKernel`
- `BASIC + ON_THE_FLY -> MatMulActKernel`
- `BASIC + FIXPIPE -> MatMulFixpipeOptiActKernel`

## Scheduler Modeling

### Aswt Scheduler

From `block_scheduler_aswt.h`:

- uses windowed snake traversal in MN
- may split the final tail tile further with `mTailCnt * nTailCnt`
- may also rebalance edge tiles through `mBaseTailSplitCnt / nBaseTailSplitCnt`

### Stream-K Scheduler

From `block_scheduler_streamk.h`:

- distinguishes `DP` and `SK` rounds
- `SK` rounds split one `(mTile, nTile)` into multiple `kTileIdx` segments

### To-Mul Scheduler

From `block_scheduler_mul.h`:

- schedules flattened vector tiles
- each AIV core advances by `tile_index = core + round * usedCoreNum`

## Current Analyzer Boundaries

The project already handles the main Ascend950 strategy branches and multiple CSV schemas.
The next refinement targets are:

- more precise transpose/NZ modeling
- better full-load branch fidelity
- load-balance candidate tracing
- FLOPs and bandwidth accounting per core

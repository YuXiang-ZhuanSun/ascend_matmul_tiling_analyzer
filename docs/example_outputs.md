# Example Outputs

This document shows the current text-report style emitted by the analyzer.
The goal is to make the output readable enough for code review, issue triage, and future GitHub presentation.

## Single Case Command

```powershell
python .\cli.py --m 2048 --k 4096 --n 256 --dtype bfloat16
```

## Example: `mat_mul_v3_M_2048_K_4096_N_256_case14`

```text
== Case ==
name: mat_mul_v3_M_2048_K_4096_N_256_case14
op: mat_mul_v3
shape: M=2048, K=4096, N=256
dtype: in=bfloat16, out=bfloat16
layout: x1=ND, x2=ND, y=ND
transpose: x1=False, x2=False

== Strategy ==
selected: basic_streamk
branch: streamk_sk
tiling_key: 0x1001 (4097)
kernel: MatMulStreamKActKernel

== Source Mapping ==
host_file: op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp
kernel_entry: op_kernel/mat_mul_v3_apt.cpp
scheduler: BlockSchedulerStreamKBuiltIn

== Tiling ==
tiling_key_fields: {'api_level': 1, 'a_trans': 0, 'b_trans': 0, 'transpose_nibble': 0, 'batch_model': 0, 'model': 1, 'full_load': 0, 'l0c2out': 0}
inter_core: {'used_core_num': 32, 'm_tile_num': 8, 'n_tile_num': 1, 'sk_k_tile_num': 4}
intra_core: {'base_m': 256, 'base_n': 256, 'base_k': 64, 'm_l1': 256, 'n_l1': 256, 'k_l1': 256}

== Load Summary ==
active_cores: 32
idle_cores: 64
total_tasks: 32
core_breakdown:
  - AIC: active=32, idle=0, tasks=32
  - AIV: active=0, idle=64, tasks=0

== Active Core Workload ==
- AIC[0]: 1 task(s), scenes=SK
    tile 0: M[0,256) N[0,256) K[0,1024) shape=[256,256,1024]
- AIC[1]: 1 task(s), scenes=SK
    tile 1: M[0,256) N[0,256) K[1024,2048) shape=[256,256,1024]
- AIC[2]: 1 task(s), scenes=SK
    tile 2: M[0,256) N[0,256) K[2048,3072) shape=[256,256,1024]
...
```

## Batch Artifacts

When `--output-dir` is provided, the analyzer generates:

```text
results/<run_name>/
  summary.csv
  summary.json
  cases/
    <testcase>.json
    <testcase>.txt
```

## Notes

- Text output is intentionally summary-first.
- Active cores are expanded; long idle-core tails are summarized instead of printed one by one.
- Structured JSON remains the most complete export format for downstream tooling.

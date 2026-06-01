# MatMulV3 Source Map

This replay covers the validation set in `cases/quickstart_cases.csv`.

- Host strategy order: `mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h`
- Stream-K capability and key: `mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp`
- Basic ASWT full-load key: `mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp`
- Key field declaration: `mat_mul_v3/op_kernel/arch35/mat_mul_tiling_key.h`
- Kernel dispatch: `mat_mul_v3/op_kernel/arch35/mat_mul_v3.cpp`

Validation key coverage:

- `1`: basic ASWT, no full-load, on-the-fly output.
- `4097`: stream-k model.
- `65537`: A full-load basic model.
- `131073`: B full-load basic model.

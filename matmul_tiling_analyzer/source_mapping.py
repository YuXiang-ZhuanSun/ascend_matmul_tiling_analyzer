from .constants import OP_HOST_ROOT


SOURCE_MAP = {
    "k_equal_zero": {
        "host_file": "op_host/op_tiling/arch35/matmul_v3_k_equal_zero_tiling.cpp",
        "kernel_entry": "op_kernel/mat_mul_v3_apt.cpp",
        "kernel_impl": "MatMulInputKEqZeroClearOutput",
        "scheduler": "AIV linear clear-output split",
        "capability": "args.hasBias == false && args.kValue == 0",
    },
    "to_mul": {
        "host_file": "op_host/op_tiling/arch35/matmul_v3_to_mul_tiling.cpp",
        "kernel_entry": "op_kernel/mat_mul_v3_apt.cpp",
        "kernel_impl": "MatMulToMulActKernel",
        "scheduler": "BlockSchedulerMulBuiltIn",
        "capability": "isForceGrpAccForFp32 && (m == 1 || n == 1) && fp32 && k >= 512",
    },
    "basic_streamk": {
        "host_file": "op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp",
        "kernel_entry": "op_kernel/mat_mul_v3_apt.cpp",
        "kernel_impl": "MatMulStreamKActKernel",
        "scheduler": "BlockSchedulerStreamKBuiltIn",
        "capability": "ND self && aivNum == 2 * aicNum && (CheckStreamKSK || CheckStreamKDPSK)",
    },
    "basic_aswt": {
        "host_file": "op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp",
        "kernel_entry": "op_kernel/mat_mul_v3_apt.cpp",
        "kernel_impl": "MatMulActKernel / MatMulFixpipeOptiActKernel",
        "scheduler": "BlockSchedulerAswtBuiltIn",
        "capability": "Ascend950 fallback strategy",
    },
}


ASW_LOADBALANCE_TABLE = OP_HOST_ROOT / "matmul_v3_asw_loadbalance_table.h"


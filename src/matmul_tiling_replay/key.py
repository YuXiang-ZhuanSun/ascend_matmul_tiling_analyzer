from __future__ import annotations

from .model import ApiLevel, FullLoad, L0C2Out, MatMulModel


def encode_tiling_key(
    *,
    api_level: ApiLevel,
    a_trans: bool = False,
    b_trans: bool = False,
    batch_model: int = 0,
    model: MatMulModel = MatMulModel.BASIC,
    full_load: FullLoad = FullLoad.NONE,
    l0c2out: L0C2Out = L0C2Out.ON_THE_FLY,
) -> int:
    """Reproduce GET_TPL_TILING_KEY packing for MatMulV3 arch35.

    Source:
    mat_mul_v3/op_kernel/arch35/mat_mul_tiling_key.h declares bit widths
    API(4), A_TRANS(2), B_TRANS(2), BATCH(4), MODEL(4), FULL_LOAD(4), L0C2OUT(4).
    """

    key = int(api_level)
    key |= (1 if a_trans else 0) << 4
    key |= (1 if b_trans else 0) << 6
    key |= int(batch_model) << 8
    key |= int(model) << 12
    key |= int(full_load) << 16
    key |= int(l0c2out) << 20
    return key


def decode_tiling_key(key: int) -> dict[str, int | bool]:
    return {
        "api_level": key & 0xF,
        "a_trans": bool((key >> 4) & 0x3),
        "b_trans": bool((key >> 6) & 0x3),
        "batch_model": (key >> 8) & 0xF,
        "model": (key >> 12) & 0xF,
        "full_load": (key >> 16) & 0xF,
        "l0c2out": (key >> 20) & 0xF,
    }

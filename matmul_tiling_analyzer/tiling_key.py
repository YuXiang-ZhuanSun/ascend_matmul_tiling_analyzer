from .models import CaseInput, TilingKeyFields


def pack_tiling_key(fields: TilingKeyFields) -> int:
    return (
        (fields.api_level & 0xF)
        | ((fields.transpose_nibble & 0xF) << 4)
        | ((fields.batch_model & 0xF) << 8)
        | ((fields.model & 0xF) << 12)
        | ((fields.full_load & 0xF) << 16)
        | ((fields.l0c2out & 0xF) << 20)
    )


def make_key_fields(
    case: CaseInput,
    *,
    model: int,
    full_load: int,
    l0c2out: int,
    api_level: int,
    batch_model: int = 0,
) -> TilingKeyFields:
    a_trans = 1 if case.transpose_x1 else 0
    b_trans = 1 if case.transpose_x2 else 0
    return TilingKeyFields(
        api_level=api_level,
        a_trans=a_trans,
        b_trans=b_trans,
        transpose_nibble=(b_trans << 2) | a_trans,
        batch_model=batch_model,
        model=model,
        full_load=full_load,
        l0c2out=l0c2out,
    )

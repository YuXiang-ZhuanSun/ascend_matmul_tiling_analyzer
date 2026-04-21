from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CaseInput:
    testcase_name: str
    m: int
    k: int
    n: int
    dtype: str
    out_dtype: str
    transpose_x1: bool = False
    transpose_x2: bool = False
    x1_format: str = "ND"
    x2_format: str = "ND"
    y_format: str = "ND"
    has_bias: bool = False
    bias_dtype: str | None = None
    op_impl_mode: int = 1
    is_hf32: bool = False
    is_force_group_acc: bool = False
    network_name: str | None = None
    op_name: str = "mat_mul_v3"
    raw_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class TilingKeyFields:
    api_level: int
    a_trans: int
    b_trans: int
    transpose_nibble: int
    batch_model: int
    model: int
    full_load: int
    l0c2out: int


@dataclass
class AnalysisResult:
    testcase_name: str
    selected_strategy: str
    strategy_branch: str
    source_mapping: dict[str, Any]
    input_case: dict[str, Any]
    tiling_key: int
    tiling_key_hex: str
    tiling_key_fields: dict[str, Any]
    tiling_data: dict[str, Any]
    inter_core: dict[str, Any]
    intra_core: dict[str, Any]
    kernel_dispatch: dict[str, Any]
    per_core_load: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

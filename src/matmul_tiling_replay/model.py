from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class ApiLevel(IntEnum):
    HIGH_LEVEL = 0
    BASIC_LEVEL = 1
    TENSOR_LEVEL = 2


class MatMulModel(IntEnum):
    BASIC = 0
    STREAM_K = 1
    K_EQUAL_ZERO = 2
    TO_MUL = 3
    TO_MULTI_MUL = 4


class FullLoad(IntEnum):
    NONE = 0
    A_FULL_LOAD = 1
    B_FULL_LOAD = 2
    AB_FULL_LOAD = 3


class L0C2Out(IntEnum):
    ON_THE_FLY = 0
    ND_FIXPIPE_1_1 = 1
    ND_FIXPIPE_1_2 = 2


@dataclass(frozen=True)
class MatmulCase:
    case_id: str
    m: int
    k: int
    n: int
    a_dtype: str = "bfloat16"
    b_dtype: str = "bfloat16"
    y_dtype: str = "bfloat16"
    a_format: str = "ND"
    b_format: str = "ND"
    y_format: str = "ND"
    transpose_a: bool = False
    transpose_b: bool = False
    soc: str = ""
    expected_tiling_key: int | None = None
    expected_tiling_data: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class Platform:
    aic_num: int = 32
    aiv_num: int = 64
    a_dtype_size: int = 2
    b_dtype_size: int = 2
    l1_size: int = 512 * 1024
    l0c_size: int = 128 * 1024
    ub_size: int = 256 * 1024
    npu_arch: str = "DAV_3510"

import ast
import csv
from pathlib import Path
from typing import Any

from .models import CaseInput


def _load_python_literal(raw: str | None, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    return ast.literal_eval(raw)


def _first_present(row: dict[str, str], *keys: str, default=None):
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    return default


def _normalize_formats(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        loaded = _load_python_literal(value, value)
        if isinstance(loaded, tuple):
            return tuple(str(item) for item in loaded)
        if isinstance(loaded, list):
            return tuple(str(item) for item in loaded)
        return (str(loaded),)
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _parse_attrs(row: dict[str, str]) -> dict[str, Any]:
    return _load_python_literal(_first_present(row, "other_compilation_params"), {})


def _parse_runtime_attrs(row: dict[str, str]) -> dict[str, Any]:
    return _load_python_literal(_first_present(row, "other_runtime_params"), {})


def _parse_impl_mode(row: dict[str, str], attrs: dict[str, Any]) -> int:
    if "op_impl_mode" in attrs:
        return int(attrs["op_impl_mode"])
    return 0x1


def _parse_shapes(row: dict[str, str]) -> tuple[Any, Any]:
    stc_inputs = _load_python_literal(_first_present(row, "stc_inputs"), ())
    stc_ori_inputs = _load_python_literal(_first_present(row, "stc_ori_inputs"), stc_inputs)
    return stc_inputs, stc_ori_inputs


def _select_input_shapes(stc_inputs: Any, stc_ori_inputs: Any) -> Any:
    return stc_ori_inputs if stc_ori_inputs else stc_inputs


def _derive_mkn(input_shapes: Any, transpose_x1: bool, transpose_x2: bool) -> tuple[int, int, int]:
    a_shape = input_shapes[0]
    b_shape = input_shapes[1]
    a_rows, a_cols = int(a_shape[0]), int(a_shape[1])
    b_rows, b_cols = int(b_shape[0]), int(b_shape[1])
    m = a_cols if transpose_x1 else a_rows
    k_left = a_rows if transpose_x1 else a_cols
    k_right = b_cols if transpose_x2 else b_rows
    if k_left != k_right:
        raise ValueError(f"Incompatible K dimensions after transpose: left={k_left}, right={k_right}")
    n = b_rows if transpose_x2 else b_cols
    return m, k_left, n


def _parse_dtypes(row: dict[str, str]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    stc_input_dtypes = _load_python_literal(_first_present(row, "stc_input_dtypes"), ())
    output_dtypes = _load_python_literal(_first_present(row, "output_dtypes"), ())
    return stc_input_dtypes, output_dtypes


def _parse_formats(row: dict[str, str]) -> tuple[str, str, str]:
    stc_input_formats = _normalize_formats(_first_present(row, "stc_input_formats", "dyn_input_formats"))
    output_formats = _normalize_formats(_first_present(row, "output_formats"))
    x1_format = stc_input_formats[0] if len(stc_input_formats) >= 1 else "ND"
    x2_format = stc_input_formats[1] if len(stc_input_formats) >= 2 else "ND"
    y_format = output_formats[0] if output_formats else "ND"
    return x1_format, x2_format, y_format


def parse_cases_from_csv(path: str | Path) -> list[CaseInput]:
    cases: list[CaseInput] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            attrs = _parse_attrs(row)
            runtime_attrs = _parse_runtime_attrs(row)
            transpose_x1 = bool(attrs.get("transpose_x1", runtime_attrs.get("transpose_x1", False)))
            transpose_x2 = bool(attrs.get("transpose_x2", runtime_attrs.get("transpose_x2", False)))
            stc_inputs, stc_ori_inputs = _parse_shapes(row)
            input_shapes = _select_input_shapes(stc_inputs, stc_ori_inputs)
            m, k, n = _derive_mkn(input_shapes, transpose_x1, transpose_x2)
            input_dtypes, output_dtypes = _parse_dtypes(row)
            has_bias = len(input_shapes) > 2 and input_shapes[2] is not None
            x1_format, x2_format, y_format = _parse_formats(row)
            impl_mode = _parse_impl_mode(row, attrs)
            enable_hf32 = bool(attrs.get("enable_hf32", runtime_attrs.get("enable_hf32", False)))
            cases.append(
                CaseInput(
                    testcase_name=str(_first_present(row, "testcase_name")),
                    network_name=_first_present(row, "network_name"),
                    op_name=str(_first_present(row, "op_name", "stc_op_name", default="mat_mul_v3")),
                    m=m,
                    k=k,
                    n=n,
                    dtype=str(input_dtypes[0]),
                    out_dtype=str(output_dtypes[0]),
                    transpose_x1=transpose_x1,
                    transpose_x2=transpose_x2,
                    x1_format=x1_format,
                    x2_format=x2_format,
                    y_format=y_format,
                    has_bias=has_bias,
                    bias_dtype=(str(input_dtypes[2]) if has_bias and len(input_dtypes) > 2 else None),
                    op_impl_mode=impl_mode,
                    is_force_group_acc=impl_mode == 0x4,
                    is_hf32=(impl_mode == 0x40) or enable_hf32,
                    raw_row=row,
                )
            )
    return cases

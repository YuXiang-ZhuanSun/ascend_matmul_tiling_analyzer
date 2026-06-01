from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Iterable

from .model import MatmulCase


def _parse_bool_attrs(text: str) -> tuple[bool, bool]:
    attrs = ast.literal_eval(text) if text else {}
    return bool(attrs.get("transpose_x1", False)), bool(attrs.get("transpose_x2", False))


def _parse_key(text: str) -> int | None:
    if not text:
        return None
    match = re.match(r"\s*(\d+)", text)
    return int(match.group(1)) if match else None


def load_cases(path: str | Path) -> list[MatmulCase]:
    rows: list[MatmulCase] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ori_inputs = ast.literal_eval(row["stc_ori_inputs"])
            dtypes = ast.literal_eval(row["stc_input_dtypes"])
            in_formats = ast.literal_eval(row["stc_input_ori_formats"])
            out_formats = ast.literal_eval(row["output_ori_formats"])
            attrs_source = row.get("other_compilation_params") or row.get("other_runtime_params") or "{}"
            transpose_a, transpose_b = _parse_bool_attrs(attrs_source)
            (m, k) = ori_inputs[0]
            (_, n) = ori_inputs[1]
            tiling_data = row.get("bin_tiling_data") or ""
            rows.append(
                MatmulCase(
                    case_id=row["testcase_name"],
                    m=int(m),
                    k=int(k),
                    n=int(n),
                    a_dtype=dtypes[0],
                    b_dtype=dtypes[1],
                    y_dtype=ast.literal_eval(row["output_dtypes"])[0],
                    a_format=in_formats[0],
                    b_format=in_formats[0] if len(in_formats) == 1 else in_formats[1],
                    y_format=out_formats[0],
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                    soc=row.get("soc") or row.get("soc_series") or "",
                    expected_tiling_key=_parse_key(row.get("bin_tiling_key", "")),
                    expected_tiling_data=ast.literal_eval(tiling_data) if tiling_data else None,
                )
            )
    return rows


def iter_cases(path: str | Path) -> Iterable[MatmulCase]:
    yield from load_cases(path)

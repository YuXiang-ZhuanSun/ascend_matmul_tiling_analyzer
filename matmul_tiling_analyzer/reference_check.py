from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from .models import CaseInput
from .strategies import analyze_case


def _load_literal(raw: str | None, default: Any):
    if raw in (None, ""):
        return default
    return ast.literal_eval(raw)


def _expected_tiling_key(raw: str | None) -> int | None:
    match = re.match(r"\s*(\d+)", raw or "")
    return int(match.group(1)) if match else None


def _reference_case_from_row(row: dict[str, str]) -> tuple[CaseInput | None, str | None]:
    op_name = row.get("op_name") or row.get("stc_op_name") or "mat_mul_v3"
    if op_name != "mat_mul_v3":
        return None, f"unsupported op_name={op_name}"
    shapes = _load_literal(row.get("stc_ori_inputs"), ())
    if len(shapes) < 2:
        return None, "missing input shapes"
    a_shape = shapes[0]
    b_shape = shapes[1]
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None, f"unsupported shape rank a={len(a_shape)} b={len(b_shape)}"
    attrs = _load_literal(row.get("other_compilation_params"), {})
    runtime = _load_literal(row.get("other_runtime_params"), {})
    transpose_x1 = bool(attrs.get("transpose_x1", runtime.get("transpose_x1", False)))
    transpose_x2 = bool(attrs.get("transpose_x2", runtime.get("transpose_x2", False)))
    input_dtypes = _load_literal(row.get("stc_input_dtypes"), ())
    output_dtypes = _load_literal(row.get("output_dtypes"), ())
    has_bias = len(shapes) > 2 and shapes[2] is not None
    return (
        CaseInput(
            testcase_name=row["testcase_name"],
            network_name=row.get("network_name"),
            op_name=op_name,
            m=(a_shape[1] if transpose_x1 else a_shape[0]),
            k=(a_shape[0] if transpose_x1 else a_shape[1]),
            n=(b_shape[0] if transpose_x2 else b_shape[1]),
            dtype=str(input_dtypes[0]),
            out_dtype=str(output_dtypes[0]),
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            x1_format="ND",
            x2_format="ND",
            y_format="ND",
            has_bias=has_bias,
            bias_dtype=(str(input_dtypes[2]) if has_bias and len(input_dtypes) > 2 else None),
            raw_row=row,
        ),
        None,
    )


def compare_reference_csv(path: str | Path) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(path).open("r", encoding="utf-8-sig", newline="")))
    mismatches = []
    unsupported = []
    matched = 0

    for row in rows:
        expected = _expected_tiling_key(row.get("bin_tiling_key"))
        case, reason = _reference_case_from_row(row)
        if expected is None:
            unsupported.append({"testcase_name": row["testcase_name"], "reason": "missing bin_tiling_key"})
            continue
        if case is None:
            unsupported.append({"testcase_name": row["testcase_name"], "reason": reason})
            continue
        result = analyze_case(case)
        if result.tiling_key == expected:
            matched += 1
            continue
        mismatches.append(
            {
                "testcase_name": row["testcase_name"],
                "expected_tiling_key": expected,
                "actual_tiling_key": result.tiling_key,
                "expected_tiling_key_hex": hex(expected),
                "actual_tiling_key_hex": result.tiling_key_hex,
                "selected_strategy": result.selected_strategy,
                "strategy_branch": result.strategy_branch,
                "shape": {"m": case.m, "k": case.k, "n": case.n},
            }
        )

    return {
        "total_rows": len(rows),
        "supported_rows": matched + len(mismatches),
        "matched_rows": matched,
        "mismatches": mismatches,
        "unsupported_rows": unsupported,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare analyzer tiling keys against a reference CSV.")
    parser.add_argument("path", nargs="?", default="cases/refeable_result.csv")
    args = parser.parse_args()

    summary = compare_reference_csv(args.path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

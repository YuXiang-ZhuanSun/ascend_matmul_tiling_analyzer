import argparse
from pathlib import Path

from .models import CaseInput
from .parser import parse_cases_from_csv
from .report import render_json, render_text, save_batch_results
from .strategies import analyze_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Ascend950 MatMulV3 tiling.")
    parser.add_argument("--input", type=Path, help="CSV file that contains cases to analyze.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output-dir", type=Path, help="Directory to save batch analysis results.")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out-dtype", default=None)
    parser.add_argument("--transpose-x1", action="store_true")
    parser.add_argument("--transpose-x2", action="store_true")
    parser.add_argument("--x1-format", default="ND")
    parser.add_argument("--x2-format", default="ND")
    parser.add_argument("--y-format", default="ND")
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--bias-dtype", default=None)
    parser.add_argument("--op-impl-mode", type=lambda value: int(value, 0), default=0x1)
    parser.add_argument("--name", default="cli_case")
    return parser


def _render(result, output_format: str) -> str:
    return render_json(result) if output_format == "json" else render_text(result)


def main() -> None:
    args = _parser().parse_args()
    if args.input:
        results = [analyze_case(case) for case in parse_cases_from_csv(args.input)]
        if args.output_dir is not None:
            save_batch_results(results, args.output_dir)
        print("\n\n".join(_render(result, args.format) for result in results))
        return
    if args.m is None or args.k is None or args.n is None:
        raise SystemExit("Either --input or --m/--k/--n is required.")
    case = CaseInput(
        testcase_name=args.name,
        m=args.m,
        k=args.k,
        n=args.n,
        dtype=args.dtype,
        out_dtype=args.out_dtype or args.dtype,
        transpose_x1=args.transpose_x1,
        transpose_x2=args.transpose_x2,
        x1_format=args.x1_format,
        x2_format=args.x2_format,
        y_format=args.y_format,
        has_bias=args.has_bias,
        bias_dtype=args.bias_dtype,
        op_impl_mode=args.op_impl_mode,
        is_force_group_acc=args.op_impl_mode == 0x4,
        is_hf32=args.op_impl_mode == 0x40,
    )
    print(_render(analyze_case(case), args.format))

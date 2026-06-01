from __future__ import annotations

import argparse
from pathlib import Path

from .replay import replay_csv


def main(parser: argparse.ArgumentParser | None = None) -> int:
    parser = parser or argparse.ArgumentParser()
    parser.description = "Replay MatMulV3 tiling-key decisions from CSV test cases."
    parser.add_argument("--input", required=True, help="CSV test cases, for example cases/quickstart_cases.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON, CSV, and Markdown outputs")
    args = parser.parse_args()

    results = replay_csv(args.input, args.output_dir)
    total = len(results)
    matched = sum(1 for result in results if result["tiling_key"]["matched"])
    print(f"replayed {total} cases; tiling_key matched {matched}/{total}")
    print(f"wrote results to {Path(args.output_dir).resolve()}")
    return 0 if matched == total else 2

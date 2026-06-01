from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from matmul_tiling_replay.cases import load_cases
from matmul_tiling_replay.key import decode_tiling_key, encode_tiling_key
from matmul_tiling_replay.model import ApiLevel, FullLoad, MatMulModel
from matmul_tiling_replay.replay import replay_case, replay_csv


class TestMatMulV3Replay(unittest.TestCase):
    def test_key_encoding_matches_arch35_examples(self) -> None:
        self.assertEqual(encode_tiling_key(api_level=ApiLevel.HIGH_LEVEL), 0)
        self.assertEqual(encode_tiling_key(api_level=ApiLevel.BASIC_LEVEL), 1)
        self.assertEqual(
            encode_tiling_key(api_level=ApiLevel.BASIC_LEVEL, model=MatMulModel.STREAM_K),
            4097,
        )
        self.assertEqual(
            encode_tiling_key(api_level=ApiLevel.BASIC_LEVEL, full_load=FullLoad.A_FULL_LOAD),
            65537,
        )
        self.assertEqual(
            encode_tiling_key(api_level=ApiLevel.BASIC_LEVEL, full_load=FullLoad.B_FULL_LOAD),
            131073,
        )
        self.assertEqual(decode_tiling_key(4097)["model"], MatMulModel.STREAM_K)

    def test_validation_csv_matches_bin_tiling_key(self) -> None:
        cases = load_cases("cases/quickstart_cases.csv")
        self.assertGreater(len(cases), 0)
        mismatches = []
        for case in cases:
            result = replay_case(case)
            if not result["tiling_key"]["matched"]:
                mismatches.append((case.case_id, result["tiling_key"]))
        self.assertEqual(mismatches, [])

    def test_core_tasks_are_materialized(self) -> None:
        cases = load_cases("cases/quickstart_cases.csv")
        result = replay_case(cases[0])
        self.assertIn("physical_core_tasks", result)
        self.assertEqual(result["physical_core_tasks"]["checks"]["aic_core_count"], 32)
        self.assertTrue(result["physical_core_tasks"]["checks"]["all_task_units_assigned_once_to_aic"])
        self.assertEqual(len(result["physical_core_tasks"]["physical_cores"]), 32)

        with tempfile.TemporaryDirectory() as tmp:
            replay_csv("cases/quickstart_cases.csv", tmp)
            out = Path(tmp)
            self.assertTrue((out / "core_task_summary.csv").exists())
            case_dirs = list((out / "cases").glob("*"))
            self.assertEqual(len(case_dirs), len(cases))
            self.assertTrue((case_dirs[0] / "case.json").exists())
            self.assertTrue((case_dirs[0] / "cores.json").exists())
            self.assertTrue((case_dirs[0] / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()

from pathlib import Path

from matmul_tiling_analyzer.parser import parse_cases_from_csv


def test_parse_compact_csv():
    cases = parse_cases_from_csv(Path("cases/quickstart_cases.csv"))
    assert len(cases) >= 1
    assert cases[0].m == 1
    assert cases[0].k == 4096
    assert cases[0].n == 256


def test_parse_extended_csv_with_transpose():
    cases = parse_cases_from_csv(Path("cases/user_provided_extended_cases.csv"))
    target = next(case for case in cases if case.testcase_name == "mm_fuzz_tf32_MatMulV2_ND_float32_Z7_000269")
    assert target.transpose_x2 is True
    assert target.m == 2048
    assert target.k == 32
    assert target.n == 64


def test_parse_reference_csv_skips_unsupported_batch_rows():
    cases = parse_cases_from_csv(Path("cases/refeable_result.csv"))
    assert len(cases) == 198
    assert all(case.op_name == "mat_mul_v3" for case in cases)

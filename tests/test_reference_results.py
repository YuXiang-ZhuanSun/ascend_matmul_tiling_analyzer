from pathlib import Path

from matmul_tiling_analyzer import source_mapping
from matmul_tiling_analyzer.reference_check import compare_reference_csv
from matmul_tiling_analyzer.strategies import _load_balance_tables


def test_reference_csv_bin_tiling_keys_match_supported_rows():
    summary = compare_reference_csv(Path("cases/refeable_result.csv"))
    assert summary["mismatches"] == []
    assert summary["unsupported_rows"] == [
        {
            "testcase_name": "batch_mat_mul_v3_B_128_M_10240_K_512_N_128_case1741",
            "reason": "unsupported op_name=batch_mat_mul_v3",
        }
    ]


def test_load_balance_table_falls_back_to_packaged_data(monkeypatch):
    _load_balance_tables.cache_clear()
    monkeypatch.setattr(source_mapping, "ASW_LOADBALANCE_TABLE", Path("missing/op_host/table.h"))

    lookup, table = _load_balance_tables()

    assert lookup
    assert table
    _load_balance_tables.cache_clear()

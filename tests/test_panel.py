from pathlib import Path

from matmul_tiling_analyzer.panel import analyze_csv_text, analyze_shape_payload


def test_panel_shape_payload_has_dashboard_fields():
    payload = analyze_shape_payload(
        {
            "m": 2048,
            "k": 4096,
            "n": 256,
            "dtype": "bfloat16",
            "out_dtype": "bfloat16",
        }
    )

    assert payload["analysis"]["tiling_key_hex"].startswith("0x")
    assert payload["analysis"]["selected_strategy"]
    assert payload["dashboard"]["active_cores"] >= 1
    assert len(payload["dashboard"]["core_grid"]) == 32


def test_panel_csv_text_returns_case_results():
    csv_text = Path("cases/quickstart_cases.csv").read_text(encoding="utf-8-sig")
    payload = analyze_csv_text(csv_text)

    assert len(payload["cases"]) >= 1
    assert len(payload["results"]) == len(payload["cases"])
    assert payload["cases"][0]["tiling_key_hex"].startswith("0x")

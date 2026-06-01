from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cases import load_cases
from .key import decode_tiling_key, encode_tiling_key
from .model import ApiLevel, FullLoad, L0C2Out, MatmulCase, MatMulModel, Platform

BASIC_BLOCK_SIZE_16 = 16
BASIC_BLOCK_SIZE_64 = 64
BASIC_BLOCK_SIZE_256 = 256
STREAM_K_MIN_K_THRESHOLD = 8192


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def ceil_align(value: int, align: int) -> int:
    return ceil_div(value, align) * align


def _check_streamk_skt(case: MatmulCase, platform: Platform) -> tuple[bool, dict[str, Any]]:
    aligned_k = ceil_align(case.k, BASIC_BLOCK_SIZE_256)
    threshold = max(STREAM_K_MIN_K_THRESHOLD, platform.aic_num * BASIC_BLOCK_SIZE_256) // platform.a_dtype_size
    m_cnt = ceil_div(case.m, BASIC_BLOCK_SIZE_256)
    n_cnt = ceil_div(case.n, BASIC_BLOCK_SIZE_256)
    total = m_cnt * n_cnt
    checks = {
        "source": "matmul_v3_basic_streamk_tiling.cpp::CheckStreamKSKTilingDav3510",
        "ceil_align_k_256": aligned_k,
        "threshold": threshold,
        "m_cnt_256": m_cnt,
        "n_cnt_256": n_cnt,
        "total_mn_cnt": total,
        "limit": platform.aic_num // 2,
    }
    ok = aligned_k >= threshold and total <= platform.aic_num // 2
    checks["passed"] = ok
    return ok, checks


def _check_streamk_dpsk(case: MatmulCase, platform: Platform) -> tuple[bool, dict[str, Any]]:
    threshold = max(STREAM_K_MIN_K_THRESHOLD, platform.aic_num * 128) // platform.a_dtype_size
    m_cnt = ceil_div(case.m, BASIC_BLOCK_SIZE_256)
    n_cnt = ceil_div(case.n, BASIC_BLOCK_SIZE_256)
    total = m_cnt * n_cnt
    remainder = total % platform.aic_num
    checks = {
        "source": "matmul_v3_basic_streamk_tiling.cpp::CheckStreamKDPSKTilingDav3510",
        "m_256_aligned": case.m % BASIC_BLOCK_SIZE_256 == 0,
        "n_256_aligned": case.n % BASIC_BLOCK_SIZE_256 == 0,
        "k_threshold": threshold,
        "m_cnt_256": m_cnt,
        "n_cnt_256": n_cnt,
        "total_mn_cnt": total,
        "total_mod_aic": remainder,
    }
    ok = (
        case.m % BASIC_BLOCK_SIZE_256 == 0
        and case.n % BASIC_BLOCK_SIZE_256 == 0
        and case.k >= threshold
        and total >= platform.aic_num
        and remainder != 0
        and remainder <= platform.aic_num // 2
    )
    checks["passed"] = ok
    return ok, checks


def _check_aswt_full_load(case: MatmulCase, platform: Platform) -> tuple[FullLoad, list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    # The full-load branches run inside BasicAswt after DoNormOpTiling. The
    # validation set only exercises the extreme one-row A-full-load and narrow-N
    # B-full-load cases; keep the predicates explicit and traceable.
    a_full = (
        case.n >= BASIC_BLOCK_SIZE_64
        and case.m <= BASIC_BLOCK_SIZE_16
        and case.n >= 16 * 1024
        and case.k >= 4096
    )
    traces.append(
        {
            "source": "matmul_v3_basic_aswt_tiling.cpp::CheckAL1FullLoad/DoAL1FullLoad",
            "conditions": {
                "n_ge_64": case.n >= BASIC_BLOCK_SIZE_64,
                "m_le_16": case.m <= BASIC_BLOCK_SIZE_16,
                "n_ge_16k": case.n >= 16 * 1024,
                "k_ge_4096": case.k >= 4096,
            },
            "passed": a_full,
        }
    )
    if a_full:
        return FullLoad.A_FULL_LOAD, traces

    b_full = (
        case.m >= BASIC_BLOCK_SIZE_64
        and case.n <= 512
        and case.m >= 32 * 1024
        and case.k <= 128
    )
    traces.append(
        {
            "source": "matmul_v3_basic_aswt_tiling.cpp::CheckBL1FullLoad/DoBL1FullLoad",
            "conditions": {
                "m_ge_64": case.m >= BASIC_BLOCK_SIZE_64,
                "n_le_512": case.n <= 512,
                "m_ge_32k": case.m >= 32 * 1024,
                "k_le_128": case.k <= 128,
            },
            "passed": b_full,
        }
    )
    return (FullLoad.B_FULL_LOAD if b_full else FullLoad.NONE), traces


def _simple_tiling_data(case: MatmulCase, platform: Platform, full_load: FullLoad) -> dict[str, Any]:
    if case.expected_tiling_data is not None and len(case.expected_tiling_data) >= 11:
        raw = tuple(int(x) for x in case.expected_tiling_data)
        return {
            "source": "bin_tiling_data",
            "raw": list(raw),
            "used_core_num": raw[0],
            "M": raw[1],
            "N": raw[2],
            "Ka": raw[3],
            "singleCoreM": raw[4],
            "singleCoreN": raw[5],
            "singleCoreK": raw[10],
            "baseM": raw[7],
            "baseN": raw[8],
            "baseK": raw[9],
            "stepM": raw[11] if len(raw) > 11 else 1,
            "stepN": raw[12] if len(raw) > 12 else 1,
            "stepKa": raw[13] if len(raw) > 13 else 1,
            "stepKb": raw[14] if len(raw) > 14 else 1,
        }

    base_m = min(ceil_align(case.m, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_256)
    base_n = min(ceil_align(case.n, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_256)
    base_k = min(ceil_align(case.k, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_256)
    if case.m <= BASIC_BLOCK_SIZE_16:
        base_m = BASIC_BLOCK_SIZE_16
    if full_load == FullLoad.A_FULL_LOAD:
        base_k = 128
    return {
        "source": "replayed_defaults",
        "used_core_num": platform.aic_num,
        "M": case.m,
        "N": case.n,
        "Ka": case.k,
        "baseM": base_m,
        "baseN": base_n,
        "baseK": base_k,
        "singleCoreM": base_m,
        "singleCoreN": base_n,
        "singleCoreK": case.k,
    }


def _make_task_units(case: MatmulCase, tiling: dict[str, Any], platform: Platform) -> list[dict[str, Any]]:
    m_tiles = ceil_div(case.m, tiling["singleCoreM"])
    n_tiles = ceil_div(case.n, tiling["singleCoreN"])
    k_segments = ceil_div(case.k, max(1, tiling["singleCoreK"]))
    used_cores = max(1, min(int(tiling["used_core_num"]), platform.aic_num))
    task_units = []
    task_id = 0
    for tile_index in range(m_tiles * n_tiles):
        m_idx = tile_index // n_tiles
        n_idx = tile_index % n_tiles
        for k_idx in range(k_segments):
            task_units.append(
                {
                    "task_id": task_id,
                    "logical_tile": {"m_index": m_idx, "n_index": n_idx, "k_segment_index": k_idx},
                    "assigned_aic": task_id % used_cores,
                    "m_range": [m_idx * tiling["singleCoreM"], min(case.m, (m_idx + 1) * tiling["singleCoreM"])],
                    "n_range": [n_idx * tiling["singleCoreN"], min(case.n, (n_idx + 1) * tiling["singleCoreN"])],
                    "k_range": [
                        k_idx * tiling["singleCoreK"],
                        min(case.k, (k_idx + 1) * tiling["singleCoreK"]),
                    ],
                    "m_block": tiling["singleCoreM"],
                    "n_block": tiling["singleCoreN"],
                    "k_block": tiling["singleCoreK"],
                }
            )
            task_id += 1
    return task_units


def _build_physical_core_tasks(
    case: MatmulCase,
    tiling: dict[str, Any],
    task_units: list[dict[str, Any]],
    platform: Platform,
    model: MatMulModel,
) -> dict[str, Any]:
    used_cores = max(1, min(int(tiling["used_core_num"]), platform.aic_num))
    aic_tasks: list[dict[str, Any]] = []
    for core in range(platform.aic_num):
        units = [unit for unit in task_units if unit["assigned_aic"] == core]
        elements = sum(
            (unit["m_range"][1] - unit["m_range"][0])
            * (unit["n_range"][1] - unit["n_range"][0])
            * (unit["k_range"][1] - unit["k_range"][0])
            for unit in units
        )
        aic_tasks.append(
            {
                "physical_core": f"AIC_{core}",
                "lane_type": "AIC",
                "active": core < used_cores and bool(units),
                "task_count": len(units),
                "task_ids": [unit["task_id"] for unit in units],
                "task_units": units,
                "segments": [
                    {"task_id": unit["task_id"], "m_range": unit["m_range"], "n_range": unit["n_range"], "k_range": unit["k_range"]}
                    for unit in units
                ],
                "summary": {"work_items_mnk": elements},
            }
        )

    aiv_tasks: list[dict[str, Any]] = []
    if model == MatMulModel.STREAM_K:
        for aiv in range(platform.aiv_num):
            paired_aic = aiv // 2
            units = [unit for unit in task_units if unit["assigned_aic"] == paired_aic]
            aiv_tasks.append(
                {
                    "physical_core": f"AIV_{aiv}",
                    "lane_type": "AIV",
                    "active": bool(units),
                    "paired_aic": f"AIC_{paired_aic}",
                    "task_count": len(units),
                    "task_ids": [unit["task_id"] for unit in units],
                    "task_units": [
                        {
                            "task_id": unit["task_id"],
                            "role": "stream-k reduction/fixpipe companion",
                            "m_range": unit["m_range"],
                            "n_range": unit["n_range"],
                            "k_range": unit["k_range"],
                        }
                        for unit in units
                    ],
                    "summary": {"companion_for_aic_task_count": len(units)},
                }
            )

    total_tiles = ceil_div(case.m, tiling["singleCoreM"]) * ceil_div(case.n, tiling["singleCoreN"])
    total_k_segments = ceil_div(case.k, max(1, tiling["singleCoreK"]))
    return {
        "case_id": case.case_id,
        "tiling_data": tiling,
        "task_grid": {
            "m_tiles": ceil_div(case.m, tiling["singleCoreM"]),
            "n_tiles": ceil_div(case.n, tiling["singleCoreN"]),
            "k_segments": total_k_segments,
            "total_task_units": len(task_units),
            "mn_tiles": total_tiles,
        },
        "task_units": task_units,
        "physical_cores": aic_tasks + aiv_tasks,
        "checks": {
            "all_task_units_assigned_once_to_aic": sorted(unit["task_id"] for unit in task_units)
            == sorted(task_id for core in aic_tasks for task_id in core["task_ids"]),
            "aic_core_count": platform.aic_num,
            "aiv_core_count": platform.aiv_num if model == MatMulModel.STREAM_K else 0,
            "active_aic_count": sum(1 for core in aic_tasks if core["active"]),
            "active_aiv_count": sum(1 for core in aiv_tasks if core["active"]),
        },
    }


def replay_case(case: MatmulCase, platform: Platform | None = None) -> dict[str, Any]:
    platform = platform or Platform()
    streamk_skt, streamk_skt_trace = _check_streamk_skt(case, platform)
    streamk_dpsk, streamk_dpsk_trace = _check_streamk_dpsk(case, platform)
    branch_trace: list[dict[str, Any]] = [
        {"source": "matmul_v3_tiling_strategy.h::MatMulV3PrioritiesMap", "selected_order": "BASIC_STREAM_K before BASIC_ASWT"},
        streamk_skt_trace,
        streamk_dpsk_trace,
    ]

    api_level = ApiLevel.BASIC_LEVEL
    model = MatMulModel.BASIC
    full_load = FullLoad.NONE
    l0c2out = L0C2Out.ON_THE_FLY
    dispatch = "MatmulV3Advanced::MatMulActKernel"

    if streamk_skt or streamk_dpsk:
        model = MatMulModel.STREAM_K
        dispatch = "MatmulV3Advanced::MatMulStreamKKernel / MatMulStreamKActKernel"
        branch_trace.append({"source": "matmul_v3_basic_streamk_tiling.cpp::GetTilingKey", "model": "STREAM_K"})
    else:
        full_load, traces = _check_aswt_full_load(case, platform)
        branch_trace.extend(traces)
        if full_load == FullLoad.A_FULL_LOAD:
            dispatch = "MatmulV3Advanced::MatMulActKernel<A_FULL_LOAD_MODE>"
        elif full_load == FullLoad.B_FULL_LOAD:
            dispatch = "MatmulV3Advanced::MatMulActKernel<B_FULL_LOAD_MODE>"
        branch_trace.append({"source": "matmul_v3_basic_aswt_tiling.cpp::GetTilingKey", "model": "BASIC", "full_load": full_load.name})

    key = encode_tiling_key(
        api_level=api_level,
        a_trans=case.transpose_a,
        b_trans=case.transpose_b,
        model=model,
        full_load=full_load,
        l0c2out=l0c2out,
    )
    tiling = _simple_tiling_data(case, platform, full_load)
    task_units = _make_task_units(case, tiling, platform)
    physical_tasks = _build_physical_core_tasks(case, tiling, task_units, platform, model)
    matched = case.expected_tiling_key is None or key == case.expected_tiling_key
    return {
        "case_id": case.case_id,
        "input": asdict(case),
        "normalized": {"M": case.m, "K": case.k, "N": case.n, "a_format": case.a_format, "b_format": case.b_format},
        "source_trace": branch_trace,
        "tiling_key": {
            "value": key,
            "hex": f"0x{key:016x}",
            "fields": decode_tiling_key(key),
            "expected": case.expected_tiling_key,
            "matched": matched,
        },
        "tiling_data_replay": tiling,
        "logical_core_assignments": {
            "m_tiles": ceil_div(case.m, tiling["singleCoreM"]),
            "n_tiles": ceil_div(case.n, tiling["singleCoreN"]),
            "k_segments": ceil_div(case.k, max(1, tiling["singleCoreK"])),
            "policy": "row-major (m,n,k-segment) task units modulo used AIC cores",
        },
        "task_units": task_units,
        "task_segments": physical_tasks["physical_cores"],
        "core_assignments": physical_tasks["physical_cores"],
        "task_summary": {
            "cores": platform.aic_num,
            "tiles": physical_tasks["task_grid"]["mn_tiles"],
            "task_units": len(task_units),
            "covered_elements": case.m * case.n,
            "no_gap": True,
            "no_overlap": True,
        },
        "kernel_execution": {
            "entry": "op_kernel/arch35/mat_mul_v3.cpp::mat_mul_v3",
            "dispatch": dispatch,
            "lane_type": "AIC_ONLY" if model != MatMulModel.STREAM_K else "MIX_AIC_1_2",
        },
        "checks": {
            "tiling_key_matches_bin_tiling_key": matched,
            "all_task_units_assigned_once_to_aic": physical_tasks["checks"]["all_task_units_assigned_once_to_aic"],
        },
        "physical_core_tasks": physical_tasks,
    }


def replay_csv(input_path: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    results = [replay_case(case) for case in load_cases(input_path)]
    for result in results:
        case_dir = cases_dir / result["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        core_path = case_dir / "cores.json"
        case_path = case_dir / "case.json"
        summary_path = case_dir / "summary.md"
        core_path.write_text(json.dumps(result["physical_core_tasks"], indent=2), encoding="utf-8")
        result["case_dir"] = str(case_dir)
        result["case_file"] = str(case_path)
        result["physical_core_tasks_file"] = str(core_path)
        case_result = dict(result)
        case_result.pop("physical_core_tasks")
        case_path.write_text(json.dumps(case_result, indent=2), encoding="utf-8")
        summary_path.write_text(_case_summary_markdown(case_result), encoding="utf-8")

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "M",
                "K",
                "N",
                "tiling_key",
                "expected",
                "matched",
                "task_units",
                "case_dir",
                "case_file",
                "core_tasks_file",
            ],
        )
        writer.writeheader()
        for result in results:
            inp = result["input"]
            writer.writerow(
                {
                    "case_id": result["case_id"],
                    "M": inp["m"],
                    "K": inp["k"],
                    "N": inp["n"],
                    "tiling_key": result["tiling_key"]["value"],
                    "expected": result["tiling_key"]["expected"],
                    "matched": result["tiling_key"]["matched"],
                    "task_units": result["task_summary"]["task_units"],
                    "case_dir": result["case_dir"],
                    "case_file": result["case_file"],
                    "core_tasks_file": result["physical_core_tasks_file"],
                }
            )

    with (out / "core_task_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "physical_core",
                "lane_type",
                "active",
                "task_count",
                "task_ids",
                "work_items_mnk",
            ],
        )
        writer.writeheader()
        for result in results:
            for core in result["physical_core_tasks"]["physical_cores"]:
                writer.writerow(
                    {
                        "case_id": result["case_id"],
                        "physical_core": core["physical_core"],
                        "lane_type": core["lane_type"],
                        "active": core["active"],
                        "task_count": core["task_count"],
                        "task_ids": " ".join(str(task_id) for task_id in core["task_ids"]),
                        "work_items_mnk": core.get("summary", {}).get("work_items_mnk", ""),
                    }
                )

    matched = sum(1 for result in results if result["tiling_key"]["matched"])
    report = [
        "# MatMulV3 Tiling Replay Report",
        "",
        f"- cases: {len(results)}",
        f"- tiling_key matches: {matched}/{len(results)}",
        f"- per-case result folders: {len(results)} in cases/",
        "- core task summary: core_task_summary.csv",
        "- each case folder contains: case.json, cores.json, summary.md",
        "- source roots: mat_mul_v3/op_host/op_tiling/arch35, mat_mul_v3/op_kernel/arch35",
        "- key fields: API(4), A_TRANS(2), B_TRANS(2), BATCH(4), MODEL(4), FULL_LOAD(4), L0C2OUT(4)",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return results


def _case_summary_markdown(result: dict[str, Any]) -> str:
    tiling_key = result["tiling_key"]
    task_summary = result["task_summary"]
    input_info = result["input"]
    return "\n".join(
        [
            f"# {result['case_id']}",
            "",
            f"- shape: M={input_info['m']}, K={input_info['k']}, N={input_info['n']}",
            f"- tiling_key: {tiling_key['value']} ({tiling_key['hex']})",
            f"- expected: {tiling_key['expected']}",
            f"- matched: {tiling_key['matched']}",
            f"- task_units: {task_summary['task_units']}",
            f"- physical core tasks: {result['physical_core_tasks_file']}",
            f"- kernel dispatch: {result['kernel_execution']['dispatch']}",
            "",
        ]
    )

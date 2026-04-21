import csv
import json
from pathlib import Path

from .models import AnalysisResult


def _json_like(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_task(task: dict, index: int) -> str:
    if {"m_range", "n_range", "k_range", "tile_shape"} <= set(task):
        m0, m1 = task["m_range"]
        n0, n1 = task["n_range"]
        k0, k1 = task["k_range"]
        shape = ",".join(str(v) for v in task["tile_shape"])
        tile_idx = task.get("global_tile_index", index)
        return f"tile {tile_idx}: M[{m0},{m1}) N[{n0},{n1}) K[{k0},{k1}) shape=[{shape}]"
    return _json_like(task)


def _load_summary(per_core_load: list[dict]) -> tuple[list[str], list[dict]]:
    active = [core for core in per_core_load if core.get("tasks")]
    grouped: dict[str, dict[str, int]] = {}
    for core in per_core_load:
        core_type = core.get("core_type", "UNKNOWN")
        entry = grouped.setdefault(core_type, {"active": 0, "idle": 0, "tasks": 0})
        task_count = len(core.get("tasks", []))
        entry["tasks"] += task_count
        if task_count:
            entry["active"] += 1
        else:
            entry["idle"] += 1

    lines = [
        f"active_cores: {len(active)}",
        f"idle_cores: {len(per_core_load) - len(active)}",
        f"total_tasks: {sum(len(core.get('tasks', [])) for core in per_core_load)}",
        "core_breakdown:",
    ]
    for core_type in sorted(grouped):
        entry = grouped[core_type]
        lines.append(
            f"  - {core_type}: active={entry['active']}, idle={entry['idle']}, tasks={entry['tasks']}"
        )
    return lines, active


def render_text(result: AnalysisResult) -> str:
    input_case = result.input_case
    load_lines, active_cores = _load_summary(result.per_core_load)
    lines = [
        "== Case ==",
        f"name: {result.testcase_name}",
        f"op: {input_case.get('op_name', 'mat_mul_v3')}",
        f"shape: M={input_case['m']}, K={input_case['k']}, N={input_case['n']}",
        f"dtype: in={input_case['dtype']}, out={input_case['out_dtype']}",
        f"layout: x1={input_case['x1_format']}, x2={input_case['x2_format']}, y={input_case['y_format']}",
        f"transpose: x1={input_case['transpose_x1']}, x2={input_case['transpose_x2']}",
        "",
        "== Strategy ==",
        f"selected: {result.selected_strategy}",
        f"branch: {result.strategy_branch}",
        f"tiling_key: {result.tiling_key_hex} ({result.tiling_key})",
        f"kernel: {result.kernel_dispatch.get('kernel_impl')}",
        "",
        "== Source Mapping ==",
        f"host_file: {result.source_mapping.get('host_file')}",
        f"kernel_entry: {result.source_mapping.get('kernel_entry')}",
        f"scheduler: {result.source_mapping.get('scheduler')}",
        "",
        "== Tiling ==",
        f"tiling_key_fields: {_json_like(result.tiling_key_fields)}",
        f"inter_core: {_json_like(result.inter_core)}",
        f"intra_core: {_json_like(result.intra_core)}",
        f"tiling_data: {_json_like(result.tiling_data)}",
        "",
        "== Load Summary ==",
    ]
    lines.extend(load_lines)
    lines.extend(["", "== Active Core Workload =="])

    if not active_cores:
        lines.append("- no active core tasks")
    for core in active_cores:
        scenes = sorted({task.get("scene", "NA") for task in core["tasks"]})
        lines.append(
            f"- {core['core_type']}[{core['core_id']}]: {len(core['tasks'])} task(s), scenes={','.join(scenes)}"
        )
        for index, task in enumerate(core["tasks"][:4]):
            lines.append(f"    {_format_task(task, index)}")
        if len(core["tasks"]) > 4:
            lines.append(f"    ... {len(core['tasks']) - 4} more task(s)")
    if result.notes:
        lines.extend(["", "== Notes =="])
        for note in result.notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


def render_json(result: AnalysisResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def save_batch_results(results: list[AnalysisResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = output_dir / "cases"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "testcase_name": result.testcase_name,
                "strategy": result.selected_strategy,
                "strategy_branch": result.strategy_branch,
                "tiling_key_hex": result.tiling_key_hex,
                "kernel_impl": result.kernel_dispatch.get("kernel_impl"),
            }
        )
        (per_case_dir / f"{result.testcase_name}.json").write_text(render_json(result), encoding="utf-8")
        (per_case_dir / f"{result.testcase_name}.txt").write_text(render_text(result), encoding="utf-8")

    (output_dir / "summary.json").write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["testcase_name", "strategy", "strategy_branch", "tiling_key_hex", "kernel_impl"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

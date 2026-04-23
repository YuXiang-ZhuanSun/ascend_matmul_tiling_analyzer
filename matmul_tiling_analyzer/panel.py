from __future__ import annotations

import argparse
import json
import tempfile
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .constants import AIC_NUM
from .models import CaseInput
from .parser import parse_cases_from_csv
from .strategies import analyze_case


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _int_value(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _case_from_payload(payload: dict[str, Any]) -> CaseInput:
    dtype = str(payload.get("dtype") or "bfloat16")
    op_impl_mode = _int_value(payload.get("op_impl_mode"), 0x1)
    return CaseInput(
        testcase_name=str(payload.get("name") or "panel_case"),
        m=_int_value(payload.get("m")),
        k=_int_value(payload.get("k")),
        n=_int_value(payload.get("n")),
        dtype=dtype,
        out_dtype=str(payload.get("out_dtype") or dtype),
        transpose_x1=_bool_value(payload.get("transpose_x1")),
        transpose_x2=_bool_value(payload.get("transpose_x2")),
        x1_format=str(payload.get("x1_format") or "ND"),
        x2_format=str(payload.get("x2_format") or "ND"),
        y_format=str(payload.get("y_format") or "ND"),
        has_bias=_bool_value(payload.get("has_bias")),
        bias_dtype=(str(payload.get("bias_dtype")) if payload.get("bias_dtype") else None),
        op_impl_mode=op_impl_mode,
        is_force_group_acc=op_impl_mode == 0x4,
        is_hf32=(op_impl_mode == 0x40) or _bool_value(payload.get("enable_hf32")),
    )


def _task_units(task: dict[str, Any]) -> int:
    shape = task.get("tile_shape")
    if isinstance(shape, list) and len(shape) == 3:
        return max(int(shape[0]), 0) * max(int(shape[1]), 0) * max(int(shape[2]), 0)
    if {"m_range", "n_range", "k_range"} <= set(task):
        m0, m1 = task["m_range"]
        n0, n1 = task["n_range"]
        k0, k1 = task["k_range"]
        return max(int(m1) - int(m0), 0) * max(int(n1) - int(n0), 0) * max(int(k1) - int(k0), 0)
    if "flat_mn_range" in task:
        start, end = task["flat_mn_range"]
        return max(int(end) - int(start), 0) * max(int(task.get("base_k", 1)), 1) * max(int(task.get("loop_k", 1)), 1)
    if "clear_range" in task:
        start, end = task["clear_range"]
        return max(int(end) - int(start), 0)
    return 0


def _normalize_core_grid(per_core_load: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aic_cores = [core for core in per_core_load if core.get("core_type") == "AIC"]
    source_cores = aic_cores if aic_cores else per_core_load
    by_id = {int(core.get("core_id", 0)): core for core in source_cores}
    grid = []
    for core_id in range(AIC_NUM):
        core = by_id.get(core_id, {"core_type": "AIC", "core_id": core_id, "tasks": []})
        tasks = core.get("tasks", [])
        units = sum(_task_units(task) for task in tasks)
        grid.append(
            {
                "core_type": core.get("core_type", "AIC"),
                "core_id": int(core.get("core_id", core_id)),
                "task_count": len(tasks),
                "work_units": units,
                "tasks": tasks,
            }
        )
    return grid


def _dashboard_payload(result: dict[str, Any]) -> dict[str, Any]:
    core_grid = _normalize_core_grid(result["per_core_load"])
    active = [core for core in core_grid if core["task_count"] > 0]
    max_units = max((core["work_units"] for core in core_grid), default=0)
    avg_active = sum(core["work_units"] for core in active) / len(active) if active else 0.0
    top_imbalanced = sorted(
        [
            {
                "core": f"{core['core_type']}[{core['core_id']}]",
                "tasks": core["task_count"],
                "work_units": core["work_units"],
                "vs_avg": ((core["work_units"] - avg_active) / avg_active if avg_active else 0.0),
            }
            for core in active
        ],
        key=lambda item: item["work_units"],
        reverse=True,
    )[:5]
    return {
        "core_grid": core_grid,
        "active_cores": len(active),
        "idle_cores": len(core_grid) - len(active),
        "max_work_units": max_units,
        "avg_active_work_units": avg_active,
        "total_tasks": sum(core["task_count"] for core in core_grid),
        "total_work_units": sum(core["work_units"] for core in core_grid),
        "top_imbalanced": top_imbalanced,
    }


def _result_payload(result) -> dict[str, Any]:
    analysis = result.to_dict()
    return {"analysis": analysis, "dashboard": _dashboard_payload(analysis)}


def analyze_shape_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _result_payload(analyze_case(_case_from_payload(payload)))


def analyze_csv_text(csv_text: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="matmul_tiling_panel_") as temp_dir:
        csv_path = Path(temp_dir) / "cases.csv"
        csv_path.write_text(csv_text, encoding="utf-8-sig")
        cases = parse_cases_from_csv(csv_path)
    results = [_result_payload(analyze_case(case)) for case in cases]
    return {
        "cases": [
            {
                "index": index,
                "name": item["analysis"]["testcase_name"],
                "shape": item["analysis"]["input_case"],
                "selected_strategy": item["analysis"]["selected_strategy"],
                "strategy_branch": item["analysis"]["strategy_branch"],
                "tiling_key_hex": item["analysis"]["tiling_key_hex"],
            }
            for index, item in enumerate(results)
        ],
        "results": results,
    }


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class PanelHandler(BaseHTTPRequestHandler):
    server_version = "MatMulTilingPanel/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            body = PANEL_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/health":
            _json_response(self, {"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            if self.path == "/api/analyze-shape":
                _json_response(self, analyze_shape_payload(payload))
                return
            if self.path == "/api/analyze-csv":
                csv_text = str(payload.get("csv_text") or "")
                if not csv_text.strip():
                    _json_response(self, {"error": "csv_text is required"}, HTTPStatus.BAD_REQUEST)
                    return
                _json_response(self, analyze_csv_text(csv_text))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def run_panel(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), PanelHandler)
    url = f"http://{host}:{port}"
    print(f"MatMul Tiling Panel is running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MatMul Tiling Panel.")
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the MatMul Tiling Analyzer local panel.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    args = parser.parse_args()
    run_panel(host=args.host, port=args.port, open_browser=not args.no_browser)


PANEL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MatMul Tiling Analyzer Panel</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #05070c;
      --panel: #101821;
      --panel-2: #151f2b;
      --line: #2c3949;
      --text: #eef3f7;
      --muted: #9eabb8;
      --cyan: #39c7d3;
      --blue: #4c8df6;
      --green: #77d36b;
      --amber: #efc84a;
      --red: #f26c4f;
      --violet: #9b68f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #0b1118;
      padding: 24px;
    }
    main { padding: 24px; }
    h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 16px; }
    label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0d141d;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }
    textarea { min-height: 120px; resize: vertical; }
    button {
      cursor: pointer;
      border-color: #315b75;
      background: linear-gradient(135deg, #1b5160, #203e72);
      font-weight: 700;
    }
    button.secondary { background: #121c27; }
    .subtle { color: var(--muted); line-height: 1.5; }
    .grid { display: grid; gap: 14px; }
    .fields { grid-template-columns: repeat(3, 1fr); }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .metric strong { display: block; margin-top: 8px; font-size: 22px; }
    .accent { color: var(--green); }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) 380px;
      gap: 14px;
    }
    .heatmap {
      display: grid;
      grid-template-columns: repeat(8, minmax(44px, 1fr));
      gap: 8px;
    }
    .core {
      min-height: 64px;
      border: 1px solid #263445;
      border-radius: 6px;
      padding: 8px;
      background: #111923;
    }
    .core button {
      height: 100%;
      min-height: 48px;
      border: 0;
      background: transparent;
      padding: 0;
      text-align: left;
    }
    .core-id { font-size: 12px; color: var(--muted); }
    .core-load { font-size: 18px; font-weight: 800; margin-top: 4px; }
    .core-tasks { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 20px 0;
    }
    .tab.active { border-color: var(--cyan); box-shadow: inset 0 0 0 1px var(--cyan); }
    .stack { display: grid; gap: 12px; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .branch { color: var(--green); font-family: Consolas, monospace; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #d9e8ef;
      font-size: 12px;
      line-height: 1.45;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    .case-list { max-height: 220px; overflow: auto; }
    .case-row {
      width: 100%;
      margin-bottom: 8px;
      text-align: left;
      background: #101922;
      border: 1px solid var(--line);
      font-weight: 600;
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .summary, .layout, .fields { grid-template-columns: 1fr; }
      .heatmap { grid-template-columns: repeat(4, minmax(44px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>MatMul Tiling Analyzer</h1>
      <div class="subtle">Local panel for shape input, testcase upload, branch tracing, tiling key decoding, and 32-core workload inspection.</div>
      <div class="tabs">
        <button id="shapeTab" class="tab active" type="button">Input Shape</button>
        <button id="csvTab" class="tab" type="button">Upload Case</button>
      </div>
      <section id="shapePane" class="stack">
        <div class="two">
          <div><label>M</label><input id="m" type="number" value="2048" min="0" /></div>
          <div><label>K</label><input id="k" type="number" value="4096" min="0" /></div>
        </div>
        <div class="two">
          <div><label>N</label><input id="n" type="number" value="256" min="0" /></div>
          <div><label>dtype</label><select id="dtype"><option>bfloat16</option><option>float16</option><option>float32</option><option>int8</option></select></div>
        </div>
        <div class="two">
          <div><label>out dtype</label><input id="outDtype" value="bfloat16" /></div>
          <div><label>op impl mode</label><input id="opImplMode" value="0x1" /></div>
        </div>
        <div class="two">
          <div><label>x1 format</label><input id="x1Format" value="ND" /></div>
          <div><label>x2 format</label><input id="x2Format" value="ND" /></div>
        </div>
        <label><input id="transposeX1" type="checkbox" style="width:auto;margin-right:8px" /> transpose x1</label>
        <label><input id="transposeX2" type="checkbox" style="width:auto;margin-right:8px" /> transpose x2</label>
        <label><input id="hasBias" type="checkbox" style="width:auto;margin-right:8px" /> has bias</label>
        <button id="analyzeShape" type="button">Analyze Shape</button>
      </section>
      <section id="csvPane" class="stack" hidden>
        <div>
          <label>CSV testcase file</label>
          <input id="csvFile" type="file" accept=".csv,text/csv" />
        </div>
        <button id="analyzeCsv" type="button">Analyze Uploaded CSV</button>
        <div id="caseList" class="case-list"></div>
      </section>
    </aside>
    <main>
      <div class="summary">
        <div class="card metric"><span class="subtle">Selected Branch</span><strong id="branch" class="branch">-</strong></div>
        <div class="card metric"><span class="subtle">Tiling Key</span><strong id="tilingKey">-</strong></div>
        <div class="card metric"><span class="subtle">Active Cores</span><strong id="activeCores">-</strong></div>
        <div class="card metric"><span class="subtle">Total Tasks</span><strong id="totalTasks">-</strong></div>
      </div>
      <div class="layout">
        <section class="card">
          <h2>32-Core Workload</h2>
          <div id="heatmap" class="heatmap"></div>
        </section>
        <section class="card stack">
          <div>
            <h2>Selected Core Detail</h2>
            <pre id="coreDetail">Run an analysis to inspect per-core tasks.</pre>
          </div>
          <div>
            <h2>Top Imbalance Cores</h2>
            <table>
              <thead><tr><th>Core</th><th>Tasks</th><th>Work</th><th>vs Avg</th></tr></thead>
              <tbody id="imbalanceRows"></tbody>
            </table>
          </div>
        </section>
      </div>
      <div class="grid fields" style="margin-top:14px">
        <section class="card"><h2>Input Shape</h2><pre id="shapeInfo">-</pre></section>
        <section class="card"><h2>Tiling Fields</h2><pre id="tilingFields">-</pre></section>
        <section class="card"><h2>Source Mapping</h2><pre id="sourceMapping">-</pre></section>
      </div>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    let currentPayload = null;

    function setMode(mode) {
      $("shapePane").hidden = mode !== "shape";
      $("csvPane").hidden = mode !== "csv";
      $("shapeTab").classList.toggle("active", mode === "shape");
      $("csvTab").classList.toggle("active", mode === "csv");
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "request failed");
      return data;
    }

    function colorFor(core, maxWork) {
      if (!core.task_count) return "#18222e";
      const ratio = maxWork ? core.work_units / maxWork : 0;
      if (ratio > 0.82) return "#77d36b";
      if (ratio > 0.55) return "#39c7d3";
      if (ratio > 0.25) return "#4c8df6";
      return "#9b68f0";
    }

    function formatNumber(value) {
      if (value >= 1_000_000_000) return (value / 1_000_000_000).toFixed(2) + "G";
      if (value >= 1_000_000) return (value / 1_000_000).toFixed(2) + "M";
      if (value >= 1_000) return (value / 1_000).toFixed(2) + "K";
      return String(Math.round(value));
    }

    function render(payload) {
      currentPayload = payload;
      const a = payload.analysis;
      const d = payload.dashboard;
      $("branch").textContent = a.strategy_branch;
      $("tilingKey").textContent = a.tiling_key_hex;
      $("activeCores").textContent = `${d.active_cores} / ${d.core_grid.length}`;
      $("totalTasks").textContent = d.total_tasks;
      $("shapeInfo").textContent = JSON.stringify(a.input_case, null, 2);
      $("tilingFields").textContent = JSON.stringify({
        tiling_key_fields: a.tiling_key_fields,
        inter_core: a.inter_core,
        intra_core: a.intra_core,
        kernel_dispatch: a.kernel_dispatch
      }, null, 2);
      $("sourceMapping").textContent = JSON.stringify(a.source_mapping, null, 2);

      const heatmap = $("heatmap");
      heatmap.innerHTML = "";
      d.core_grid.forEach((core) => {
        const wrapper = document.createElement("div");
        wrapper.className = "core";
        wrapper.style.borderColor = colorFor(core, d.max_work_units);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.innerHTML = `<div class="core-id">${core.core_type}[${core.core_id}]</div><div class="core-load">${formatNumber(core.work_units)}</div><div class="core-tasks">${core.task_count} task(s)</div>`;
        btn.onclick = () => {
          $("coreDetail").textContent = JSON.stringify(core, null, 2);
        };
        wrapper.appendChild(btn);
        heatmap.appendChild(wrapper);
      });
      $("coreDetail").textContent = JSON.stringify(d.core_grid.find((core) => core.task_count > 0) || d.core_grid[0], null, 2);
      $("imbalanceRows").innerHTML = d.top_imbalanced.map((item) => {
        const pct = (item.vs_avg * 100).toFixed(1) + "%";
        return `<tr><td>${item.core}</td><td>${item.tasks}</td><td>${formatNumber(item.work_units)}</td><td>${pct}</td></tr>`;
      }).join("");
    }

    $("shapeTab").onclick = () => setMode("shape");
    $("csvTab").onclick = () => setMode("csv");
    $("analyzeShape").onclick = async () => {
      try {
        const payload = await postJson("/api/analyze-shape", {
          name: "panel_case",
          m: $("m").value,
          k: $("k").value,
          n: $("n").value,
          dtype: $("dtype").value,
          out_dtype: $("outDtype").value || $("dtype").value,
          op_impl_mode: $("opImplMode").value,
          x1_format: $("x1Format").value,
          x2_format: $("x2Format").value,
          transpose_x1: $("transposeX1").checked,
          transpose_x2: $("transposeX2").checked,
          has_bias: $("hasBias").checked
        });
        render(payload);
      } catch (error) {
        alert(error.message);
      }
    };
    $("analyzeCsv").onclick = async () => {
      try {
        const file = $("csvFile").files[0];
        if (!file) throw new Error("Choose a CSV file first.");
        const csvText = await file.text();
        const payload = await postJson("/api/analyze-csv", {csv_text: csvText});
        const list = $("caseList");
        list.innerHTML = payload.cases.map((item) => `<button class="case-row" type="button" data-index="${item.index}">${item.name}<br><span class="subtle">${item.tiling_key_hex} / ${item.strategy_branch}</span></button>`).join("");
        list.querySelectorAll("button").forEach((button) => {
          button.onclick = () => render(payload.results[Number(button.dataset.index)]);
        });
        if (payload.results.length) render(payload.results[0]);
      } catch (error) {
        alert(error.message);
      }
    };
    $("analyzeShape").click();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()

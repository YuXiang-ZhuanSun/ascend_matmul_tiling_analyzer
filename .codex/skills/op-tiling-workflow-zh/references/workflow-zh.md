# 端到端工作流

这份清单用于指导 agent 从算子源码和参考用例创建一个新的 tiling 复刻工具。执行时按阶段推进；前一阶段信息不足时，不要用猜测填补后一阶段。

## 0. 定义任务边界

先确认并写入工具文档：

- 源码根目录：至少识别 `op_api`、`op_host`、`op_kernel` 或等价目录。
- 参考用例：CSV/JSON/YAML/脚本都可以，但要整理成 quickstart 能直接读取的 `cases/quickstart_cases.csv`。
- 目标算子/API：写清算子名、public API、主要 shape 维度、dtype/layout/sparse/mask 等限制。
- 目标工具目录：它应当是一个可独立发布的 Python 项目，而不是工作区根目录里的临时脚本。
- fixture 策略：没有 `fixtures/`，或复制完整算子快照；不要只复制 host 子树。

## 1. 建源码地图

源码地图是 replay 的依据，优先级高于写代码。至少产出 `docs/source_map.json` 或等价文档，包含：

- 文件清单：参与 tiling/dispatch 的源码文件、相对路径、来源版本。
- 符号清单：API 入口、host tiling 函数、结构体、常量、setter、kernel 入口、dispatch 函数、模板入口。
- 调用链：从 public API 到 host tiling，再到 kernel dispatch 的主链路。
- 字段映射：case 输入、host 中间变量、tiling data 字段、kernel 读取字段之间的对应关系。
- 分支表：每个关键 `if/switch/template specialization` 的源码位置、条件表达式、可能取值、当前支持状态。
- tiling-key 表：组成字段、编码方式、候选 key、选中规则、下游 kernel 模板或 dispatch 分支。

对每个分支标注 `supported`、`recognized_unsupported` 或 `unknown`。`unknown` 不能被当作通过；它必须推动继续读源码或收窄支持范围。

## 2. 复刻 host tiling

按源码顺序实现，不按输出倒推：

- 保留整数除法、向上取整、对齐、边界裁剪、默认值和覆写逻辑。
- 将每个关键判断记录进 `tiling_trace.branches`，包含 `branch_id`、源码位置、表达式、输入值、结果和说明。
- 将常量和阈值放在命名结构里，附上来源文件/符号，避免散落 magic number。
- 遇到尚未支持的分支，抛出带源码位置和 unsupported reason 的错误，或在结果里明确标记失败；不要静默回退到默认路径。

host 复刻的产物至少包括：

- 归一化后的 case 参数。
- tiling 参数结构。
- split/分块/分核中间值。
- `logical_core_assignments`。
- host 侧 branch trace。
- tiling-key 组件和候选值。

## 3. 连接 tiling-key 与 kernel dispatch

tiling-key 不能只作为数字输出，必须解释它怎么决定 kernel 路径：

- 从 kernel 源码提取入口函数、dispatch 表、模板参数、宏分支和 lane 合同。
- 输出 `kernel_dispatch.candidates`，列出可能路径及其条件。
- 输出 `kernel_dispatch.selected`，指出当前 case 命中的入口、模板、tiling-key、lane 类型和源码位置。
- 如果 host 选出的 key 与 kernel 表无法匹配，应当失败并报告 mismatch，不要继续生成负载结果。

## 4. 展开 AIC/AIV 或物理 core 负载

先区分两层概念：

- `logical_core_assignments`：host 侧逻辑分核结果。
- `core_assignments`：kernel 侧物理执行单元，通常需要展开为 AIC/AIV、vector/cube lane 或硬件 core。

每个物理执行单元至少输出：

- `physical_core_id` 或等价 id。
- `lane_type`：例如 `AIC`、`AIV`、`cube`、`vector`。
- `logical_core_id`：回指 host 分核。
- `task_units`：不可再拆的计算/搬运任务。
- `task_segments`：按主二维块平面压缩后的连续段。
- `task_summary`：块数、token/element 数、有效负载、padding/空转、首尾边界。
- `kernel_execution`：kernel 入口、dispatch 分支、tiling-key、模板参数、lane 合同、读取的 tiling 字段。
- `visualization_ref`：指向对应 SVG/Markdown/HTML 可视化。

覆盖检查至少包含：

- `no_gap`：目标块平面没有缺口。
- `no_overlap`：不同 core 不重复覆盖同一有效块，除非源码明确允许。
- `coverage_ok`：覆盖全集与源码目标一致。
- `weighted_coverage_ok`：按有效 token/element 加权后仍一致。
- `load_balance`：每个 AIC/AIV 的负载统计和最大/最小/均值。

## 5. 搭建项目骨架

推荐结构：

```text
tiling-replay-tool/
|-- README.md
|-- pyproject.toml
|-- cli.py
|-- cases/
|   `-- quickstart_cases.csv
|-- src/
|   `-- <package_name>/
|       |-- __init__.py
|       |-- cases.py
|       |-- source_map.py
|       |-- replay.py
|       |-- dispatch.py
|       |-- workloads.py
|       `-- visualize.py
|-- tests/
|-- docs/
|-- examples/
`-- fixtures/                 # optional; only complete snapshots
```

`cli.py` 可以很薄，但必须稳定调用包内实现。默认命令固定为：

```bash
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
```

## 6. 组织输出

一次 replay 的 JSON、CSV、SVG、Markdown 必须来自同一份内存结果，避免多个命令各算一次导致不一致。推荐输出：

```text
results/quickstart/
|-- summary.csv
|-- summary.json
|-- source_map_used.json
|-- case_<case_id>/
|   |-- replay.json
|   |-- tiling_trace.json
|   |-- kernel_dispatch.json
|   |-- core_assignments.json
|   |-- checks.json
|   `-- visualizations/
|       `-- core_blocks.svg
`-- index.md
```

重复 `case_id` 要自动去重或加序号，不能覆盖结果。

## 7. 验证与收口

最少验证：

- `python -m pip install -e .`
- quickstart 命令能生成完整结果目录。
- `python -m unittest discover -s tests -v`
- 全量参考用例跑通，或未支持 case 有明确 unsupported reason。
- 至少一个 golden case 校验 branch trace、tiling-key、kernel dispatch、AIC/AIV 负载和覆盖检查。
- README 中的命令与真实命令完全一致。

如果在实现中发现新的分支、字段复用、fixture 边界或输出约束，必须同步更新 skill/reference 或项目文档，避免经验只留在代码里。

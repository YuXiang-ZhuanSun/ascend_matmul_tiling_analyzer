---
name: op-tiling-workflow-zh
description: 从算子实现源码和参考用例创建完整的 tiling 复刻与分析工具。适用于 C++ / CUDA / Ascend C / Ascend 算子工程，需要不重不漏建立 op_host/op_kernel 源码地图，解析真实执行 branch、tiling-key、kernel dispatch，并输出每个 AIC/AIV 或物理 core 的负载；同时搭建可 python -m pip install -e . 安装、可 python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart 运行的 GitHub 风格 Python CLI 项目，补齐 README、测试、示例和交付文档。
---

# 算子 Tiling 复刻工具工作流

把输入的算子源码和参考用例，交付成一个能独立安装、运行、验证、阅读的 tiling 复刻工具项目。目标不是“猜一个近似 tiling”，而是让工具能回答：

- 这个 case 走了源码里的哪条 branch。
- host 侧生成了什么 tiling 参数和 tiling-key。
- kernel 侧 dispatch 到了哪个模板/入口/执行路径。
- 每个 AIC/AIV 或物理 core 实际承担了哪些块、任务和执行语境。

## 必读材料

- 端到端工作流：读 [references/workflow-zh.md](references/workflow-zh.md)。
- 项目形态、输出 schema、README 契约：读 [references/delivery-contract-zh.md](references/delivery-contract-zh.md)。
- 目标算子是 PromptFlashAttention / FlashAttention 时，再读 [references/fpa-v2-zh.md](references/fpa-v2-zh.md)。

## 北极星

交付物必须像一个完整 GitHub 项目，而不是脚本堆：

- `python -m pip install -e .` 能安装依赖与本地包。
- `python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart` 是默认 quickstart 入口。
- README 简洁、有张力，第一屏说明价值、命令、输出，不写流水账。
- `tests/` 能验证源码地图、replay、golden case 和输出一致性。
- `docs/` 记录源码地图、分支覆盖、tiling-key/dispatch 关系和已知限制。
- `examples/` 或 `results/quickstart` 展示一组可复现的 JSON/CSV/SVG/Markdown 结果。

## 工作顺序

1. 先定边界：确认源码根、参考用例、目标工具目录、支持的算子/API/shape 范围，以及 fixture 是否采用完整快照。
2. 先建源码地图：同时覆盖 host 和 kernel，记录文件、符号、调用链、条件分支、字段映射、tiling-key 组成和 dispatch 入口。
3. 再写复刻器：按源码顺序复现整数除法、边界、默认值、覆写逻辑和 split 逻辑；不能凭输出反推一个“看起来差不多”的结果。
4. 串起 branch -> tiling-key -> kernel：每个 case 都输出 branch trace、tiling-key 候选/选中值、kernel dispatch 候选/选中路径。
5. 展开到 AIC/AIV：区分 logical assignment 与 physical lane，给每个 AIC/AIV 或物理 core 输出 task units、segments、summary、kernel_execution 和二维块图。
6. 做成项目：补 `pyproject.toml`、`cli.py`、`src/`、`tests/`、`docs/`、`cases/quickstart_cases.csv`、README 和示例结果。
7. 验证收口：跑全量用例、单元测试、golden case、覆盖检查；失败就修复或明确标为 unsupported，不允许静默跳过。

## 不重不漏原则

- 不只分析 `op_host`。host 负责 tiling 参数，kernel 负责执行合同，两边必须连起来。
- 不只保留“当前命中的 happy path”。源码地图要列出已支持、已识别但未支持、未触达的分支。
- 不用猜测代替源码依据。每条关键规则都要能指回文件、符号、条件或常量。
- 不用范围代替负载。每个物理执行单元必须有任务明细、摘要和可视化。
- 不复制残缺 fixture。要么没有 `fixtures/`，要么保留完整算子快照并写清来源和同步方式。

## 输出底线

每个 case 至少输出这些信息：

- `case_id`、输入参数、归一化后的有效参数。
- `source_trace` 或 `tiling_trace`：branch 判定、条件真假、源码位置、关键中间值。
- `tiling_key`：组成字段、候选值、最终值、与 kernel dispatch 的连接。
- `logical_core_assignments`：host 侧逻辑分核结果。
- `core_assignments`：AIC/AIV 或物理 core 级展开结果。
- `task_units`、`task_segments`、`task_summary`：每个物理执行单元的负载。
- `kernel_execution`：入口、dispatch 分支、lane 类型、模板/tiling-key 语境。
- `checks`：no gap、no overlap、coverage、weighted coverage、unsupported reason。
- `visualizations`：至少能看出每个 core 在主二维块平面上的覆盖。

## 完成定义

完成时必须能在目标工具目录运行：

```bash
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
python -m unittest discover -s tests -v
```

并且 README 能让第一次进入仓库的人在 60 秒内理解：这个工具复刻哪个算子、怎么运行、会产出什么、当前覆盖到哪里、哪些分支尚未支持。

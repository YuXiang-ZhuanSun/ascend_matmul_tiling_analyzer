# 交付契约

这份契约定义“完成”的样子。交付物要像一个可以公开阅读的 GitHub 项目：入口清楚、结果有解释、测试能跑、限制诚实。

## 1. 项目形态

目标工具目录至少包含：

```text
README.md
pyproject.toml
cli.py
cases/quickstart_cases.csv
src/<package_name>/
tests/
docs/
examples/
```

可选目录：

- `fixtures/`：只允许完整算子快照，并在 `docs/fixtures.md` 或 README 中写明来源、复制时间、同步方式。
- `results/`：默认是本地生成目录；如果提交示例结果，应放在 `examples/`，避免把临时结果当源码。

`cli.py` 是用户入口，包内源码放在 `src/<package_name>/`。不要把全部逻辑塞进 `cli.py`。

## 2. 安装与 quickstart

README 和测试都必须以这组命令为准：

```bash
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
```

可额外提供 console script，但不能替代上面的入口。

## 3. README 风格

README 要简洁、有张力，第一屏完成三件事：

- 一句话说明：这个工具复刻哪个算子的 tiling，并把 branch、tiling-key、AIC/AIV 负载串起来。
- 一段 quickstart：只给可复制运行的最短命令。
- 一个结果预览：输出目录树、字段摘要或一张 core block 可视化。

推荐结构：

````markdown
# <Operator> Tiling Replay

Replays <operator> tiling from source code: branch trace, tiling-key, kernel dispatch, and per-AIC/AIV workloads.

## Quickstart

```bash
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
```

## What You Get

## Supported Scope

## Output Schema

## Validation

## Known Limits
````

风格要求：

- 少写背景，多写价值和证据。
- 不造假 badge，不夸大覆盖范围。
- 已支持和未支持分开写，限制要具体到分支/shape/dtype/layout。
- 命令、路径、字段名必须和仓库真实内容一致。

## 4. 输出 schema

每个 case 的 `replay.json` 至少包含：

```json
{
  "case_id": "...",
  "inputs": {},
  "normalized_inputs": {},
  "source_trace": {
    "source_revision": "...",
    "branches": []
  },
  "tiling_trace": {
    "host_path": "...",
    "branches": [],
    "intermediate_values": {},
    "tiling_key": {}
  },
  "kernel_dispatch": {
    "candidates": [],
    "selected": {}
  },
  "logical_core_assignments": [],
  "core_assignments": [],
  "checks": {},
  "visualizations": []
}
```

`core_assignments[]` 至少包含：

```json
{
  "physical_core_id": 0,
  "lane_type": "AIC-or-AIV",
  "logical_core_id": 0,
  "range": {},
  "task_units": [],
  "task_segments": [],
  "task_summary": {},
  "kernel_execution": {},
  "visualization_ref": "visualizations/core_blocks.svg"
}
```

字段可以扩展，但不要删除这些基础层级。

## 5. 源码覆盖文档

`docs/` 至少写清：

- `source_map`：host/kernel 文件、符号、调用链、字段映射。
- `branch_coverage`：支持分支、已识别未支持分支、未知分支。
- `tiling_key_dispatch`：tiling-key 组成、候选、选中逻辑、kernel dispatch 入口。
- `validation`：跑过哪些 cases、哪些检查通过、哪些 case unsupported。

## 6. 测试门槛

测试至少覆盖：

- case parser 能读 quickstart CSV。
- source map 里包含 host 和 kernel 关键符号。
- golden case 的 branch trace 和 tiling-key 稳定。
- golden case 的 AIC/AIV 负载覆盖无 gap/overlap。
- CLI 能在临时目录生成 JSON/CSV/可视化。

如果有真实参考输出或硬件日志，增加对比测试；没有参考输出时，也要验证内部一致性和源码分支解释。

## 7. 清理要求

交付前清理：

- 删除被新项目取代的散落原型脚本、重复 README、重复示例和缓存目录。
- 不提交 `__pycache__`、临时结果、大型无来源二进制。
- 保留的 fixture 必须能解释来源；解释不了就不要提交。
- 根目录如果只是工作区，不要伪装成主 Git 仓库；真正交付目录内保持自洽。

## 8. 不可接受的交付

- 只解析 `op_host`，没有 kernel dispatch 和执行语境。
- 只输出 core 范围，没有 task units、summary、kernel_execution。
- README 很长但没有 quickstart 或结果预览。
- quickstart 命令和真实 CLI 不一致。
- 对未支持分支静默走默认逻辑。
- fixture 只复制部分源码却声称是完整来源。

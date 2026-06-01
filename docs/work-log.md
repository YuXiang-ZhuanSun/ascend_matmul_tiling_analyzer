# 项目工作纪要

这份纪要是对本次协作聊天记录的简化整理，保留对项目有长期价值的要求、决策和验收结果。它不是逐字聊天备份，而是后续维护者可以快速理解上下文的工作记录。

## 初始目标

用户要求使用 `op-tiling-workflow-zh` skill，对 `mat_mul_v3` 目录下的 MatMulV3 算子实现进行 tiling 复刻。

核心验收要求：

- 读取 `mat_mul_v3` 的 host tiling 与 kernel dispatch 源码。
- 使用 `cases/quickstart_cases.csv` 作为复刻后的验证集。
- 复刻出的 `tiling_key` 必须和验证集中的 `bin_tiling_key` 一致。
- 不只输出 tiling-key，还要说明源码分支、kernel dispatch、每个 AIC/AIV 或物理 core 的任务。

## 已完成工作

项目新增了一个可安装、可运行、可验证的 Python CLI 工具：

```bash
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
python -m unittest discover -s tests -v
```

主要实现：

- `src/matmul_tiling_replay/key.py`：复刻 arch35 MatMulV3 tiling-key 字段编码和解码。
- `src/matmul_tiling_replay/cases.py`：读取 CSV 验证集并解析 shape、dtype、format、期望 key 和 tiling data。
- `src/matmul_tiling_replay/replay.py`：复刻源码分支、生成 key、展开 task units、生成物理核任务。
- `cli.py`：命令行入口。
- `tests/test_replay.py`：验证 key 编码、全量 CSV key 匹配、结果文件结构。

验证结果：

- `cases/quickstart_cases.csv` 共 198 个 case。
- `tiling_key` 匹配 `bin_tiling_key`：198/198。
- 单元测试：3 个测试全部通过。

## 关键源码依据

复刻重点来自以下源码：

- 策略顺序：`mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h`
- Stream-K 判断：`mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp`
- Basic ASWT full-load 判断：`mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp`
- tiling-key 字段：`mat_mul_v3/op_kernel/arch35/mat_mul_tiling_key.h`
- tiling-key 常量：`mat_mul_v3/op_kernel/arch35/mat_mul_v3_tiling_key_public.h`
- kernel dispatch：`mat_mul_v3/op_kernel/arch35/mat_mul_v3.cpp`

当前验证集中实际覆盖的 key：

| tiling_key | 解释 |
| --- | --- |
| `1` | Basic ASWT，非 full-load |
| `4097` | Stream-K model |
| `65537` | Basic ASWT，A full-load |
| `131073` | Basic ASWT，B full-load |

## 物理核任务完善过程

最初版本只做了轻量的 M/N tile 到 core 的映射。用户指出这不满足 skill 中“每个 AIC/AIV 或物理 core 实际承担哪些块、任务”的要求。

随后补齐：

- 使用 `bin_tiling_data` 中的 `singleCoreM / singleCoreN / singleCoreK` 作为任务块粒度。
- 将每个 case 展开为 `(m_tile, n_tile, k_segment)` task unit。
- 将 task unit 分配到具体 `AIC_x`。
- Stream-K 场景额外记录 paired `AIV_x` companion lane。
- 空闲物理核也记录为 `active=false`，避免静默跳过。
- 生成全局 `core_task_summary.csv` 和每个 case 的 `cores.json`。

## 输出结构调整

第一次实现中，根目录下有一堆 `<case_id>.json`，同时 `core_tasks/` 下也有一堆 per-case JSON。用户反馈“同一个用例的结果应该放在一起”。

最终输出结构调整为：

```text
results/quickstart/
  report.md
  summary.csv
  core_task_summary.csv
  cases/
    <case_id>/
      summary.md
      case.json
      cores.json
```

设计意图：

- 全局索引留在 run 根目录，方便批量查看。
- 单个 case 的所有结果聚合在 `cases/<case_id>/`，方便定位、归档和对比。
- `case.json` 放源码 trace、tiling-key、dispatch 和检查结果。
- `cores.json` 放物理核任务明细。
- `summary.md` 给人工快速阅读。

## README 中文化

用户要求将 README 改成中文并完善。已完成：

- 项目目标。
- 快速开始。
- 输出结构。
- tiling-key 覆盖范围。
- 物理核任务展开规则。
- 源码地图。
- 当前限制。
- 测试方式。

注意：PowerShell 的 `Get-Content` 可能显示中文乱码，但文件本身是 UTF-8，Python 读取确认正常。

## 当前边界

当前工具以验证集覆盖为准，已保证 `cases/quickstart_cases.csv` 中所有 case 的 `bin_tiling_key` 匹配。

暂未完整覆盖：

- 转置输入。
- NZ / 非 ND format。
- batch matmul、fused matmul、特殊 bias batch model。
- K=0、to-mul、to-multi-mul 等验证集中未出现分支。
- ASWT load-balance 内部所有 base block 搜索细节。

这些后续可以基于新的验证集继续扩展。

## 后续维护建议

新增验证集时，建议按这个顺序维护：

1. 先把新 CSV 放入 `cases/` 或 `test/`。
2. 跑 CLI，观察 `summary.csv` 中不匹配 case。
3. 回到对应 host tiling 源码补充分支规则。
4. 确保 `case.json` 的 `source_trace` 能指回源码位置。
5. 确保 `cores.json` 中每个 task unit 都被分配到物理核。
6. 增加或更新单元测试。

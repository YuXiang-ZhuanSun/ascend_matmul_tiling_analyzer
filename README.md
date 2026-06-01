# MatMulV3 Tiling 复刻工具

本项目用于复刻 `mat_mul_v3` 算子的 tiling-key 决策链路，并把验证集中的每个 case 展开到物理核任务级别。工具会从 CSV 用例读取输入 shape、dtype、format、`bin_tiling_key` 和 `bin_tiling_data`，按源码中的 host tiling 分支复现 `tiling_key`，再和验证集的 `bin_tiling_key` 做逐 case 校验。

当前验证集已对齐：

- 用例数：198
- `tiling_key` 匹配：198/198
- 输入范围：`bfloat16 x bfloat16 -> bfloat16`
- format：ND
- transpose：当前验证集均为非转置
- 目标路径：`mat_mul_v3/op_host/op_tiling/arch35` 与 `mat_mul_v3/op_kernel/arch35`

## 快速开始

在仓库根目录执行：

```bash
python -m pip install -e .
python cli.py --input=cases/quickstart_cases.csv --output-dir=results/quickstart
python -m unittest discover -s tests -v
```

运行成功后，CLI 会输出类似：

```text
replayed 198 cases; tiling_key matched 198/198
wrote results to .../results/quickstart
```

## 输出结构

一次运行的结果保存在 `results/<run_name>/` 下。默认 quickstart 输出为：

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

根目录文件：

- `report.md`：本次运行的总览，包含 case 数、tiling-key 匹配数、源码根路径和 key 字段说明。
- `summary.csv`：一行一个 case，便于快速筛选 shape、预测 key、期望 key、是否匹配、case 文件路径。
- `core_task_summary.csv`：一行一个物理核，便于快速查看每个 `AIC_x` / `AIV_x` 的任务数量、task id 和工作量摘要。

每个 case 目录：

- `summary.md`：单个 case 的人类可读摘要。
- `case.json`：输入参数、归一化 shape、源码分支 trace、tiling-key 字段、kernel dispatch、检查结果和文件引用。
- `cores.json`：物理核任务明细，包含每个 `(m_tile, n_tile, k_segment)` task unit、M/N/K 范围、分配到的 `AIC_x`，以及 Stream-K 场景下 paired `AIV_x` companion lane。

## 复刻范围

工具当前复刻并验证了验证集中出现的四类 tiling-key：

| tiling_key | 含义 | 源码依据 |
| --- | --- | --- |
| `1` | Basic ASWT，非 full-load，on-the-fly 输出 | `matmul_v3_basic_aswt_tiling.cpp::GetTilingKey` |
| `4097` | Stream-K model | `matmul_v3_basic_streamk_tiling.cpp::CheckStreamK* / GetTilingKey` |
| `65537` | Basic ASWT，A full-load | `matmul_v3_basic_aswt_tiling.cpp::CheckAL1FullLoad / DoAL1FullLoad` |
| `131073` | Basic ASWT，B full-load | `matmul_v3_basic_aswt_tiling.cpp::CheckBL1FullLoad / DoBL1FullLoad` |

`tiling_key` 字段编码来自：

- `mat_mul_v3/op_kernel/arch35/mat_mul_tiling_key.h`
- `mat_mul_v3/op_kernel/arch35/mat_mul_v3_tiling_key_public.h`

字段布局：

```text
API_LEVEL(4), A_TRANS(2), B_TRANS(2), BATCH_MODEL(4),
MODEL(4), FULL_LOAD(4), L0C2OUT_MODEL(4)
```

## 物理核任务展开

物理核任务展开优先使用验证集中的 `bin_tiling_data`：

- `singleCoreM`：tuple index 4
- `singleCoreN`：tuple index 5
- `singleCoreK`：tuple index 10
- `baseM/baseN/baseK`：tuple index 7/8/9

工具会按 `singleCoreM / singleCoreN / singleCoreK` 将计算空间展开为 task units：

```text
(m_tile, n_tile, k_segment)
```

再按 used AIC 核数分配到具体 `AIC_x`。非 Stream-K kernel 记录 AIC 任务；Stream-K kernel 额外记录 `AIV_x` companion lane，用于描述 reduction/fixpipe 伴随任务。

每个 `cores.json` 至少包含：

- `task_grid`：M tiles、N tiles、K segments、总 task units。
- `task_units`：所有逻辑任务单元及其 M/N/K 范围。
- `physical_cores`：每个物理核的任务列表、task id、active 状态和工作量摘要。
- `checks`：任务是否都被分配到 AIC、AIC/AIV 数量、活跃核数量。

## 源码地图

关键源码位置：

- host 策略顺序：`mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h`
- Stream-K capability：`mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp`
- Basic ASWT full-load：`mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp`
- tiling-key public 常量：`mat_mul_v3/op_kernel/arch35/mat_mul_v3_tiling_key_public.h`
- kernel dispatch：`mat_mul_v3/op_kernel/arch35/mat_mul_v3.cpp`

更详细的说明见：

- `docs/source-map.md`
- `docs/core-task-schema.md`
- `docs/work-log.md`

## 当前限制

当前工具以验证集覆盖为准，已经保证 `cases/quickstart_cases.csv` 中 198 个 case 的 `tiling_key` 全量匹配。尚未覆盖或未完整建模的范围包括：

- 转置输入场景。
- NZ / 非 ND format 场景。
- batch matmul、fused matmul、bias 特殊 batch 模型。
- K=0、to-mul、to-multi-mul 等验证集中未出现的分支。
- 完整复刻 ASWT load-balance 内部所有 base block 搜索细节。

这些分支在源码地图中保留了入口位置，后续可以继续按验证集扩展。

## 测试

单元测试覆盖：

- arch35 tiling-key 编码示例。
- `cases/quickstart_cases.csv` 全量 `bin_tiling_key` 对齐。
- 每个 case 的 `case.json / cores.json / summary.md` 输出结构。

运行：

```bash
python -m unittest discover -s tests -v
```

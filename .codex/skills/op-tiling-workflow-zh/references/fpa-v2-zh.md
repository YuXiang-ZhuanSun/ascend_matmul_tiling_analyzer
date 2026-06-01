# FPA / PromptFlashAttention 适配备注

这份备注只在目标算子是 PromptFlashAttention / FlashAttention 时读取。它记录当前项目里已经踩过的关键事实，用来防止重新退化成 host-only 分析。

## 1. 已验证范围

- 已落地算子：`PromptFlashAttentionTilingV2`。
- public API / testcase 路径：`aclnnPromptFlashAttentionV3`。
- 当前主覆盖 split 路径：`SPLIT_NBS_CUBE`。
- 当前工具需要同时覆盖 host tiling、kernel entry、dispatch branch、tiling-key 和每个物理 lane 的执行语境。

## 2. Host 侧字段复用

`PromptAttentionSeqParams` 在 FPA 中存在字段复用，不能只按字面字段名解释。当前已验证映射：

- `CoreHeadNumTail -> coreNidStart`
- `actualS1 -> coreNidEnd`
- `actualCoreNums -> coreSidStart`
- `singleCoreHeadNumSize -> coreSidEnd`
- `coreSeqPosStart -> coreSposStart`
- `coreSeqPosEnd -> coreSposEnd`

复刻时要把这种映射写入 `source_map` 和 `tiling_trace.intermediate_values`，否则读者会误以为字段名就是语义。

## 3. Kernel 侧事实

- kernel 入口函数在 `op_kernel/prompt_flash_attention.cpp`。
- 主 dispatch 入口在 `op_kernel/prompt_flash_attention_arch32.h`。
- 当前主路径 `SPLIT_NBS_CUBE` 会把一个 logical assignment 展开为 `vector + cube` 两个物理 lane。
- 结果里要能从 `tiling_key.selected` 追到 `kernel_dispatch.selected`，再追到每个 `core_assignments[].kernel_execution`。

## 4. FPA 输出重点

FPA 结果必须让读者同时看到：

- `logical_core_assignments`：host 侧逻辑分组。
- `tiling_trace`：host split 分支、DN 覆写状态、tiling-key 组件、候选 dispatch、最终选中 key。
- `core_assignments`：物理 lane 展开后的 AIC/AIV 或 vector/cube 结果。
- `core_assignments[].kernel_execution`：该物理 lane 的入口、dispatch、tiling-key、模板/路径语境。
- `Q x KV block` 可视化：每个物理 lane 在二维块平面上的覆盖。

只输出 `coreSposStart/coreSposEnd` 不够；必须结合 branch trace、task units、task segments、kernel execution 和可视化说明负载。

## 5. Fixture 提醒

- 不要只复制 `op_host` 到 fixture。
- 至少要能解释 `op_api`、`op_host`、`op_kernel` 的来源；更稳妥做法是复制完整算子快照。
- 如果 fixture 是工作区源码的冻结副本，要写清同步边界：哪些内容来自根目录输入，哪些内容随工具交付。

## 6. README 中的 FPA 说法

不要写成泛泛的 “FlashAttention analyzer”。更好的说法是：

- 支持 `PromptFlashAttentionTilingV2` 的指定 testcase 范围。
- 当前重点覆盖 `SPLIT_NBS_CUBE`。
- 输出 host branch trace、tiling-key、kernel dispatch、AIC/AIV lane 负载和 `Q x KV block` 图。
- 未覆盖路径列在 Known Limits，不要暗示全量 FPA 都已复刻。

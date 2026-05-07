import ast
import re
from functools import lru_cache

from .constants import *
from .models import AnalysisResult, CaseInput
from .schedulers import simulate_basic_aswt, simulate_basic_streamk, simulate_to_mul
from .source_mapping import SOURCE_MAP, read_asw_loadbalance_table_text
from .tiling_key import make_key_fields, pack_tiling_key
from .utils import ceil_align, ceil_div, floor_align, min_sqrt_factor


def _build_result(case: CaseInput, strategy: str, branch: str, fields, tiling_data: dict, inter_core: dict, intra_core: dict, kernel_dispatch: dict, per_core_load: list[dict], notes: list[str]) -> AnalysisResult:
    tiling_key = pack_tiling_key(fields)
    return AnalysisResult(
        testcase_name=case.testcase_name,
        selected_strategy=strategy,
        strategy_branch=branch,
        source_mapping=SOURCE_MAP[strategy],
        input_case=case.__dict__,
        tiling_key=tiling_key,
        tiling_key_hex=hex(tiling_key),
        tiling_key_fields=fields.__dict__,
        tiling_data=tiling_data,
        inter_core=inter_core,
        intra_core=intra_core,
        kernel_dispatch=kernel_dispatch,
        per_core_load=per_core_load,
        notes=notes,
    )


def _base_state(case: CaseInput) -> dict:
    return {
        "used_core_num": AIC_NUM,
        "base_m": BASIC_BLOCK_SIZE_256,
        "base_n": BASIC_BLOCK_SIZE_256,
        "base_k": BASIC_BLOCK_K_128_BYTE // DTYPE_SIZE[case.dtype],
        "step_m": 1,
        "step_n": 1,
        "step_ka": 1,
        "step_kb": 1,
        "depth_a1": 1,
        "depth_b1": 1,
        "single_core_m": BASIC_BLOCK_SIZE_256,
        "single_core_n": BASIC_BLOCK_SIZE_256,
        "single_core_k": case.k,
        "m_tail_cnt": 1,
        "n_tail_cnt": 1,
        "k_tail_cnt": 1,
        "m_base_tail_split_cnt": 1,
        "n_base_tail_split_cnt": 1,
        "m_tail_main": 0,
        "n_tail_main": 0,
        "l1_buffer_num": 2,
        "l0c_db": 1,
        "ub_db": 1,
    }


def _fill_basic_tiling_data(case: CaseInput, state: dict, workspace_bytes: int) -> dict:
    step_ka = min(4, min(state["step_ka"], state["step_kb"]))
    return {
        "usedCoreNum": state["used_core_num"],
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "mL1": min(ceil_align(case.m, BASIC_BLOCK_SIZE_16), state["base_m"] * state["step_m"]),
        "nL1": min(ceil_align(case.n, BASIC_BLOCK_SIZE_16), state["base_n"] * state["step_n"]),
        "kL1": state["base_k"] * step_ka,
        "baseM": state["base_m"],
        "baseN": state["base_n"],
        "baseK": state["base_k"],
        "skSingleCoreK": state["single_core_k"],
        "mTailCnt": state["m_tail_cnt"],
        "nTailCnt": state["n_tail_cnt"],
        "mBaseTailSplitCnt": state["m_base_tail_split_cnt"],
        "nBaseTailSplitCnt": state["n_base_tail_split_cnt"],
        "mTailMain": state["m_tail_main"],
        "nTailMain": state["n_tail_main"],
        "isHf32": int(case.is_hf32),
        "l1BufferNum": state["l1_buffer_num"],
        "l0cDB": state["l0c_db"],
        "ubDB": state["ub_db"],
        "workspaceBytes": workspace_bytes,
    }


def _cal_l1_tiling(case: CaseInput, state: dict, *, has_bias: bool = False) -> None:
    reserve_bt = BIAS_TABLE_NUM * DATA_SIZE_FP32 if has_bias else 0
    depth_a1 = max(L1_SIZE // 2 // state["base_m"] // state["base_k"] // DTYPE_SIZE[case.dtype], 1)
    depth_b1 = max(L1_SIZE // 2 // state["base_n"] // state["base_k"] // DTYPE_SIZE[case.dtype], 1)
    if depth_a1 * state["base_m"] * state["base_k"] * DTYPE_SIZE[case.dtype] + depth_b1 * state["base_n"] * state["base_k"] * DTYPE_SIZE[case.dtype] > L1_SIZE - reserve_bt:
        if state["base_m"] <= state["base_n"]:
            depth_a1 = max(depth_a1 // 2, 1)
        else:
            depth_b1 = max(depth_b1 // 2, 1)
    step_ka = max(depth_a1 // DB_SIZE, 1)
    step_kb = max(depth_b1 // DB_SIZE, 1)
    if state["base_m"] == BASIC_BLOCK_SIZE_256 and state["base_n"] == BASIC_BLOCK_SIZE_256 and case.m % 16 == 0 and case.n % 16 == 0 and case.k % 16 == 0 and state["single_core_k"] <= BASIC_BLOCK_SIZE_256:
        step_ka = min(step_ka, 2)
        step_kb = min(step_kb, 2)
    if step_ka >= step_kb:
        step_ka = max((step_ka // step_kb) * step_kb, 1)
    else:
        step_kb = max((step_kb // step_ka) * step_ka, 1)
    state["step_ka"] = step_ka
    state["step_kb"] = step_kb
    state["depth_a1"] = step_ka * DB_SIZE
    state["depth_b1"] = step_kb * DB_SIZE
    state["single_core_m"] = state["base_m"]
    state["single_core_n"] = state["base_n"]


def _get_l0c2out_aswt(case: CaseInput, state: dict) -> int:
    is_valid_mkn = case.k <= BASIC_BLOCK_SIZE_256 and case.m >= BASIC_BLOCK_SIZE_256
    m_cnt = ceil_div(case.m, state["single_core_m"])
    n_cnt = ceil_div(case.n, state["single_core_n"])
    is_multi_round = m_cnt * n_cnt >= 2 * AIC_NUM
    is_unaligned_n = (case.n * DTYPE_SIZE[case.out_dtype]) % 128 != 0 and case.n * DTYPE_SIZE[case.out_dtype] > BASIC_BLOCK_SIZE_256
    if not (is_valid_mkn and is_multi_round and is_unaligned_n and AIV_NUM == AIC_NUM * 2):
        return MAT_MUL_ON_THE_FLY
    return MAT_MUL_1V1_ND_ALIG_FIXPIPE if case.dtype in {"float16", "bfloat16"} else MAT_MUL_1V2_ND_ALIG_FIXPIPE


def _get_l0c2out_streamk(case: CaseInput) -> int:
    if case.n > BASIC_BLOCK_SIZE_64 and case.n % BASIC_BLOCK_SIZE_16 != 0 and case.m > 2 and case.m * case.n >= BASIC_BLOCK_SIZE_256:
        return MAT_MUL_1V2_ND_ALIG_FIXPIPE
    return MAT_MUL_ON_THE_FLY


def _is_single_round(case: CaseInput, state: dict) -> bool:
    m_cnt = ceil_div(case.m, state["single_core_m"])
    n_cnt = ceil_div(case.n, state["single_core_n"])
    return m_cnt * n_cnt <= AIC_NUM


def _check_a_full_load(case: CaseInput, state: dict, l0c2out: int) -> bool:
    if l0c2out != MAT_MUL_ON_THE_FLY or case.n < CACHELINE or _is_single_round(case, state):
        return False
    max_step_m = max(min_sqrt_factor(AIC_NUM) - 1, 1)
    if case.m >= max_step_m * state["base_m"]:
        return False
    m_aligned = ceil_align(case.m, BASIC_BLOCK_SIZE_16)
    k_align = BASIC_BLOCK_SIZE_16 if case.transpose_x1 else BLOCK_BYTE_SIZE // DTYPE_SIZE[case.dtype]
    k_aligned = ceil_align(case.k, k_align)
    a_l1_size = k_aligned * m_aligned * DTYPE_SIZE[case.dtype]
    bias_size = 0
    if case.has_bias and case.bias_dtype is not None:
        basic_n_l1 = min(ceil_align(case.n, BASIC_BLOCK_SIZE_16), state["base_n"] * DB_SIZE)
        bias_size = basic_n_l1 * DTYPE_SIZE.get(case.bias_dtype, DTYPE_SIZE[case.out_dtype])
    if a_l1_size + bias_size > (L1_SIZE * 3) // 4:
        return False
    m_cnt = ceil_div(case.m, state["single_core_m"])
    n_cnt = ceil_div(case.n, state["single_core_n"])
    a_l1_full_mte2 = case.m * AIC_NUM + case.n * m_cnt
    base_mte2 = case.m * n_cnt + case.n * m_cnt
    if case.m > BASIC_BLOCK_SIZE_256 and float(base_mte2) < 1.2 * float(a_l1_full_mte2):
        return False
    return True


def _check_b_full_load(case: CaseInput, state: dict) -> bool:
    if case.m < CACHELINE or _is_single_round(case, state):
        return False
    max_step_n = max(min_sqrt_factor(AIC_NUM) - 1, 1)
    if case.n >= max_step_n * state["base_n"]:
        return False
    k_align = BLOCK_BYTE_SIZE // DTYPE_SIZE[case.dtype] if case.transpose_x2 else BASIC_BLOCK_SIZE_16
    k_aligned = ceil_align(case.k, k_align)
    n_aligned = ceil_align(case.n, BASIC_BLOCK_SIZE_16)
    b_l1_size = k_aligned * n_aligned * DTYPE_SIZE[case.dtype]
    bias_size = n_aligned * DTYPE_SIZE.get(case.bias_dtype, DTYPE_SIZE[case.out_dtype]) if case.has_bias and case.bias_dtype is not None else 0
    if b_l1_size + bias_size > (L1_SIZE * 3) // 4:
        return False
    m_cnt = ceil_div(case.m, state["single_core_m"])
    n_cnt = ceil_div(case.n, state["single_core_n"])
    b_l1_full_mte2 = case.n * AIC_NUM + case.m * n_cnt
    base_mte2 = case.m * n_cnt + case.n * m_cnt
    if case.n > BASIC_BLOCK_SIZE_256 and float(base_mte2) < 1.2 * float(b_l1_full_mte2):
        return False
    return True


def _reset_base(case: CaseInput, state: dict) -> None:
    state.update(_base_state(case))


def _calc_basic_block(case: CaseInput, state: dict) -> None:
    m_core = ceil_div(case.m, state["base_m"])
    n_core = ceil_div(case.n, state["base_n"])
    if m_core <= n_core:
        state["base_m"] = ceil_align(ceil_div(case.m, m_core), BASIC_BLOCK_SIZE_16)
        m_core = ceil_div(case.m, state["base_m"])
        n_core = max(state["used_core_num"] // m_core, 1)
        state["base_n"] = ceil_align(ceil_div(case.n, n_core), BASIC_BLOCK_SIZE_16)
    else:
        state["base_n"] = ceil_align(ceil_div(case.n, n_core), BASIC_BLOCK_SIZE_16)
        n_core = ceil_div(case.n, state["base_n"])
        m_core = max(state["used_core_num"] // n_core, 1)
        state["base_m"] = ceil_align(ceil_div(case.m, m_core), BASIC_BLOCK_SIZE_16)
    while state["base_n"] >= state["base_m"] * 2 and n_core < state["used_core_num"] // 2:
        n_core *= 2
        m_core = max(state["used_core_num"] // n_core, 1)
        state["base_m"] = ceil_align(ceil_div(case.m, m_core), BASIC_BLOCK_SIZE_16)
        state["base_n"] = ceil_align(ceil_div(case.n, n_core), BASIC_BLOCK_SIZE_16)
        m_core = ceil_div(case.m, state["base_m"])
        n_core = ceil_div(case.n, state["base_n"])
    while state["base_m"] >= state["base_n"] * 2 and m_core < state["used_core_num"] // 2:
        m_core *= 2
        n_core = max(state["used_core_num"] // m_core, 1)
        state["base_m"] = ceil_align(ceil_div(case.m, m_core), BASIC_BLOCK_SIZE_16)
        state["base_n"] = ceil_align(ceil_div(case.n, n_core), BASIC_BLOCK_SIZE_16)
        m_core = ceil_div(case.m, state["base_m"])
        n_core = ceil_div(case.n, state["base_n"])


def _formulate_basic_block(case: CaseInput, state: dict) -> None:
    m_core = ceil_div(case.m, state["base_m"])
    n_core = ceil_div(case.n, state["base_n"])
    if m_core * n_core >= AIC_NUM:
        state["base_m"] = min(ceil_align(case.m, BASIC_BLOCK_SIZE_16), state["base_m"])
        state["base_n"] = min(ceil_align(case.n, BASIC_BLOCK_SIZE_16), state["base_n"])
        return
    _calc_basic_block(case, state)
    m_core = ceil_div(case.m, state["base_m"])
    n_core = ceil_div(case.n, state["base_n"])
    state["used_core_num"] = m_core * n_core
    k_aligned = ceil_align(case.k, BASIC_BLOCK_SIZE_16)
    k_max = floor_align(L0A_SIZE_2 // DB_SIZE // DTYPE_SIZE[case.dtype] // max(state["base_m"], state["base_n"]), BASIC_BLOCK_SIZE_16)
    state["base_k"] = min(k_aligned, k_max)


def _get_outer_axis_tail_cnt(case: CaseInput, state: dict, n_load_balance: bool) -> tuple[int, int]:
    x, y = (case.n, case.m) if n_load_balance else (case.m, case.n)
    base_x, base_y = (state["base_n"], state["base_m"]) if n_load_balance else (state["base_m"], state["base_n"])
    x_cnt = ceil_div(x, base_x)
    y_cnt = ceil_div(y, base_y)
    x_tail = x % base_x
    asw_window_len = min_sqrt_factor(AIC_NUM)
    total_windows = ceil_div(x_cnt * y_cnt, AIC_NUM)
    main_windows = ceil_div((x_cnt - 1) * y_cnt + y_cnt % AIC_NUM, AIC_NUM)
    if y_cnt % AIC_NUM == 0 and (x_cnt % asw_window_len == 0 or asw_window_len % x_cnt == 0):
        main_windows = total_windows
    tail_windows = total_windows - main_windows
    perf_res = main_windows * base_x + tail_windows * x_tail
    base_tail_cnt_max = min((base_x - x_tail) // BASIC_BLOCK_SIZE_16, x_cnt)
    best_split, best_tail_main = 1, 0
    for merge_len in range(1, base_tail_cnt_max):
        new_tail_main = ceil_align(ceil_div(merge_len * base_x + x_tail, merge_len + 1), BASIC_BLOCK_SIZE_16)
        new_tail_last = merge_len * (base_x - new_tail_main) + x_tail
        new_main_round = 0
        new_tail_round = 0
        if merge_len < x_cnt - 1:
            new_main_round = ceil_div((x_cnt - 1 - merge_len) * y_cnt + ((merge_len + 1) * y_cnt) % AIC_NUM, AIC_NUM)
        if merge_len > 0:
            new_tail_round = min(ceil_div(merge_len * y_cnt + y_cnt % AIC_NUM, AIC_NUM), total_windows - new_main_round)
        cur_perf = new_main_round * base_x + new_tail_round * new_tail_main + (total_windows - new_main_round - new_tail_round) * new_tail_last
        if cur_perf < perf_res or (not n_load_balance and cur_perf == perf_res):
            perf_res = cur_perf
            best_split = merge_len + 1
            best_tail_main = new_tail_main
    return best_split, best_tail_main


def _optimize_edge_basic_block(case: CaseInput, state: dict) -> None:
    m_core = ceil_div(case.m, state["base_m"])
    n_core = ceil_div(case.n, state["base_n"])
    if m_core * n_core < AIC_NUM or m_core == 1 or n_core == 1:
        return
    m_base_tail = case.m % state["base_m"]
    n_base_tail = case.n % state["base_n"]
    balance_post_fixp = case.k <= BASIC_BLOCK_SIZE_256 and case.n % BLOCK_BYTE_SIZE == 0
    if m_base_tail > 0 and not case.transpose_x1 and (n_base_tail == 0 or m_base_tail <= n_base_tail or balance_post_fixp):
        state["m_base_tail_split_cnt"], state["m_tail_main"] = _get_outer_axis_tail_cnt(case, state, False)
    elif n_base_tail > 0 and case.transpose_x2 and not balance_post_fixp:
        state["n_base_tail_split_cnt"], state["n_tail_main"] = _get_outer_axis_tail_cnt(case, state, True)


def _calc_tail_basic_block(case: CaseInput, state: dict) -> None:
    mn_cnt = ceil_div(case.m, state["base_m"]) * ceil_div(case.n, state["base_n"])
    tail_cnt = 0 if mn_cnt <= AIC_NUM else mn_cnt % AIC_NUM
    state["m_tail_cnt"], state["n_tail_cnt"] = 1, 1
    if tail_cnt:
        while (state["m_tail_cnt"] + 1) * state["n_tail_cnt"] * tail_cnt <= AIC_NUM:
            state["m_tail_cnt"] += 1
            if state["m_tail_cnt"] * (state["n_tail_cnt"] + 1) * tail_cnt <= AIC_NUM:
                state["n_tail_cnt"] += 1


@lru_cache(maxsize=1)
def _load_balance_tables():
    text = read_asw_loadbalance_table_text()
    lookup_match = re.search(r"BLOCK_LOOKUP_TABLE = \{(.*?)\};", text, re.S)
    table_match = re.search(r"BLOCK_TABLE = \{(.*?)\};", text, re.S)
    if lookup_match is None or table_match is None:
        raise ValueError("Failed to locate load balance tables in source header.")

    def _parse_entries(block_text: str, expected_len: int) -> list[tuple]:
        entries = []
        for raw_entry in re.findall(r"\{([^{}]+)\}", block_text):
            parts = [part.strip() for part in raw_entry.split(",")]
            if len(parts) != expected_len:
                raise ValueError(f"Unexpected load balance entry: {raw_entry}")
            parsed = []
            for part in parts:
                if "." in part:
                    parsed.append(float(part))
                else:
                    parsed.append(int(part))
            entries.append(tuple(parsed))
        return entries

    lookup = _parse_entries(lookup_match.group(1), 3)
    table = _parse_entries(table_match.group(1), 5)
    return lookup, table


def _find_load_balance_info(block: int):
    lookup, _ = _load_balance_tables()
    for block_value, count, start in lookup:
        if block_value == block:
            return start, count
    return None


def _calc_multi_core_balance(m: int, n: int, core_num: int, base_m: int, base_n: int) -> float:
    m_cnt = ceil_div(m, base_m)
    n_cnt = ceil_div(n, base_n)
    m_tail = m % base_m
    n_tail = n % base_n
    m_main_cnt = m // base_m
    n_main_cnt = n // base_n
    total_tiles = m_cnt * n_cnt
    total_main_tiles = m_main_cnt * n_main_cnt
    if total_tiles == 0:
        return 0.0
    avg_load = float(m * n) / core_num
    single_max_tail = max(base_m * n_tail, base_n * m_tail)

    def _div(base: int) -> int:
        if base % BASIC_BLOCK_SIZE_256 == 0:
            return BASIC_BLOCK_SIZE_16
        if base % BASIC_BLOCK_SIZE_128 == 0:
            return BASIC_BLOCK_SIZE_16 // 2
        if base % BASIC_BLOCK_SIZE_64 == 0:
            return 4
        if base % BLOCK_BYTE_SIZE == 0:
            return 2
        return 1

    core_tail_num = total_tiles % core_num
    tail_ratio = min(core_num // core_tail_num if core_tail_num else 1, _div(base_m) * _div(base_n))
    tail_num = ceil_div(total_tiles, core_num) - ceil_div(total_main_tiles, core_num)
    no_div_tail_num = tail_num - 1 if tail_num >= 2 else 0
    if tail_num == 0:
        max_load = float(total_main_tiles // core_num) * base_m * base_n + float(ceil_div(total_main_tiles, core_num) - total_main_tiles // core_num) * base_m * base_n / tail_ratio
    else:
        max_load = float(ceil_div(total_main_tiles, core_num) * base_m * base_n + no_div_tail_num * single_max_tail + (tail_num - no_div_tail_num) * single_max_tail / tail_ratio)
    return avg_load / max_load


def _calc_redundant_data_movement(base_m: int, base_n: int, m_value: int, n_value: int) -> int:
    return ceil_div(m_value, base_m) * n_value + ceil_div(n_value, base_n) * m_value - (m_value + n_value)


def _calc_single_x(case: CaseInput) -> tuple[int, int]:
    data = float(case.m * case.n) / AIC_NUM
    while data > float(BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256):
        data /= 2
    best = data ** 0.5
    return ceil_align(int(best + 0.999999), BASIC_BLOCK_SIZE_16), floor_align(int(best), BASIC_BLOCK_SIZE_16)


def _formulate_load_balance_block(case: CaseInput, state: dict) -> None:
    state["base_m"] = min(ceil_align(case.m, BASIC_BLOCK_SIZE_16), state["base_m"])
    state["base_n"] = min(ceil_align(case.n, BASIC_BLOCK_SIZE_16), state["base_n"])
    default_balance = _calc_multi_core_balance(case.m, case.n, AIC_NUM, state["base_m"], state["base_n"])
    redundant = _calc_redundant_data_movement(state["base_m"], state["base_n"], case.m, case.n)
    m_core = ceil_div(case.m, state["base_m"])
    n_core = ceil_div(case.n, state["base_n"])
    single_block_num = float(m_core * n_core) / AIC_NUM
    if 1.0 <= single_block_num <= MAX_SINGLE_CORE_ROUND and default_balance < LOAD_BALANCE_RATE_LIMIT:
        higher, lower = _calc_single_x(case)
        min_mn = ceil_align(min(case.m, case.n), BASIC_BLOCK_SIZE_16)
        is_m_larger = case.m > case.n
        _, block_table = _load_balance_tables()
        candidates = []
        for start, end, step in ((lower, MIN_BASE_BLOCK, -BASIC_BLOCK_SIZE_16), (higher, min(MAX_BASE_BLOCK, min_mn), BASIC_BLOCK_SIZE_16)):
            base_x = start
            while (step < 0 and base_x >= end) or (step > 0 and base_x <= end):
                info = _find_load_balance_info(base_x)
                if info is not None:
                    index, count = info
                    for i in range(count):
                        x1, x2, x3, x4, _ = block_table[index + i]
                        cur_m = x2 if is_m_larger else x1
                        cur_n = x1 if is_m_larger else x2
                        balance = _calc_multi_core_balance(case.m, case.n, AIC_NUM, cur_m, cur_n) / x4
                        remove_ratio = float(_calc_redundant_data_movement(cur_m, cur_n, case.m, case.n)) / redundant
                        if balance - default_balance > remove_ratio - BALANCE_REDUNDANT_THRESHOLD:
                            candidates.append((balance, cur_m, cur_n, x3))
                            if balance > LOAD_BALANCING_THRESHOLD:
                                state["base_m"], state["base_n"], state["base_k"] = cur_m, cur_n, x3
                                break
                base_x += step
        if candidates and state["base_m"] == BASIC_BLOCK_SIZE_256 and state["base_n"] == BASIC_BLOCK_SIZE_256:
            best = max(candidates, key=lambda item: item[0] if item[0] > MIN_EQUALIZATION_COEFFICIENT * default_balance else -1)
            if best[0] > 0:
                state["base_m"], state["base_n"], state["base_k"] = best[1], best[2], best[3]
    if single_block_num < 1.0:
        _calc_basic_block(case, state)
    state["base_m"] = ceil_align(state["base_m"], BASIC_BLOCK_SIZE_16)
    state["base_n"] = ceil_align(state["base_n"], BASIC_BLOCK_SIZE_16)
    state["l0c_db"] = DB_SIZE if state["base_m"] * state["base_n"] * DATA_SIZE_FP32 * DB_SIZE <= L0C_SIZE else 1
    m_core = ceil_div(case.m, state["base_m"])
    n_core = ceil_div(case.n, state["base_n"])
    state["used_core_num"] = min(m_core * n_core, AIC_NUM)
    base_k_align = BLOCK_BYTE_SIZE // DTYPE_SIZE[case.dtype] if (not case.transpose_x1 and case.transpose_x2 and case.dtype == "float32") else BASIC_BLOCK_SIZE_16
    state["base_k"] = min(ceil_align(case.k, base_k_align), floor_align(L0A_SIZE_2 // DB_SIZE // DTYPE_SIZE[case.dtype] // max(state["base_m"], state["base_n"]), base_k_align))


def _check_streamk_sk(case: CaseInput) -> bool:
    if case.dtype == "float32" and not case.is_hf32 and case.k > 2_000_000:
        return False
    if ceil_align(case.k, BASIC_BLOCK_SIZE_256) < max(8192, AIC_NUM * BASIC_BLOCK_K_256_BYTE) // DTYPE_SIZE[case.dtype]:
        return False
    align_value = BASIC_BLOCK_SIZE_32 if case.dtype == "float32" and not case.is_hf32 else BASIC_BLOCK_SIZE_256
    return ceil_div(case.m, align_value) * ceil_div(case.n, align_value) <= AIC_NUM // 2


def _check_streamk_dpsk(case: CaseInput) -> bool:
    if case.m % BASIC_BLOCK_SIZE_256 != 0 or case.n % BASIC_BLOCK_SIZE_256 != 0:
        return False
    if case.k < max(8192, AIC_NUM * BASIC_BLOCK_K_128_BYTE) // DTYPE_SIZE[case.dtype]:
        return False
    if case.dtype == "float32" and not case.is_hf32:
        return False
    total_mn = ceil_div(case.m, BASIC_BLOCK_SIZE_256) * ceil_div(case.n, BASIC_BLOCK_SIZE_256)
    return total_mn >= AIC_NUM and total_mn % AIC_NUM != 0 and total_mn % AIC_NUM <= AIC_NUM // 2


def analyze_k_equal_zero(case: CaseInput) -> AnalysisResult:
    fields = make_key_fields(case, model=MAT_MUL_K_EQUAL_ZERO, full_load=MAT_MUL_NO_FULL_LOAD, l0c2out=MAT_MUL_ON_THE_FLY, api_level=MAT_MUL_HIGH_LEVEL)
    total = case.m * case.n
    chunk = ceil_div(total, AIV_NUM)
    per_core = [{"core_type": "AIV", "core_id": core, "tasks": [{"clear_range": [core * chunk, min(total, (core + 1) * chunk)]}]} for core in range(AIV_NUM) if core * chunk < total]
    return _build_result(case, "k_equal_zero", "k_equal_zero_clear_output", fields, {"totalDataAmount": total, "aivNum": AIV_NUM}, {"used_core_num": AIV_NUM, "mode": "vector clear output"}, {"operation": "clear output"}, {"kernel_impl": "MatMulInputKEqZeroClearOutput", "kernel_type": "AIV_ONLY"}, per_core, ["Matches `MatMulV3KEqZeroTiling::DoOpTiling`."])


def analyze_to_mul(case: CaseInput) -> AnalysisResult:
    shape_mn = case.m * case.n
    ub_count = UB_SIZE // 4
    data_copy_mode = (case.m == 1 and not case.transpose_x2) or (case.n == 1 and case.transpose_x1)
    base_mn = TO_MUL_BASE_MN
    base_k = TO_MUL_BASE_K
    tmp_mn = ceil_div(shape_mn, AIV_NUM)
    if data_copy_mode:
        base_mn = min(ceil_align(tmp_mn, TO_MUL_ALIGN_NUM), TO_MUL_BASE_MN)
        bias_count = base_mn if case.has_bias else 0
        base_k = (ub_count - base_mn - bias_count) // 5 // base_mn
    else:
        bias_count = tmp_mn if case.has_bias else 0
        tmp_k = 0 if ub_count <= (tmp_mn + bias_count) else (ub_count - tmp_mn - bias_count) // 5 // tmp_mn
        base_k = max(floor_align(tmp_k, TO_MUL_ALIGN_NUM), TO_MUL_BASE_K)
        bias_count = 1 if case.has_bias else 0
        base_mn = min(tmp_mn, ub_count // (5 * base_k + bias_count + 1))
    tiling_data = {"usedCoreNum": min(ceil_div(shape_mn, base_mn), AIV_NUM), "tileNum": ceil_div(shape_mn, base_mn), "m": case.m, "n": case.n, "k": case.k, "baseMN": base_mn, "tailMN": shape_mn % base_mn, "baseK": base_k, "tailK": case.k % base_k, "loopK": ceil_div(case.k, base_k), "dataCopyMode": data_copy_mode}
    fields = make_key_fields(case, model=MAT_MUL_TO_MUL, full_load=MAT_MUL_NO_FULL_LOAD, l0c2out=MAT_MUL_ON_THE_FLY, api_level=MAT_MUL_BASIC_LEVEL)
    return _build_result(case, "to_mul", "basic_to_mul", fields, tiling_data, {"used_core_num": tiling_data["usedCoreNum"], "tile_num": tiling_data["tileNum"], "core_type": "AIV"}, {"base_mn": tiling_data["baseMN"], "base_k": tiling_data["baseK"], "loop_k": tiling_data["loopK"]}, {"kernel_impl": "MatMulToMulActKernel", "kernel_type": "AIV_ONLY"}, simulate_to_mul(case, tiling_data), ["Matches `MatMulV3ToMulTiling::DoOpTiling`."])


def analyze_basic_streamk(case: CaseInput) -> AnalysisResult:
    state = _base_state(case)
    m_cnt = ceil_div(case.m, state["base_m"])
    n_cnt = ceil_div(case.n, state["base_n"])
    total_mn = m_cnt * n_cnt
    if total_mn <= AIC_NUM // 2:
        if AIC_NUM // 3 < m_cnt < AIC_NUM // 2:
            m_cnt = AIC_NUM // 2
        if AIC_NUM // 3 < n_cnt < AIC_NUM // 2:
            n_cnt = AIC_NUM // 2
        total_mn = m_cnt * n_cnt
        state["base_m"] = ceil_align(ceil_div(case.m, m_cnt), BASIC_BLOCK_SIZE_16)
        state["base_n"] = ceil_align(ceil_div(case.n, n_cnt), BASIC_BLOCK_SIZE_16)
        state["k_tail_cnt"] = max(AIC_NUM // max(total_mn, 1), 1)
        state["single_core_k"] = ceil_div(case.k, state["k_tail_cnt"])
    else:
        state["k_tail_cnt"] = AIC_NUM // (total_mn % AIC_NUM)
        sk_single_core_k = ceil_div(case.k, max(state["k_tail_cnt"], 1))
        state["k_tail_cnt"] = ceil_div(case.k, sk_single_core_k)
        state["single_core_k"] = sk_single_core_k
    base_k_align = BASIC_BLOCK_SIZE_128 // DTYPE_SIZE[case.dtype] if (not case.transpose_x1 or case.transpose_x2) else BASIC_BLOCK_SIZE_16
    state["base_k"] = min(state["single_core_k"], floor_align(L0A_SIZE_2 // DB_SIZE // DTYPE_SIZE[case.dtype] // max(state["base_m"], state["base_n"]), base_k_align))
    _cal_l1_tiling(case, state, has_bias=case.has_bias)
    if state["base_m"] == state["base_n"] and state["depth_b1"] == state["depth_a1"] * 2:
        state["depth_a1"] *= 2
        state["depth_b1"] //= 2
        state["step_kb"] = state["depth_b1"] // DB_SIZE
        state["step_ka"] = state["depth_a1"] // DB_SIZE
    if total_mn > AIC_NUM and case.has_bias:
        state["step_ka"] = state["step_kb"] = 3
    l0c2out = _get_l0c2out_streamk(case)
    fields = make_key_fields(case, model=MAT_MUL_STREAM_K, full_load=MAT_MUL_NO_FULL_LOAD, l0c2out=l0c2out, api_level=MAT_MUL_BASIC_LEVEL)
    tiling_data = _fill_basic_tiling_data(case, state, AIC_NUM * BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256 * 4 + RPC_WORKSIZE_MB * 1024 * 1024)
    branch = "streamk_sk" if _check_streamk_sk(case) else "streamk_dpsk"
    return _build_result(case, "basic_streamk", branch, fields, tiling_data, {"used_core_num": tiling_data["usedCoreNum"], "m_tile_num": ceil_div(case.m, tiling_data["baseM"]), "n_tile_num": ceil_div(case.n, tiling_data["baseN"]), "sk_k_tile_num": ceil_div(case.k, tiling_data["skSingleCoreK"])}, {"base_m": tiling_data["baseM"], "base_n": tiling_data["baseN"], "base_k": tiling_data["baseK"], "m_l1": tiling_data["mL1"], "n_l1": tiling_data["nL1"], "k_l1": tiling_data["kL1"]}, {"kernel_impl": "MatMulStreamKActKernel", "kernel_type": "MIX_AIC_1_2" if l0c2out else "AIC_ONLY", "l0c2out": l0c2out}, simulate_basic_streamk(case, tiling_data), [f"Capability branch: {'CheckStreamKSKTiling' if branch == 'streamk_sk' else 'CheckStreamKDPSKTiling'}."])


def analyze_basic_aswt(case: CaseInput) -> AnalysisResult:
    state = _base_state(case)
    _reset_base(case, state)
    _formulate_basic_block(case, state)
    _optimize_edge_basic_block(case, state)
    _calc_tail_basic_block(case, state)
    _cal_l1_tiling(case, state, has_bias=case.has_bias)
    l0c2out = _get_l0c2out_aswt(case, state)
    full_load = MAT_MUL_NO_FULL_LOAD
    branch = "basic_aswt"
    notes = ["Initial branch follows `DoNormOpTiling`."]
    if _check_a_full_load(case, state, l0c2out):
        full_load = MAT_MUL_A_FULL_LOAD
        branch = "basic_aswt_a_full_load"
        notes.append("Matched `CheckAL1FullLoad`.")
    elif _check_b_full_load(case, state):
        full_load = MAT_MUL_B_FULL_LOAD
        branch = "basic_aswt_b_full_load"
        notes.append("Matched `CheckBL1FullLoad`.")
    elif l0c2out == MAT_MUL_ON_THE_FLY:
        _reset_base(case, state)
        _formulate_load_balance_block(case, state)
        if state["base_m"] == BASIC_BLOCK_SIZE_256 and state["base_n"] == BASIC_BLOCK_SIZE_256:
            _optimize_edge_basic_block(case, state)
        _calc_tail_basic_block(case, state)
        _cal_l1_tiling(case, state, has_bias=case.has_bias)
        remain = L1_SIZE - (BIAS_TABLE_NUM * DATA_SIZE_FP32 if case.has_bias else 0)
        state["step_ka"] = max(remain // 2 // ((state["base_m"] + state["base_n"]) * state["base_k"]) // DTYPE_SIZE[case.dtype], 1)
        state["step_kb"] = state["step_ka"]
        state["depth_a1"] = state["step_ka"] * DB_SIZE
        state["depth_b1"] = state["step_kb"] * DB_SIZE
        notes.append("Non-full-load ON_THE_FLY branch recomputed by `MatMulV3AswTiling::DoOpTiling`.")
    tiling_data = _fill_basic_tiling_data(case, state, RPC_WORKSIZE_MB * 1024 * 1024)
    fields = make_key_fields(case, model=MAT_MUL_BASIC, full_load=full_load, l0c2out=l0c2out, api_level=MAT_MUL_BASIC_LEVEL)
    kernel_impl = "MatMulFixpipeOptiActKernel" if l0c2out else "MatMulActKernel"
    return _build_result(case, "basic_aswt", branch, fields, tiling_data, {"used_core_num": tiling_data["usedCoreNum"], "m_tile_num": ceil_div(case.m, tiling_data["mL1"]), "n_tile_num": ceil_div(case.n, tiling_data["nL1"]), "tail_split": {"m_tail_cnt": tiling_data["mTailCnt"], "n_tail_cnt": tiling_data["nTailCnt"], "m_base_tail_split_cnt": tiling_data["mBaseTailSplitCnt"], "n_base_tail_split_cnt": tiling_data["nBaseTailSplitCnt"]}}, {"base_m": tiling_data["baseM"], "base_n": tiling_data["baseN"], "base_k": tiling_data["baseK"], "m_l1": tiling_data["mL1"], "n_l1": tiling_data["nL1"], "k_l1": tiling_data["kL1"], "step_ka": state["step_ka"], "step_kb": state["step_kb"]}, {"kernel_impl": kernel_impl, "kernel_type": "MIX_AIC_1_2" if l0c2out else "AIC_ONLY", "full_load": full_load, "l0c2out": l0c2out}, simulate_basic_aswt(case, tiling_data, {"full_load": full_load, "l0c2out": l0c2out}), notes)


def analyze_case(case: CaseInput) -> AnalysisResult:
    if case.k == 0 and not case.has_bias:
        return analyze_k_equal_zero(case)
    if case.is_force_group_acc and (case.m == 1 or case.n == 1) and case.dtype == "float32" and case.k >= TO_MUL_LIMIT_K:
        return analyze_to_mul(case)
    if case.x1_format == "ND" and AIV_NUM == AIC_NUM * 2 and (_check_streamk_sk(case) or _check_streamk_dpsk(case)):
        return analyze_basic_streamk(case)
    return analyze_basic_aswt(case)

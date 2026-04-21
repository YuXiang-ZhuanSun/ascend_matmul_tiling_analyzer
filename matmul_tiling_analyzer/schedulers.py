from .constants import AIV_NUM, WINDOW_LEN
from .utils import ceil_div


def _aswt_mn_tile(tile_idx: int, m_tile_num: int, n_tile_num: int) -> tuple[int, int, int]:
    main_window = min(WINDOW_LEN, m_tile_num)
    main_row = m_tile_num // main_window - 1
    tail_window = m_tile_num - main_row * main_window
    row_idx = tile_idx // n_tile_num // main_window
    if row_idx < main_row:
        m_tile_idx = row_idx * main_window + tile_idx % main_window
        n_tile_idx = (tile_idx // main_window) % n_tile_num
    else:
        row_idx = main_row
        tail_index = tile_idx - main_row * main_window * n_tile_num
        m_tile_idx = main_row * main_window + tail_index % tail_window
        n_tile_idx = (tail_index // tail_window) % n_tile_num
    if row_idx % 2 != 0:
        n_tile_idx = n_tile_num - 1 - n_tile_idx
    return m_tile_idx, n_tile_idx, row_idx


def _tail_shape(tile_idx: int, per_core_block_num: int, used_core_num: int, tail_cnt: int) -> bool:
    if tail_cnt == 1:
        return False
    return tile_idx // used_core_num == per_core_block_num - 1


def simulate_basic_aswt(case, tiling_data: dict, dispatch: dict) -> list[dict]:
    used_core_num = tiling_data["usedCoreNum"]
    m_l1 = tiling_data["mL1"]
    n_l1 = tiling_data["nL1"]
    m_tile_num = ceil_div(case.m, m_l1)
    n_tile_num = ceil_div(case.n, n_l1)
    tile_num = m_tile_num * n_tile_num
    tail_tile_num = tile_num % used_core_num
    m_tail_cnt = max(tiling_data["mTailCnt"], 1)
    n_tail_cnt = max(tiling_data["nTailCnt"], 1)
    tail_cnt = m_tail_cnt * n_tail_cnt
    effective_tile_num = tile_num + (tail_cnt - 1) * tail_tile_num if tail_tile_num else tile_num
    per_core_block_num = ceil_div(effective_tile_num, used_core_num)
    m_norm_cnt = m_tile_num - tiling_data["mBaseTailSplitCnt"]
    n_norm_cnt = n_tile_num - tiling_data["nBaseTailSplitCnt"]
    tail_l1_m = case.m - m_norm_cnt * m_l1
    tail_l1_n = case.n - n_norm_cnt * n_l1
    m_tail_main = tail_l1_m if tiling_data["mBaseTailSplitCnt"] == 1 else tiling_data["mTailMain"]
    n_tail_main = tail_l1_n if tiling_data["nBaseTailSplitCnt"] == 1 else tiling_data["nTailMain"]
    m_tail_last = tail_l1_m - (tiling_data["mBaseTailSplitCnt"] - 1) * m_tail_main
    n_tail_last = tail_l1_n - (tiling_data["nBaseTailSplitCnt"] - 1) * n_tail_main

    per_core = []
    for core in range(used_core_num):
        tasks = []
        for local_round in range(per_core_block_num):
            raw_idx = core + local_round * used_core_num
            if raw_idx >= effective_tile_num:
                continue
            tile_idx = raw_idx
            split_info = None
            if _tail_shape(raw_idx, per_core_block_num, used_core_num, tail_cnt):
                tile_idx = (per_core_block_num - 1) * used_core_num + core // tail_cnt
                split_info = {
                    "m_split_idx": (core % tail_cnt) % m_tail_cnt,
                    "n_split_idx": (core % tail_cnt) // m_tail_cnt,
                }
            logical_tile_idx = tile_idx % tile_num
            m_tile_idx, n_tile_idx, row_idx = _aswt_mn_tile(logical_tile_idx, m_tile_num, n_tile_num)
            blk_m = m_l1
            blk_n = n_l1
            if m_tile_idx >= m_norm_cnt:
                blk_m = m_tail_last if m_tile_idx == m_tile_num - 1 else m_tail_main
            if n_tile_idx >= n_norm_cnt:
                blk_n = n_tail_last if n_tile_idx == n_tile_num - 1 else n_tail_main
            m_offset = m_tile_idx * m_l1 if m_tile_idx <= m_norm_cnt else m_norm_cnt * m_l1 + (m_tile_idx - m_norm_cnt) * m_tail_main
            n_offset = n_tile_idx * n_l1 if n_tile_idx <= n_norm_cnt else n_norm_cnt * n_l1 + (n_tile_idx - n_norm_cnt) * n_tail_main
            if split_info is not None:
                split_blk_m = ceil_div(blk_m, m_tail_cnt)
                split_blk_n = ceil_div(blk_n, n_tail_cnt)
                m_split_offset = split_info["m_split_idx"] * split_blk_m
                n_split_offset = split_info["n_split_idx"] * split_blk_n
                if m_split_offset >= blk_m or n_split_offset >= blk_n:
                    continue
                blk_m = min(blk_m - m_split_offset, split_blk_m)
                blk_n = min(blk_n - n_split_offset, split_blk_n)
                m_offset += m_split_offset
                n_offset += n_split_offset
                split_info["m_split_offset"] = m_split_offset
                split_info["n_split_offset"] = n_split_offset
            tasks.append(
                {
                    "round": local_round,
                    "raw_tile_index": raw_idx,
                    "logical_tile_index": logical_tile_idx,
                    "row_idx": row_idx,
                    "m_tile_idx": m_tile_idx,
                    "n_tile_idx": n_tile_idx,
                    "m_range": [m_offset, m_offset + blk_m],
                    "n_range": [n_offset, n_offset + blk_n],
                    "k_range": [0, case.k],
                    "tile_shape": [blk_m, blk_n, case.k],
                    "split": split_info,
                }
            )
        per_core.append({"core_type": "AIC", "core_id": core, "tasks": tasks})
    return per_core


def simulate_basic_streamk(case, tiling_data: dict) -> list[dict]:
    used_core_num = tiling_data["usedCoreNum"]
    base_m = tiling_data["baseM"]
    base_n = tiling_data["baseN"]
    single_core_k = tiling_data["skSingleCoreK"]
    m_tile_num = ceil_div(case.m, base_m)
    n_tile_num = ceil_div(case.n, base_n)
    sk_k_tile_num = ceil_div(case.k, single_core_k)
    tail_mn_tile_num = (m_tile_num * n_tile_num) % used_core_num
    total_mn_dp = m_tile_num * n_tile_num - tail_mn_tile_num
    total_tile_num = total_mn_dp + tail_mn_tile_num * sk_k_tile_num

    per_core = []
    for core in range(used_core_num):
        tasks = []
        idx = core
        while idx < total_tile_num:
            is_sk = ceil_div(idx + 1, used_core_num) == ceil_div(total_tile_num, used_core_num)
            cur_k_tile_num = sk_k_tile_num if is_sk else 1
            if is_sk:
                k_tile_idx = (idx % used_core_num) % cur_k_tile_num
                mn_idx = (idx % used_core_num) // cur_k_tile_num + total_mn_dp
            else:
                k_tile_idx = 0
                mn_idx = idx
            m_tile_idx, n_tile_idx, row_idx = _aswt_mn_tile(mn_idx, m_tile_num, n_tile_num)
            m_offset = m_tile_idx * base_m
            n_offset = n_tile_idx * base_n
            k_offset = k_tile_idx * single_core_k
            tile_m = min(base_m, case.m - m_offset)
            tile_n = min(base_n, case.n - n_offset)
            tile_k = min(single_core_k if is_sk else case.k, case.k - k_offset)
            tasks.append(
                {
                    "global_tile_index": idx,
                    "scene": "SK" if is_sk else "DP",
                    "row_idx": row_idx,
                    "m_tile_idx": m_tile_idx,
                    "n_tile_idx": n_tile_idx,
                    "k_tile_idx": k_tile_idx,
                    "m_range": [m_offset, m_offset + tile_m],
                    "n_range": [n_offset, n_offset + tile_n],
                    "k_range": [k_offset, k_offset + tile_k],
                    "tile_shape": [tile_m, tile_n, tile_k],
                }
            )
            idx += used_core_num
        per_core.append({"core_type": "AIC", "core_id": core, "tasks": tasks})
    for core in range(AIV_NUM):
        per_core.append({"core_type": "AIV", "core_id": core, "tasks": []})
    return per_core


def simulate_to_mul(case, tiling_data: dict) -> list[dict]:
    used_core_num = tiling_data["usedCoreNum"]
    tile_num = tiling_data["tileNum"]
    per_core_round = ceil_div(tile_num, used_core_num)
    per_core = []
    for core in range(used_core_num):
        tasks = []
        for local_round in range(per_core_round):
            tile_idx = core + local_round * used_core_num
            if tile_idx >= tile_num:
                continue
            start = tile_idx * tiling_data["baseMN"]
            end = min(case.m * case.n, start + tiling_data["baseMN"])
            tasks.append(
                {
                    "round": local_round,
                    "tile_index": tile_idx,
                    "flat_mn_range": [start, end],
                    "base_mn": tiling_data["baseMN"],
                    "tail_mn": tiling_data["tailMN"],
                    "base_k": tiling_data["baseK"],
                    "tail_k": tiling_data["tailK"],
                    "loop_k": tiling_data["loopK"],
                    "data_copy_mode": tiling_data["dataCopyMode"],
                }
            )
        per_core.append({"core_type": "AIV", "core_id": core, "tasks": tasks})
    return per_core

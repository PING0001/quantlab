# -*- coding: utf-8 -*-
"""池成员资格（时点）单源模块（池时点化阶段 2，2026-08-24）。

数据源：pool_snapshots(effective_date, cutoff_date, code)——半年度快照
（pools/build_pool_history.py 产出）。语义：日期 d 的池 = 满足
eff <= d < next_eff 的档的成员集；d 早于首档 -> 空集。

消费端（训练/筛选/泄漏断言/回测基准/LIVE 报告/日更）一律经本模块取
成员，不再读 pools/*.json 的历史并集（并集宇宙已废）。
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH

_IVS = None   # 进程级缓存：[(eff, next_eff|None, frozenset)]


def _load(con: duckdb.DuckDBPyConnection | None = None):
    global _IVS
    if _IVS is not None:
        return _IVS
    own = con is None
    c = con or duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = c.execute(
            "SELECT effective_date, list(code) FROM pool_snapshots "
            "GROUP BY effective_date ORDER BY effective_date").fetchall()
    finally:
        if own:
            c.close()
    effs = [r[0] for r in rows]
    ivs = [(r[0], effs[i + 1] if i + 1 < len(rows) else None, frozenset(r[1]))
           for i, r in enumerate(rows)]
    _IVS = ivs
    return ivs


def refresh():
    """快照重建后清缓存。"""
    global _IVS
    _IVS = None


def _norm_date(d) -> str:
    if isinstance(d, str):
        return d[:10]
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def member_mask(dates, codes, con=None) -> np.ndarray:
    """逐行判定 (date, code) 是否当期池成员。向量化（merge），无 python 行循环。"""
    ivs = _load(con)
    effs = np.array([e for e, _, _ in ivs])
    d_str = pd.Series(dates).map(_norm_date).to_numpy()
    iv_id = np.searchsorted(effs, d_str, side="right") - 1
    valid = iv_id >= 0
    members = pd.DataFrame(
        {"iv": np.repeat(np.arange(len(ivs)), [len(s) for _, _, s in ivs]),
         "code": np.concatenate([sorted(s) for _, _, s in ivs]) if ivs else []})
    rows = pd.DataFrame({"iv": np.where(valid, iv_id, -1), "code": list(codes)})
    if members.empty:
        return np.zeros(len(rows), dtype=bool)
    hit = rows.merge(members.drop_duplicates(), on=["iv", "code"], how="left",
                     indicator=True)
    return (hit["_merge"] == "both").to_numpy()


def union_codes(since: str | None = None, con=None) -> list[str]:
    """since 起各档成员并集（数据加载用：标签 K 线、因子行）。"""
    ivs = _load(con)
    s = set()
    for eff, _, codes in ivs:
        if since is None or eff >= since:
            s |= codes
    return sorted(s)


def latest_codes(con=None) -> frozenset:
    return _load(con)[-1][2]


def codes_on(date, con=None) -> frozenset:
    d = _norm_date(date)
    for eff, nxt, codes in reversed(_load(con)):
        if d >= eff and (nxt is None or d < nxt):
            return codes
    return frozenset()


def reset_points(test_start, test_end, con=None) -> list[tuple[str, frozenset]]:
    """基准半年重置点（用户裁定 A）：测试首日 + 窗口内各生效日 ->
    [(date, members)] 升序。"""
    ts, te = _norm_date(test_start), _norm_date(test_end)
    pts = [(ts, codes_on(ts, con))]
    for eff, _, codes in _load(con):
        if ts < eff <= te:
            pts.append((eff, codes))
    return pts

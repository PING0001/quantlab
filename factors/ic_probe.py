# -*- coding: utf-8 -*-
"""IC 初探（bench mainboard_all 数据层验收件，2026-08-26）。

全主板池 vs 微盘池的公式因子 rank IC 对照：同一批 SELECTED_FACTORS、
同一标签构造（compute_median_open 20d, next_open 锚）、同一排除口径
（次日开盘封板 + 当日 IsST），仅横截面参考系不同（rank 因子在各自池内
分组计算）。评估窗 2020-01-01 起（用户裁定 2026-08-26）。

用法：
    QUANTLAB_POOL=mainboard_all python -m factors.ic_probe
    python -m factors.ic_probe            # 默认微盘
输出：终端对照表 + data/ic_probe_{pool}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, SELECTED_FACTORS, get_factor_table
from pools.membership import union_codes, member_mask
from strategies.labels import compute_median_open, compute_nextopen_limit_mask
from factors.select_factors import _rank_ic_np

EVAL_START = "2020-01-01"
LABEL_WINDOW = (16, 20)       # 20d 模型（主 horizon）
MIN_STOCKS_PER_DATE = 50      # 每日期最少样本（全主板截面宽，门槛水涨船高）


def load_pool_panel(con, pool: str, table: str) -> pd.DataFrame:
    # 先查表内可用列，SELECT 限定所需（省内存，缺失列显式跳过）
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table]).fetchall()}
    wanted = [f for f in dict.fromkeys(SELECTED_FACTORS + ["IsST"]) if f in cols]
    codes = union_codes(since=EVAL_START, con=con, pool=pool)
    ph = ",".join(["?"] * len(codes))
    col_sql = ", ".join(f'"{c}"' for c in wanted)
    df = con.execute(
        f"SELECT code, date, {col_sql} FROM {table} WHERE code IN ({ph})",
        codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"]).sort_index()


def main():
    print(f"Pool: {POOL_NAME} | label: median_open T+{LABEL_WINDOW[0]}..{LABEL_WINDOW[1]} "
          f"(next_open 锚) | eval since {EVAL_START}")
    table = get_factor_table()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Loading factor panel ...")
    fv = load_pool_panel(con, POOL_NAME, table)
    available = [f for f in SELECTED_FACTORS if f in fv.columns]
    missing = [f for f in SELECTED_FACTORS if f not in fv.columns]
    if missing:
        print(f"  WARNING: {len(missing)} factors missing from {table}: {missing[:8]}...")
    fv = fv[available]

    # 池时点化：仅当期档成员行（与 select_factors 同口径）
    mm = member_mask(fv.index.get_level_values("date"),
                     fv.index.get_level_values("code"), con=con, pool=POOL_NAME)
    fv = fv.loc[mm]

    print("Loading kline & labels ...")
    codes = sorted(fv.index.get_level_values("code").unique())
    ph = ",".join(["?"] * len(codes))
    # compute_median_open / compute_nextopen_limit_mask 均吃平铺列 kline
    kline = con.execute(
        f"SELECT code, date, open, close FROM daily_kline "
        f"WHERE code IN ({ph}) ORDER BY code, date", codes).fetchdf()
    kline["date"] = pd.to_datetime(kline["date"])
    labels = compute_median_open(kline, start_day=LABEL_WINDOW[0],
                                 end_day=LABEL_WINDOW[1], baseline="next_open")
    labels.name = "label"

    common = fv.index.intersection(labels.index)
    fv, labels = fv.loc[common], labels.loc[common]

    # 排除：标签 NaN + 当日 IsST + 次日开盘封板（系统 IC 口径）
    exclude = ~labels.notna()
    if "IsST" in fv.columns:
        exclude |= fv["IsST"].fillna(0).astype(bool)
    limit_mask = compute_nextopen_limit_mask(
        kline, st_series=fv["IsST"].fillna(0).astype(bool) if "IsST" in fv.columns else None)
    exclude |= limit_mask.reindex(labels.index).fillna(False)
    fv, labels = fv.loc[~exclude], labels.loc[~exclude]

    dates = fv.index.get_level_values("date")
    fv = fv.loc[dates >= pd.Timestamp(EVAL_START)]
    labels = labels.loc[fv.index]
    n_dates = fv.index.get_level_values("date").nunique()
    print(f"  Aligned: {len(fv)} rows, {n_dates} dates, {len(available)} factors")

    # 逐日期 rank IC
    print("Computing rank IC ...")
    dates_arr = fv.index.get_level_values("date").values
    uniq, start_idx, counts = np.unique(dates_arr, return_index=True, return_counts=True)
    F = fv[available].values.astype(np.float64)
    F[~np.isfinite(F)] = np.nan
    L = labels.values.astype(np.float64)
    n_factors = len(available)
    ic_mat = np.full((len(uniq), n_factors), np.nan)
    for i, (s, c) in enumerate(zip(start_idx, counts)):
        f_block, l_block = F[s:s + c], L[s:s + c]
        for j in range(n_factors):
            ic_mat[i, j] = _rank_ic_np(f_block[:, j], l_block)

    out = {}
    for j, name in enumerate(available):
        series = pd.Series(ic_mat[:, j], index=pd.DatetimeIndex(uniq)).dropna()
        if len(series) < 60:
            continue
        monthly = series.resample("ME").mean().dropna()
        out[name] = {
            "ic_mean": round(float(series.mean()), 4),
            "ic_std": round(float(series.std()), 4),
            "icir": round(float(series.mean() / series.std()), 3) if series.std() else None,
            "monthly_win_rate": round(float((monthly > 0).mean()), 3) if len(monthly) else None,
            "n_dates": int(len(series)),
        }

    payload = {
        "pool": POOL_NAME, "table": table, "eval_start": EVAL_START,
        "label": "median_open(16,20,next_open)", "n_rows": len(fv),
        "n_dates": n_dates, "factors": out,
    }
    out_path = Path(__file__).resolve().parent.parent / "data" / f"ic_probe_{POOL_NAME}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{'因子':32s} {'IC均值':>8s} {'ICIR':>7s} {'月胜率':>7s}")
    for name, st in sorted(out.items(), key=lambda kv: -abs(kv[1]["ic_mean"]))[:25]:
        print(f"{name:32s} {st['ic_mean']:8.4f} {st['icir'] or 0:7.3f} "
              f"{st['monthly_win_rate'] or 0:7.1%}")
    print(f"\n-> {out_path}  ({len(out)} factors)")
    con.close()


if __name__ == "__main__":
    main()

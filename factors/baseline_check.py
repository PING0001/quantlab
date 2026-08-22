# -*- coding: utf-8 -*-
"""
评估器回归门禁：8 个基准因子（factors/baseline_alphas.py）的固定窗口
rank IC 汇总，与冻结参考值（factors/baseline_reference.json）比对。

用途：评估口径（select_factors 的 IC 计算 / labels 前向收益 / 未来
factor foundry 准入逻辑）变更后，跑本工具确认"已知因子"的 IC 没有
非预期漂移。手动工具，不进每日流水线。

    python -m factors.baseline_check           # 比对模式，超限 exit 1
    python -m factors.baseline_check --init    # 冻结/刷新参考值

口径与 select_factors 主线一致：固定训练窗（2018-01-01 ~ 2025-06-01）、
逐日截面 Spearman rank IC、剔除 IsST=1 与退市填充(-1.0)与 NaN 标签行、
截面最少 30 只。标签为 close 锚 20d 前向收益（经典口径）。

repaint 注意：daily_kline 为前复权 VIEW，新除权事件会使历史 IC 缓慢
漂移——容差带（ic_mean ±0.01 / icir ±0.10）用于吸收；无评估器变更却
持续超限时，用 --init 重新冻结并记录原因。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes
from factors.baseline_alphas import compute_baseline_factors, BASELINE_FACTORS
from factors.select_factors import MIN_STOCKS_PER_DATE, _rank_ic_np
from strategies.labels import compute_forward_returns

TRAIN_START = "2018-01-01"
TRAIN_END = "2025-06-01"
HORIZON = 20
LOAD_FROM = "2017-09-01"   # 滚动窗预热（最长回看 ~30 交易日）
LOAD_TO = "2025-08-31"     # 标签需 T+20 收盘
IC_MEAN_TOL = 0.01
ICIR_TOL = 0.10

REF_PATH = Path(__file__).resolve().parent / "baseline_reference.json"


def load_kline_pd(con, codes):
    ph = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, close FROM daily_kline "
        f"WHERE code IN ({ph}) AND date >= '{LOAD_FROM}' AND date <= '{LOAD_TO}' "
        f"ORDER BY code, date", codes
    ).fetchdf()
    return df


def load_delist_info(con) -> dict:
    try:
        df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        return {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def load_isst(con, codes) -> pd.Series:
    ph = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, IsST FROM factor_values WHERE code IN ({ph})",
        codes,
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"])["IsST"]


def compute_metrics() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = get_pool_codes()

    # 行情（polars 面板，供因子计算）
    ph = ",".join(["?"] * len(codes))
    kdf = con.execute(
        f"SELECT code, date, open, high, low, close, volume FROM daily_kline "
        f"WHERE code IN ({ph}) AND date >= '{LOAD_FROM}' AND date <= '{LOAD_TO}' "
        f"ORDER BY code, date", codes
    ).fetchdf()
    kdf["date"] = kdf["date"].astype(str)
    panel = pl.from_pandas(kdf).rename({"code": "vt_symbol", "date": "datetime"})
    panel = panel.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap")
    )
    res = compute_baseline_factors(panel)

    # 标签（close 锚 20d，退市感知）
    kline_pd = load_kline_pd(con, codes)
    data_max_date = str(kline_pd["date"].max())
    delist_info = load_delist_info(con)
    fwd = compute_forward_returns(kline_pd, horizon=HORIZON, delist_info=delist_info)

    # IsST（factor_values 现值）
    isst = load_isst(con, codes)
    con.close()

    # 对齐到 (date, code) 索引
    fdf = res.rename({"vt_symbol": "code", "datetime": "date"}).to_pandas()
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf.set_index(["date", "code"]).sort_index()
    fdf = fdf.loc[~fdf.index.duplicated(keep="last")]

    common = fdf.index.intersection(fwd.index)
    fdf, fwd = fdf.loc[common], fwd.loc[common]

    st = isst.reindex(common).fillna(0).astype(bool)
    exclude = st | ~fwd.notna() | (fwd == -1.0)
    fdf, fwd = fdf.loc[~exclude], fwd.loc[~exclude]

    fdf = fdf.loc[(fdf.index.get_level_values("date") >= pd.Timestamp(TRAIN_START))
                  & (fdf.index.get_level_values("date") < pd.Timestamp(TRAIN_END))]

    dates = fdf.index.get_level_values("date").unique().sort_values()
    F = fdf[BASELINE_FACTORS].values.astype(np.float64)
    F[~np.isfinite(F)] = np.nan
    date_arr = fdf.index.get_level_values("date").values
    uniq, start_idx, counts = np.unique(date_arr, return_index=True, return_counts=True)
    label_arr = fwd.loc[fdf.index].values

    ics = {f: [] for f in BASELINE_FACTORS}
    f_idx = {f: i for i, f in enumerate(BASELINE_FACTORS)}
    for g in range(len(uniq)):
        s, c = start_idx[g], counts[g]
        if c < MIN_STOCKS_PER_DATE:
            continue
        l_vals = label_arr[s:s + c]
        f_block = F[s:s + c]
        for f in BASELINE_FACTORS:
            ic = _rank_ic_np(f_block[:, f_idx[f]], l_vals)
            if not np.isnan(ic):
                ics[f].append(ic)

    metrics = {}
    for f in BASELINE_FACTORS:
        arr = np.array(ics[f])
        mean, std = float(arr.mean()), float(arr.std())
        metrics[f] = {
            "ic_mean": mean,
            "ic_std": std,
            "icir": mean / std if std > 0 else float("nan"),
            "pos_rate": float((arr > 0).mean()),
            "n_dates": int(len(arr)),
        }
    metrics["_meta"] = {
        "pool": POOL_NAME,
        "train_window": [TRAIN_START, TRAIN_END],
        "horizon": HORIZON,
        "data_max_date": data_max_date,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluator regression gate")
    parser.add_argument("--init", action="store_true",
                        help="freeze/refresh the reference values")
    args = parser.parse_args()

    t0 = time.time()
    metrics = compute_metrics()

    if args.init:
        ref = {
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tolerances": {"ic_mean_abs": IC_MEAN_TOL, "icir_abs": ICIR_TOL},
            **metrics,
        }
        REF_PATH.write_text(json.dumps(ref, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"Reference frozen -> {REF_PATH}")
        for f in BASELINE_FACTORS:
            m = metrics[f]
            print(f"  {f:<12} ic_mean={m['ic_mean']:+.4f}  icir={m['icir']:+.3f}"
                  f"  pos={m['pos_rate']:.2f}  n={m['n_dates']}")
        return

    if not REF_PATH.exists():
        print(f"ERROR: {REF_PATH} not found. Run with --init first.")
        sys.exit(2)

    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    print(f"reference frozen at {ref.get('frozen_at')}, "
          f"data_max_date={ref.get('_meta', {}).get('data_max_date')}")
    print(f"current  data_max_date={metrics['_meta']['data_max_date']}")
    print(f"\n{'factor':<12} {'ic_mean':>16} {'icir':>18} {'n':>6}")
    ok = True
    for f in BASELINE_FACTORS:
        m, r = metrics[f], ref.get(f)
        if r is None:
            print(f"{f:<12}  MISSING IN REFERENCE")
            ok = False
            continue
        d_mean = m["ic_mean"] - r["ic_mean"]
        d_icir = m["icir"] - r["icir"]
        flag = ""
        if abs(d_mean) > IC_MEAN_TOL or abs(d_icir) > ICIR_TOL:
            flag = "  <-- BREACH"
            ok = False
        print(f"{f:<12} {m['ic_mean']:+.4f} ({d_mean:+.4f})"
              f" {m['icir']:+.3f} ({d_icir:+.3f}) {m['n_dates']:>6}{flag}")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""gb_gap1d：boost 族机器学习因子——XGBoost 预测 T+1 隔夜跳空 open(T+1)/close(T)−1
（2026-08-23 用户口述规格）。

命名约定（用户裁定）：gb_ 前缀 = gradient boosting 族（XGBoost/LightGBM），
mlp_ 前缀保留给深度学习族；列所有权归本脚本。

规格：
  输入  = 11 个公式因子（无原始 OHLCV 列）：
            Volatility(波动率) / WinnerRate(筹码) / Return_5d(动量) /
            Reversal_60d(反转) / GZ2000_return_5d(国证2000) /
            Turnover_3d_rank + Return_1d_rank(排序横截面) /
            Bollinger_width(布林带) + Stochastic_K(KDJ) /
            Gap_pct(当日跳空) + Intraday_return(当日日内收益)
  目标  = open(T+1) / close(T) − 1（后复权口径，除权日不失真；次日停牌样本剔除）
  模型  = XGBoost 回归（reg:squarederror，保守小树）
  管道  = expanding walk-forward OOF：2015 起，≥960 交易日出 OOS，每 120 交易日
          重训，训练尾段截 label_buffer=6（目标最远引用 T+1，1+5 安全边际）
  产出  = 池因子表（spec.factor_table）列 gb_gap1d（本脚本拥有列所有权；全历史重建覆盖旧值）

验收基准（过去实测）：最强单因子 Gap_pct 的 IC 0.089；gap1d 主模型
（35 因子 LightGBM）IC ~0.19；早期带 OHLCV 的迭代版（100 列 + 9 因子）
IC 0.1912（已归档删除）。

Usage:
    python -m factors.build_gb_gap1d                     # 默认池（env/微盘）
    python -m factors.build_gb_gap1d --pool mainboard_all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH
from pools.spec import get_pool, PoolSpec
from pools.membership import union_codes
from factors import store

TRAIN_START = "2015-01-01"
MIN_TRAIN_DAYS = 960
CADENCE = 120
LABEL_BUFFER = 6               # 目标最远引用 T+1，1+5 安全边际
COL = "gb_gap1d"
FORMULA_FACTORS = [
    "Volatility", "WinnerRate", "Return_5d", "Reversal_60d",
    "GZ2000_return_5d", "Turnover_3d_rank", "Return_1d_rank",
    "Bollinger_width", "Stochastic_K",
    "Gap_pct", "Intraday_return",              # 用户裁定加入
]

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=4, n_estimators=300,
    learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, tree_method="hist",
    random_state=42, n_jobs=-1,
)


def load_panel(spec: PoolSpec) -> tuple[pd.DataFrame, pd.Series]:
    # kline 仅用于构造目标：次行开盘 / 当日收盘 − 1（后复权）
    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = union_codes(con=con, pool=spec.name)   # 池时点化：快照全历史成员并集
    ph = ",".join(["?"] * len(codes))
    k = con.execute(
        f"SELECT code, date, open, close, adj_factor "
        f"FROM daily_raw WHERE code IN ({ph}) AND date >= ? ORDER BY code, date",
        [*codes, TRAIN_START]).fetchdf()
    k["date"] = pd.to_datetime(k["date"])
    for c in ("open", "close"):
        k[c] = k[c] * k["adj_factor"]
    g = k.groupby("code", sort=False)
    nxt_open = g["open"].shift(-1)
    kidx = pd.MultiIndex.from_arrays([k["code"], k["date"]], names=["code", "date"])
    y = pd.Series(nxt_open.to_numpy() / k["close"].to_numpy() - 1.0, index=kidx)
    y = y.where(nxt_open.notna().to_numpy())   # 次日停牌剔除
    y.name = "target"

    # 输入 = 11 个公式因子列，无原始 OHLCV（因子表 SQL 走 store 单点）
    fv = store.load_panel(con, spec, codes=codes, cols=FORMULA_FACTORS,
                          start=TRAIN_START)
    con.close()
    fv["date"] = pd.to_datetime(fv["date"])
    X = fv.set_index(["code", "date"]).sort_index()
    return X, y


def walk_forward_oof(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    dates = X.index.get_level_values("date")
    uniq = dates.unique().sort_values()
    n = len(uniq)
    preds_parts = []
    last = None
    i = MIN_TRAIN_DAYS
    while i < n:
        cut_ts = uniq[max(i - LABEL_BUFFER, 0)]
        seg_dates = set(uniq[i:i + CADENCE])
        seg_mask = dates.isin(seg_dates)
        train_mask = np.asarray(dates < cut_ts)
        yt = y.reindex(X.index)
        ok = train_mask & yt.notna().to_numpy()
        if ok.sum() > 10000:
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X.loc[ok].values, yt[ok].values)
            last = m
        if last is not None:
            Xs = X.loc[seg_mask]
            if len(Xs):
                preds_parts.append(pd.Series(last.predict(Xs.values), index=Xs.index))
        i += CADENCE
    return pd.concat(preds_parts) if preds_parts else pd.Series(dtype=float)


def main():
    ap = argparse.ArgumentParser(description="gb_gap1d 构建XGBoost 隔夜跳空因子")
    ap.add_argument("--pool", default=None,
                    help="目标池（默认 env QUANTLAB_POOL / 微盘）")
    args = ap.parse_args()
    spec = get_pool(args.pool)

    t0 = time.time()
    print(f"[{COL}] 池 {spec.name} 加载面板（{len(FORMULA_FACTORS)} 公式因子，无原始 OHLCV）...")
    X, y = load_panel(spec)
    print(f"  面板 {len(X):,} 行 × {X.shape[1]} 列；有效目标 {y.notna().sum():,} 行")

    preds = walk_forward_oof(X, y)
    print(f"  OOS 预测 {len(preds):,} 行，训练耗时 {(time.time() - t0) / 60:.1f} 分钟")

    # 评估：日均横截面 rank IC，剔除次日开盘 ±10% 板样本（近似口径，
    # 与 gap1d 主模型的过滤同族；ST ±5% 未细分）
    yv = y.reindex(preds.index)
    ratio_next_open = None
    ok = preds.notna() & yv.notna()
    df_ok = pd.DataFrame({"p": preds[ok], "y": yv[ok]})
    ics_all, ics_safe = [], []
    for _, ch in df_ok.groupby(level="date", sort=True):
        if len(ch) <= 30:
            continue
        ic = spearmanr(ch["p"].values, ch["y"].values).statistic
        if not np.isnan(ic):
            ics_all.append(ic)
        safe = ch[(ch["y"] > -0.095) & (ch["y"] < 0.095)]   # y 即跳空，±10% 板近似
        if len(safe) > 30:
            ic2 = spearmanr(safe["p"].values, safe["y"].values).statistic
            if not np.isnan(ic2):
                ics_safe.append(ic2)
    for tag, ics in (("全样本", ics_all), ("剔±10%板", ics_safe)):
        a = np.asarray(ics)
        print(f"  OOS rank IC[{tag}]: mean={a.mean():+.4f} "
              f"IC IR={a.mean() / a.std():.2f} t={a.mean() / (a.std() / np.sqrt(len(a))):.1f} "
              f"({len(a)} 日)")

    con_w = duckdb.connect(str(DB_PATH))
    s = preds.dropna().copy()
    s.name = "value"
    pdf = s.reset_index()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    pdf = pdf.rename(columns={"value": COL})
    n_match = store.update_columns(con_w, spec, pdf)
    con_w.execute("CHECKPOINT")
    con_w.close()
    print(f"  写入 {COL}@{spec.factor_table}: matched {n_match:,} rows | "
          f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()

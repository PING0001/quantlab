# -*- coding: utf-8 -*-
"""gb_30d_turn5d：boost 族机器学习因子——XGBoost 吃 30 日变换 OHLCV，
预测未来 5 日平均市值换手率（2026-08-31 用户口述规格）。

命名约定（用户裁定）：gb_ 前缀 = gradient boosting 族（XGBoost）；
{30d} = 输入窗口，{turn5d} = 预测目标；列所有权归本脚本。

规格：
  输入  = 30 日 × 5 变换 = 150 特征（t=1..30，1=最近一日；全在脚本内现算）：
            gap_t = 后复权 open[D-t+1] / 后复权 close[D-t] − 1（跳空百分比）
            hi_t  = high/open − 1（盘中最高相对开盘涨幅）
            lo_t  = low/open − 1（带负号，≤0）
            cl_t  = close/open − 1（日内收盘相对开盘）
            to_t  = amount / circ_mv / 10（市值换手率，项目换手率单源公式；
                    circ_mv 来自 daily_basic（单位万元），daily_raw.float_mv 全空不可用）
          同日内价比 (hi/lo/cl) 复权因子自动相消；跨日 gap 用 hfq。
          输入白名单 = 裸 OHLCV + circ_mv 市场数据，无公式因子、无模型输出（DAG 无环）。
  目标  = y = mean( to[D+1..D+5] )（未来 5 个交易日平均市值换手率；
          部分窗口 ≥3 日有效才取，否则 NaN）
  模型  = XGBoost 回归（超参指纹 = XGB_PARAMS，逐字复用 build_gb_gap1d/gb_4d_open2d）
  管道  = expanding walk-forward OOF：2015 起，≥960 交易日出 OOS，每 120 交易日
          重训，训练尾段截 label_buffer=10（目标最远引用 T+5，5+5 安全边际）
  产出  = 池因子表（spec.factor_table）列 gb_30d_turn5d（本脚本拥有列所有权；
          全历史重建覆盖旧值）

Usage:
    python -m factors.build_gb_30d_turn5d                     # 默认池（env/微盘）
    python -m factors.build_gb_30d_turn5d --pool mainboard_all
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
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, ROOT
from pools.spec import get_pool, PoolSpec
from pools.membership import union_codes
from factors import store

TRAIN_START = "2015-01-01"
WARMUP_START = "2014-10-01"    # 31 日窗 + 首段训练预热，早于 TRAIN_START
MIN_TRAIN_DAYS = 960
CADENCE = 120
LABEL_BUFFER = 10              # 目标最远引用 T+5，5+5 安全边际
FWD_DAYS = 5
WINDOW = 30
COL = "gb_30d_turn5d"
PREFIXES = ("gap", "hi", "lo", "cl", "to")
FEATURES = [f"{p}{t}" for p in PREFIXES for t in range(1, WINDOW + 1)]
TH_INCREMENTAL = 0.75          # <0.75 增量 / 0.75~0.95 边缘 / >0.95 冗余（用户标准）
TH_REDUNDANT = 0.95

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=4, n_estimators=300,
    learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, tree_method="hist",
    random_state=42, n_jobs=-1,
)


def load_panel(spec: PoolSpec) -> tuple[pd.DataFrame, pd.Series]:
    """30 日变换 OHLCV 特征（float32）+ 未来 5 日平均换手率目标。"""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = union_codes(con=con, pool=spec.name)   # 池时点化：快照全历史成员并集
    ph = ",".join(["?"] * len(codes))
    k = con.execute(
        f"SELECT r.code, r.date, r.open, r.high, r.low, r.close, r.amount, "
        f"r.adj_factor, b.circ_mv "
        f"FROM daily_raw r LEFT JOIN daily_basic b ON r.code = b.code AND r.date = b.date "
        f"WHERE r.code IN ({ph}) AND r.date >= ? ORDER BY r.code, r.date",
        [*codes, WARMUP_START]).fetchdf()
    con.close()
    k["date"] = pd.to_datetime(k["date"])
    for c in ("open", "high", "low", "close"):
        k[c] = k[c] * k["adj_factor"]              # hfq（跨日 gap 需要复权诚实）

    g = k.groupby("code", sort=False)
    k["gap"] = k["open"] / g["close"].shift(1) - 1.0
    k["hi"] = k["high"] / k["open"] - 1.0
    k["lo"] = k["low"] / k["open"] - 1.0
    k["cl"] = k["close"] / k["open"] - 1.0
    k["to"] = k["amount"] / k["circ_mv"] / 10.0

    # 目标：未来 5 个交易日平均换手率（停牌自然跳到下一交易日行；≥3 日有效）
    fwd = pd.concat([g["to"].shift(-d) for d in range(1, FWD_DAYS + 1)], axis=1)
    y = fwd.mean(axis=1).where(fwd.notna().sum(axis=1) >= 3)

    feats = {}
    for p in PREFIXES:
        for t in range(1, WINDOW + 1):
            feats[f"{p}{t}"] = g[p].shift(t - 1)
    kidx = pd.MultiIndex.from_arrays([k["code"], k["date"]], names=["code", "date"])
    # 必须用 numpy 值构造（位置对齐）：dict-of-Series + 显式 index 会按标签
    # 对齐，RangeIndex 对 MultiIndex 全 NaN（曾产出常数模型的实锤 bug）
    X = pd.DataFrame({n: s.to_numpy() for n, s in feats.items()},
                     index=kidx, dtype=np.float32).sort_index()
    del k, fwd, feats
    nan_rate = float(np.isnan(X.to_numpy()).mean())
    assert nan_rate < 0.5, f"FATAL 特征 NaN 率 {nan_rate:.1%}（面板构造错位）"
    y = pd.Series(y.to_numpy(), index=kidx, name="target").reindex(X.index)
    return X, y


def walk_forward_oof(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """expanding walk-forward OOF（协议与 build_gb_gap1d 逐字同构）。"""
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


def eval_ic(preds: pd.Series, y: pd.Series) -> pd.Series:
    """逐日截面 rank IC 序列（≥30 只/日）。"""
    yv = y.reindex(preds.index)
    ok = preds.notna().to_numpy() & yv.notna().to_numpy()
    df_ok = pd.DataFrame({"p": preds[ok].to_numpy(), "y": yv[ok].to_numpy(),
                          "date": preds.index.get_level_values("date")[ok]})
    ics = {}
    for d, ch in df_ok.groupby("date", sort=True):
        if len(ch) <= 30:
            continue
        ic = spearmanr(ch["p"].values, ch["y"].values).statistic
        if not np.isnan(ic):
            ics[d] = ic
    return pd.Series(ics, name="ic")


def eval_redundancy(spec: PoolSpec, preds: pd.Series) -> None:
    """对三主模型现役清单并集的逐日截面 rank 相关均值 → max|corr| 判定。"""
    incumbents = set()
    for m in ("open2d", "6d", "20d"):
        p = ROOT / "factors" / f"selected_{spec.name}_{m}.json"
        incumbents |= set(json.loads(p.read_text())["selected_factors"])
    incumbents = sorted(incumbents)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    fv = store.load_panel(con, spec, cols=incumbents,
                          start=str(preds.index.get_level_values("date").min())[:10])
    con.close()
    fv["date"] = pd.to_datetime(fv["date"])
    fv = fv.set_index(["code", "date"]).sort_index()
    j = fv.join(preds.rename("pred"), how="inner")
    rows = []
    for f in incumbents:
        if f not in j.columns:
            rows.append((f, np.nan, 0))
            continue
        per_day = j.groupby(level="date").apply(
            lambda d: d["pred"].rank().corr(d[f].rank()) if len(d) >= 30 else np.nan,
            include_groups=False).dropna()
        rows.append((f, float(per_day.mean()) if len(per_day) else np.nan, len(per_day)))
    print("\n  冗余门（vs 三清单并集逐日截面 rank 相关，报 |corr| 最高的前 10）：")
    best = 0.0
    shown = 0
    for f, c, n in sorted(rows, key=lambda r: -abs(r[1] if not np.isnan(r[1]) else 0)):
        if np.isnan(c) or shown >= 10:
            continue
        print(f"    {f:<24} mean_corr={c:+.3f}  ({n} 日)")
        best = max(best, abs(c))
        shown += 1
    verdict = ("增量" if best < TH_INCREMENTAL
               else "边缘" if best <= TH_REDUNDANT else "冗余")
    print(f"    max|corr| = {best:.3f} → {verdict}（{TH_INCREMENTAL}/{TH_REDUNDANT} 线）")


def main():
    ap = argparse.ArgumentParser(description="gb_30d_turn5d 构建：XGBoost 吃 30 日变换 OHLCV 预测未来 5 日平均换手率")
    ap.add_argument("--pool", default=None,
                    help="目标池（默认 env QUANTLAB_POOL / 微盘）")
    args = ap.parse_args()
    spec = get_pool(args.pool)

    t0 = time.time()
    print(f"[{COL}] 池 {spec.name} 加载 30 日变换 OHLCV 面板（{len(FEATURES)} 特征，自 {WARMUP_START} 预热）...")
    X, y = load_panel(spec)
    print(f"  面板 {len(X):,} 行 × {X.shape[1]} 列（float32）；有效目标 {y.notna().sum():,} 行")

    preds = walk_forward_oof(X, y)
    del X
    print(f"  OOS 预测 {len(preds):,} 行，训练耗时 {(time.time() - t0) / 60:.1f} 分钟")

    ics = eval_ic(preds, y)
    if ics.empty:
        sys.exit("FATAL 逐日 IC 全空（预测或目标异常），拒绝继续")
    a = ics.to_numpy()
    print(f"  OOS rank IC: mean={a.mean():+.4f} IC IR={a.mean() / a.std():.2f} "
          f"t={a.mean() / (a.std() / np.sqrt(len(a))):.1f} ({len(a)} 日, "
          f"{ics.index.min().date()} ~ {ics.index.max().date()})")
    print("  分年 mean IC:")
    for yr, seg in ics.groupby(ics.index.year):
        print(f"    {yr}: {seg.mean():+.4f}  ({len(seg)} 日)")

    eval_redundancy(spec, preds)

    con_w = duckdb.connect(str(DB_PATH))
    s = preds.dropna().copy()
    s.name = "value"
    pdf = s.reset_index()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    pdf = pdf.rename(columns={"value": COL})
    n_match = store.update_columns(con_w, spec, pdf)
    n_col = con_w.execute(
        f"SELECT count(*), sum(CASE WHEN date >= '2020-01-01' THEN 1 ELSE 0 END) "
        f"FROM {spec.factor_table} WHERE {COL} IS NOT NULL").fetchone()
    con_w.execute("CHECKPOINT")
    con_w.close()
    print(f"  写入 {COL}@{spec.factor_table}: matched {n_match:,} rows | "
          f"列非空 {n_col[0]:,}（2020+ {n_col[1]:,}）| "
          f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()

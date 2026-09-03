# -*- coding: utf-8 -*-
"""gb_4d_open2d：boost 族机器学习因子——XGBoost 吃 4 日裸 K（OHLCV）预测 open2d
主模型目标（2026-08-28 用户口述规格，计划已批准）。

命名约定（用户裁定）：gb_ 前缀 = gradient boosting 族；{4d} = 输入窗口，
{open2d} = 预测目标；列所有权归本脚本。

规格：
  输入  = 20 个裸 K 特征（本脚本从 daily_raw 现算，后复权 hfq 口径，永不重绘；
          无公式因子、无模型输出引用——DAG 无环）：
            o_t/h_t/l_t/c_t（t=1..4，1=最近一根）= 后复权价[D-t+1] / 后复权 close[D-t] - 1
            v_t（t=1..4）= volume[D-t+1] / volume 自身 20 日均值
              （MA20 min_periods=20，不足为 NaN；volume 不复权，比率自归一）
          XGBoost 原生容忍 NaN（次新/长停牌不杀行）
  目标  = compute_median_open(kline, 2, 2, "next_open") = open[T+2]/open[T+1] - 1
          （strategies/labels.py 单源，与 open2d 主模型训练目标逐位同构）
  模型  = XGBoost 回归（超参指纹 = XGB_PARAMS，逐字复用 build_gb_gap1d）
  管道  = expanding walk-forward OOF：2015 起，≥960 交易日出 OOS，每 120 交易日
          重训，训练尾段截 label_buffer=7（目标最远引用 T+2，2+5 安全边际）
  产出  = 池因子表（spec.factor_table）列 gb_4d_open2d（本脚本拥有列所有权；
          全历史重建覆盖旧值）

Usage:
    python -m factors.build_gb_4d_open2d                     # 默认池（env/微盘）
    python -m factors.build_gb_4d_open2d --pool mainboard_all
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
from strategies.labels import compute_median_open

TRAIN_START = "2015-01-01"
WARMUP_START = "2014-10-01"    # MA20 + 4 日窗预热，早于 TRAIN_START
MIN_TRAIN_DAYS = 960
CADENCE = 120
LABEL_BUFFER = 7               # 目标最远引用 T+2，2+5 安全边际
COL = "gb_4d_open2d"
K4_LAGS = (1, 2, 3, 4)
FEATURES = [f"{p}{t}" for p in ("o", "h", "l", "c", "v") for t in K4_LAGS]
TH_INCREMENTAL = 0.75          # <0.75 增量 / 0.75~0.95 边缘 / >0.95 冗余（用户标准）
TH_REDUNDANT = 0.95

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=4, n_estimators=300,
    learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, tree_method="hist",
    random_state=42, n_jobs=-1,
)


def load_panel(spec: PoolSpec) -> tuple[pd.DataFrame, pd.Series]:
    """裸 K 特征 + open2d 目标，(code, date) MultiIndex。

    特征全在脚本内现算（输入白名单 = 裸 hfq OHLCV，不读因子表任何列）；
    OHLC 一律后复权（raw × adj_factor），volume 不复权（比率自归一）。
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = union_codes(con=con, pool=spec.name)   # 池时点化：快照全历史成员并集
    ph = ",".join(["?"] * len(codes))
    k = con.execute(
        f"SELECT code, date, open, high, low, close, volume, adj_factor "
        f"FROM daily_raw WHERE code IN ({ph}) AND date >= ? ORDER BY code, date",
        [*codes, WARMUP_START]).fetchdf()
    con.close()
    k["date"] = pd.to_datetime(k["date"])
    for c in ("open", "high", "low", "close"):
        k[c] = k[c] * k["adj_factor"]

    g = k.groupby("code", sort=False)
    ma20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    for t in K4_LAGS:
        pc = g["close"].shift(t)                   # 该根 K 的昨收（分母）
        k[f"o{t}"] = g["open"].shift(t - 1) / pc - 1.0
        k[f"h{t}"] = g["high"].shift(t - 1) / pc - 1.0
        k[f"l{t}"] = g["low"].shift(t - 1) / pc - 1.0
        k[f"c{t}"] = g["close"].shift(t - 1) / pc - 1.0
        k[f"v{t}"] = g["volume"].shift(t - 1) / ma20.shift(t - 1)

    kidx = pd.MultiIndex.from_arrays([k["code"], k["date"]], names=["code", "date"])
    X = k.set_index(kidx)[FEATURES].sort_index()

    y = compute_median_open(k, start_day=2, end_day=2, baseline="next_open")
    y.index = y.index.swaplevel()                  # (date, code) → (code, date)
    y = y.reindex(X.index)
    y.name = "target"
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


def eval_ic(preds: pd.DataFrame, y: pd.Series) -> pd.Series:
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
    """对 open2d 现役清单（含 gb_gap1d）的逐日截面 rank 相关均值 → max|corr| 判定。"""
    list_path = ROOT / "factors" / f"selected_{spec.name}_open2d.json"
    incumbents = json.loads(list_path.read_text())["selected_factors"]
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
    print("\n  冗余门（对 open2d 现役清单逐日截面 rank 相关均值）：")
    best = 0.0
    for f, c, n in sorted(rows, key=lambda r: -abs(r[1] if not np.isnan(r[1]) else 0)):
        tag = "n/a" if np.isnan(c) else f"{c:+.3f}"
        print(f"    {f:<24} mean_corr={tag:>7}  ({n} 日)")
        if not np.isnan(c):
            best = max(best, abs(c))
    verdict = ("增量" if best < TH_INCREMENTAL
               else "边缘" if best <= TH_REDUNDANT else "冗余")
    print(f"    max|corr| = {best:.3f} → {verdict}（{TH_INCREMENTAL}/{TH_REDUNDANT} 线）")


def main():
    ap = argparse.ArgumentParser(description="gb_4d_open2d 构建：XGBoost 吃 4 日裸 K 预测 open2d")
    ap.add_argument("--pool", default=None,
                    help="目标池（默认 env QUANTLAB_POOL / 微盘）")
    args = ap.parse_args()
    spec = get_pool(args.pool)

    t0 = time.time()
    print(f"[{COL}] 池 {spec.name} 加载裸 K 面板（20 特征，自 {WARMUP_START} 预热）...")
    X, y = load_panel(spec)
    print(f"  面板 {len(X):,} 行 × {X.shape[1]} 列；有效目标 {y.notna().sum():,} 行")

    preds = walk_forward_oof(X, y)
    print(f"  OOS 预测 {len(preds):,} 行，训练耗时 {(time.time() - t0) / 60:.1f} 分钟")

    ics = eval_ic(preds, y)
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

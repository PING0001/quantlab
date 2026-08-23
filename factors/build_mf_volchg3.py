# -*- coding: utf-8 -*-
"""mf_volchg3：机器学习因子——XGBoost 预测未来 3 日波动率变化率（2026-08-23 用户口述规格）。

规格（忠实实现，勿自作主张加料）：
  输入  = 最近 10 个交易日的 OHLCV（后复权价格 + 原始量，含当日，50 列）
          + 现有因子 Volatility（20 日滚动日收益 std，factor_values 列，1 列）
  目标  = 未来 3 个交易日（T+1..T+3）的 Volatility 均值 ÷ 今日 Volatility − 1
          （变化率口径：惯性基线恒等于 0，模型所学即增量——2026-08-23 用户裁定，
          取代被 persistence 淹没的水平版 mf_vol3，旧列已归档删除）
  模型  = XGBoost 回归（reg:squarederror，小树保守参数）
  管道  = expanding 窗口 walk-forward OOF：训练起点 2015-01，≥960 交易日起
          出 OOS，每 120 交易日重训，训练尾段截 label_buffer=8 交易日
          （目标最远引用 T+3 的价格，8 = 3+5 安全边际）
  产出  = factor_values 新列 mf_volchg3（本脚本拥有列所有权；mf_ 前缀惯例）

Usage:
    python -m factors.build_mf_volchg3
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, get_pool_codes

TRAIN_START = "2015-01-01"
MIN_TRAIN_DAYS = 960          # ~4 年交易日起出 OOS
CADENCE = 120                 # 重训间隔（交易日）
LABEL_BUFFER = 8              # 目标最远引用 T+3，3+5 安全边际
N_LAGS = 10                   # 输入的 OHLCV 天数（含当日）
HORIZON = 3                   # 预测未来几日的波动率
COL = "mf_volchg3"

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=4, n_estimators=300,
    learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, tree_method="hist",
    random_state=42, n_jobs=-1,
)


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    """返回 (特征帧 X, 目标序列 y)，均为 (code,date) MultiIndex。"""
    codes = get_pool_codes()
    ph = ",".join(["?"] * len(codes))
    con = duckdb.connect(str(DB_PATH), read_only=True)

    k = con.execute(
        f"SELECT code, date, open, high, low, close, volume, adj_factor "
        f"FROM daily_raw WHERE code IN ({ph}) AND date >= ? ORDER BY code, date",
        [*codes, TRAIN_START]).fetchdf()
    k["date"] = pd.to_datetime(k["date"])
    for c in ("open", "high", "low", "close"):
        k[c] = k[c] * k["adj_factor"]          # 后复权：点时诚实、永不重绘
    con.close()

    g = k.groupby("code", sort=False)
    feats = {}
    for lag in range(N_LAGS):                  # lag 0 = 当日
        for c, short in (("open", "o"), ("high", "h"), ("low", "l"),
                         ("close", "c"), ("volume", "v")):
            feats[f"{short}{lag}"] = g[c].shift(lag)
    X = pd.DataFrame(feats)
    X.index = pd.MultiIndex.from_arrays([k["code"], k["date"]], names=["code", "date"])

    # 目标：Volatility（20d 滚动 std）在未来 HORIZON 个交易行的均值
    # （按个股自身交易日序列前移，停牌自然跳过；三日须齐才算有效训练样本）
    con = duckdb.connect(str(DB_PATH), read_only=True)
    fv = con.execute(
        f"SELECT code, date, Volatility FROM factor_values "
        f"WHERE code IN ({ph}) AND date >= ?", [*codes, TRAIN_START]).fetchdf()
    con.close()
    fv["date"] = pd.to_datetime(fv["date"])
    fv = fv.set_index(["code", "date"]).sort_index()
    vol = fv["Volatility"]
    gg = vol.groupby(level="code", sort=False)
    fut = pd.concat([gg.shift(-i) for i in range(1, HORIZON + 1)], axis=1)
    # 变化率口径：未来 HORIZON 日均值 ÷ 今日 − 1（惯性基线恒为 0）；
    # 三日须齐 + 今日波动率有效非零才算有效训练样本
    y = fut.mean(axis=1) / vol - 1.0
    y = y.where(fut.notna().all(axis=1) & (vol > 1e-12))
    y.name = "target"

    X = X.join(vol, how="inner")               # 输入并列 Volatility 列
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
    t0 = time.time()
    print(f"[{COL}] 加载面板（10d OHLCV + Volatility，{TRAIN_START} 起）...")
    X, y = load_panel()
    print(f"  面板 {len(X):,} 行 × {X.shape[1]} 列；有效目标 {y.notna().sum():,} 行")

    print(f"  walk-forward OOF（cadence={CADENCE}d, buffer={LABEL_BUFFER}d）...")
    preds = walk_forward_oof(X, y)
    print(f"  OOS 预测 {len(preds):,} 行，耗时 {(time.time() - t0) / 60:.1f} 分钟")

    yv = y.reindex(preds.index)
    ok = preds.notna() & yv.notna()
    pear = float(np.corrcoef(preds[ok], yv[ok])[0, 1])
    # 变化率口径下排序能力比相关系数更关键：补逐日横截面 rank IC 均值
    from scipy.stats import spearmanr
    df_ok = pd.DataFrame({"p": preds[ok], "y": yv[ok]})
    daily_ic = [spearmanr(ch["p"].values, ch["y"].values).statistic
                for _, ch in df_ok.groupby(level="date", sort=True)
                if len(ch) > 30]
    print(f"  OOS Pearson={pear:+.4f} (n={ok.sum():,})")
    ics = np.asarray(daily_ic)
    print(f"  OOS 日均横截面 rank IC={ics.mean():+.4f} "
          f"(IC IR={ics.mean() / ics.std():.2f}, t={ics.mean() / (ics.std() / np.sqrt(len(ics))):.1f}, "
          f"{len(ics)} 日)")
    yr = preds[ok].groupby(preds[ok].index.get_level_values("date").year).apply(
        lambda s: float(np.corrcoef(s, yv.reindex(s.index)[s.index])[0, 1]))
    print("  逐年 Pearson:")
    print(yr.to_string())

    # 写回 factor_values（mf_ 列所有权归本脚本；写前确认无并发写库进程）
    con_w = duckdb.connect(str(DB_PATH))
    con_w.execute(f"ALTER TABLE factor_values ADD COLUMN IF NOT EXISTS {COL} DOUBLE")
    s = preds.dropna().copy()
    s.name = "value"
    pdf = s.reset_index()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    con_w.execute("CREATE OR REPLACE TEMP TABLE _mf_upd AS SELECT * FROM pdf")
    n_match = con_w.execute("""
        SELECT COUNT(*) FROM _mf_upd p JOIN factor_values f
          ON f.code = p.code AND f.date = p.date""").fetchone()[0]
    con_w.execute(f"""
        UPDATE factor_values f SET {COL} = p.value
        FROM _mf_upd p WHERE f.code = p.code AND f.date = p.date""")
    con_w.execute("CHECKPOINT")
    con_w.close()
    print(f"  写入 {COL}: matched {n_match:,} rows | 总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()

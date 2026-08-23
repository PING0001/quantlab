# -*- coding: utf-8 -*-
"""nn_gap1d：神经网络因子——MLP 预测 T+1 隔夜跳空 open(T+1)/close(T)−1（2026-08-24 用户口述规格）。

规格：
  输入  = 最近 30 个交易日的 OHLCV，归一化为窗口比值（价格÷当日收盘−1、
          量÷30日均量−1，中心化到 0，150 列）
          + 12 个公式因子（归一化后并入，修复量纲混杂）：
            · 10 个个股级（Volatility/WinnerRate/Return_5d/Reversal_60d/
              Turnover_3d_rank/Return_1d_rank/Bollinger_width/Stochastic_K/
              Gap_pct/Intraday_return）→ 逐日横截面百分位排名 −0.5
            · GZ2000_return_5d（同日全池同值，横截面排名退化）→ 时序
              z-score（滚动 252 日，只用过去）
            · DaysToNextTrading（距下一交易日休市天数，同为同日同值）
              → 线性缩放 /10 −0.5
  训练段 = 面板起点 = 筹码类因子（WinnerRate）存在起点（运行时动态查询，
          约 2018 年初；kline 多载 60 自然日供 30 日滞后预热）
  目标  = open(T+1) / close(T) − 1（后复权；次日停牌剔除；训练时 ×100）
  模型  = sklearn MLPRegressor：(64,32) ReLU，Adam lr=1e-3，batch 40960，
          早停放宽（耐心 10/阈值 1e-5/上限 40 轮），L2 alpha=1e-4
  管道  = expanding walk-forward OOF：起点起 ≥960 交易日（约 4 年）出 OOS，
          每 120 交易日重训，训练尾段截 label_buffer=6
  缺失值 = 不填补，任一输入缺失的样本直接输出缺失（用户裁定：下游 Boost
          容忍 NaN）
  重建  = 列 nn_gap1d 直接 DROP+ADD 全量重建，不做备份（用户裁定：
          单因子实验以配方可复现为纪律，复现不了说明配方逻辑不对）

对照基准：gb_gap1d（XGBoost 11 因子）rank IC 0.198（OOS 2018-12 起）；
纯 20d OHLCV 版 0.178。注意本版 OOS 起点 ≈2021 年末，对比须用共同窗口。

Usage:
    python -m factors.build_nn_gap1d
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, get_pool_codes

MIN_TRAIN_DAYS = 960           # 从筹码起点起算（约 4 年）
CADENCE = 120
LABEL_BUFFER = 6               # 目标最远引用 T+1，1+5 安全边际
N_LAGS = 30
COL = "nn_gap1d"

STOCK_FACTORS = [              # 个股级：逐日横截面百分位排名 −0.5
    "Volatility", "WinnerRate", "Return_5d", "Reversal_60d",
    "Turnover_3d_rank", "Return_1d_rank",
    "Bollinger_width", "Stochastic_K", "Gap_pct", "Intraday_return",
]
MARKET_FACTOR = "GZ2000_return_5d"     # 同日同值：时序 z-score
CALENDAR_FACTOR = "DaysToNextTrading"  # 同日同值：/10 −0.5

MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    learning_rate_init=1e-3, batch_size=40960, alpha=1e-4,
    max_iter=40, early_stopping=True, validation_fraction=0.10,
    n_iter_no_change=10, tol=1e-5, random_state=42,
)


def load_panel() -> tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
    codes = get_pool_codes()
    ph = ",".join(["?"] * len(codes))
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # 训练段起点 = 筹码因子存在起点（用户裁定，运行时动态查询防硬编码漂移）
    chips_start = pd.Timestamp(con.execute(
        "SELECT min(date) FROM factor_values WHERE WinnerRate IS NOT NULL"
    ).fetchone()[0])
    load_from = (chips_start - pd.Timedelta(days=60)).strftime("%Y-%m-%d")  # 滞后预热

    k = con.execute(
        f"SELECT code, date, open, high, low, close, volume, adj_factor "
        f"FROM daily_raw WHERE code IN ({ph}) AND date >= ? ORDER BY code, date",
        [*codes, load_from]).fetchdf()
    k["date"] = pd.to_datetime(k["date"])
    for c in ("open", "high", "low", "close"):
        k[c] = k[c] * k["adj_factor"]          # 后复权：永不重绘

    # 公式因子（起点起）+ 归一化
    fv = con.execute(
        f"SELECT code, date, {', '.join(STOCK_FACTORS + [MARKET_FACTOR, CALENDAR_FACTOR])} "
        f"FROM factor_values WHERE code IN ({ph}) AND date >= ?",
        [*codes, chips_start.strftime("%Y-%m-%d")]).fetchdf()
    con.close()
    fv["date"] = pd.to_datetime(fv["date"])
    fv = fv.set_index(["code", "date"]).sort_index()

    # 个股级：逐日横截面百分位 −0.5（rank 保留 NaN → 缺失传导）
    fv[STOCK_FACTORS] = fv.groupby(level="date")[STOCK_FACTORS].rank(pct=True) - 0.5
    # 市场级：时序 z-score（滚动 252 日，只用过去，无前视）
    gz = fv.reset_index().drop_duplicates("date").set_index("date")[MARKET_FACTOR]
    m = gz.rolling(252, min_periods=60).mean()
    sd = gz.rolling(252, min_periods=60).std()
    z = (gz - m) / sd
    fv[MARKET_FACTOR] = fv.index.get_level_values("date").map(z)
    # 日历级：线性缩放
    fv[CALENDAR_FACTOR] = fv[CALENDAR_FACTOR] / 10.0 - 0.5

    idx_all = pd.MultiIndex.from_arrays([k["code"], k["date"]], names=["code", "date"])
    g = k.groupby("code", sort=False)

    close0 = k["close"]
    vol_mean = g["volume"].transform(lambda s: s.rolling(N_LAGS, min_periods=1).mean())
    feats = {}
    for lag in range(N_LAGS):
        for c, short in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")):
            feats[f"{short}{lag}"] = g[c].shift(lag) / close0 - 1.0
        feats[f"v{lag}"] = g["volume"].shift(lag) / (vol_mean + 1.0) - 1.0
    X = pd.DataFrame(feats)                   # 先按 RangeIndex 构建（直接传 index= 会触发对齐 → 全 NaN）
    X.index = idx_all
    X = X.astype(np.float32)

    # 面板裁到筹码起点（预热行丢弃）
    in_panel = np.asarray(idx_all.get_level_values("date") >= chips_start)
    X = X.loc[in_panel]

    # 目标：次行开盘 / 当日收盘 − 1
    nxt_open = g["open"].shift(-1)
    y = pd.Series(nxt_open.to_numpy() / k["close"].to_numpy() - 1.0, index=idx_all)
    y = y.where(nxt_open.notna().to_numpy() & in_panel)
    y.name = "target"

    X = X.join(fv, how="inner").astype(np.float32)
    return X, y, chips_start


def walk_forward_oof(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    dates = X.index.get_level_values("date")
    uniq = dates.unique().sort_values()
    n = len(uniq)
    preds_parts = []
    last = None
    i = MIN_TRAIN_DAYS
    # 目标 ×100（百分比单位）：原始目标 ~±0.01 的平方损失 ~1e-5 会瞬间触发
    # 早停容差，网络不学直接塌常数（历史教训）；预测除回
    Y_SCALE = 100.0
    feat_ok = X.notna().all(axis=1).to_numpy()   # 缺失样本直接输出缺失（不填补）
    seg_i = 0
    while i < n:
        cut_ts = uniq[max(i - LABEL_BUFFER, 0)]
        seg_dates = set(uniq[i:i + CADENCE])
        seg_mask = dates.isin(seg_dates)
        train_mask = np.asarray(dates < cut_ts)
        yt = y.reindex(X.index)
        ok = train_mask & feat_ok & yt.notna().to_numpy()
        if ok.sum() > 10000:
            m = MLPRegressor(**MLP_PARAMS)
            m.fit(X.loc[ok].values, (yt[ok].values * Y_SCALE))
            last = m
            print(f"    重训#{seg_i}: 训练行 {ok.sum():,}, 实际轮数 {m.n_iter_}")
        if last is not None:
            Xs = X.loc[seg_mask & feat_ok]
            if len(Xs):
                preds_parts.append(
                    pd.Series(last.predict(Xs.values) / Y_SCALE, index=Xs.index))
        i += CADENCE
        seg_i += 1
    return pd.concat(preds_parts) if preds_parts else pd.Series(dtype=float)


def main():
    t0 = time.time()
    print(f"[{COL}] 加载面板（{N_LAGS}d OHLCV 比值 + 12 归一化公式因子）...")
    X, y, chips_start = load_panel()
    print(f"  面板 {len(X):,} 行 × {X.shape[1]} 列；有效目标 {y.notna().sum():,} 行"
          f"；训练起点（筹码存在起点）={chips_start.date()}")

    preds = walk_forward_oof(X, y)
    print(f"  OOS 预测 {len(preds):,} 行，训练耗时 {(time.time() - t0) / 60:.1f} 分钟")

    yv = y.reindex(preds.index)
    ok = preds.notna() & yv.notna()
    df_ok = pd.DataFrame({"p": preds[ok], "y": yv[ok]})
    ics_all, ics_safe = [], []
    for _, ch in df_ok.groupby(level="date", sort=True):
        if len(ch) <= 30:
            continue
        ic = spearmanr(ch["p"].values, ch["y"].values).statistic
        if not np.isnan(ic):
            ics_all.append(ic)
        safe = ch[(ch["y"] > -0.095) & (ch["y"] < 0.095)]
        if len(safe) > 30:
            ic2 = spearmanr(safe["p"].values, safe["y"].values).statistic
            if not np.isnan(ic2):
                ics_safe.append(ic2)
    for tag, ics in (("全样本", ics_all), ("剔±10%板", ics_safe)):
        a = np.asarray(ics)
        print(f"  OOS rank IC[{tag}]: mean={a.mean():+.4f} "
              f"IC IR={a.mean() / a.std():.2f} t={a.mean() / (a.std() / np.sqrt(len(a))):.1f} "
              f"({len(a)} 日)")
    s_all = pd.Series(ics_all, index=sorted({d for d in df_ok.index.get_level_values("date")
                                              if len(df_ok.xs(d, level="date")) > 30}))
    yearly = s_all.groupby(s_all.index.year).agg(["mean", "count"])
    print("  逐年 rank IC（均值 / 天数）:")
    print(yearly.to_string())

    con_w = duckdb.connect(str(DB_PATH))
    con_w.execute(f"ALTER TABLE factor_values DROP COLUMN IF EXISTS {COL}")   # 不备份（用户裁定：配方可复现即纪律）
    con_w.execute(f"ALTER TABLE factor_values ADD COLUMN {COL} DOUBLE")
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

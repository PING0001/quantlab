# -*- coding: utf-8 -*-
"""模型因子构建管道（第2层，2026-08-23 全新构建，不复用 ai_gz2000 遗产）。

范式（见 factors/registry.py）：模型因子 = LightGBM walk-forward OOF 预测，
只吃公式因子 + 后复权 OHLCV（白名单强制校验），输出为 (code, date) 普通
列（mf_ 前缀，本脚本拥有列所有权），与公式因子同位入池，无结构特权。

纪律：expanding window、每 cadence 交易日重训、训练尾段截 label_buffer
防标签前视、超参指纹存注册表（冻结）。历史值逐位可复现（hfq 永不重绘）。

Usage:
    python -m factors.build_model_factors            # 构建全部已注册模型因子
    python -m factors.build_model_factors --only <mf_name>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, get_pool_codes
from factors.registry import MODEL_FACTORS, validate_entry

TRAIN_START = "2015-01-01"
MIN_TRAIN_DAYS = 960          # ~4 年交易日起 OOS
SAMPLE_CAP_PER_DAY = None     # 横截面全保留（池仅 ~1100 只）


# ---------------------------------------------------------------- targets
def build_targets(k_hfq: pd.DataFrame, k_amt: pd.DataFrame) -> dict[str, pd.Series]:
    """由后复权 K 线与成交额构造预测目标。返回 {(code,date) 索引: 目标值}。

    所有目标都引用未来数据——只能用于训练标签（walk-forward 会截 buffer）。
    """
    targets: dict[str, pd.Series] = {}

    # realized_vol_20d：T+1..T+20 的 hfq 日收益 std（年化前）
    ret = k_hfq.groupby("code", sort=False)["close_hfq"].pct_change()
    fut_vars = pd.concat(
        [ret.groupby(k_hfq["code"], sort=False).shift(-i) for i in range(1, 21)],
        axis=1)
    vol20 = fut_vars.std(axis=1)
    vol20.index = pd.MultiIndex.from_arrays(
        [k_hfq["code"], k_hfq["date"]], names=["code", "date"])
    targets["realized_vol_20d"] = vol20

    # volume_surprise_5d：T+1..T+5 总额 / (5 × 近 20 日均额) − 1
    amt = k_amt.set_index(pd.MultiIndex.from_arrays(
        [k_amt["code"], k_amt["date"]], names=["code", "date"]))["amount"]
    g = amt.groupby(level="code", sort=False)
    base20 = g.transform(lambda s: s.rolling(20).mean())
    fut_amt = pd.concat([g.shift(-i) for i in range(1, 6)], axis=1).sum(axis=1)
    surp = fut_amt / (5.0 * base20) - 1.0
    targets["volume_surprise_5d"] = surp
    return targets


# ---------------------------------------------------------------- features
def load_inputs(con: duckdb.DuckDBPyConnection, entry: dict
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (X 因子特征, K 后复权K线)，均为 (code,date) MultiIndex 帧。"""
    codes = get_pool_codes()
    ph = ",".join(["?"] * len(codes))

    fcols = entry["inputs"]["factors"]
    fv = con.execute(
        f"SELECT code, date, {', '.join(fcols)} FROM factor_values "
        f"WHERE code IN ({ph}) AND date >= ?", codes + [TRAIN_START]).fetchdf()
    fv["date"] = pd.to_datetime(fv["date"])
    fv = fv.set_index(["code", "date"]).sort_index()

    k = con.execute(
        f"SELECT code, date, open, high, low, close, volume, amount, adj_factor "
        f"FROM daily_raw WHERE code IN ({ph}) AND date >= ? ORDER BY code, date",
        codes + [TRAIN_START]).fetchdf()
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        k[f"{c}_hfq"] = k[c] * k["adj_factor"]

    # 白名单 OHLCV 派生特征（一律比值形态；hfq 水平时点诚实但横截面不可比，
    # 故不做水平特征）
    gcode = k["code"]
    cl = k["close_hfq"]
    feats = {}
    feats["ret_1d"] = cl.groupby(gcode, sort=False).pct_change(1)
    feats["ret_5d"] = cl.groupby(gcode, sort=False).pct_change(5)
    feats["ret_20d"] = cl.groupby(gcode, sort=False).pct_change(20)
    feats["gap_1d"] = k["open_hfq"] / cl.groupby(gcode, sort=False).shift(1) - 1
    rng = (k["high_hfq"] - k["low_hfq"]) / k["open_hfq"]
    feats["range_pct"] = rng
    feats["range_mean_20d"] = rng.groupby(gcode, sort=False).transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    feats["volume_ratio_20d"] = k["volume"] / k["volume"].groupby(
        gcode, sort=False).transform(lambda s: s.rolling(20, min_periods=5).mean())
    feats["amount_ratio_20d"] = k["amount"] / k["amount"].groupby(
        gcode, sort=False).transform(lambda s: s.rolling(20, min_periods=5).mean())

    ohlcv_df = pd.DataFrame(feats)
    ohlcv_df.index = pd.MultiIndex.from_arrays(
        [k["code"], k["date"]], names=["code", "date"])

    use_ohlcv = [o for o in entry["inputs"]["ohlcv"]]
    X = fv.join(ohlcv_df[use_ohlcv], how="left")
    return X, k


# ---------------------------------------------------------------- walk-forward
def walk_forward_predict(X: pd.DataFrame, y: pd.Series,
                         cadence: int, label_buffer: int, params: dict
                         ) -> pd.Series:
    """expanding window OOF：每 cadence 交易日重训，训练尾段截 label_buffer。
    返回 OOS 段 (code,date) -> 预测。"""
    dates = X.index.get_level_values("date")
    uniq = dates.unique().sort_values()
    n = len(uniq)
    preds_parts = []
    i = MIN_TRAIN_DAYS
    last = None
    while i < n:
        cut_ts = uniq[max(i - label_buffer, 0)]
        seg_dates = set(uniq[i:i + cadence])
        seg_mask = dates.isin(seg_dates)
        train_mask = dates < cut_ts
        if train_mask.sum() > 0:
            Xt = X.loc[train_mask]
            yt = y.reindex(Xt.index)
            ok = yt.notna()
            if ok.sum() > 10000:
                m = lgb.LGBMRegressor(**params)
                m.fit(Xt.loc[ok].values, yt[ok].values)
                last = m
        if last is not None:
            Xs = X.loc[seg_mask]
            if len(Xs):
                p = pd.Series(last.predict(Xs.values), index=Xs.index)
                preds_parts.append(p)
        i += cadence
    return pd.concat(preds_parts) if preds_parts else pd.Series(dtype=float)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(MODEL_FACTORS), default=None)
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    fv_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='factor_values'").fetchall()}

    todo = [args.only] if args.only else list(MODEL_FACTORS)
    results = {}
    for name in todo:
        entry = MODEL_FACTORS[name]
        errs = validate_entry(name, entry, fv_cols)
        if errs:
            print(f"[{name}] 白名单校验失败，拒绝构建：")
            for e in errs:
                print(f"  - {e}")
            continue

        t0 = time.time()
        X, k = load_inputs(con, entry)
        targets = build_targets(k[["code", "date", "open_hfq", "close_hfq"]],
                                k[["code", "date", "amount"]])
        y = targets[entry["target_key"]]
        common = X.index.intersection(y.dropna().index)
        y = y.loc[common]
        Xc = X.loc[common].replace([np.inf, -np.inf], np.nan)

        print(f"[{name}] target={entry['target_key']} rows={len(Xc)} "
              f"cadence={entry['cadence']}d buffer={entry['label_buffer']}d")
        preds = walk_forward_predict(Xc, y, entry["cadence"],
                                     entry["label_buffer"], entry["params"])
        # 目标自身的可预测性参照（OOS Pearson）
        yv = y.reindex(preds.index)
        ok = preds.notna() & yv.notna()
        pear = np.corrcoef(preds[ok], yv[ok])[0, 1]
        print(f"  OOS n={ok.sum()} 目标Pearson={pear:+.3f} 耗时{time.time()-t0:.0f}s")
        results[name] = preds

    con.close()
    if not results:
        return

    print("\n写入 factor_values（mf_ 列所有权归本脚本）...")
    con_w = duckdb.connect(str(DB_PATH))
    for name, preds in results.items():
        con_w.execute(f"ALTER TABLE factor_values ADD COLUMN IF NOT EXISTS {name} DOUBLE")
        s = preds.dropna().copy()
        s.name = "value"
        pdf = s.reset_index()
        pdf["date"] = pdf["date"].astype(str).str[:10]
        con_w.execute("CREATE OR REPLACE TEMP TABLE _mf_upd AS SELECT * FROM pdf")
        n_match = con_w.execute("""
            SELECT COUNT(*) FROM _mf_upd p JOIN factor_values f
              ON f.code = p.code AND f.date = p.date""").fetchone()[0]
        con_w.execute(f"""
            UPDATE factor_values f SET {name} = p.value
            FROM _mf_upd p WHERE f.code = p.code AND f.date = p.date""")
        print(f"  {name}: matched {n_match} rows")
    con_w.execute("CHECKPOINT")
    con_w.close()
    print("Done.")


if __name__ == "__main__":
    main()

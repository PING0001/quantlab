# -*- coding: utf-8 -*-
"""
AI Factor Builder — 训练 LightGBM 预测国证2000指数收益，输出作为因子。

Features: GZ2000 自身技术指标 + SHIBOR 利率
Targets: GZ2000 20d 前向收益 + median_5d 峰值中位数
Output: ai_gz2000_20d / ai_gz2000_median_5d 写入 factor_values
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

TEST_START = pd.Timestamp("2025-06-01")
TRAIN_START = pd.Timestamp("2020-01-01")

# GZ2000 index-level features
GZ2000_FEATURES = [
    "GZ2000_return_1d", "GZ2000_return_5d", "GZ2000_return_20d",
    "GZ2000_reversal_60d", "GZ2000_pricepos_252d",
    "GZ2000_atr_14d", "GZ2000_boll_width",
    "GZ2000_vol_10d", "GZ2000_vol_60d",
]

LGB_KWARGS = dict(
    num_leaves=31,
    max_depth=5,
    learning_rate=0.02,
    n_estimators=2000,
    min_child_samples=10,
    reg_alpha=1.0,
    reg_lambda=1.0,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    objective="regression_l1",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def load_daily_features(con) -> pd.DataFrame:
    """Load GZ2000 index-level features and SHIBOR, one row per date."""
    cols = ", ".join([f'MAX({c}) AS {c}' for c in GZ2000_FEATURES])
    df = con.execute(f"""
        SELECT date, {cols}
        FROM factor_values
        WHERE GZ2000_return_1d IS NOT NULL
        GROUP BY date ORDER BY date
    """).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # SHIBOR
    shibor = con.execute(
        "SELECT date, shibor_on, shibor_1m FROM macro_daily ORDER BY date"
    ).fetchdf()
    if not shibor.empty:
        shibor["date"] = pd.to_datetime(shibor["date"])
        shibor = shibor.set_index("date")
        df = df.join(shibor, how="left")

    return df


def load_gz2000_labels(con) -> pd.DataFrame:
    """Compute GZ2000 forward returns: 20d and median_5d."""
    df = con.execute(
        "SELECT date, close FROM index_daily WHERE code='399303' ORDER BY date"
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["gz2000_20d"] = df["close"].shift(-20) / df["close"] - 1

    # median_5d: median of T+16 ~ T+20 closes relative to T
    for i in range(1, 21):
        df[f"_f_{i}"] = df["close"].shift(-i) / df["close"] - 1
    fwd_cols = [f"_f_{i}" for i in range(16, 21)]
    df["gz2000_median_5d"] = df[fwd_cols].median(axis=1)

    return df[["gz2000_20d", "gz2000_median_5d"]].dropna()


def main():
    print("=" * 60)
    print("  AI Factor: LightGBM → GZ2000 收益预测")
    print(f"  Features: {GZ2000_FEATURES + ['shibor_on', 'shibor_1m']}")
    print("=" * 60)

    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("\n[1/3] Loading daily features ...")
    t0 = time.time()
    X = load_daily_features(con)
    print(f"  {len(X)} dates, {len(X.columns)} features")

    print("  Loading GZ2000 labels ...")
    labels = load_gz2000_labels(con)
    con.close()

    # Align
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    labels = labels.loc[common]
    print(f"  aligned: {len(X)} dates")
    print(f"  date range: {X.index[0].date()} ~ {X.index[-1].date()}")

    # Filter
    mask = X.index >= TRAIN_START
    X = X.loc[mask]
    labels = labels.loc[mask]

    X = X.fillna(0)

    horizons = ["gz2000_20d", "gz2000_median_5d"]
    print(f"\n[2/3] Training LGB models (train < {TEST_START.date()}) ...")

    train_mask = X.index < TEST_START
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]

    preds_all = pd.DataFrame(index=X.index)

    for h in horizons:
        print(f"  horizon {h} ...")
        y_train = labels.loc[X_train.index, h].values
        y_test = labels.loc[X_test.index, h].values

        model = lgb.LGBMRegressor(**LGB_KWARGS)
        model.fit(X_train, y_train)

        preds_all[h] = model.predict(X)

        train_corr = np.corrcoef(model.predict(X_train), y_train)[0, 1]
        test_corr = np.corrcoef(model.predict(X_test), y_test)[0, 1]
        print(f"    Pearson corr: train={train_corr:.4f}, test={test_corr:.4f}")

    # ---- broadcast ----
    print(f"\n[3/3] Broadcasting AI factors to factor_values ...")
    con_w = duckdb.connect(str(DB_PATH))

    con_w.execute("ALTER TABLE factor_values ADD COLUMN IF NOT EXISTS ai_gz2000_20d DOUBLE")
    con_w.execute("ALTER TABLE factor_values ADD COLUMN IF NOT EXISTS ai_gz2000_median_5d DOUBLE")

    preds_all = preds_all.reset_index()
    preds_all["date"] = preds_all["date"].astype(str)
    con_w.execute("CREATE OR REPLACE TEMP TABLE _ai_preds AS SELECT * FROM preds_all")

    con_w.execute("""
        UPDATE factor_values f SET 
            ai_gz2000_20d = (SELECT p.gz2000_20d FROM _ai_preds p WHERE p.date = f.date::DATE),
            ai_gz2000_median_5d = (SELECT p.gz2000_median_5d FROM _ai_preds p WHERE p.date = f.date::DATE)
        WHERE f.date::DATE IN (SELECT date::DATE FROM _ai_preds)
    """)

    n = con_w.execute("SELECT count(*) FROM factor_values WHERE ai_gz2000_20d IS NOT NULL").fetchone()[0]
    print(f"  Updated: {n} rows")
    con_w.execute("CHECKPOINT")
    con_w.close()

    elapsed = time.time() - t0
    print(f"\n  Done. ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

"""
Multi-horizon MLP walk-forward training.

Trains three separate models for 3d, 4d, and 5d forward returns,
persists each model + predictions, and reports per-horizon IC.

Usage:
    python run_mlp_multi.py
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path as P
from datetime import datetime

import duckdb
import pandas as pd

# --- project root ---
ROOT = P.cwd()
sys.path.insert(0, str(ROOT))

from strategies import MLPStrategy, walk_forward, rank_ic, pearson_ic, ic_summary
from strategies.labels import compute_forward_returns


SELECTED_FACTORS = [
    # Momentum (4)
    "Return_1d", "Return_5d", "Return_20d", "Reversal_60d",
    # Volatility (5)
    "ATR", "Volatility", "Volatility_60d", "Bollinger_width", "alpha060",
    # Price position / other (6)
    "Price_position_252d", "Stochastic_K", "Return_skew_20d",
    "Trend_strength", "SMA", "MACD_signal",
    # Pattern / intraday (3)
    "Gap_pct", "Body_pct", "Intraday_range_pct",
    # Volume / liquidity (2)
    "Volume_ratio", "Amihud_illiquidity",
    # Alpha composite (13)
    "alpha001", "alpha002", "alpha003", "alpha006", "alpha009",
    "alpha012", "alpha013", "alpha014", "alpha019", "alpha020",
    "alpha050", "alpha101", "alpha191",
]


DB_PATH = ROOT / "data" / "ashare.duckdb"
MODEL_DIR = ROOT / "models"

# --- config ---
TEST_START = pd.Timestamp("2025-05-01")
TEST_END   = pd.Timestamp("2026-06-26")
WARMUP_DAYS = 100
TRAIN_WINDOW = 252
MIN_TRAIN = 252

HORIZONS = [3, 4, 5, 10]
WEIGHTS = {3: 0.25, 4: 0.25, 5: 0.25, 10: 0.25}

MLP_KWARGS = dict(
    hidden_layer_sizes=(25, 12, 8),
    dropout=0.2,
    alpha=0.0001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    learning_rate=0.001,
    batch_size=16384*2,
    random_state=42,
)


def load_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load factor panel from DuckDB."""
    df = con.execute("SELECT * FROM factor_values").fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_kline(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load kline data for label computation."""
    return con.execute(
        "SELECT code, date, close FROM daily_kline ORDER BY code, date"
    ).fetchdf()


def main():
    print("=== Loading factor data (shared across horizons) ===")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("  loading factors ...")
    factors_raw = load_factors(con)

    print("  loading kline ...")
    kline = load_kline(con)
    con.close()

    # Restrict to the curated factors and drop NaN rows
    available = [f for f in SELECTED_FACTORS if f in factors_raw.columns]
    missing  = [f for f in SELECTED_FACTORS if f not in factors_raw.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected factors not in DB: {missing}")
    factor_cols = available
    factors_raw = factors_raw[factor_cols].copy()
    print(f"  factors: {len(factor_cols)} available: {factor_cols}")
    print(f"  full date range: {factors_raw.index.get_level_values('date').min()} ~ {factors_raw.index.get_level_values('date').max()}")
    print(f"  total stocks: {factors_raw.index.get_level_values('code').nunique()}")

    all_results = {}

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"=== Horizon = {horizon}d forward return ===")
        print(f"{'='*60}")
        t0 = time.time()

        # ---- labels ----
        print(f"  computing labels ({horizon}d forward return) ...")
        labels_raw = compute_forward_returns(kline, horizon=horizon)

        # ---- align ----
        common = factors_raw.index.intersection(labels_raw.index)
        X = factors_raw.loc[common]
        y = labels_raw.loc[common]

        # Drop NaN labels
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        print(f"  aligned samples: {len(X)}")

        # ---- strategy ----
        strategy = MLPStrategy(
            factor_names=factor_cols,
            **MLP_KWARGS,
        )
        print(f"  strategy: {strategy.name}, hidden={MLP_KWARGS['hidden_layer_sizes']}")

        # ---- walk-forward ----
        print(f"  walk-forward: train_window={TRAIN_WINDOW}, test={TEST_START.date()}~{TEST_END.date()}, warmup={WARMUP_DAYS}")
        preds = walk_forward(
            strategy,
            X, y,
            train_window=TRAIN_WINDOW,
            min_train=MIN_TRAIN,
            warmup_days=WARMUP_DAYS,
            test_start=TEST_START,
            test_end=TEST_END,
        )
        t_pred = time.time()
        n_pred = len(preds)
        n_dates = preds.index.get_level_values("date").nunique() if isinstance(preds.index, pd.MultiIndex) else (1 if n_pred > 0 else 0)
        print(f"  predictions: {n_pred} rows over {n_dates} dates ({t_pred - t0:.1f}s)")

        # ---- train-set IC (overfitting check) ----
        train_mask = X.index.get_level_values("date") < TEST_START
        preds_train = strategy.predict(X.loc[train_mask])
        ric_train = rank_ic(preds_train, y)
        s_train = ic_summary(ric_train)
        print(f"  train-set Rank IC: mean_ic={s_train['mean_ic']:.4f}, ir={s_train['ir']:.3f}, hit_rate={s_train['hit_rate']:.2%}, {s_train['n_periods']} dates")

        if n_pred > 0:
            ric_test = rank_ic(preds, y)
            pic_test = pearson_ic(preds, y)
            s_test = ic_summary(ric_test)
            p_test = ic_summary(pic_test)
            print(f"  test-set  Rank IC: mean_ic={s_test['mean_ic']:.4f}, ir={s_test['ir']:.3f}, hit_rate={s_test['hit_rate']:.2%}, {s_test['n_periods']} dates")
            print(f"  test-set Pearson IC: mean_ic={p_test['mean_ic']:.4f}, ir={p_test['ir']:.3f}, hit_rate={p_test['hit_rate']:.2%}, {p_test['n_periods']} dates")
            print(f"  IC gap (train - test): {s_train['mean_ic'] - s_test['mean_ic']:.4f}")
        else:
            s_test = {"n_periods": 0}
            p_test = {"n_periods": 0}
            print(f"  test-set: NO predictions (check data range vs min_train)")

        # ---- persist model ----
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / f"mlp_horizon{horizon}.pt"
        saved = strategy.save(model_path)
        print(f"  model saved: {saved}")

        # ---- persist predictions ----
        pred_path = ROOT / "data" / f"predictions_h{horizon}.parquet"
        pred_df = preds.to_frame("prediction")
        pred_df.to_parquet(pred_path)
        print(f"  predictions saved: {pred_path} ({len(pred_df)} rows)")

        # ---- collect results ----
        all_results[horizon] = {
            "model_path": str(model_path),
            "predictions_path": str(pred_path),
            "train_ic": s_train,
            "test_ic": s_test,
            "test_pearson_ic": p_test,
            "n_predictions": n_pred,
            "n_pred_dates": n_dates,
            "weight": WEIGHTS[horizon],
        }

    # ---- save combined meta ----
    meta = {
        "horizons": HORIZONS,
        "weights": WEIGHTS,
        "factor_names": factor_cols,
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "train_window": TRAIN_WINDOW,
        "mlp_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in MLP_KWARGS.items()},
        "results_per_horizon": {
            str(h): all_results[h] for h in HORIZONS
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    META_PATH = ROOT / "data" / "predictions_multi_meta.json"
    META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  combined meta saved: {META_PATH}")

    # ---- final summary ----
    print(f"\n{'='*60}")
    print(f"=== Summary ===")
    print(f"{'='*60}")
    for h in HORIZONS:
        r = all_results[h]
        st = r["test_ic"]
        print(f"  Horizon {h}d: train_ic={r['train_ic']['mean_ic']:.4f}, test_rank_ic={st.get('mean_ic', float('nan')):.4f}, test_pearson_ic={sp.get('mean_ic', float('nan')):.4f}, predictions={r['n_predictions']}")
    print("\nDone.")


if __name__ == "__main__":
    main()

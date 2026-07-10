"""
Multi-horizon MLP walk-forward training (single multi-head model).

Trains one MLP with shared backbone + per-horizon output heads for
1d / 3d / 5d / 10d forward returns.  Persists a single model file and
a single predictions parquet with all horizon columns.

Usage:
    python run_mlp_multi.py
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd

# --- project root ---
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ROOT, DB_PATH, POOL_NAME, get_pool_codes, get_model_dir, get_model_path, get_predictions_path, get_predictions_meta_path

from strategies import MLPStrategy, walk_forward, rank_ic, pearson_ic, ic_summary
from strategies.labels import compute_forward_returns, compute_nextopen_limit_mask


SELECTED_FACTORS = [
    # Momentum (3)
    "Return_5d", "Return_20d", "Reversal_60d",
    # Volatility (5)
    "ATR", "Volatility", "Volatility_60d", "Bollinger_width", "alpha060",
    # Price position / other (6)
    "Price_position_252d", "Stochastic_K", "Return_skew_20d",
    "Trend_strength", "SMA", "MACD_signal",
    # Pattern / intraday (3)
    "Gap_pct", "Body_pct", "Intraday_range_pct",
    # Volume / liquidity (2)
    "Volume_ratio", "Amihud_illiquidity",
    # Alpha composite (22)
    "alpha001", "alpha002", "alpha003", "alpha006", "alpha007",
    "alpha009", "alpha012", "alpha013", "alpha014", "alpha017",
    "alpha018", "alpha019", "alpha020", "alpha028", "alpha035",
    "alpha038", "alpha046", "alpha050", "alpha057",
    "alpha101", "alpha191",
    # Market cap / amount (3)
    "AvgAmount_90d", "LnMktCap", "LnFloatCap",
    # Turnover (2)
    "Turnover_3d", "Turnover_3d_ratio",
    # Intraday (1)
    "Intraday_return",
    # Market state (5)
    "CSI_return_1d", "CSI_return_20d", "CSI_volatility_20d",
    "HS300_return_1d", "HS300_return_20d",
    # Cross-sectional ranks (3)
    "Return_1d_rank",  "Return_20d_rank","Turnover_3d_rank",
    # ST status (1)
    "IsST",
    # Firm age (1)
    # "LnAge",
    # Chip / position cost (3)
    # "WinnerRate", "CostPosition", "ChipDispersion",
]


# --- config ---
TRAIN_START = pd.Timestamp("2015-01-01")
TEST_START = pd.Timestamp("2025-06-01")
TEST_END   = pd.Timestamp("2026-06-01")
WARMUP_DAYS = 90
TRAIN_WINDOW = 252
MIN_TRAIN = 252

HORIZONS = [3, 5, 10, 20]
WEIGHTS = {3: 0.25, 5: 0.25, 10: 0.25, 20: 0.25}

MLP_KWARGS = dict(
    hidden_layer_sizes=(48, 24,12),
    dropout=0.5,
    alpha=0.0001,
    early_stopping=True,
    validation_fraction=0.05,
    n_iter_no_change=20,
    learning_rate=0.001,
    batch_size=4096,
    random_state=42,
)


def load_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load factor panel from DuckDB, filtered to current pool's stock codes."""
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT * FROM factor_values WHERE code IN ({placeholders})"
    df = con.execute(query, pool_codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_kline(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load kline data for label computation, filtered to current pool."""
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date"
    return con.execute(query, pool_codes).fetchdf()


def load_delist_info(con: duckdb.DuckDBPyConnection) -> dict[str, pd.Timestamp]:
    try:
        df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        if df.empty:
            return {}
        return {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def main():
    print(f"=== Loading factor data (pool: {POOL_NAME}) ===")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("  loading factors ...")
    factors_raw = load_factors(con)

    print("  loading kline ...")
    kline = load_kline(con)

    print("  loading delist_info ...")
    delist_info = load_delist_info(con)
    print(f"  delisted stocks: {len(delist_info)}")
    con.close()

    # Restrict to the curated factors and drop NaN rows
    available = [f for f in SELECTED_FACTORS if f in factors_raw.columns]
    missing = [f for f in SELECTED_FACTORS if f not in factors_raw.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected factors not in DB: {missing}")
    factor_cols = available
    factors_raw = factors_raw[factor_cols].copy()
    print(f"  factors: {len(factor_cols)} available: {factor_cols}")
    print(f"  full date range: {factors_raw.index.get_level_values('date').min()} ~ {factors_raw.index.get_level_values('date').max()}")
    print(f"  total stocks: {factors_raw.index.get_level_values('code').nunique()}")

    t0 = time.time()

    # ---- labels: all horizons at once ----
    print(f"  computing labels for horizons {HORIZONS} ...")
    label_dfs = {}
    for h in HORIZONS:
        label_dfs[h] = compute_forward_returns(kline, horizon=h, delist_info=delist_info)
    labels_raw = pd.DataFrame({h: label_dfs[h] for h in HORIZONS})

    # ---- align ----
    common = factors_raw.index.intersection(labels_raw.index)
    X = factors_raw.loc[common]
    y = labels_raw.loc[common]

    # Drop rows where ANY horizon label is NaN
    mask = y.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]

    # Restrict to training window start
    date_level = X.index.get_level_values("date")
    mask = date_level >= TRAIN_START
    X, y = X.loc[mask], y.loc[mask]

    print(f"  aligned samples: {len(X)}")
    print(f"  date range: {date_level.min().date()} ~ {date_level.max().date()}")

    # ---- limit mask (for test IC filtering) ----
    st_series = factors_raw["IsST"].astype(bool) if "IsST" in factors_raw.columns else None
    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)
    print(f"  limit-hit predictions (next-open): {limit_mask.sum()}")

    # ---- strategy (multi-head) ----
    strategy = MLPStrategy(
        factor_names=factor_cols,
        horizons=tuple(HORIZONS),
        **MLP_KWARGS,
    )
    print(f"  strategy: {strategy.name}, horizons={HORIZONS}, hidden={MLP_KWARGS['hidden_layer_sizes']}")

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

    if isinstance(preds, pd.DataFrame) and not preds.empty:
        n_pred = len(preds)
        n_dates = preds.index.get_level_values("date").nunique() if isinstance(preds.index, pd.MultiIndex) else 1
        print(f"  predictions: {n_pred} rows over {n_dates} dates ({t_pred - t0:.1f}s)")
        print(f"  columns: {list(preds.columns)}")
    else:
        n_pred = 0
        n_dates = 0
        print(f"  predictions: NONE ({t_pred - t0:.1f}s)")

    # ---- train-set IC (overfitting check) ----
    train_mask = X.index.get_level_values("date") < TEST_START
    preds_train = strategy.predict(X.loc[train_mask])

    all_results = {}
    for h in HORIZONS:
        col = f"pred_{h}d"
        print(f"\n--- Horizon {h}d ---")
        ric_train = rank_ic(preds_train[col], y[h])
        s_train = ic_summary(ric_train)
        print(f"  train-set Rank IC: mean_ic={s_train['mean_ic']:.4f}, ir={s_train['ir']:.3f}, hit_rate={s_train['hit_rate']:.2%}, {s_train['n_periods']} dates")

        if n_pred > 0 and col in preds.columns:
            test_pred = preds[col]
            test_y = y.reindex(preds.index)[h]

            safe = ~limit_mask.reindex(preds.index, fill_value=False)
            n_limit = (~safe).sum()

            if st_series is not None:
                st_mask = st_series.reindex(preds.index, fill_value=False)
                safe = safe & ~st_mask
                n_st = st_mask.sum()
            else:
                n_st = 0

            ric_test = rank_ic(test_pred.loc[safe], test_y.loc[safe])
            pic_test = pearson_ic(test_pred.loc[safe], test_y.loc[safe])
            s_test = ic_summary(ric_test)
            p_test = ic_summary(pic_test)
            print(f"  test-set  Rank IC: mean_ic={s_test['mean_ic']:.4f}, ir={s_test['ir']:.3f}, hit_rate={s_test['hit_rate']:.2%}, {s_test['n_periods']} dates")
            print(f"  test-set Pearson IC: mean_ic={p_test['mean_ic']:.4f}, ir={p_test['ir']:.3f}, hit_rate={p_test['hit_rate']:.2%}, {p_test['n_periods']} dates")
            print(f"  test-set  excluded: {n_limit} limit-hit + {n_st} ST = {n_limit + n_st} observations")
            print(f"  IC gap (train - test): {s_train['mean_ic'] - s_test['mean_ic']:.4f}")
        else:
            s_test = {"n_periods": 0}
            p_test = {"n_periods": 0}

        all_results[h] = {
            "train_ic": s_train,
            "test_ic": s_test,
            "test_pearson_ic": p_test,
        }

    # ---- persist single model ----
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path()
    strategy.save(model_path)
    print(f"\n  model saved: {model_path}")

    # ---- persist single predictions parquet ----
    pred_path = get_predictions_path()
    if isinstance(preds, pd.DataFrame) and not preds.empty:
        preds.to_parquet(pred_path)
        print(f"  predictions saved: {pred_path} ({len(preds)} rows)")

    # ---- save combined meta ----
    meta = {
        "horizons": HORIZONS,
        "weights": WEIGHTS,
        "factor_names": factor_cols,
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "train_window": TRAIN_WINDOW,
        "mlp_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in MLP_KWARGS.items()},
        "results_per_horizon": {str(h): all_results[h] for h in HORIZONS},
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    META_PATH = get_predictions_meta_path()
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  meta saved: {META_PATH}")

    # ---- final summary ----
    print(f"\n{'=' * 60}")
    print("=== Summary ===")
    print(f"{'=' * 60}")
    for h in HORIZONS:
        r = all_results[h]
        st = r["test_ic"]
        print(f"  Horizon {h}d: train_ic={r['train_ic']['mean_ic']:.4f}, "
              f"test_rank_ic={st.get('mean_ic', float('nan')):.4f}")
    avg_train = sum(all_results[h]["train_ic"]["mean_ic"] for h in HORIZONS) / len(HORIZONS)
    avg_test = sum(
        all_results[h]["test_ic"].get("mean_ic", float("nan")) for h in HORIZONS
    ) / len(HORIZONS)
    print(f"  Avg across horizons: train_ic={avg_train:.4f}, test_rank_ic={avg_test:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

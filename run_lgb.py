"""
Multi-horizon LightGBM walk-forward training.

Trains one LGBMRegressor per horizon (5d / 10d / 20d / 30d).

Usage:
    python run_lgb.py
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH, POOL_NAME, get_pool_codes, SELECTED_FACTORS, get_lgb_model_path, get_lgb_predictions_path, get_lgb_predictions_meta_path

from strategies import LGBStrategy, walk_forward, rank_ic, pearson_ic, ic_summary
from strategies.labels import compute_forward_returns, compute_median_close, compute_nextopen_limit_mask


# --- config ---
TRAIN_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2025-06-01")
TEST_END   = pd.Timestamp("2026-06-01")
WARMUP_DAYS = 90
TRAIN_WINDOW = 252
MIN_TRAIN = 252

HORIZONS = ['median_5d', 20]
WEIGHTS = {'median_5d': 0.50, 20: 0.50}

LGB_KWARGS = dict(
    num_leaves=8,
    max_depth=4,
    learning_rate=0.02,
    n_estimators=3000,
    min_child_samples=2000,
    reg_alpha=10.0,
    reg_lambda=10.0,
    subsample=0.6,
    subsample_freq=1,
    colsample_bytree=0.3,
    objective="regression_l1",
    categorical_feature=["sw_l3"],
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=50,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def load_industry_sw_l3(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Load SW L3 codes for all stocks, factorize into integer categories."""
    df = con.execute(
        "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pd.Series(dtype=int)
    codes, uniques = pd.factorize(df["sw_l3_code"])
    return pd.Series(codes, index=df["code"], name="sw_l3")


def load_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT * FROM factor_values WHERE code IN ({placeholders})"
    df = con.execute(query, pool_codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_kline(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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

    print("  loading industry sw_l3 ...")
    industry_sw_l3 = load_industry_sw_l3(con)
    print(f"  industry categories: {industry_sw_l3.nunique()}")

    con.close()

    selected_path = Path(__file__).resolve().parent / "factors" / f"selected_{POOL_NAME}.json"
    if selected_path.exists():
        selected_data = json.loads(selected_path.read_text())
        use_factors = selected_data["selected_factors"]
        print(f"  Using {len(use_factors)} pre-selected factors from {selected_path.name}")
    else:
        use_factors = SELECTED_FACTORS

    available = [f for f in use_factors if f in factors_raw.columns]
    missing = [f for f in use_factors if f not in factors_raw.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected factors not in DB: {missing}")
    factor_cols = available
    factors_raw = factors_raw[factor_cols].copy()
    
    # ---- add sw_l3 categorical feature ----
    if not industry_sw_l3.empty:
        idx_codes = factors_raw.index.get_level_values("code")
        factors_raw["sw_l3"] = idx_codes.map(industry_sw_l3).fillna(-1).astype(int)
        if "sw_l3" not in factor_cols:
            factor_cols = factor_cols + ["sw_l3"]
        n_cats = industry_sw_l3.nunique()
        print(f"  sw_l3 categorical: {n_cats} categories")
    else:
        print(f"  sw_l3 categorical: not available")

    print(f"  factors: {len(factor_cols)} available")
    print(f"  full date range: {factors_raw.index.get_level_values('date').min()} ~ {factors_raw.index.get_level_values('date').max()}")
    print(f"  total stocks: {factors_raw.index.get_level_values('code').nunique()}")

    t0 = time.time()

    # ---- labels ----
    print(f"  computing labels for horizons {HORIZONS} ...")
    label_dfs = {}
    label_dfs['median_5d'] = compute_median_close(kline, start_day=16, end_day=20, delist_info=delist_info)
    label_dfs[20] = compute_forward_returns(kline, horizon=20, delist_info=delist_info)
    labels_raw = pd.DataFrame({h: label_dfs[h] for h in HORIZONS})

    # ---- align ----
    common = factors_raw.index.intersection(labels_raw.index)
    X = factors_raw.loc[common]
    y = labels_raw.loc[common]

    mask = y.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]

    date_level = X.index.get_level_values("date")
    mask = date_level >= TRAIN_START
    X, y = X.loc[mask], y.loc[mask]

    print(f"  aligned samples: {len(X)}")
    print(f"  date range: {date_level.min().date()} ~ {date_level.max().date()}")

    # ---- exclude ST + delisted observations from training ----
    st_series = factors_raw["IsST"].astype(bool) if "IsST" in factors_raw.columns else None
    if st_series is not None:
        st_mask = st_series.reindex(X.index, fill_value=False)
    else:
        st_mask = pd.Series(False, index=X.index)
    idx_date = X.index.get_level_values("date")
    idx_code = X.index.get_level_values("code")
    delist_series = pd.Series(delist_info)
    delist_dates = idx_code.map(delist_series)
    delist_mask = (idx_date >= delist_dates.values)
    delist_mask = pd.Series(delist_mask, index=X.index).fillna(False)
    
    # ---- limit mask (for training and test filtering) ----
    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)
    limit_mask = limit_mask.reindex(X.index, fill_value=False)
    print(f"  limit-hit predictions (next-open): {limit_mask.sum()}")

    exclude = st_mask | delist_mask | limit_mask
    if exclude.any():
        X, y = X.loc[~exclude], y.loc[~exclude]
        print(f"  excluded from training: {exclude.sum()} ST/delist/limit observations")
    else:
        print(f"  excluded from training: 0 ST/delist/limit observations")

    # ---- strategy ----
    strategy = LGBStrategy(
        factor_names=factor_cols,
        horizons=tuple(HORIZONS),
        **LGB_KWARGS,
    )
    print(f"  strategy: {strategy.name}, horizons={HORIZONS}")

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
    else:
        n_pred = 0
        n_dates = 0
        print(f"  predictions: NONE ({t_pred - t0:.1f}s)")

    # ---- train-set IC ----
    train_mask = X.index.get_level_values("date") < TEST_START
    preds_train = strategy.predict(X.loc[train_mask])

    def _pred_col(h):
        if isinstance(h, str):
            return f"pred_{h}"
        return f"pred_{h}d"

    def _h_label(h):
        if isinstance(h, str):
            return h
        return f"{h}d"

    all_results = {}
    for h in HORIZONS:
        col = _pred_col(h)
        h_label = _h_label(h)
        print(f"\n--- Horizon {h_label} ---")
        ric_train = rank_ic(preds_train[col], y[h])
        s_train = ic_summary(ric_train)
        if s_train.get("n_periods", 0) > 0:
            print(f"  train-set Rank IC: mean_ic={s_train['mean_ic']:.4f}, ir={s_train['ir']:.3f}, hit_rate={s_train['hit_rate']:.2%}, {s_train['n_periods']} dates")
        else:
            print(f"  train-set Rank IC: NO DATA")

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

    # ---- persist model ----
    model_path = get_lgb_model_path()
    strategy.save(model_path)
    print(f"\n  model saved: {model_path}")

    # ---- persist predictions ----
    pred_path = get_lgb_predictions_path()
    if isinstance(preds, pd.DataFrame) and not preds.empty:
        preds.to_parquet(pred_path)
        print(f"  predictions saved: {pred_path} ({len(preds)} rows)")

    # ---- save meta ----
    meta = {
        "model": "LightGBM",
        "horizons": HORIZONS,
        "weights": WEIGHTS,
        "factor_names": factor_cols,
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "train_window": TRAIN_WINDOW,
        "lgb_kwargs": LGB_KWARGS,
        "results_per_horizon": {str(h): all_results[h] for h in HORIZONS},
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = get_lgb_predictions_meta_path()
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  meta saved: {meta_path}")

    # ---- final summary ----
    print(f"\n{'=' * 60}")
    print("=== Summary ===")
    print(f"{'=' * 60}")
    for h in HORIZONS:
        r = all_results[h]
        st = r["test_ic"]
        h_label = _h_label(h)
        print(f"  Horizon {h_label}: train_ic={r['train_ic']['mean_ic']:.4f}, "
              f"test_rank_ic={st.get('mean_ic', float('nan')):.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

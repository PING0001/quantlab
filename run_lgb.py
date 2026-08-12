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
import numpy as np
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

HORIZONS = ['label']
WEIGHTS = {'label': 1.0}

LGB_KWARGS = dict(
    num_leaves=8,
    max_depth=4,
    learning_rate=0.05,
    n_estimators=3000,
    min_child_samples=2000,
    reg_alpha=3.0,
    reg_lambda=3.0,
    subsample=0.6,
    subsample_freq=1,
    colsample_bytree=0.3,
    model_type="classifier",
    categorical_feature=["sw_l3"],
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=50,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def load_industry_sw_l3(con: duckdb.DuckDBPyConnection) -> tuple[pd.Series, dict[str, int]]:
    """Load SW L3 codes for all stocks, encode into deterministic integers.

    Returns (series, mapping) where series maps stock_code → integer,
    and mapping is {sw_l3_code: integer} with sorted codes for reproducibility.
    """
    df = con.execute(
        "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pd.Series(dtype=int), {}

    categories = sorted(df["sw_l3_code"].astype(str).unique())
    mapping = {code: i for i, code in enumerate(categories)}
    codes = df["sw_l3_code"].map(mapping).fillna(-1).astype(int)
    return pd.Series(codes.values, index=df["code"], name="sw_l3"), mapping


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
    industry_sw_l3, sw_l3_mapping = load_industry_sw_l3(con)
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
    # IsST may not be in the selected-factors JSON; capture it before column
    # pruning so ST exclusion and the next-open limit mask still work.
    st_series = factors_raw["IsST"].astype(bool) if "IsST" in factors_raw.columns else None
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
    # 分类目标：T+16~T+20 中位数收盘 vs T日收盘
    #   >= +8% → 1,  <= -4% → -1,  否则 → 0
    print(f"  computing labels: median_5d close return (T+16~T+20) ...")
    median_ret = compute_median_close(kline, start_day=16, end_day=20, delist_info=delist_info)
    
    label_20d = compute_forward_returns(kline, horizon=20, delist_info=delist_info)

    def _classify(ret):
        if ret >= 0.08:
            return 1
        if ret <= -0.04:
            return -1
        return 0

    # 分类标签
    y_class = median_ret.apply(_classify)
    y_class.name = "label"

    # 同时保留连续值用于 IC 参考
    labels_raw = pd.DataFrame({"label": y_class, "ret_20d": label_20d, "ret_median": median_ret})

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
    # st_series captured before column pruning above (it may be absent from
    # the selected-factors JSON and would otherwise be silently dropped).
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
    if sw_l3_mapping:
        strategy._category_mappings["sw_l3"] = sw_l3_mapping
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

    # ---- evaluation ----
    all_results = {}
    h = 'label'
    col = f"pred_{h}"
    print(f"\n--- Classification (label: >=8%→+1, <=-4%→-1, else→0) ---")

    # train classification accuracy
    train_mask = X.index.get_level_values("date") < TEST_START
    preds_train = strategy.predict(X.loc[train_mask])
    train_pred = preds_train[col].values
    train_true = y.loc[train_mask, h].values

    train_pred_class = np.sign(train_pred).astype(int)
    train_acc = np.mean(train_pred_class == train_true)
    print(f"  train-set accuracy: {train_acc:.4f}")
    print(f"  train label dist: +1={sum(train_true==1)}, 0={sum(train_true==0)}, -1={sum(train_true==-1)}")

    if n_pred > 0 and col in preds.columns:
        test_pred = preds[col]
        test_true = y.reindex(preds.index)[h]

        safe = ~limit_mask.reindex(preds.index, fill_value=False)
        if st_series is not None:
            safe = safe & ~st_series.reindex(preds.index, fill_value=False)
        n_excl = (~safe).sum()

        tp = test_pred.loc[safe].values
        tt = test_true.loc[safe].values

        test_pred_class = np.sign(tp).astype(int)
        test_acc = np.mean(test_pred_class == tt)
        print(f"\n  test-set  accuracy: {test_acc:.4f}")
        print(f"  test label dist: +1={sum(tt==1)}, 0={sum(tt==0)}, -1={sum(tt==-1)}")
        print(f"  excluded: {n_excl} obs")

        # Rank IC vs continuous median return (reference)
        ret_median = y.reindex(preds.index)["ret_median"]
        ric = rank_ic(test_pred.loc[safe], ret_median.loc[safe])
        s = ic_summary(ric)
        print(f"  ref IR vs ret_median: mean_ic={s['mean_ic']:.4f}, ir={s['ir']:.3f}, hit={s['hit_rate']:.2%}")

        ret_20d = y.reindex(preds.index)["ret_20d"]
        ric20 = rank_ic(test_pred.loc[safe], ret_20d.loc[safe])
        s20 = ic_summary(ric20)
        print(f"  ref IR vs ret_20d:    mean_ic={s20['mean_ic']:.4f}, ir={s20['ir']:.3f}, hit={s20['hit_rate']:.2%}")

        all_results[h] = {
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "ref_ic_median": s,
            "ref_ic_20d": s20,
        }
    else:
        all_results[h] = {"train_acc": float(train_acc)}

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
        "model": "LightGBM (classification)",
        "horizons": HORIZONS,
        "factor_names": factor_cols,
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "train_window": TRAIN_WINDOW,
        "lgb_kwargs": LGB_KWARGS,
        "results": {str(k): all_results[k] for k in all_results},
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
    r = all_results.get("label", {})
    print(f"  Train accuracy: {r.get('train_acc', 0):.4f}")
    print(f"  Test  accuracy: {r.get('test_acc', 0):.4f}")
    ref = r.get("ref_ic_median", {})
    print(f"  Ref IC (ret_median): ir={ref.get('ir', 0):.3f}, hit={ref.get('hit_rate', 0):.2%}")
    print("\nDone.")


if __name__ == "__main__":
    main()

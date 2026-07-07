"""
Synthetic-data smoke test for MLP strategy + IC evaluation.
Also tests fixed test-set split and warmup_days.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from strategies import MLPStrategy, walk_forward, rank_ic, pearson_ic, ic_summary


def build_synthetic_panel(n_dates=100, n_codes=50, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    codes = [f"{i:06d}" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])

    factor1 = rng.normal(0, 1, size=len(idx))
    factor2 = rng.normal(0, 1, size=len(idx))
    factor3 = rng.normal(0, 1, size=len(idx))

    factors = pd.DataFrame(
        {"f1": factor1, "f2": factor2, "f3": factor3},
        index=idx,
    )

    noise = rng.normal(0, 0.5, size=len(idx))
    forward_ret = 0.3 * factor3 + noise
    forward_ret = pd.Series(forward_ret, index=idx, name="forward_ret")

    return factors, forward_ret


def main():
    print("=== building synthetic panel ===")
    factors, fwd_ret = build_synthetic_panel(n_dates=100)
    print(f"factors shape: {factors.shape}")
    print(f"forward returns shape: {fwd_ret.shape}")

    # Test 1: MLP with anti-overfitting defaults
    print("\n=== Test 1: MLP with anti-overfitting defaults ===")
    mlp = MLPStrategy(
        factor_names=["f1", "f2", "f3"],
        max_iter=50,
        random_state=42,
    )
    preds = walk_forward(
        mlp, factors, fwd_ret,
        train_window=30, min_train=20,
    )
    n_dates = preds.index.get_level_values("date").nunique()
    print(f"predictions: {len(preds)} rows over {n_dates} dates")

    ric = rank_ic(preds, fwd_ret)
    pic = pearson_ic(preds, fwd_ret)
    print(f"Rank  IC summary: {ic_summary(ric)}")
    print(f"Pearson IC summary: {ic_summary(pic)}")

    # Test 2: MLP with only noise factors
    print("\n=== Test 2: MLP with only noise factors ===")
    mlp_noise = MLPStrategy(
        factor_names=["f1", "f2"],
        max_iter=50,
        random_state=42,
    )
    preds_noise = walk_forward(
        mlp_noise, factors, fwd_ret,
        train_window=30, min_train=20,
    )
    ric_noise = rank_ic(preds_noise, fwd_ret)
    print(f"Rank IC summary: {ic_summary(ric_noise)}")

    # Test 3: Fixed test-set split
    print("\n=== Test 3: Fixed test-set split ===")
    test_start = pd.Timestamp("2020-03-02")
    test_end = pd.Timestamp("2020-05-01")
    mlp_fixed = MLPStrategy(
        factor_names=["f1", "f2", "f3"],
        max_iter=50,
        random_state=42,
    )
    preds_fixed = walk_forward(
        mlp_fixed, factors, fwd_ret,
        train_window=30, min_train=20,
        test_start=test_start, test_end=test_end,
    )
    n_fixed = preds_fixed.index.get_level_values("date").nunique()
    print(f"fixed test-set predictions: {len(preds_fixed)} rows over {n_fixed} dates")

    pred_dates = sorted(preds_fixed.index.get_level_values("date").unique())
    test_in_range = [dt for dt in pred_dates if test_start <= dt <= test_end]
    print(f"  Test-set dates (in range): {len(test_in_range)} / {len(pred_dates)} total pred dates")
    assert len(test_in_range) > 0, "No predictions within test range"
    print("  Test-set predictions exist: OK")

    ric_fixed = rank_ic(preds_fixed, fwd_ret)
    print(f"Rank IC summary: {ic_summary(ric_fixed)}")

    # Test 4: warmup_days
    print("\n=== Test 4: warmup_days ===")
    mlp_warmup = MLPStrategy(
        factor_names=["f1", "f2", "f3"],
        max_iter=50,
        random_state=42,
    )
    preds_w = walk_forward(
        mlp_warmup, factors, fwd_ret,
        train_window=30, min_train=20,
        warmup_days=10,
    )
    n_dates_w = preds_w.index.get_level_values("date").nunique()
    first_pred = min(preds_w.index.get_level_values("date"))
    all_dates = sorted(factors.index.get_level_values("date").unique())
    print(f"warmup=10: {n_dates_w} pred dates, first={first_pred}")
    print(f"  earliest data date: {all_dates[0]}, first pred date: {first_pred}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()

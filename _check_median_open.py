# -*- coding: utf-8 -*-
"""
Verification for compute_median_open (spec §3.2, plan Task 1).

Independently recomputes median-open labels row-by-row (naive loop) for
three sample stocks — normal (most rows), delisted, long-suspended — and
asserts exact equality with strategies.labels.compute_median_open across
both label windows and all three baselines.

Also asserts the documented delisting semantics:
  - partial windows keep median over available opens
  - fully missing window -> NaN (tail of exactly `start_day` rows)
  - no -1.0 values anywhere (legacy fill is dead code, must not reappear)

Run (workbuddy-env python, from project root):
    python _check_median_open.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH, get_pool_codes
from strategies.labels import compute_median_open

WINDOWS = {"20d": dict(start_day=16, end_day=20), "6d": dict(start_day=4, end_day=6)}
BASELINES = ("next_open", "open", "close")


def pick_samples(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    pool = get_pool_codes()
    ph = ",".join(["?"] * len(pool))

    normal = con.execute(
        f"SELECT code, count(*) AS n FROM daily_kline WHERE code IN ({ph}) "
        f"GROUP BY code ORDER BY n DESC LIMIT 1",
        pool,
    ).fetchone()

    delist = con.execute(
        f"SELECT d.code, count(*) AS n, max(d.delist_date) AS dd "
        f"FROM delist_info d JOIN daily_kline k ON k.code = d.code "
        f"WHERE d.code IN ({ph}) "
        f"GROUP BY d.code HAVING n > 200 ORDER BY dd DESC LIMIT 1",
        pool,
    ).fetchone()

    susp = con.execute(
        f"""
        WITH s AS (
          SELECT code, min(date) AS d0, max(date) AS d1, count(*) AS n
          FROM daily_kline WHERE code IN ({ph})
          GROUP BY code HAVING min(date) < '2024-01-01'
        ), cal AS (
          SELECT date FROM trading_calendar WHERE CAST(is_open AS INT) = 1
        )
        SELECT s.code,
               (SELECT count(*) FROM cal WHERE cal.date BETWEEN s.d0 AND s.d1) - s.n AS gap
        FROM s ORDER BY gap DESC LIMIT 1
        """,
        pool,
    ).fetchone()

    print(f"samples: normal={normal[0]} ({normal[1]} rows) | "
          f"delisted={delist[0]} ({delist[1]} rows, delist_date={delist[2]}) | "
          f"suspended={susp[0]} (missing {susp[1]} trading days)")
    return {"normal": normal[0], "delisted": delist[0], "suspended": susp[0]}


def load_one(con: duckdb.DuckDBPyConnection, code: str) -> pd.DataFrame:
    df = con.execute(
        "SELECT code, date, open, close FROM daily_kline WHERE code = ? ORDER BY date",
        [code],
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    return df


def naive(df: pd.DataFrame, start_day: int, end_day: int, baseline: str) -> np.ndarray:
    """Independent re-implementation of the SPEC semantics (not of the code):
    window = opens at rows t+start_day..t+end_day that exist AND are finite
    (NULL adjusted prices — missing adj_factor — count as unavailable, same
    skipna rule as the implementation); empty window -> NaN; NULL baseline
    -> NaN."""
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n = len(opens)
    out = np.full(n, np.nan)
    for t in range(n):
        vals = [opens[t + d] for d in range(start_day, end_day + 1)
                if t + d < n and np.isfinite(opens[t + d])]
        if not vals:
            continue
        med = float(np.median(vals))
        if baseline == "next_open":
            b = opens[t + 1] if t + 1 < n else np.nan
        elif baseline == "open":
            b = opens[t]
        else:
            b = closes[t]
        if not np.isfinite(b) or b == 0.0:
            continue
        out[t] = med / b - 1.0
    return out


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    samples = pick_samples(con)
    klines = {name: load_one(con, code) for name, code in samples.items()}
    con.close()

    # multi-code frame: the call shape training/backtest actually use
    # (single-code input makes pandas collapse the groupby result to a
    # plain date index, so always evaluate through the pool-style frame)
    all_df = pd.concat(klines.values(), ignore_index=True)
    for name, df in klines.items():
        n_null = int((df["open"].isna() | df["close"].isna()).sum())
        print(f"{name:>10} ({df['code'].iloc[0]}): NULL-ohlc rows = {n_null}")
    results = {(wname, base): compute_median_open(all_df, baseline=base, **win)
               for wname, win in WINDOWS.items() for base in BASELINES}

    fail = 0
    for name, df in klines.items():
        code = df["code"].iloc[0]
        for wname, win in WINDOWS.items():
            for base in BASELINES:
                got = results[(wname, base)].xs(code, level="code").to_numpy(dtype=float)
                exp = naive(df, win["start_day"], win["end_day"], base)

                mismatch = (~np.isclose(got, exp, equal_nan=True, rtol=1e-10, atol=1e-12)).sum()
                tail_nan = int(np.isnan(exp[-win["start_day"]:]).all())
                n_nan = int(np.isnan(got).sum())
                n_neg1 = int((got == -1.0).sum())
                print(f"[{name:>10}/{wname}/{base:>9}] n={len(got)} mismatch={mismatch} "
                      f"nan={n_nan} tail{win['start_day']}d_all_nan={tail_nan} n_neg1={n_neg1}")

                if mismatch:
                    fail += 1
                if n_neg1:
                    fail += 1
                if not tail_nan:
                    fail += 1

        # delisted stock: partial-window extremes visible (eyeball)
        if name == "delisted":
            got = results[("20d", "next_open")].xs(code, level="code")
            print(f"  delisted min label (20d/next_open): {got.min():.4f}")

        # suspended stock: show one label whose window crosses the suspension
        if name == "suspended":
            dates = pd.to_datetime(df["date"])
            gaps = dates.diff().dt.days
            gi = int(gaps.idxmax())
            code = df["code"].iloc[0]
            t = max(gi - WINDOWS["20d"]["start_day"], 0)
            print(f"  suspension around {dates.iloc[gi].date()} "
                  f"(gap {gaps.iloc[gi]:.0f} calendar days); "
                  f"T={dates.iloc[t].date()} label uses opens from row {t + 16}..{t + 20} "
                  f"=> dates {[str(d.date()) for d in dates.iloc[t + 16:t + 21].tolist()]}")

    if fail:
        print(f"\nFAILED: {fail} check groups mismatched")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

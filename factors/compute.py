# -*- coding: utf-8 -*-
"""
Factor computation pipeline.

Loads OHLCV + supplementary data from DuckDB, computes all 101 Alpha101 factors
and ~34 non-alpha factors, stores results to the `factor_values` table.

Usage:
    python -m factors.compute                  # full rebuild
    python -m factors.compute --from 2024-01-01  # from specific date
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes

from .alpha101 import ALPHA_EXPRESSIONS
from .utility import calculate_by_expression
from .extra_factors import compute_non_alpha_factors, apply_ind_neutralize

log = logging.getLogger(__name__)


# ---- Data Loading ----

def _load_ohlcv(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pl.DataFrame:
    """Load daily kline data for given codes, compute VWAP."""
    placeholders = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, open, high, low, close, volume "
        f"FROM daily_kline WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    df["date"] = df["date"].astype(str)

    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol", "date": "datetime"})

    # VWAP approximation
    result = result.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap")
    )

    return result


def _load_market_cap(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pl.DataFrame:
    """Load total_mv, circ_mv from daily_basic (code without suffix)."""
    pure_codes = [c.replace(".SH", "").replace(".SZ", "") for c in codes]
    code_map = dict(zip(pure_codes, codes))

    placeholders = ",".join(["?"] * len(pure_codes))
    df = con.execute(
        f"SELECT code, date, total_mv, circ_mv "
        f"FROM daily_basic WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        pure_codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()

    df["date"] = df["date"].astype(str)
    df["code"] = df["code"].map(code_map)

    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol", "date": "datetime"})

    result = result.with_columns([
        pl.col("total_mv").cast(pl.Float64),
        pl.col("circ_mv").cast(pl.Float64),
    ])

    return result


def _load_industry(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load 申万 L3 industry classification."""
    df = con.execute(
        "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame(schema={"vt_symbol": pl.Utf8, "sw_l3_code": pl.Utf8})

    # Filter to pool stocks later; for now load all
    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol"})
    return result


def _load_cyq(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pl.DataFrame:
    """Load chip distribution raw data."""
    placeholders = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, winner_rate, cost_5pct, cost_15pct, cost_50pct, "
        f"cost_85pct, cost_95pct, weight_avg "
        f"FROM cyq_perf WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()

    df["date"] = df["date"].astype(str)
    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol", "date": "datetime"})
    return result


def _load_index_data(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load CSI (000985) and HS300 (000300) market state features."""
    df = con.execute(
        "SELECT code, date, close FROM index_daily "
        "WHERE code IN ('000985', '000300', '399303') ORDER BY code, date"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    df["date"] = df["date"].astype(str)

    r = {}
    for code, prefix in [("000985", "CSI"), ("000300", "HS300"), ("399303", "GZ2000")]:
        part = df[df["code"] == code][["date", "close"]].copy()
        if part.empty:
            continue
        pl_df = pl.from_pandas(part).rename({"date": "datetime"})
        feats = pl_df.select([
            pl.col("datetime"),
            (pl.col("close") / pl.col("close").shift(1) - 1).alias(f"{prefix}_return_1d"),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias(f"{prefix}_return_20d"),
        ])
        if prefix == "CSI":
            vol = pl_df.select([
                pl.col("datetime"),
                (pl.col("close") / pl.col("close").shift(20) - 1)
            ]).select([
                pl.col("datetime"),
                pl.col("close").rolling_std(20, min_samples=1).alias("CSI_volatility_20d")
            ])
            feats = feats.join(vol, on="datetime", how="left")
        r[prefix] = feats

    market = r.get("CSI", pl.DataFrame())
    if "HS300" in r:
        market = market.join(r["HS300"], on="datetime", how="left") if not market.is_empty() else r["HS300"]
    return market


def _load_shibor(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load SHIBOR daily rates (on, 1m) for macro feature."""
    df = con.execute(
        "SELECT date, shibor_on, shibor_1m FROM macro_daily ORDER BY date"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    df["date"] = df["date"].astype(str)
    return pl.from_pandas(df).rename({"date": "datetime"})


def _load_stock_info(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load list_date for LnAge computation."""
    df = con.execute(
        "SELECT code, strftime(list_date, '%Y-%m-%d') AS list_date FROM stock_info"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol"})
    return result


def _compute_isst(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Compute IsST factor from namechange table."""
    df = con.execute("""
        SELECT code, start_date, end_date, change_reason
        FROM namechange
        WHERE change_reason IN ('ST', '*ST')
        ORDER BY code, start_date
    """).fetchdf()

    if df.empty:
        return pl.DataFrame()

    # For each (code, date) in daily_kline date range, check if in ST period
    date_range = con.execute("SELECT DISTINCT date FROM daily_kline ORDER BY date").fetchdf()
    if date_range.empty:
        return pl.DataFrame()

    all_dates = sorted(date_range["date"].astype(str).tolist())

    records = []
    for _, row in df.iterrows():
        code = row["code"]
        start = str(row["start_date"])
        end = str(row["end_date"]) if row["end_date"] else "9999-12-31"
        for d in all_dates:
            if start <= d <= end:
                records.append({"vt_symbol": code, "datetime": d, "IsST": 1})

    if not records:
        return pl.DataFrame(schema={"vt_symbol": pl.Utf8, "datetime": pl.Utf8, "IsST": pl.Int32})

    return pl.DataFrame(records)


# ---- Main Pipeline ----

def compute_panel(
    con: duckdb.DuckDBPyConnection,
    codes: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Compute all factors for the given stock codes and date range.

    Returns a wide-format Polars DataFrame with columns:
    [code, date, alpha1...alpha101, non_alpha_factors..., IsST, ...]
    """
    t0 = time.time()

    # 1. Load data
    log.info("Loading OHLCV data ...")
    df = _load_ohlcv(con, codes)
    if df.is_empty():
        log.warning("No OHLCV data loaded.")
        return pl.DataFrame()

    log.info("Loading market cap data ...")
    mktcap = _load_market_cap(con, codes)
    if not mktcap.is_empty():
        df = df.join(mktcap, on=["datetime", "vt_symbol"], how="left")

    log.info("Loading stock info ...")
    info = _load_stock_info(con)
    if not info.is_empty():
        df = df.join(info, on="vt_symbol", how="left")

    # Filter date range
    if start_date:
        df = df.filter(pl.col("datetime") >= start_date)
    if end_date:
        df = df.filter(pl.col("datetime") <= end_date)

    # Sort for expression engine
    df = df.sort(["vt_symbol", "datetime"])

    n_stocks = df["vt_symbol"].n_unique()
    n_dates = df["datetime"].n_unique()
    log.info("Data loaded: %d stocks × %d dates = %d rows", n_stocks, n_dates, len(df))

    # 2. Prepare for expression engine: add cap + pre-computed returns
    if "total_mv" in df.columns:
        df = df.with_columns(pl.col("total_mv").alias("cap"))

    # Pre-compute daily returns (used by 50+ alpha expressions via RETURNS_EXPR)
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("ret")
    )

    # 3. Compute Alpha101 factors (sequential – IPC overhead dominates with multiprocessing)
    log.info("Computing %d Alpha101 factors ...", len(ALPHA_EXPRESSIONS))
    from tqdm import tqdm

    alpha_results: dict[str, pl.Series] = {}
    expressions = list(ALPHA_EXPRESSIONS.items())

    for name, expr in tqdm(expressions, desc="Alpha101"):
        _, series = _compute_one_alpha((df, name, expr))
        alpha_results[name] = series

    # 4. Build alpha factor DataFrame
    id_cols = df[["datetime", "vt_symbol"]]
    alpha_df = id_cols.clone()
    for name in sorted(alpha_results.keys()):
        alpha_df = alpha_df.with_columns(alpha_results[name].alias(name))
    log.info("Alpha factors computed: %d columns", len(alpha_results))

    # 5. Apply IndNeutralize
    log.info("Applying IndNeutralize (申万 L3) ...")
    industry_map = _load_industry(con)
    if not industry_map.is_empty():
        alpha_df = apply_ind_neutralize(alpha_df, industry_map)

    # 6. Compute non-alpha factors
    log.info("Computing non-alpha factors ...")
    extra_df = compute_non_alpha_factors(df)
    extra_df = alpha_df[["datetime", "vt_symbol"]].join(
        extra_df, on=["datetime", "vt_symbol"], how="left"
    )

    # 7. Merge supplementary factors (chip, market state, IsST)
    log.info("Loading supplementary factors ...")

    cyq_df = _load_cyq(con, codes)
    if not cyq_df.is_empty():
        # Derive chip factors from raw cyq columns
        # WinnerRate: directly from cyq_perf
        cyq_df = cyq_df.with_columns(
            pl.col("winner_rate").alias("WinnerRate")
        )
        # CostPosition: (close - cost_50pct) / (cost_95pct - cost_5pct)
        cyq_df = cyq_df.with_columns(
            ((pl.col("cost_50pct") - pl.col("weight_avg")) / 
             (pl.col("cost_95pct") - pl.col("cost_5pct") + 1e-10)).alias("CostPosition")
        )
        # ChipDispersion: (cost_85pct - cost_15pct) / cost_50pct
        cyq_df = cyq_df.with_columns(
            ((pl.col("cost_85pct") - pl.col("cost_15pct")) / 
             (pl.col("cost_50pct") + 1e-10)).alias("ChipDispersion")
        )
        # ChipSkew: (cost_50pct - weight_avg) / (cost_85pct - cost_15pct)
        cyq_df = cyq_df.with_columns(
            ((pl.col("cost_50pct") - pl.col("weight_avg")) / 
             (pl.col("cost_85pct") - pl.col("cost_15pct") + 1e-10)).alias("ChipSkew")
        )
        chip_cols = ["datetime", "vt_symbol", "WinnerRate", "CostPosition",
                     "ChipDispersion", "ChipSkew"]
        cyq_df = cyq_df.select(chip_cols)
        extra_df = extra_df.join(cyq_df, on=["datetime", "vt_symbol"], how="left")

    market_df = _load_index_data(con)
    if not market_df.is_empty():
        # Broadcast: add vt_symbol to market, then join
        symbols = extra_df.select("vt_symbol").unique()
        market_df = symbols.join(market_df, how="cross")
        extra_df = extra_df.join(market_df, on=["datetime", "vt_symbol"], how="left")

    shibor_df = _load_shibor(con)
    if not shibor_df.is_empty():
        dates = extra_df.select("datetime").unique()
        shibor_df = dates.join(shibor_df, on="datetime", how="left")
        symbols = extra_df.select("vt_symbol").unique()
        shibor_df = symbols.join(shibor_df, how="cross")
        extra_df = extra_df.join(shibor_df, on=["datetime", "vt_symbol"], how="left")

    isst_df = _compute_isst(con)
    if not isst_df.is_empty():
        extra_df = extra_df.join(isst_df, on=["datetime", "vt_symbol"], how="left")
        extra_df = extra_df.with_columns(pl.col("IsST").fill_null(0).cast(pl.Int32))
    else:
        extra_df = extra_df.with_columns(pl.lit(0).cast(pl.Int32).alias("IsST"))

    # 8. Merge alpha + extra (single join, not loop)
    log.info("Merging alpha + non-alpha factors ...")
    extra_fact_cols = [c for c in extra_df.columns if c not in ("datetime", "vt_symbol")]
    merged = alpha_df.join(extra_df, on=["datetime", "vt_symbol"], how="left")

    # 9. Final cleanup: clip extreme values, fill remaining NaN
    # Exclude intermediate columns (ret, cap, and any with _ prefix)
    excluded = {"ret", "cap"}
    factor_columns = [c for c in sorted(alpha_results.keys()) + extra_fact_cols
                      if c not in excluded and not c.startswith("_")]
    for col in factor_columns:
        if col in merged.columns:
            merged = merged.with_columns(
                pl.when(
                    pl.col(col).is_infinite() | pl.col(col).is_nan()
                ).then(None).otherwise(pl.col(col)).alias(col)
            )

    # Drop intermediate columns
    id_cols = ["datetime", "vt_symbol"]
    merged = merged.select(id_cols + factor_columns)

    # Convert to wide format: (code, date, factors...)
    merged = merged.rename({"vt_symbol": "code", "datetime": "date"})

    elapsed = time.time() - t0
    n_factors = len(factor_columns)
    log.info("Total: %d factors, %d rows, %.1f seconds", n_factors, len(merged), elapsed)

    return merged


def _compute_one_alpha(args: tuple) -> tuple[str, pl.Series]:
    """Compute a single alpha factor expression (for multiprocessing)."""
    df, name, expr = args

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            result_df = calculate_by_expression(df, expr)
            return name, result_df["data"]
        except Exception:
            return name, pl.Series(name, [None] * len(df))


# ---- Storage ----

def store_factor_values(con: duckdb.DuckDBPyConnection, panel: pl.DataFrame):
    """Store factor panel into DuckDB factor_values table."""
    if panel.is_empty():
        log.warning("Empty panel, nothing to store.")
        return

    # Deduplicate on (code, date)
    panel = panel.unique(subset=["code", "date"], keep="last")

    con.execute("DROP TABLE IF EXISTS factor_values")

    # Build schema from panel columns
    pandas_df = panel.to_pandas()
    pandas_df = pandas_df.sort_values(["date", "code"])

    # Create table automatically from pandas
    con.execute("CREATE TABLE factor_values AS SELECT * FROM pandas_df")

    con.execute("CHECKPOINT")
    log.info("factor_values table created with %d rows, %d columns",
             len(pandas_df), len(pandas_df.columns))


# ---- Main Entry ----

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Compute factor panel")
    parser.add_argument("--from", dest="start_date", default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_date", default=None,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    codes = get_pool_codes()
    log.info("Pool: %d stocks", len(codes))

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")

    panel = compute_panel(con, codes, start_date=args.start_date,
                          end_date=args.end_date)

    if not panel.is_empty():
        store_factor_values(con, panel)

    con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()

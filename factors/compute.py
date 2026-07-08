# -*- coding: utf-8 -*-
"""
Factor computation pipeline: read kline data from DuckDB, compute all factors, store.
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd
import numpy as np

from config import DB_PATH
from .factors import FACTOR_HUB

log = logging.getLogger(__name__)


def compute_factors_for_stock(df_stock, factor_hub=None):
    """Compute all registered factors for one stock's kline DataFrame."""
    if factor_hub is None:
        factor_hub = FACTOR_HUB
    code = df_stock.get("code", pd.Series([None])).iloc[0]

    results = {}
    for name, func in factor_hub.items():
        try:
            results[name] = func(df_stock)
        except Exception as e:
            log.warning("Factor %s failed for %s: %s", name, code, e)
            results[name] = np.nan

    out = pd.DataFrame(results, index=df_stock.index)
    out["date"] = df_stock["date"].values if "date" in df_stock.columns else df_stock.index
    if code is not None:
        out["code"] = code
    return out


def load_all_stocks(con):
    """Load all kline data sorted by (code, date)."""
    return con.execute("""
        SELECT code, date, open, high, low, close, volume, amount,
               turn, pct_chg
        FROM daily_kline
        ORDER BY code, date
    """).fetchdf()


def compute_panel(con=None, db_path=None, codes=None, max_stocks=None):
    """Load kline data and compute all factors for every stock.

    Returns DataFrame with MultiIndex (date, code) and one column per factor.
    """
    if con is None:
        con = duckdb.connect(str(db_path or DB_PATH))
        should_close = True
    else:
        should_close = False

    df_all = load_all_stocks(con)
    if codes is not None:
        df_all = df_all[df_all["code"].isin(codes)]
    if max_stocks is not None:
        keep_codes = df_all["code"].unique()[:max_stocks]
        df_all = df_all[df_all["code"].isin(keep_codes)]

    log.info("Loaded %d rows, %d stocks", len(df_all), df_all["code"].nunique())

    panels = []
    for code, grp in df_all.groupby("code", sort=False):
        grp = grp.sort_values("date")
        factor_df = compute_factors_for_stock(grp)
        panels.append(factor_df)

    if not panels:
        if should_close:
            con.close()
        return pd.DataFrame()

    result = pd.concat(panels, ignore_index=True)
    result = result.set_index(["date", "code"]).sort_index()

    if should_close:
        con.close()
    return result


def store_factors(factor_panel, table_name="factor_values", con=None,
                  db_path=None, if_exists="replace"):
    """Store factor panel into DuckDB as a wide table.

    Schema: (code, date, <factor1>, <factor2>, ...) with PK (code, date).
    """
    own_con = False
    if con is None:
        con = duckdb.connect(str(db_path or DB_PATH))
        own_con = True

    df_to_store = factor_panel.reset_index()

    if if_exists == "replace":
        con.execute("DROP TABLE IF EXISTS " + table_name)

    con.execute("CREATE TABLE IF NOT EXISTS " + table_name + " AS SELECT * FROM df_to_store WHERE 1=0")
    con.execute("INSERT INTO " + table_name + " SELECT * FROM df_to_store")

    n = len(df_to_store)
    log.info("Stored %d rows in table '%s'", n, table_name)

    if own_con:
        con.close()
    return n
def load_all_stocks_since(con, min_date):
    """Load kline data from min_date onwards, sorted by (code, date)."""
    return con.execute("""
        SELECT code, date, open, high, low, close, volume, amount,
               turn, pct_chg
        FROM daily_kline
        WHERE date >= ?
        ORDER BY code, date
    """, (min_date,)).fetchdf()


def compute_panel_incremental(con=None, lookback_trading_days=252):
    """Compute factors only for dates not yet in factor_values.

    Uses a lookback window so lookback-dependent factors
    (e.g. Volatility_60d, Price_position_252d) have enough history.

    Returns
    -------
    DataFrame with MultiIndex (date, code), only rows NOT yet in factor_values.
    Empty DataFrame if nothing new to compute.
    """
    own_con = False
    if con is None:
        con = duckdb.connect(str(DB_PATH))
        own_con = True

    try:
        raw = con.execute("SELECT MAX(date) FROM factor_values").fetchone()[0]
        last_date = pd.Timestamp(raw) if raw is not None else None
    except Exception:
        last_date = None

    if last_date is None:
        log.info("factor_values empty or missing, doing full compute ...")
        result = compute_panel(con=con)
        if own_con:
            con.close()
        return result

    # Check if there is new kline data after last_date
    kline_max = con.execute("SELECT MAX(date) FROM daily_kline").fetchone()[0]
    # Normalise both to date for safe comparison
    kline_max_date = pd.Timestamp(kline_max).date() if kline_max is not None else None
    last_date_ts = pd.Timestamp(last_date).date()
    if kline_max_date is None or kline_max_date <= last_date_ts:
        if own_con:
            con.close()
        return pd.DataFrame()

    lookback_start = last_date - pd.Timedelta(days=int(lookback_trading_days * 1.6))
    df_all = load_all_stocks_since(con, lookback_start)

    log.info("Loaded %d rows, %d stocks (since %s)",
             len(df_all), df_all["code"].nunique(), lookback_start.date())

    panels = []
    for code, grp in df_all.groupby("code", sort=False):
        grp = grp.sort_values("date")
        factor_df = compute_factors_for_stock(grp)
        panels.append(factor_df)

    if not panels:
        if own_con:
            con.close()
        return pd.DataFrame()

    result = pd.concat(panels, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index(["date", "code"]).sort_index()

    # Keep only rows after the last date in factor_values
    new_data = result[result.index.get_level_values("date") > pd.Timestamp(last_date_ts)]

    n_new = len(new_data)
    n_dates = new_data.index.get_level_values("date").nunique()
    log.info("Computed %d new factor rows (%d new date(s))", n_new, n_dates)

    if own_con:
        con.close()
    return new_data

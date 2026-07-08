# -*- coding: utf-8 -*-
"""
Factor computation pipeline: read kline data from DuckDB, compute all factors, store.
"""
from __future__ import annotations

import logging
import re

import duckdb
import pandas as pd
import numpy as np

from config import DB_PATH
from .factors import FACTOR_HUB
from .ops import decay_linear

log = logging.getLogger(__name__)


def _cs_rank(series):
    return series.rank(pct=True) - 0.5


_CS_RANK_BASES = {
    "close": "close_cs",
    "volume": "volume_cs",
    "low": "low_cs",
    "dc1": "dc1_cs",
    "dv1": "dv1_cs",
    "ret1d": "ret1d_cs",
}

_CS_RANK_POST_FACTORS = [
    "alpha001", "alpha013", "alpha014", "alpha018", "alpha019", "alpha020",
    "alpha050", "alpha101", "alpha191",
]

# deferred alphas: factor function stores raw intermediates (prefixed _d{N}_)
# _compose_deferred_alphas applies cs_rank and composes final values
_DEFERRED_ALPHAS = {
    "alpha017": {"cols": ["_d17_a", "_d17_b", "_d17_c"]},
    "alpha038": {"cols": ["_d38_a", "_d38_b"]},
    "alpha057": {"cols": ["_d57_a", "_d57_b"]},
}


def _compute_cs_rank_cols(df_all):
    """Compute derived base columns and their cross-sectional ranks.

    Adds dc1, dv1, ret1d and their _cs-ranked siblings to the full panel.
    Modifies df_all in place.
    """
    grp = df_all.groupby("code")
    df_all["dc1"] = grp["close"].transform(lambda x: x - x.shift(1))
    df_all["dv1"] = grp["volume"].transform(lambda x: x - x.shift(1))
    df_all["ret1d"] = grp["close"].pct_change()

    for src, dst in _CS_RANK_BASES.items():
        df_all[dst] = df_all.groupby("date")[src].transform(_cs_rank)

    return df_all


def _apply_cs_rank_post(result):
    """Apply cross-sectional rank to outer-rank alpha factors."""
    for col in _CS_RANK_POST_FACTORS:
        if col in result.columns:
            result[col] = result.groupby("date")[col].transform(_cs_rank)
    return result


def _compose_deferred_alphas(result):
    """Compose deferred alphas from raw intermediate columns.

    Alpha factor functions store raw sub-expressions in _d{N}_* columns.
    This function applies cross-sectional rank to each intermediate and
    composes the final alpha value per the formulas in _DEFERRED_ALPHAS.
    """
    ptn = re.compile(r"_d\d+_\w+")
    raw_cols = [c for c in result.columns if ptn.match(c)]
    if not raw_cols:
        return result

    # step 1: cs_rank all raw intermediates
    cs_map = {}
    for col in raw_cols:
        cs_name = "_cs" + col
        result[cs_name] = result.groupby("date")[col].transform(_cs_rank)
        cs_map[col] = cs_name

    # step 2: compose each deferred alpha
    for alpha, cfg in _DEFERRED_ALPHAS.items():
        raw = cfg["cols"]
        if not all(c in cs_map for c in raw):
            continue

        if alpha == "alpha057":
            # alpha057: -1 * (_d57_b) / decay_linear( cs_rank(_d57_a) , 2 )
            cs_a = result[cs_map["_d57_a"]]
            decayed = cs_a.groupby(result.index.get_level_values("code")).transform(
                lambda s: pd.Series(decay_linear(s, 2).values, index=s.index)
            )
            result[alpha] = (-result["_d57_b"] / decayed.replace(0, np.nan)).clip(-1e10, 1e10)
        else:
            # generic product with negation: -1 * cs_rank(a) * cs_rank(b) * cs_rank(c)
            parts = [result[cs_map[c]] for c in raw]
            prod = parts[0]
            for p in parts[1:]:
                prod = prod * p
            result[alpha] = -prod

    # step 3: clean up intermediate columns
    drop_cols = [c for c in raw_cols if c in result.columns]
    drop_cols += [cs_map[c] for c in raw_cols if cs_map[c] in result.columns]
    result.drop(columns=drop_cols, inplace=True, errors="ignore")
    return result


def _merge_rank_factors(result):
    """Generate cross-sectional rank versions of selected factors."""
    rank_map = {
        "Return_1d": "Return_1d_rank",
        "Return_20d": "Return_20d_rank",
        "Turnover_3d": "Turnover_3d_rank",
    }
    for base, rank_name in rank_map.items():
        if base in result.columns:
            result[rank_name] = result.groupby("date")[base].transform(_cs_rank)
    return result


def compute_factors_for_stock(df_stock, factor_hub=None):
    """Compute all registered factors for one stock's kline DataFrame."""
    if factor_hub is None:
        factor_hub = FACTOR_HUB
    code = df_stock.get("code", pd.Series([None])).iloc[0]

    results = {}
    for name, func in factor_hub.items():
        try:
            raw = func(df_stock)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    results[k] = v
            else:
                results[name] = raw
        except Exception as e:
            log.warning("Factor %s failed for %s: %s", name, code, e)
            results[name] = np.nan

    out = pd.DataFrame(results, index=df_stock.index)
    out["date"] = df_stock["date"].values if "date" in df_stock.columns else df_stock.index
    if code is not None:
        out["code"] = code
    return out


def load_all_stocks(con):
    """Load all kline data joined with market cap sorted by (code, date)."""
    return con.execute("""
        SELECT k.code, k.date, k.open, k.high, k.low, k.close,
               k.volume, k.amount, k.pct_chg,
               k.volume * k.close / NULLIF(b.circ_mv, 0) AS turn,
               b.total_mv, b.circ_mv
        FROM daily_kline k
        LEFT JOIN daily_basic b ON k.code = b.code AND k.date = b.date
        ORDER BY k.code, k.date
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

    df_all = _compute_cs_rank_cols(df_all)

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

    result = result.clip(-1e10, 1e10).replace([np.inf, -np.inf], np.nan)
    result = _apply_cs_rank_post(result)
    result = _compose_deferred_alphas(result)
    result = _merge_rank_factors(result)
    result = _merge_market_features(result, con)

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
    """Load kline data joined with market cap from min_date onwards."""
    return con.execute("""
        SELECT k.code, k.date, k.open, k.high, k.low, k.close,
               k.volume, k.amount, k.pct_chg,
               k.volume * k.close / NULLIF(b.circ_mv, 0) AS turn,
               b.total_mv, b.circ_mv
        FROM daily_kline k
        LEFT JOIN daily_basic b ON k.code = b.code AND k.date = b.date
        WHERE k.date >= ?
        ORDER BY k.code, k.date
    """, (min_date,)).fetchdf()


def compute_market_features(con):
    """Compute market state features from 中证全指 (000985) index data."""
    idx = con.execute("""
        SELECT date, pct_chg, close FROM index_daily
        WHERE code = '000985' ORDER BY date
    """).fetchdf()
    if idx.empty:
        return pd.DataFrame()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")

    idx["CSI_return_1d"] = idx["pct_chg"] / 100.0
    idx["CSI_return_5d"] = idx["close"].pct_change(5)
    idx["CSI_return_20d"] = idx["close"].pct_change(20)
    idx["CSI_volatility_20d"] = idx["pct_chg"].rolling(20, min_periods=5).std() / 100.0

    return idx[["date", "CSI_return_1d", "CSI_return_5d", "CSI_return_20d",
                "CSI_volatility_20d"]]


def _merge_market_features(result, con):
    """Broadcast market features to every (date, code) row in result."""
    market = compute_market_features(con)
    if market.empty:
        return result
    result = result.reset_index()
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(market, on="date", how="left")
    return result.set_index(["date", "code"]).sort_index()


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

    df_all = _compute_cs_rank_cols(df_all)

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

    result = result.clip(-1e10, 1e10).replace([np.inf, -np.inf], np.nan)
    result = _apply_cs_rank_post(result)
    result = _compose_deferred_alphas(result)
    result = _merge_rank_factors(result)

    # Broadcast market features
    result = _merge_market_features(result, con)

    # Keep only rows after the last date in factor_values
    new_data = result[result.index.get_level_values("date") > pd.Timestamp(last_date_ts)]

    n_new = len(new_data)
    n_dates = new_data.index.get_level_values("date").nunique()
    log.info("Computed %d new factor rows (%d new date(s))", n_new, n_dates)

    if own_con:
        con.close()
    return new_data

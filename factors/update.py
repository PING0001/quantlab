"""
Incremental factor update.

Computes factors only for dates not yet in the factor_values table,
then appends the new rows.

Usage:
    python -m factors.update
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes
from .compute import compute_panel_incremental

log = logging.getLogger(__name__)

_CURATED_FACTORS = [
    "Return_1d", "Return_5d", "Return_20d", "Reversal_60d",
    "ATR", "Volatility", "Volatility_60d", "Bollinger_width", "alpha060",
    "Price_position_252d", "Stochastic_K", "Return_skew_20d",
    "Trend_strength", "SMA", "MACD_signal",
    "Gap_pct", "Body_pct", "Intraday_range_pct",
    "Volume_ratio", "Amihud_illiquidity",
    "alpha001", "alpha002", "alpha003", "alpha006", "alpha007",
    "alpha009", "alpha012", "alpha013", "alpha014", "alpha017",
    "alpha018", "alpha019", "alpha020", "alpha028", "alpha035",
    "alpha038", "alpha046", "alpha050", "alpha057",
    "alpha101", "alpha191",
        "LnMktCap", "LnFloatCap", "AvgAmount_90d",
        "Turnover_3d", "Turnover_3d_ratio",
        "Intraday_return",
        "CSI_return_1d", "CSI_return_5d", "CSI_return_20d", "CSI_volatility_20d",
        "HS300_return_1d", "HS300_return_20d",
        "Return_1d_rank", "Return_20d_rank", "Turnover_3d_rank",
        "LnAge",
        "WinnerRate", "CostPosition", "ChipDispersion", "ChipSkew",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    con = duckdb.connect(str(DB_PATH))
    pool_codes = get_pool_codes()

    panel = compute_panel_incremental(con=con, codes=pool_codes)
    if panel.empty:
        log.info("factor_values is already up to date. Nothing to do.")
        con.close()
        return

    available = [f for f in _CURATED_FACTORS if f in panel.columns]
    missing = [f for f in _CURATED_FACTORS if f not in panel.columns]
    if missing:
        log.warning("Factors not in computed panel: %s", missing)

    selected = panel[available]
    df_to_store = selected.reset_index()

    # Create table if it doesn't exist
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='factor_values'"
    ).fetchone()[0]
    if not exists:
        log.info("Creating factor_values table ...")
        con.execute("CREATE TABLE factor_values AS SELECT * FROM df_to_store WHERE 1=0")

    log.info("Appending %d rows to factor_values ...", len(df_to_store))
    con.execute("INSERT INTO factor_values SELECT * FROM df_to_store")
    con.execute("CHECKPOINT")

    new_count = con.execute("SELECT count(*) FROM factor_values").fetchone()[0]
    log.info("Done. factor_values now has %d rows.", new_count)
    con.close()


if __name__ == "__main__":
    main()

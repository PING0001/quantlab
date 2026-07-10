"""
Build the microcap stock pool via semi-annual circ_mv screening
with inflation-adjusted ranges (M2-GDP proxy).

Base: 2026-06 -> 1e < circ_mv < 20e
2015-06 -> ~0.57e < circ_mv < ~11.36e  (11/20 = 0.55 ratio over 11 years)

Output: tmp/mainboard_microcap_tmp.json (does NOT overwrite the live pool)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ashare.duckdb"
POOL_PATH = ROOT / "pools" / "mainboard_microcap.json"
OUT_PATH = ROOT / "tmp" / "mainboard_microcap_tmp.json"

# Base year and range (2026 CNY)
BASE_YEAR = 2026
BASE_LOW_YI = 1.0     # 1 亿
BASE_HIGH_YI = 20.0   # 20 亿

# Inflation proxy: 2015's 11e ≈ 2026's 20e  ->  ratio 0.55 over 11 years
TOTAL_YEARS = 11.0
DECAY_RATIO = 11.0 / 20.0  # 0.55


def _decay_factor(cp_year: float) -> float:
    """Return the scaling factor for a given checkpoint decimal year."""
    t = max(0.0, BASE_YEAR - cp_year)
    return DECAY_RATIO ** (t / TOTAL_YEARS)


def _cp_decimal_year(cp_str: str) -> float:
    """Convert 'YYYY-MM-DD' to decimal year."""
    dt = datetime.strptime(cp_str, "%Y-%m-%d")
    days_in_year = 366 if (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0) else 365
    return dt.year + (dt.timetuple().tm_yday - 1) / days_in_year


def _generate_checkpoints(start: datetime, end: datetime, months: int = 6):
    pts = []
    d = start
    while d <= end:
        pts.append(d.strftime("%Y-%m-%d"))
        d = d + relativedelta(months=months)
    return pts


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ---- 1. Generate checkpoints ----
    checkpoints = _generate_checkpoints(datetime(2015, 6, 15), datetime(2026, 6, 15))
    log.info("Checkpoints (%d): %s ... %s", len(checkpoints), checkpoints[0], checkpoints[-1])

    # ---- 2. For each checkpoint, screen with decay-adjusted ranges ----
    all_codes = set()
    cp_counts = {}
    cp_ranges = {}

    for cp in checkpoints:
        # Find nearest prior trading day
        row = con.execute(
            "SELECT MAX(date) FROM daily_basic WHERE date <= ?", [cp]
        ).fetchone()
        if row[0] is None:
            log.warning("  %s -> no data, skipping", cp)
            continue

        trade_date = row[0]

        # Calculate inflation-adjusted range for this checkpoint
        cp_year = _cp_decimal_year(cp)
        factor = _decay_factor(cp_year)
        low_wan = round(BASE_LOW_YI * factor * 10000, 0)
        high_wan = round(BASE_HIGH_YI * factor * 10000, 0)
        cp_ranges[cp] = {
            "year": round(cp_year, 2),
            "factor": round(factor, 4),
            "low_yi": round(BASE_LOW_YI * factor, 4),
            "high_yi": round(BASE_HIGH_YI * factor, 2),
        }

        codes = con.execute(
            """
            SELECT DISTINCT b.code
            FROM daily_basic b
            JOIN stock_info s ON b.code = s.code
            WHERE s.market = '主板'
              AND b.date = ?
              AND b.circ_mv > ?
              AND b.circ_mv < ?
            """,
            [trade_date, low_wan, high_wan],
        ).fetchall()

        s = set(c[0] for c in codes)
        cp_counts[cp] = len(s)
        all_codes.update(s)
        log.info(
            "  %s -> trade %s | range %se-%se | found %d | cumul %d",
            cp, trade_date,
            round(BASE_LOW_YI * factor, 2),
            round(BASE_HIGH_YI * factor, 2),
            len(s), len(all_codes),
        )

    log.info("Union unique stocks across all %d checkpoints: %d", len(checkpoints), len(all_codes))

    # ---- 3. Load existing pool for coverage comparison ----
    if POOL_PATH.exists():
        with open(POOL_PATH, "r", encoding="utf-8") as f:
            existing_pool = json.load(f)
        existing_codes = set(s["code"] for s in existing_pool["stocks"])
        existing_industry = {s["code"]: s.get("industry", "") for s in existing_pool["stocks"]}
        log.info("Existing pool: %d stocks", len(existing_codes))
    else:
        existing_codes = set()
        existing_industry = {}
        log.info("No existing pool found, creating fresh")

    # ---- 4. Coverage analysis ----
    missing = existing_codes - all_codes
    new_codes = all_codes - existing_codes

    log.info("===== COVERAGE REPORT =====")
    if missing:
        log.warning(
            "Existing stocks NOT covered by union (%d/%d): %s",
            len(missing), len(existing_codes), sorted(missing),
        )
    else:
        log.info("All %d existing stocks ARE covered by the semi-annual union.", len(existing_codes))
    log.info("New stocks from union not in existing pool: %d", len(new_codes))
    log.info("============================")

    # ---- 5. Fetch full info for union codes ----
    merged_codes = sorted(all_codes)
    log.info("Final pool: %d stocks", len(merged_codes))

    # Latest trade date for cap snapshot
    latest_date = con.execute("SELECT MAX(date) FROM daily_basic").fetchone()[0]
    log.info("Latest daily_basic date: %s", str(latest_date))

    # stock_info
    info_df = con.execute(
        f"""
        SELECT code, name, full_code
        FROM stock_info
        WHERE code IN ({','.join(['?'] * len(merged_codes))})
        ORDER BY code
        """,
        merged_codes,
    ).df()
    info_map = {r["code"]: r for _, r in info_df.iterrows()}

    # Latest market cap data
    basic_df = con.execute(
        f"""
        SELECT code, total_mv, circ_mv, pe
        FROM daily_basic
        WHERE date = ?
          AND code IN ({','.join(['?'] * len(merged_codes))})
        """,
        [str(latest_date)] + merged_codes,
    ).df()
    basic_map = {r["code"]: r for _, r in basic_df.iterrows()}

    con.close()

    # ---- 6. Build stocks list ----
    stocks = []
    for code in merged_codes:
        info = info_map.get(code, {})
        basic = basic_map.get(code, {})

        full_code = info.get("full_code", "")
        if "." in full_code:
            market = full_code.split(".")[1]
        elif code.startswith("6"):
            market = "SH"
        else:
            market = "SZ"

        circ_mv_wan = basic.get("circ_mv")
        total_mv_wan = basic.get("total_mv")

        stocks.append({
            "code": code,
            "name": info.get("name", ""),
            "market": market,
            "full_code": full_code or f"{code}.{market}",
            "circ_mv_yi": round(circ_mv_wan / 10000, 4) if circ_mv_wan is not None else None,
            "total_mv_yi": round(total_mv_wan / 10000, 4) if total_mv_wan is not None else None,
            "pe": round(float(basic.get("pe")), 4) if basic.get("pe") is not None else None,
            "industry": existing_industry.get(code, ""),
        })

    # ---- 7. Write to TMP file ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_pool = {
        "filter": "main_board + 1e < circ_mv < 20e @2026, decay-adjusted by year (M2-GDP proxy: 11e@2015 -> 20e@2026)",
        "trade_date": str(latest_date),
        "total": len(stocks),
        "checkpoints": {
            "start": "2015-06-15",
            "end": "2026-06-15",
            "frequency": "6 months",
            "base_range": f"{BASE_LOW_YI}e - {BASE_HIGH_YI}e @ {BASE_YEAR}",
            "decay_ratio": f"{DECAY_RATIO} over {TOTAL_YEARS} years (11e@2015 -> 20e@2026)",
            "ranges": cp_ranges,
            "counts": cp_counts,
        },
        "coverage": {
            "union_unique": len(all_codes),
            "existing_total": len(existing_codes),
            "existing_covered": len(existing_codes - missing) if missing else len(existing_codes),
            "missing": sorted(missing) if missing else [],
            "new_from_union": sorted(new_codes),
        },
        "stocks": stocks,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_pool, f, ensure_ascii=False, indent=2)

    log.info("Written %s with %d stocks", OUT_PATH, len(stocks))

    # ---- 8. Summary ----
    print("\n" + "=" * 60)
    print(f"  Output:        {OUT_PATH}")
    print(f"  Checkpoints:   {len(checkpoints)} ({checkpoints[0]} -> {checkpoints[-1]})")
    print(f"  Union unique:  {len(all_codes)}")
    print(f"  Existing pool: {len(existing_codes)}")
    print(f"  Covered:       {len(existing_codes) - len(missing)}/{len(existing_codes)}")
    if missing:
        print(f"  Not covered:   {len(missing)}")
    print(f"  New from union: {len(new_codes)}")
    print("=" * 60)

    # ---- 9. Print adjusted ranges table ----
    print(f"\n  {'Checkpoint':>12s}  {'Factor':>7s}  {'Low(e)':>8s}  {'High(e)':>8s}  {'N':>5s}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*5}")
    for cp in checkpoints:
        if cp in cp_ranges:
            r = cp_ranges[cp]
            n = cp_counts.get(cp, 0)
            print(f"  {cp:>12s}  {r['factor']:>7.4f}  {r['low_yi']:>8.4f}  {r['high_yi']:>8.2f}  {n:>5d}")
    print()


if __name__ == "__main__":
    main()

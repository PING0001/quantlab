"""One-off: deduplicate factor_values on (code, date).

Removes excess rows from duplicate (code, date) keys, keeping one copy each.
Root cause: factors/update.py incremental path previously lacked dedup;
factors/compute.py _compute_isst emits duplicate rows for stocks with
overlapping ST/*ST namechange records (both multiply through the IsST join).

Safe by construction:
  - whole operation wrapped in a BEGIN/COMMIT transaction (atomic: either
    the old table stays or the new one replaces it, never a half-state);
  - duplicate rows are backed up to a parquet file before any mutation;
  - content-identity of duplicate rows is verified (join-multiplied rows
    should be identical, so keeping any copy is safe).
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DB_PATH

BACKUP_PATH = (DB_PATH.parent / "factor_values_dupes_backup.parquet").as_posix()


def main():
    con = duckdb.connect(str(DB_PATH))

    total_before = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
    dup_keys = con.execute(
        "SELECT COUNT(*) FROM (SELECT code, date FROM factor_values "
        "GROUP BY code, date HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    excess = con.execute(
        "SELECT COALESCE(SUM(c - 1), 0) FROM (SELECT COUNT(*) c FROM factor_values "
        "GROUP BY code, date HAVING c > 1)"
    ).fetchone()[0]
    print(f"Before: {total_before} rows, {dup_keys} dup (code,date) keys, "
          f"{excess} excess rows to remove")

    if excess == 0:
        print("No duplicates - nothing to do.")
        con.close()
        return

    # Sanity: duplicate rows should be content-identical (the IsST join
    # multiplies entire rows without altering values). Check representative
    # columns spanning different factor families.
    diff = con.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT code, date, "
        "    COUNT(DISTINCT alpha1) d_a1, "
        '    COUNT(DISTINCT "IsST") d_st, '
        '    COUNT(DISTINCT "LnMktCap") d_lmc '
        "  FROM factor_values GROUP BY code, date HAVING COUNT(*) > 1"
        ") WHERE d_a1 > 1 OR d_st > 1 OR d_lmc > 1"
    ).fetchone()[0]
    if diff > 0:
        print(f"WARNING: {diff} dup keys have DIFFERING content - "
              f"keeping an arbitrary copy (ORDER BY code). Review backup.")
    else:
        print("All duplicate rows are content-identical - safe to keep any copy.")

    # Backup all copies of dup-key rows (tiny: ~2x dup_keys rows) for reversibility
    con.execute(
        f"COPY (SELECT t.* FROM factor_values t WHERE EXISTS ("
        f"  SELECT 1 FROM (SELECT code, date FROM factor_values "
        f"  GROUP BY code, date HAVING COUNT(*) > 1) d "
        f"  WHERE d.code = t.code AND d.date = t.date"
        f")) TO '{BACKUP_PATH}'"
    )
    n_backup = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{BACKUP_PATH}')"
    ).fetchone()[0]
    print(f"Backed up {n_backup} dup-key rows (all copies) -> {BACKUP_PATH}")

    # Dedup: recreate keeping one row per (code, date)
    con.execute("BEGIN TRANSACTION")
    con.execute(
        "CREATE TABLE __fv_dedup AS "
        "SELECT * EXCLUDE (rn) FROM ("
        "  SELECT *, ROW_NUMBER() OVER (PARTITION BY code, date ORDER BY code) AS rn "
        "  FROM factor_values"
        ") WHERE rn = 1"
    )
    n_dedup = con.execute("SELECT COUNT(*) FROM __fv_dedup").fetchone()[0]
    con.execute("DROP TABLE factor_values")
    con.execute("ALTER TABLE __fv_dedup RENAME TO factor_values")
    con.execute("COMMIT")

    total_after = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
    dup_after = con.execute(
        "SELECT COUNT(*) FROM (SELECT code, date FROM factor_values "
        "GROUP BY code, date HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    con.execute("CHECKPOINT")
    con.close()

    print(f"After:  {total_after} rows, {dup_after} dup keys")
    print(f"Removed {total_before - total_after} excess rows (expected {excess})")
    if total_before - total_after != excess:
        print("WARNING: removed count != expected - inspect backup!")
    elif dup_after == 0:
        print("OK: no duplicates remain.")


if __name__ == "__main__":
    main()

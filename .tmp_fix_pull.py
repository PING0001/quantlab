from pathlib import Path
p = Path("data/pull_adj.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

# Find the block to replace: line 221 (0-indexed: 220) if not all_daily:
# to line 247 (0-indexed: 246) the daily_basic comment
# Replace with version that has no early return

# Current lines 220-249 (0-indexed)
old_start = 220  # "    if not all_daily:\n"
old_end = 249    # "    # ---- daily_basic 增量更新 ----\n" (exclusive)

new_block = """    if not all_daily:
        log.info("无新日线数据")
    else:
        daily = pd.concat(all_daily, ignore_index=True)
        adj = pd.concat(all_adj, ignore_index=True) if all_adj else pd.DataFrame()
        if not adj.empty:
            daily = daily.merge(adj, on=["code", "date"], how="left")
        else:
            daily["adj_factor"] = None
        if "turn" not in daily.columns:
            daily["turn"] = None

        log.info("合并写入 daily_raw（%d 行）...", len(daily))
        con.execute(\"""
            INSERT OR REPLACE INTO daily_raw
                (code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor)
            SELECT code, date, open, high, low, close,
                   volume, amount, pct_chg, turn, adj_factor
            FROM daily
        \""")
        con.execute("CHECKPOINT")
        row_count = con.execute("SELECT count(*) FROM daily_raw").fetchone()[0]
        log.info("完成：daily_raw 当前 %d 行", row_count)

"""

# Replace
new_lines = lines[:old_start] + [new_block] + lines[old_end:]
# Remove the \n inside lines that already have keepends=True
# Actually, the new_block has its own newlines, but the lines in the list already have \n

p.write_text("".join(new_lines), encoding="utf-8")
print("done")

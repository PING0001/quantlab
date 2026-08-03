"""One-off: quantify the label-side 未来函数 (train/test boundary leak).

Shows exactly which training rows have labels (compute_median_close T+16~T+20)
reaching into the test period (>= TEST_START=2025-06-01), and counts them.
"""
import sys
from pathlib import Path
import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DB_PATH, get_pool_codes

TEST_START = pd.Timestamp("2025-06-01")
LABEL_START, LABEL_END = 16, 20  # compute_median_close(kline, 16, 20)

con = duckdb.connect(str(DB_PATH), read_only=True)
dates = sorted(con.execute("SELECT DISTINCT date FROM daily_kline ORDER BY date").fetchdf()["date"])
dates = pd.to_datetime(dates)
idx = {d: i for i, d in enumerate(dates)}

test_days = [d for d in dates if d >= TEST_START]
train_days = [d for d in dates if d < TEST_START]
first_test = test_days[0]
last_train = train_days[-1]

# Contaminated when the label window [D+16, D+20] ANY day reaches into test
# (median over the 5 closes: one test-period price pollutes it), i.e. D+20 >= first_test.
cutoff_i = idx[first_test] - LABEL_END
contam = [d for d in train_days if idx[d] >= cutoff_i]

print(f"TEST_START             : {TEST_START.date()}")
print(f"first test trading day : {first_test.date()}")
print(f"last train trading day : {last_train.date()}")
print(f"label window           : T+{LABEL_START}~T+{LABEL_END}")
print()
print(f"contaminated train window: {contam[0].date()} ~ {contam[-1].date()} ({len(contam)} trading days)")
print("sample D -> label uses close at:")
for d in [contam[0], contam[len(contam) // 2], contam[-1]]:
    i = idx[d]
    in_test = dates[i + LABEL_START] >= first_test
    print(f"  D={d.date()}  ->  close[{dates[i+LABEL_START].date()} .. {dates[i+LABEL_END].date()}]  in_test={in_test}")

n_pool = len(get_pool_codes())
total_train_rows = len(train_days) * n_pool
print()
print(f"pool size             : {n_pool} stocks")
print(f"contaminated rows     : ~{len(contam) * n_pool} ({len(contam)} days x {n_pool})")
print(f"total training rows   : ~{total_train_rows} ({len(train_days)} days x {n_pool})")
print(f"contaminated share    : ~{len(contam) * n_pool / total_train_rows:.2%}")
con.close()

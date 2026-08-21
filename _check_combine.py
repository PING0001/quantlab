# -*- coding: utf-8 -*-
"""
Synthetic-data checks for strategies/combine.py (v4: single score, no pct).

Run: python _check_combine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies.combine import combine_scores

fail = 0


def check(name: str, cond: bool):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fail += 1


def series(values: dict) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(list(values), names=["date", "code"])
    return pd.Series(list(values.values()), index=idx, dtype=float)


D1 = pd.Timestamp("2026-01-05")

p20 = series({(D1, "A"): 0.10, (D1, "B"): 0.05, (D1, "C"): -0.02, (D1, "D"): np.nan})
p6 = series({(D1, "A"): 0.03, (D1, "B"): np.nan, (D1, "C"): -0.01, (D1, "D"): 0.02})

s = combine_scores(p20, p6)

# 公式：0.6*20d + 0.4*6d（2026-08-21 用户改回 20d 主导）
check("blend A = 0.6*0.10+0.4*0.03", np.isclose(s.loc[(D1, "A")], 0.6 * 0.10 + 0.4 * 0.03))
check("blend C = 0.6*(-0.02)+0.4*(-0.01)", np.isclose(s.loc[(D1, "C")], 0.6 * -0.02 + 0.4 * -0.01))
# 缺 6d -> 重归一为 20d 原值；缺 20d -> 重归一为 6d 原值
check("missing 6d -> p20 renorm", np.isclose(s.loc[(D1, "B")], 0.05))
check("missing 20d -> p6 renorm", np.isclose(s.loc[(D1, "D")], 0.02))
# 双缺失 -> NaN
both_nan = combine_scores(series({(D1, "E"): np.nan}), series({(D1, "E"): np.nan}))
check("both-missing -> NaN", bool(both_nan.isna().all()))
check("no inf", bool(np.isfinite(s.dropna()).all()))
check("name is score", s.name == "score")
try:
    combine_scores(p20, p6, w20=0.5, w6=0.6)
    check("bad weights raise", False)
except ValueError:
    check("bad weights raise", True)

print("\nRESULT:", "ALL CHECKS PASSED" if fail == 0 else f"{fail} FAILED")
sys.exit(1 if fail else 0)

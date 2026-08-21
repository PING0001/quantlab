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

# v4 公式：0.4*20d + 0.6*6d
check("blend A = 0.4*0.10+0.6*0.03", np.isclose(s.loc[(D1, "A")], 0.4 * 0.10 + 0.6 * 0.03))
check("blend C = 0.4*(-0.02)+0.6*(-0.01)", np.isclose(s.loc[(D1, "C")], 0.4 * -0.02 + 0.6 * -0.01))
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

# ---- v8 三模型融合（2026-08-22）----
from strategies.combine import combine_scores3

p2 = series({(D1, "A"): 0.02, (D1, "B"): 0.01, (D1, "C"): -0.01, (D1, "D"): np.nan, (D1, "E"): 0.0})
s3 = combine_scores3(p2, p6, p20)

check("v8 A = 0.4*0.02+0.35*0.03+0.25*0.10",
      np.isclose(s3.loc[(D1, "A")], 0.4 * 0.02 + 0.35 * 0.03 + 0.25 * 0.10))
# B 缺 6d：按 0.4/0.25 重归一
check("v8 B missing 6d -> (0.4*p2+0.25*p20)/0.65",
      np.isclose(s3.loc[(D1, "B")], (0.4 * 0.01 + 0.25 * 0.05) / 0.65))
# D 缺 2d 与 20d：只剩 6d 原值
check("v8 D missing 2d/20d -> p6 renorm", np.isclose(s3.loc[(D1, "D")], 0.02))
# E 只有 2d：原值
check("v8 E only 2d -> p2 renorm", np.isclose(s3.loc[(D1, "E")], 0.0))
check("v8 no inf", bool(np.isfinite(s3.dropna()).all()))
try:
    combine_scores3(p2, p6, p20, w2d=0.5, w6=0.35, w20=0.25)
    check("v8 bad weights raise", False)
except ValueError:
    check("v8 bad weights raise", True)

print("\nRESULT:", "ALL CHECKS PASSED" if fail == 0 else f"{fail} FAILED")
sys.exit(1 if fail else 0)

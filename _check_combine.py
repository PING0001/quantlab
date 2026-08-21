# -*- coding: utf-8 -*-
"""
Synthetic-data checks for strategies/combine.py (plan Task 6).

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
    idx = pd.MultiIndex.from_tuples(values, names=["date", "code"])
    return pd.Series([v for _, v in values], index=idx, dtype=float)


D1 = pd.Timestamp("2026-01-05")
D2 = pd.Timestamp("2026-01-06")

# ---- 1. weighted percentile + exec on a clean day ----
p20 = series({(D1, "A"): 0.10, (D1, "B"): 0.05, (D1, "C"): -0.02, (D1, "D"): 0.01,
              (D2, "A"): 0.01, (D2, "B"): 0.01, (D2, "C"): 0.01, (D2, "D"): 0.01})
p6 = series({(D1, "A"): 0.03, (D1, "B"): 0.02, (D1, "C"): -0.01, (D1, "D"): 0.00,
             (D2, "A"): 0.02, (D2, "B"): 0.02, (D2, "C"): 0.02, (D2, "D"): 0.02})

out = combine_scores(p20, p6)

# 手算：D1 两个模型的百分位排序一致 [A=1.0, B=0.75, C=0.25, D=0.5]
exp_rank = {(D1, "A"): 1.00, (D1, "B"): 0.75, (D1, "C"): 0.25, (D1, "D"): 0.50}
got_rank = out["rank_score"].loc[list(exp_rank)].to_dict()
check("weighted rank (clean day)", np.allclose(sorted(got_rank.values()), sorted(exp_rank.values())))

exp_exec_A = 0.6 * 0.10 + 0.4 * 0.03
check("exec = raw weighted sum", np.isclose(out["exec_score"].loc[(D1, "A")], exp_exec_A))

# ---- 2. 单模型缺失：按可用权重重归一 ----
p6_missing = p6.drop((D1, "B"))
out2 = combine_scores(p20, p6_missing)
check("missing 6d -> rank renorm to 20d pct",
      np.isclose(out2["rank_score"].loc[(D1, "B")], 0.75))
check("missing 6d -> exec renorm to 20d raw",
      np.isclose(out2["exec_score"].loc[(D1, "B")], 0.05))
check("missing 6d -> other rows unaffected",
      np.isclose(out2["rank_score"].loc[(D1, "A")], 1.00))

# ---- 3. 全持平日：不制造离散 ----
d2_rank = out["rank_score"].xs(D2, level="date")
check("all-tied day collapses to 0.5", bool((d2_rank == 0.5).all()))
d2_exec = out["exec_score"].xs(D2, level="date")
check("all-tied day exec = 0.6*0.01+0.4*0.02",
      bool(np.allclose(d2_exec, 0.6 * 0.01 + 0.4 * 0.02)))

# ---- 4. 双缺失 -> NaN；值域检查 ----
p20_extra = pd.concat([p20, series({(D2, "E"): np.nan})])
p6_extra = pd.concat([p6, series({(D2, "E"): np.nan})])
out4 = combine_scores(p20_extra, p6_extra)
check("both-missing -> NaN", bool(out4.loc[(D2, "E")].isna().all()))
rs = out["rank_score"].dropna()
check("rank_score in [0,1]", bool(((rs >= 0) & (rs <= 1)).all()))
check("no inf anywhere", bool(np.isfinite(out.dropna().to_numpy()).all()))

# ---- 5. 权重校验 ----
try:
    combine_scores(p20, p6, w20=0.7, w6=0.5)
    check("bad weights raise", False)
except ValueError:
    check("bad weights raise", True)

print("\nRESULT:", "ALL CHECKS PASSED" if fail == 0 else f"{fail} FAILED")
sys.exit(1 if fail else 0)

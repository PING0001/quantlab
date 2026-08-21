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
    idx = pd.MultiIndex.from_tuples(list(values), names=["date", "code"])
    return pd.Series(list(values.values()), index=idx, dtype=float)


D1 = pd.Timestamp("2026-01-05")
D2 = pd.Timestamp("2026-01-06")

# ---- 1. weighted percentile + exec on a clean day ----
p20 = series({(D1, "A"): 0.10, (D1, "B"): 0.05, (D1, "C"): -0.02, (D1, "D"): 0.01,
              (D2, "A"): 0.01, (D2, "B"): 0.01, (D2, "C"): 0.01, (D2, "D"): 0.01})
p6 = series({(D1, "A"): 0.03, (D1, "B"): 0.02, (D1, "C"): -0.01, (D1, "D"): 0.00,
             (D2, "A"): 0.02, (D2, "B"): 0.02, (D2, "C"): 0.02, (D2, "D"): 0.02})

out = combine_scores(p20, p6)

# 手算（项目约定 (rank-0.5)/n）：D1 两模型排序一致
#   ranks: C=1, D=2, B=3, A=4 -> pct: C=0.125, D=0.375, B=0.625, A=0.875
exp_rank = {(D1, "A"): 0.875, (D1, "B"): 0.625, (D1, "C"): 0.125, (D1, "D"): 0.375}
got_rank = out["rank_score"].loc[list(exp_rank)].to_dict()
check("weighted rank (clean day)", np.allclose([got_rank[k] for k in exp_rank],
                                               [exp_rank[k] for k in exp_rank]))

exp_exec_A = 0.6 * 0.10 + 0.4 * 0.03
check("exec = raw weighted sum", np.isclose(out["exec_score"].loc[(D1, "A")], exp_exec_A))

# ---- 2. 单模型缺失：按可用权重重归一 ----
# D1 的 6d 横截面变 3 只（A=0.03,C=-0.01,D=0.00）-> pct: A=2.5/3, C=0.5/3, D=1.5/3
p6_missing = p6.drop((D1, "B"))
out2 = combine_scores(p20, p6_missing)
check("missing 6d -> rank renorm to 20d pct",
      np.isclose(out2["rank_score"].loc[(D1, "B")], 0.625))
check("missing 6d -> exec renorm to 20d raw",
      np.isclose(out2["exec_score"].loc[(D1, "B")], 0.05))
check("other rows use shrunk 6d cross-section",
      np.isclose(out2["rank_score"].loc[(D1, "A")], 0.6 * 0.875 + 0.4 * (2.5 / 3)))

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

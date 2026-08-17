# -*- coding: utf-8 -*-
"""Strategy performance report — single-page HTML with cumulative return curves."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Data ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / "backtest" / "mainboard_microcap"
OUTPUT_PATH = ROOT / "document" / "strategy_report_20260717.html"

equity_df = pd.read_csv(BACKTEST_DIR / "equity_lgb_20d_5d_rebalance.csv", index_col=0)
bench_df = pd.read_csv(BACKTEST_DIR / "benchmark.csv", index_col=0)

equity_df.index = pd.to_datetime(equity_df.index)
bench_df.index = pd.to_datetime(bench_df.index)

port_eq = equity_df["Equity"]
port_dd = equity_df["Drawdown"]
bench_eq = bench_df["equity"]
bench_ret = bench_df["daily_ret"].dropna()

# ── Metrics ──────────────────────────────────────────────────────────────────
def _metrics(eq: pd.Series, rf: float = 0.025) -> dict:
    ret = eq.pct_change().dropna()
    n = len(ret)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = float((1 + total) ** (252 / n) - 1) if n > 0 else 0
    ann_std = float(ret.std() * np.sqrt(252))
    sharpe = float((ret.mean() * 252 - rf) / ann_std) if ann_std > 0 else 0
    downside = ret[ret < 0]
    sortino = float(ret.mean() * np.sqrt(252) / downside.std()) if len(downside) > 0 else 0
    dd = (eq - eq.cummax()) / eq.cummax()
    return {
        "total": total, "cagr": cagr, "sharpe": sharpe,
        "sortino": sortino, "max_dd": float(dd.min()),
        "win_rate": float((ret > 0).mean()), "n_days": n,
    }

p = _metrics(port_eq)
b = _metrics(bench_eq)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 5.5), dpi=120,
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
    facecolor="white",
)

ax1.plot(port_eq.index, port_eq / port_eq.iloc[0],
         label=f"Portfolio (Total: +{p['total']:.1%})",
         color="#1a73e8", linewidth=1.5)
ax1.plot(bench_eq.index, bench_eq,
         label=f"Benchmark (Total: +{b['total']:.1%})",
         color="#999999", linewidth=1.0, alpha=0.8)
ax1.set_ylabel("Cumulative Return (×)", fontsize=9)
ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax1.grid(True, alpha=0.15)
ax1.tick_params(axis="both", labelsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.fill_between(port_dd.index, port_dd * 100, 0,
                  color="#d32f2f", alpha=0.3, label="Portfolio Drawdown")
ax2.set_ylabel("Drawdown (%)", fontsize=9)
ax2.grid(True, alpha=0.15)
ax2.tick_params(axis="both", labelsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.tight_layout(pad=2)

buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight")
plt.close()
img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

# ── HTML ─────────────────────────────────────────────────────────────────────
period_start = str(port_eq.index[0].date())
period_end = str(port_eq.index[-1].date())
n_days = int(p["n_days"])

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantlab Strategy Report — 2026-07-17</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f4f5f7; color: #1a1a2e; line-height: 1.6; padding: 40px 20px;
  }}
  .container {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 8px; }}
  .subtitle {{ color: #555; font-size: 13px; margin-bottom: 20px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 24px 0 10px; }}
  .chart {{ background: #fff; border-radius: 8px; padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .chart img {{ width: 100%; border-radius: 4px; }}
  .desc {{ background: #fff; border-radius: 6px; padding: 14px 18px;
            font-size: 13px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
            line-height: 1.8; margin-top: 10px; }}
  .metrics {{ display: flex; gap: 20px; margin-top: 16px; }}
  .metric {{ background: #fff; border-radius: 6px; padding: 14px 24px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); flex: 1; text-align: center; }}
  .metric .val {{ font-size: 22px; font-weight: 700; }}
  .metric .lbl {{ font-size: 12px; color: #777; margin-top: 4px; }}
  .pos {{ color: #0a8f4a; }}
  .neg {{ color: #c0392b; }}
  footer {{ margin-top: 32px; font-size: 11px; color: #aaa; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>Quantlab Strategy Report</h1>
  <p class="subtitle">{period_start} ~ {period_end} · {n_days} 个交易日</p>

  <h2>累计收益</h2>
  <div class="chart"><img src="data:image/png;base64,{img_b64}" alt="Cumulative Returns"></div>

  <h2>策略说明</h2>
  <p class="desc">
    主板微盘股（流通市值 1-20 亿），LightGBM 三分类模型预测个股 20 日周期中位数收益率，
    5 个交易日调仓，每次持仓 Top 10，限价单次日执行。
  </p>

  <h2>绩效指标</h2>
  <div class="metrics">
    <div class="metric">
      <div class="val pos">+{p['total']:.2%}</div>
      <div class="lbl">总收益率</div>
    </div>
    <div class="metric">
      <div class="val">{p['sharpe']:.2f}</div>
      <div class="lbl">夏普比率</div>
    </div>
    <div class="metric">
      <div class="val neg">{p['max_dd']:.2%}</div>
      <div class="lbl">最大回撤</div>
    </div>
  </div>

  <footer>Quantlab · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>
</div>
</body>
</html>"""

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(html, encoding="utf-8")
print(f"Report written to: {OUTPUT_PATH}")

"""
forecast_display -- Load three saved MLP models (3d / 4d / 5d) and predict scores
for the latest trading day.  Outputs a self-contained HTML report with
per-horizon predictions and a weighted-composite ranking.

Usage (from project root):
    python forecast_display/generate.py

All files are contained inside forecast_display/ -- deleting this folder
leaves the rest of the project untouched.
"""
from __future__ import annotations

import sys
import datetime
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies import MLPStrategy

DB_PATH = ROOT / "data" / "ashare.duckdb"

MODEL_PATHS = {
    3: ROOT / "models" / "mlp_horizon3.pt",
    4: ROOT / "models" / "mlp_horizon4.pt",
    5: ROOT / "models" / "mlp_horizon5.pt",
    10: ROOT / "models" / "mlp_horizon10.pt",
}
WEIGHTS = {3: 0.25, 4: 0.25, 5: 0.25, 10: 0.25}
HORIZONS = [3, 4, 5, 10]

HERE = Path(__file__).resolve().parent
HTML_DIR = HERE / "html"


def load_factors(con):
    df = con.execute("SELECT * FROM factor_values").fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_name_map(con):
    info = con.execute("SELECT code, name FROM stock_info").fetchdf()
    return dict(zip(info["code"], info["name"]))


def load_and_predict():
    print("=== forecast_display: Multi-horizon MLP prediction (3 loaded models) ===\n")

    for h in HORIZONS:
        if not MODEL_PATHS[h].exists():
            print(f"ERROR: No saved model at {MODEL_PATHS[h]}")
            print("Run `python run_mlp_multi.py` first to train and persist all models.")
            sys.exit(1)

    models = {}
    for h in HORIZONS:
        print(f"[1/3] Loading model for horizon={h}d ...")
        strat = MLPStrategy.load(MODEL_PATHS[h])
        models[h] = strat
        print(f"      layers={strat._config['hidden_layer_sizes']}, factors={len(strat.factor_names)}")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("\n[2/3] Loading factor data ...")
    factors = load_factors(con)

    print("      loading stock names ...")
    name_map = load_name_map(con)
    con.close()

    ref_cols = models[3].factor_names
    available_cols = [c for c in ref_cols if c in factors.columns]
    factors = factors[available_cols].dropna()

    all_dates = sorted(factors.index.get_level_values("date").unique())
    latest_date = all_dates[-1]
    n_latest = (factors.index.get_level_values("date") == latest_date).sum()
    print(f"      date range: {all_dates[0].date()} ~ {latest_date.date()}")
    print(f"      stocks on latest date: {n_latest}")

    print(f"\n[3/3] Predicting scores for {latest_date.date()} ...")
    X_latest = factors.xs(latest_date, level="date", drop_level=False)

    for h in HORIZONS:
        preds = models[h].predict(X_latest)
        if isinstance(preds.index, pd.MultiIndex):
            preds = preds.droplevel("date")

        if h == 3:
            all_codes = preds.index
            results = pd.DataFrame({"code": all_codes})
            results["name"] = results["code"].map(name_map).fillna(results["code"])

        results[f"score_{h}d"] = results["code"].map(preds.to_dict()).astype(float)

    results["composite"] = (
        WEIGHTS[3] * results["score_3d"]
        + WEIGHTS[4] * results["score_4d"]
        + WEIGHTS[5] * results["score_5d"]
    + WEIGHTS[10] * results["score_10d"]
    )

    score_cols = [f"score_{h}d" for h in HORIZONS] + ["composite"]
    results = results.dropna(subset=score_cols).reset_index(drop=True)

    results = results.sort_values("composite", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)
    results = results[["rank", "code", "name", "score_3d", "score_4d", "score_5d", "score_10d", "composite"]]

    named_count = sum(
        1 for n in results["name"] if n.isdigit() is False
    )
    print(f"      predictions: {len(results)} stocks scored ({named_count} with names)")

    meta = {
        "prediction_date": str(latest_date.date()),
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_stocks": len(results),
        "horizons": HORIZONS,
        "weights": WEIGHTS,
        "model_info": {
            str(h): {
                "layers": str(models[h]._config["hidden_layer_sizes"]),
                "factor_count": len(models[h].factor_names),
            }
            for h in HORIZONS
        },
        "positive_count": int((results["composite"] > 0).sum()),
        "negative_count": int((results["composite"] <= 0).sum()),
    }

    print(f"\n    Top 5 by composite score:")
    for _, row in results.head(5).iterrows():
        print(f"      {row['rank']:3d}. {row['code']:6s}  {row['name']:8s}  "
              f"3d={row['score_3d']:+.6f} 4d={row['score_4d']:+.6f} 5d={row['score_5d']:+.6f} 10d={row['score_10d']:+.6f} "
              f"composite={row['composite']:+.6f}")

    return results, meta


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLP Multi-Horizon Prediction -- {prediction_date}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f4f5f7; color: #1a1a2e; line-height: 1.5; padding: 24px 16px;
  }}
  .container {{ max-width: 1300px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; }}
  header h1 {{ font-size: 22px; font-weight: 600; color: #0d0d1a; }}
  header .meta {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; font-size: 13px; color: #555;
  }}
  header .meta span {{ background: #e8ecf1; padding: 3px 10px; border-radius: 4px; }}
  header .stats {{ display: flex; gap: 20px; margin-top: 14px; }}
  .stat-box {{
    background: #fff; border: 1px solid #e0e3e8; border-radius: 6px;
    padding: 10px 18px; min-width: 100px; text-align: center;
  }}
  .stat-box .val {{ font-size: 20px; font-weight: 700; }}
  .stat-box .lbl {{ font-size: 11px; color: #777; margin-top: 2px; }}
  .stat-box.pos .val {{ color: #0a8f4a; }}
  .stat-box.neg .val {{ color: #c0392b; }}
  .toolbar {{ display: flex; gap: 8px; margin-bottom: 14px; align-items: center; flex-wrap: wrap; }}
  .toolbar input {{
    padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; width: 220px;
  }}
  .toolbar input:focus {{ outline: none; border-color: #4a6cf7; }}
  .toolbar .hint {{ font-size: 12px; color: #999; }}
  table {{
    width: 100%; border-collapse: collapse; background: #fff;
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}
  thead {{ background: #2c2c3a; color: #f0f0f5; }}
  th {{
    padding: 10px 8px; text-align: right; font-size: 12px; font-weight: 500; white-space: nowrap;
  }}
  th:first-child {{ text-align: center; width: 44px; }}
  th:nth-child(2) {{ text-align: left; }}
  th:nth-child(3) {{ text-align: left; }}
  td {{
    padding: 8px; font-size: 12px; border-bottom: 1px solid #eef0f4; text-align: right; white-space: nowrap;
  }}
  td:first-child {{ text-align: center; color: #888; font-size: 11px; width: 44px; }}
  td:nth-child(2) {{ text-align: left; font-family: "SF Mono", "Cascadia Code", "Consolas", monospace; }}
  td:nth-child(3) {{ text-align: left; }}
  tbody tr:hover {{ background: #f7f8fb; }}
  .score-pos {{ color: #0a8f4a; font-weight: 600; }}
  .score-neg {{ color: #c0392b; font-weight: 600; }}
  .score-bar {{
    display: inline-block; height: 7px; border-radius: 3px;
    vertical-align: middle; margin-right: 4px; min-width: 2px;
  }}
  .hidden {{ display: none; }}
  footer {{
    margin-top: 24px; font-size: 11px; color: #aaa; text-align: center;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>MLP Multi-Horizon Prediction</h1>
    <div class="meta">
      <span>Pred Date: {prediction_date}</span>
      {model_tags}
      <span>Weight: {weight_display}</span>
      <span>Generated: {generated_at}</span>
    </div>
    <div class="stats">
      <div class="stat-box"><div class="val">{n_stocks}</div><div class="lbl">Total</div></div>
      <div class="stat-box pos"><div class="val">{positive_count}</div><div class="lbl">Bullish (composite &gt; 0)</div></div>
      <div class="stat-box neg"><div class="val">{negative_count}</div><div class="lbl">Bearish (composite &le; 0)</div></div>
    </div>
  </header>
  <div class="toolbar">
    <input type="text" id="filter" placeholder="Search code or name..." oninput="doFilter()">
    <span class="hint">Ranked by composite score (descending)</span>
  </div>
  <table>
    <thead><tr>
      <th>#</th>
      <th>Code</th>
      <th>Name</th>
      <th>3d Pred</th>
      <th>4d Pred</th>
      <th>5d Pred</th>
      <th>10d Pred</th>
      <th>Composite</th>
    </tr></thead>
    <tbody id="table-body">
{table_rows}
    </tbody>
  </table>
  <footer>
    Composite = (3d + 4d + 5d + 10d) / 4 &middot;
    Three independent MLP models &middot; For reference only
  </footer>
</div>
<script>
  function doFilter() {{
    var q = document.getElementById("filter").value.trim().toLowerCase();
    var rows = document.querySelectorAll("#table-body tr");
    rows.forEach(function(tr) {{
      tr.classList.toggle("hidden", q !== "" && tr.textContent.toLowerCase().indexOf(q) === -1);
    }});
  }}
</script>
</body>
</html>"""


def build_model_tags(meta):
    tags = []
    info = meta["model_info"]
    for h in meta["horizons"]:
        i = info[str(h)]
        tags.append(f'<span>Model {h}d: MLP{i["layers"]}, {i["factor_count"]} factors</span>')
    return " ".join(tags)


def _score_cell(val):
    cls = "score-pos" if val > 0 else "score-neg"
    px_per_unit = 1200  # 1% (0.01) -> 12px
    max_px = 150
    bar_w = max(int(min(abs(val) * px_per_unit, max_px)), 0)
    bar_w = max(bar_w, 2) if val != 0 else 0
    bar_color = "#0a8f4a" if val > 0 else "#c0392b"
    bar = f'<span class="score-bar" style="width:{bar_w}px;background:{bar_color};"></span>'
    return f'<td class="{cls}">{bar}{val:+.6f}</td>'


def build_html(scored, meta):
    rows = []

    for _, r in scored.iterrows():
        row = (
            f"<tr>"
            f"<td>{r['rank']}</td>"
            f"<td>{r['code']}</td>"
            f"<td>{r['name']}</td>"
            f"{_score_cell(r['score_3d'])}"
            f"{_score_cell(r['score_4d'])}"
            f"{_score_cell(r['score_5d'])}"
            f"{_score_cell(r['score_10d'])}"
            f"{_score_cell(r['composite'])}"
            f"</tr>"
        )
        rows.append(row)

    weight_display = "/".join(f"{WEIGHTS[h]:.2f}" for h in HORIZONS)

    return HTML_TEMPLATE.format(
        table_rows="\n".join(rows),
        model_tags=build_model_tags(meta),
        weight_display=weight_display,
        **meta,
    )


def main():
    print("Generating multi-horizon forecast HTML ...")
    scored, meta = load_and_predict()
    html = build_html(scored, meta)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    pred_date = meta["prediction_date"]
    html_path = HTML_DIR / f"{pred_date}_forecast.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\n=== HTML written to: {html_path} ===")
    print(f"    Open with: start {html_path}")


if __name__ == "__main__":
    main()

"""
forecast_display (LightGBM) -- Load the LightGBM model and predict scores
for the latest trading day across 1d/3d/5d/10d horizons.
Outputs a self-contained HTML report.

Usage (from project root):
    python forecast_display/generate_lgb.py
"""
from __future__ import annotations

import sys
import datetime
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies import LGBStrategy

from config import DB_PATH, POOL_NAME, get_pool_codes, get_lgb_model_path, get_forecast_lgb_dir

MODEL_PATH = get_lgb_model_path()
WEIGHTS = {5: 0.25, 10: 0.25, 20: 0.25, 30: 0.25}

HTML_DIR = get_forecast_lgb_dir()


def load_factors(con):
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    df = con.execute(
        f"SELECT * FROM factor_values WHERE code IN ({placeholders})",
        pool_codes,
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_name_map(con):
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    info = con.execute(
        f"SELECT code, name FROM stock_info WHERE code IN ({placeholders})",
        pool_codes,
    ).fetchdf()
    return dict(zip(info["code"], info["name"]))


def load_sw_l3(con, mapping: dict[str, int]):
    """Load SW L3 integer categories using the model's saved mapping.
    Unknown categories → -1.
    """
    df = con.execute(
        "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pd.Series(dtype=int)
    df["sw_l3"] = df["sw_l3_code"].map(mapping).fillna(-1).astype(int)
    return pd.Series(df["sw_l3"].values, index=df["code"], name="sw_l3")


def load_and_predict(target_date=None):
    print("=== forecast_display: LightGBM prediction ===\n")

    if not MODEL_PATH.exists():
        print(f"ERROR: No saved model at {MODEL_PATH}")
        print("Run `python run_lgb.py` first to train and persist the LightGBM model.")
        sys.exit(1)

    print(f"[1/3] Loading LightGBM model from {MODEL_PATH} ...")
    model = LGBStrategy.load(MODEL_PATH)
    n_trees = {}
    for h in model.horizons:
        if h in model._models:
            n_trees[h] = model._models[h].booster_.current_iteration()
    tree_info = " ".join(f"{h}d={n_trees.get(h, '?')}" for h in model.horizons)
    print(f"      horizons={model.horizons}, factors={len(model.factor_names)}, trees={tree_info}")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("\n[2/3] Loading factor data ...")
    factors = load_factors(con)

    print("      loading stock names ...")
    name_map = load_name_map(con)

    available_cols = [c for c in model.factor_names if c in factors.columns]
    factors = factors[available_cols]

    # ---- add sw_l3 categorical feature if model expects it ----
    if hasattr(model, '_categorical_feature') and model._categorical_feature:
        saved_mapping = model._category_mappings.get("sw_l3")
        if not saved_mapping:
            raise RuntimeError(
                "模型缺少 sw_l3 类别映射，请使用当前代码重新训练模型"
            )
        sw_l3 = load_sw_l3(con, mapping=saved_mapping)
        if not sw_l3.empty:
            idx_codes = factors.index.get_level_values("code")
            factors["sw_l3"] = idx_codes.map(sw_l3).fillna(-1).astype(int)

    all_dates = sorted(factors.index.get_level_values("date").unique())
    if target_date is not None:
        target_ts = pd.Timestamp(target_date)
        if target_ts not in all_dates:
            print(f"ERROR: target date {target_date} not found in factor data")
            print(f"Available dates: {[str(d.date()) for d in all_dates[-10:]]}")
            sys.exit(1)
        latest_date = target_ts
    else:
        latest_date = all_dates[-1]

    # Build exclusion set: ST/退 + stocks listed < 90 days
    excluded = {c for c, n in name_map.items() if "ST" in n or "退" in n}
    cutoff = (latest_date - pd.Timedelta(days=90)).date()
    warmup = con.execute(
        "SELECT code FROM stock_info WHERE list_date > ?",
        [cutoff],
    ).fetchall()
    for (c,) in warmup:
        excluded.add(c)
    print(f"      excluded (ST/退: {len(excluded) - len(warmup)} + listed<90d: {len(warmup)} = {len(excluded)} stocks)")
    con.close()

    n_latest = (factors.index.get_level_values("date") == latest_date).sum()
    print(f"      date range: {all_dates[0].date()} ~ {latest_date.date()}")
    print(f"      stocks on latest date: {n_latest}")

    print(f"\n[3/3] Predicting scores for {latest_date.date()} ...")
    X_latest = factors.xs(latest_date, level="date", drop_level=False)
    X_latest = X_latest.fillna(0)
    pred_df = model.predict(X_latest)

    if isinstance(pred_df.index, pd.MultiIndex):
        pred_df.index = pred_df.index.droplevel("date")

    # Exclude ST/delisting stocks from display
    keep_mask = ~pred_df.index.isin(excluded)
    pred_df = pred_df.loc[keep_mask]

    results = pd.DataFrame({"code": pred_df.index})
    results["name"] = results["code"].map(name_map).fillna(results["code"])

    # Use model's actual horizon columns
    model_horizons = list(pred_df.columns)
    weights: dict = {}
    total_w = 0
    for col in model_horizons:
        h = col.replace("pred_", "")
        if h == "median_5d":
            w = 0.5
        elif h.endswith("d"):
            try:
                w = WEIGHTS.get(int(h[:-1]), 0.25)
            except (ValueError, TypeError):
                w = 0.25
        else:
            w = 0.25
        weights[col] = w
        total_w += w
    if total_w > 0:
        for col in model_horizons:
            weights[col] /= total_w

    for col in model_horizons:
        results[f"score_{col.replace('pred_', '')}"] = results["code"].map(pred_df[col].to_dict()).astype(float)

    score_cols = [f"score_{col.replace('pred_', '')}" for col in model_horizons]
    results["composite"] = sum(
        results[f"score_{col.replace('pred_', '')}"] * weights[col]
        for col in model_horizons
    )
    results = results.dropna(subset=score_cols + ["composite"]).reset_index(drop=True)

    results = results.sort_values("composite", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)

    display_cols = ["rank", "code", "name"] + score_cols + ["composite"]
    results = results[display_cols]

    named_count = sum(1 for n in results["name"] if not n.isdigit())
    print(f"      predictions: {len(results)} stocks scored ({named_count} with names)")

    # Build display-friendly horizon labels
    display_horizons = [col.replace("pred_", "") for col in model_horizons]
    display_weights = {h: weights[f"pred_{h}"] for h in display_horizons}

    cfg = model._config
    meta = {
        "prediction_date": str(latest_date.date()),
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_stocks": len(results),
        "horizons": display_horizons,
        "weights": display_weights,
        "model_info": {
            "path": str(MODEL_PATH),
            "num_leaves": cfg["num_leaves"],
            "learning_rate": cfg["learning_rate"],
            "min_child_samples": cfg["min_child_samples"],
            "reg_alpha": cfg["reg_alpha"],
            "reg_lambda": cfg["reg_lambda"],
            "factor_count": len(model.factor_names),
            "trees": tree_info,
        },
        "positive_count": int((results["composite"] > 0).sum()),
        "negative_count": int((results["composite"] <= 0).sum()),
    }

    print(f"\n    Top 5 by composite score:")
    for _, row in results.head(5).iterrows():
        scores = " ".join(f"{h}={row[f'score_{h}']:+.6f}" for h in display_horizons)
        print(f"      {row['rank']:3d}. {row['code']:6s}  {row['name']:8s}  "
              f"{scores}  composite={row['composite']:+.6f}")

    return results, meta


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LightGBM Multi-Horizon Prediction -- {prediction_date}</title>
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
    <h1>LightGBM Multi-Horizon Prediction</h1>
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
{header_cols}
      <th>Composite</th>
    </tr></thead>
    <tbody id="table-body">
{table_rows}
    </tbody>
  </table>
  <footer>
    Composite = {footer_formula} &middot;
    LightGBM model &middot; For reference only
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
    info = meta["model_info"]
    return (
        f"<span>LightGBM: leaves={info['num_leaves']}, "
        f"lr={info['learning_rate']}, "
        f"{info['factor_count']} factors</span>"
    )


def _score_cell(val):
    cls = "score-pos" if val > 0 else "score-neg"
    px_per_unit = 1200
    max_px = 150
    bar_w = max(int(min(abs(val) * px_per_unit, max_px)), 0)
    bar_w = max(bar_w, 2) if val != 0 else 0
    bar_color = "#0a8f4a" if val > 0 else "#c0392b"
    bar = f'<span class="score-bar" style="width:{bar_w}px;background:{bar_color};"></span>'
    return f'<td class="{cls}">{bar}{val:+.6f}</td>'


def build_html(scored, meta):
    rows = []
    horizons = meta["horizons"]
    weights = meta["weights"]
    score_cols = [f"score_{h}" for h in horizons]
    for _, r in scored.iterrows():
        cells = "".join(_score_cell(r[col]) for col in score_cols)
        row = (
            f"<tr>"
            f"<td>{r['rank']}</td>"
            f"<td>{r['code']}</td>"
            f"<td>{r['name']}</td>"
            f"{cells}"
            f"{_score_cell(r['composite'])}"
            f"</tr>"
        )
        rows.append(row)

    meta["table_rows"] = "\n".join(rows)
    meta["model_tags"] = build_model_tags(meta)
    meta["weight_display"] = "/".join(f"{weights[h]:.2f}" for h in horizons)
    meta["header_cols"] = "".join(f"<th>{h} Pred</th>" for h in horizons)
    meta["footer_formula"] = " + ".join(f"{weights[h]:.2f}*{h}" for h in horizons)

    return HTML_TEMPLATE.format(**meta)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LightGBM forecast HTML")
    parser.add_argument("--date", type=str, default=None,
                        help="Target prediction date (YYYY-MM-DD). Default: latest available.")
    args = parser.parse_args()

    print("Generating LightGBM forecast HTML ...")
    scored, meta = load_and_predict(target_date=args.date)
    html = build_html(scored, meta)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    pred_date = meta["prediction_date"]
    html_path = HTML_DIR / f"{pred_date}_forecast_lgb.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\n=== HTML written to: {html_path} ===")
    print(f"    Open with: start {html_path}")


if __name__ == "__main__":
    main()

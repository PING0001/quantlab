"""
Generate LightGBM prediction history HTML for last 20 trading days.
Shows composite score evolution (heatmap) for all pool stocks, sorted
by latest date's composite score. ST/delisting stocks are excluded.

Usage: python forecast_display/generate_lgb_history.py
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
WEIGHTS = {3: 0.25, 5: 0.35, 10: 0.25, 20: 0.15}
HORIZONS = [3, 5, 10, 20]
N_DAYS = 20

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


def load_and_predict():
    print(f"=== LightGBM {N_DAYS}-Day Prediction History ===\n")

    if not MODEL_PATH.exists():
        print(f"ERROR: No saved model at {MODEL_PATH}")
        sys.exit(1)

    print(f"[1/3] Loading LightGBM model from {MODEL_PATH} ...")
    model = LGBStrategy.load(MODEL_PATH)
    tree_info = " ".join(
        f"{h}d={model._models[h].booster_.current_iteration()}"
        for h in model.horizons if h in model._models
    )
    print(f"      horizons={model.horizons}, factors={len(model.factor_names)}, trees={tree_info}")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    print(f"\n[2/3] Loading factor data ...")
    factors = load_factors(con)
    name_map = load_name_map(con)

    # Exclude ST/delisting stocks
    excluded = {c for c, n in name_map.items() if "ST" in n or "退" in n}
    print(f"      excluded (ST/退): {len(excluded)} stocks")
    con.close()

    available_cols = [c for c in model.factor_names if c in factors.columns]
    factors = factors[available_cols]

    all_dates = sorted(factors.index.get_level_values("date").unique())
    target_dates = all_dates[-N_DAYS:]
    print(f"      date range: {all_dates[0].date()} ~ {all_dates[-1].date()}")
    print(f"      predicting {len(target_dates)} dates: {target_dates[0].date()} ~ {target_dates[-1].date()}")

    print(f"\n[3/3] Predicting scores for {len(target_dates)} dates ...")

    records = []
    for dt in target_dates:
        X_dt = factors.xs(dt, level="date", drop_level=False)
        X_dt = X_dt.fillna(0)
        pred_df = model.predict(X_dt)

        if isinstance(pred_df.index, pd.MultiIndex):
            pred_df.index = pred_df.index.droplevel("date")

        # Exclude ST/delisting
        keep = ~pred_df.index.isin(excluded)
        pred_df = pred_df.loc[keep]

        # Composite score
        composite = np.zeros(len(pred_df))
        for h in HORIZONS:
            col = f"pred_{h}d"
            if col in pred_df.columns:
                composite += WEIGHTS[h] * pred_df[col].values
        records.append((dt, composite, pred_df.index))

    # Build pivot table: rows=codes, columns=dates, values=composite
    all_codes = set()
    for _, _, codes in records:
        all_codes.update(codes)

    latest_dt = target_dates[-1]
    latest_scores = {}
    for codes, comp in [(r[2], r[1]) for r in records if r[0] == latest_dt]:
        for c, v in zip(codes, comp):
            latest_scores[c] = v

    # Create matrix
    code_list = sorted(all_codes, key=lambda c: latest_scores.get(c, -999), reverse=True)
    n_stocks = len(code_list)

    # Build score matrix: rows=stocks, cols=dates
    score_matrix = np.full((n_stocks, len(target_dates)), np.nan)
    for j, (dt, comp, codes) in enumerate(records):
        code_to_score = {c: v for c, v in zip(codes, comp)}
        for i, c in enumerate(code_list):
            if c in code_to_score:
                score_matrix[i, j] = code_to_score[c]

    # Stats
    n_positive = (score_matrix[:, -1] > 0).sum()
    n_negative = (score_matrix[:, -1] <= 0).sum()

    print(f"      predictions: {n_stocks} stocks x {len(target_dates)} dates")

    # === build HTML ===
    return build_html(code_list, score_matrix, target_dates, name_map, model,
                      n_stocks, n_positive, n_negative, tree_info)


def _color_scale(val):
    """Map a value to a background color: red(-) -> white(0) -> green(+)."""
    if np.isnan(val):
        return "#f0f0f0"
    cap = 0.05
    v = max(-cap, min(cap, val)) / cap
    if v >= 0:
        r = int(255 * (1 - v))
        g = 255
        b = int(255 * (1 - v))
    else:
        r = 255
        g = int(255 * (1 + v))
        b = int(255 * (1 + v))
    return f"rgb({r},{g},{b})"


def build_html(codes, scores, dates, name_map, model, n_stocks, n_pos, n_neg, tree_info):
    cfg = model._config
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred_start = dates[0].strftime("%Y-%m-%d")
    pred_end = dates[-1].strftime("%Y-%m-%d")

    # Date headers
    date_headers = "".join(
        f'<th>{d.strftime("%m-%d")}</th>' for d in dates
    )

    # Table rows
    rows_html = []
    for i in range(n_stocks):
        code = codes[i]
        name = name_map.get(code, code)
        cells = ""
        for j in range(len(dates)):
            v = scores[i, j]
            bg = _color_scale(v)
            txt = f"{v:+.4f}" if not np.isnan(v) else ""
            cells += f'<td style="background:{bg}">{txt}</td>'
        rows_html.append(
            f'<tr>'
            f'<td>{i + 1}</td>'
            f'<td>{code}</td>'
            f'<td>{name}</td>'
            f'{cells}'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LightGBM Prediction History -- {pred_start} ~ {pred_end}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f4f5f7; color: #1a1a2e; line-height: 1.5; padding: 16px 8px;
  }}
  .container {{ max-width: 1800px; margin: 0 auto; }}
  header {{ margin-bottom: 20px; }}
  header h1 {{ font-size: 20px; font-weight: 600; }}
  header .meta {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; font-size: 12px; color: #555;
  }}
  header .meta span {{ background: #e8ecf1; padding: 2px 8px; border-radius: 4px; }}
  .stats {{ display: flex; gap: 16px; margin-top: 10px; }}
  .stat-box {{
    background: #fff; border: 1px solid #e0e3e8; border-radius: 6px;
    padding: 8px 14px; min-width: 80px; text-align: center;
  }}
  .stat-box .val {{ font-size: 18px; font-weight: 700; }}
  .stat-box .lbl {{ font-size: 11px; color: #777; }}
  .stat-box.pos .val {{ color: #0a8f4a; }}
  .stat-box.neg .val {{ color: #c0392b; }}
  .toolbar {{ display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }}
  .toolbar input {{
    padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; width: 200px;
  }}
  .toolbar .hint {{ font-size: 11px; color: #999; }}
  .wrapper {{ overflow-x: auto; }}
  table {{
    width: 100%; border-collapse: collapse; background: #fff;
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06);
    font-size: 10px;
  }}
  thead {{ background: #2c2c3a; color: #f0f0f5; position: sticky; top: 0; z-index: 1; }}
  th {{
    padding: 6px 5px; text-align: center; font-size: 10px; font-weight: 500; white-space: nowrap;
  }}
  th:first-child {{ width: 30px; }}
  th:nth-child(2) {{ text-align: left; }}
  th:nth-child(3) {{ text-align: left; white-space: nowrap; }}
  td {{
    padding: 3px 5px; font-size: 9px; border-bottom: 1px solid #eef0f4;
    text-align: center; white-space: nowrap; font-family: monospace;
  }}
  td:first-child {{ color: #888; width: 30px; font-family: sans-serif; }}
  td:nth-child(2) {{ text-align: left; font-family: monospace; }}
  td:nth-child(3) {{ text-align: left; white-space: nowrap; font-family: sans-serif; }}
  tbody tr:hover {{ outline: 2px solid #4a6cf7; outline-offset: -2px; }}
  .hidden {{ display: none; }}
  footer {{ margin-top: 20px; font-size: 11px; color: #aaa; text-align: center; }}
  .legend {{ display: flex; gap: 4px; align-items: center; font-size: 10px; color: #777; margin-bottom: 8px; }}
  .legend .bar {{ width: 100px; height: 14px; border-radius: 3px;
    background: linear-gradient(to right, #c0392b, #fff, #0a8f4a); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>LightGBM Prediction History ({N_DAYS} Days)</h1>
    <div class="meta">
      <span>Period: {pred_start} ~ {pred_end}</span>
      <span>Horizons: {'/'.join(str(h)+'d' for h in HORIZONS)}</span>
      <span>Weights: {'/'.join(f'{WEIGHTS[h]:.2f}' for h in HORIZONS)}</span>
      <span>Factors: {len(model.factor_names)}</span>
      <span>Generated: {gen_time}</span>
    </div>
    <div class="stats">
      <div class="stat-box"><div class="val">{n_stocks}</div><div class="lbl">Total Stocks</div></div>
      <div class="stat-box pos"><div class="val">{n_pos}</div><div class="lbl">Latest Bullish</div></div>
      <div class="stat-box neg"><div class="val">{n_neg}</div><div class="lbl">Latest Bearish</div></div>
    </div>
  </header>
  <div class="toolbar">
    <input type="text" id="filter" placeholder="Search code or name..." oninput="doFilter()">
    <span class="hint">Composite = {' + '.join(f'{WEIGHTS[h]:.2f}x{h}d' for h in HORIZONS)} &middot;
    Sorted by latest ({dates[-1].strftime('%m-%d')}) composite &middot; ST/退 excluded</span>
  </div>
  <div class="legend">
    <span>Bearish</span>
    <div class="bar"></div>
    <span>Bullish</span>
    <span style="margin-left: 8px">(cap: ±0.05)</span>
  </div>
  <div class="wrapper">
  <table>
    <thead><tr>
      <th>#</th>
      <th>Code</th>
      <th>Name</th>
      {date_headers}
    </tr></thead>
    <tbody id="table-body">
{"\n".join(rows_html)}
    </tbody>
  </table>
  </div>
  <footer>
    LightGBM model (leaves={cfg['num_leaves']}, lr={cfg['learning_rate']}, trees={tree_info})
    &middot; For reference only
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

    return html


def main():
    scored_html = load_and_predict()
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    path = HTML_DIR / f"{today}_forecast_lgb_history.html"
    path.write_text(scored_html, encoding="utf-8")
    print(f"\n=== HTML written to: {path} ===")
    print(f"    Open with: start {path}")


if __name__ == "__main__":
    main()

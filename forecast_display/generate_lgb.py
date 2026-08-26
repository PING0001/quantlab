# -*- coding: utf-8 -*-
"""
forecast_display (dual-regression v8) — 三模型融合预测报告 + 三级降级。

报告主体：读三个预测 parquet（open2d / 6d / 20d，均 next_open 锚）+ 各自
meta，融合分 score = 0.40*p_open2d + 0.35*p_6d + 0.25*p_20d（与回测同款
combine_scores3，缺失侧重归一），Top 榜按 score 降序，附三成分分列与
卖出阈值（Σ wᵢ×calib_medianᵢ，来自各 meta；降级时按可用权重重归一）。
gap1d 为独立实验模型（不接 score/回测），不入报告。

三级降级链（任何情况不 exit 1 断流——本脚本是夜间流水线末段）：
  L1 三 parquet+meta 齐全            → 完整报告
  L2 部分缺失（文件/列/当日无行）    → 可用成分重归一融合，顶部 DOWNGRADED
                                       横幅注明缺哪个
  L3 全缺 / 目标日无效 / 任何异常    → 红色警告占位报告（"预测缺失，数据/
                                       训练待查"），仍写 HTML、exit 0

⚠️ 刻意不回退旧三分类 lgb_multi.joblib：该模型是 2026-08-13 训练的带泄漏
版本（标签前视 + 前复权 VIEW 公式错误两项审计都在），不作出数来源。

ST/退市三层防御与 backtest/run_lgb.py 同款：
  ① 名称快照含 "ST"/"退"（兜底） ② IsST 因子（当日时点，主力）
  ③ delist_info（date >= delist_date）

注意：预测 parquet 止于 TEST_END（walk_forward 硬裁剪），报告日期 =
产物中最新可用日，非自然"今天"。

Usage:
    python forecast_display/generate_lgb.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import datetime
import json
import sys
import traceback
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DB_PATH, POOL_NAME, get_lgb_predictions_path,
                    get_lgb_predictions_meta_path, get_forecast_lgb_dir)

# 融合权重与预测列名单源：直接 import 回测入口，防两处漂移暗改有效权重
# （v8 的核心教训--幅度/权重漂移曾在主窗口摆动 ±20pp）
from backtest.run_lgb import PRED_COLS, W2D, W6D, W20D
from strategies.lgb import combine_scores3

# 报告成分（gap1d 独立实验模型，刻意排除）
COMPONENTS: dict[str, float] = {"open2d": W2D, "6d": W6D, "20d": W20D}
HTML_DIR = get_forecast_lgb_dir()


# ============================================================================
# 数据加载（每层独立 try，任何一层失败都只影响该层/该成分，不致命）
# ============================================================================
def load_components() -> dict[str, dict]:
    """读三成分 parquet+meta；单成分任何问题 → 该成分缺席（进入 L2）。"""
    comps: dict[str, dict] = {}
    for m, w in COMPONENTS.items():
        ppath = get_lgb_predictions_path(m)
        mpath = get_lgb_predictions_meta_path(m)
        if not (ppath.exists() and mpath.exists()):
            continue
        try:
            meta = json.loads(mpath.read_text())
            pdf = pd.read_parquet(ppath)
            col = PRED_COLS[m]
            if col not in pdf.columns:
                continue
            s = pdf[col].dropna()
            if s.empty:
                continue
            comps[m] = {"series": s, "meta": meta, "w": w}
        except Exception as e:  # noqa: BLE001 — 单成分损坏不拖垮整份报告
            print(f"  WARNING: component {m} load failed: {e}")
    return comps


def load_name_map(codes: list[str]) -> dict[str, str]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        ph = ",".join(["?"] * len(codes))
        info = con.execute(
            f"SELECT code, name FROM stock_info WHERE code IN ({ph})", codes
        ).fetchdf()
        return dict(zip(info["code"], info["name"]))
    finally:
        con.close()


def load_st_delist_excluded(target_date: pd.Timestamp,
                            codes: list[str]) -> tuple[set[str], list[str]]:
    """三层防御的 ②③：IsST 当日时点 + delist_date。①名称快照由调用方合并。

    返回 (排除集, 各层计数描述)；DB 读失败时返回空集（只丢防御层，不致命）。
    """
    excluded: set[str] = set()
    parts: list[str] = []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        ph = ",".join(["?"] * len(codes))
        day = str(target_date.date())

        isst = con.execute(
            f"SELECT code FROM factor_values WHERE code IN ({ph}) "
            f"AND date = ? AND IsST = 1",
            [*codes, day],
        ).fetchall()
        st_set = {c for (c,) in isst}
        excluded |= st_set
        parts.append(f"IsST@{day}={len(st_set)}")

        dl = con.execute(
            f"SELECT code, delist_date FROM delist_info WHERE code IN ({ph})",
            codes,
        ).fetchdf()
        if not dl.empty:
            dl_dates = pd.to_datetime(dl["delist_date"])
            dl_set = set(dl.loc[dl_dates <= target_date, "code"])
            excluded |= dl_set
            parts.append(f"delisted={len(dl_set)}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"DB 读失败（防御降级）: {e}")
    finally:
        con.close()
    return excluded, parts


# ============================================================================
# 报告构建
# ============================================================================
def live_day_predictions(target_date: pd.Timestamp,
                         comps: dict[str, dict]) -> tuple[dict[str, pd.Series], list[str]]:
    """前沿日期实时推理：冻结 joblib × calib_slope（parquet 止于 TEST_END，不含前沿）。

    模型与训练完全一致（只加载不重训）；输出与 parquet 同口径（已乘校准斜率）。
    单成分失败只降级该成分（走 L2 缺失重归一），不炸整报；
    SQL 加池过滤防池外行混入（2026-08-24 审计 #11）。
    返回 ({model: 当日预测 Series(code 索引)}, 缺失/降级披露)。"""
    from strategies.lgb import LGBStrategy
    from config import get_lgb_model_path
    from pools.membership import latest_codes

    pool_codes = sorted(latest_codes())   # 池时点化：LIVE 过滤 = 最新档成员
    ph = ",".join(["?"] * len(pool_codes))
    day = str(target_date.date())
    day_series: dict[str, pd.Series] = {}
    notes: list[str] = []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sw = con.execute(
            "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
        ).fetchdf()
        sw_map = dict(zip(sw["code"], sw["sw_l3_code"]))
        for m, c in comps.items():
            try:
                strategy = LGBStrategy.load(get_lgb_model_path(m))
                meta = c["meta"]
                fnames = list(strategy.factor_names)
                fcols = [f for f in fnames if f != "sw_l3"]
                df = con.execute(
                    f"SELECT code, {', '.join(fcols)} FROM factor_values "
                    f"WHERE date = ? AND code IN ({ph})",
                    [day, *pool_codes]).fetchdf()
                if df.empty:
                    notes.append(f"{m}: 因子表当日无池内行")
                    continue
                miss = [f for f in fcols if df[f].isna().all()]
                if miss:
                    notes.append(f"{m} 当日整列缺失（树容忍）: {', '.join(miss)}")
                X = df.set_index("code")
                if "sw_l3" in fnames:
                    mapping = strategy._category_mappings.get("sw_l3", {})
                    X["sw_l3"] = (X.index.map(sw_map).map(mapping)
                                  if mapping else X.index.map(sw_map))
                    X["sw_l3"] = pd.to_numeric(X["sw_l3"], errors="coerce").fillna(-1).astype(int)
                X = X[fnames]
                pred = strategy.predict(X)
                pcol = [c2 for c2 in pred.columns if c2.startswith("pred_")][0]
                k = float(meta.get("results", {}).get("calib_slope", 1.0))
                day_series[m] = pred[pcol] * k
            except Exception as e:  # noqa: BLE001 — 单成分失败只降级该成分
                notes.append(f"{m}: 实时推理失败（{type(e).__name__}: {e}），按缺失成分处理")
    finally:
        con.close()
    return day_series, notes


def build_day_frame(comps: dict[str, dict], target_date: pd.Timestamp,
                    name_st: set[str], day_series: dict[str, pd.Series] | None = None,
                    live_notes: list[str] | None = None, is_live: bool = False,
                    ) -> tuple[pd.DataFrame, dict] | None:
    """指定日融合 + 三层防御过滤。返回 (results, report_meta) 或 None（当日无任何行）。

    day_series 传入时跳过 parquet 切片（前沿实时推理路径）。"""
    if day_series is None:
        day_series = {}
        for m, c in comps.items():
            try:
                day_series[m] = c["series"].xs(target_date, level="date")
            except KeyError:
                continue
    if not day_series:
        return None

    missing = [m for m in COMPONENTS if m not in day_series]
    avail_w = sum(COMPONENTS[m] for m in day_series)
    empty = pd.Series(dtype=float)
    score = combine_scores3(
        day_series.get("open2d", empty),
        day_series.get("6d", empty),
        day_series.get("20d", empty),
        w2d=W2D, w6=W6D, w20=W20D,
    )

    # 卖出零点：Σ (wᵢ/w_avail) × calib_medianᵢ（降级时与融合同口径重归一；
    # meta 缺 calib_median 的成分按 0 贡献——与 backtest 回退语义一致）
    sell_zero = 0.0
    med_parts: list[str] = []
    for m, s in day_series.items():
        med = comps[m]["meta"].get("results", {}).get("calib_median")
        if med is not None:
            w_eff = COMPONENTS[m] / avail_w
            sell_zero += w_eff * float(med)
            med_parts.append(f"{m}={float(med):+.5f}")

    codes = sorted(score.index.astype(str))
    name_map = load_name_map(codes)
    st_delist, layer_notes = load_st_delist_excluded(target_date, codes)
    excluded = name_st & set(codes) | st_delist

    df = pd.DataFrame({"code": score.index.astype(str), "score": score.values})
    for m in ("open2d", "6d", "20d"):
        if m in day_series:
            s = day_series[m]
            df[m] = df["code"].map(s.to_dict())
    df["name"] = df["code"].map(name_map).fillna(df["code"])
    df = df[~df["code"].isin(excluded)].dropna(subset=["score"])
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    model_cards = []
    for m, s in day_series.items():
        meta = comps[m]["meta"]
        ic = meta.get("results", {}).get("test_ic", {})
        model_cards.append({
            "model": m,
            "ic": ic.get("mean_ic", float("nan")),
            "ir": ic.get("ir", float("nan")),
            "slope": meta.get("results", {}).get("calib_slope"),
            "median": meta.get("results", {}).get("calib_median"),
            "train_end": meta.get("train_end"),
            "n_factors": len(meta.get("factor_names", [])),
            "w_eff": COMPONENTS[m] / avail_w,
        })

    report_meta = {
        "prediction_date": str(target_date.date()),
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "missing": missing,
        "degraded": bool(missing),
        "live": is_live,
        "live_notes": live_notes or [],
        "sell_zero": sell_zero,
        "med_parts": med_parts,
        "layer_notes": layer_notes,
        "models": model_cards,
    }
    return df, report_meta


# ============================================================================
# HTML
# ============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dual-Regression Forecast — {prediction_date}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f4f5f7; color: #1a1a2e; line-height: 1.5; padding: 24px 16px;
  }}
  .container {{ max-width: 1300px; margin: 0 auto; }}
  .banner {{ border-radius: 6px; padding: 10px 16px; margin-bottom: 16px;
            font-size: 14px; font-weight: 600; }}
  .banner.downgraded {{ background: #fff4e0; border: 1px solid #e8a23d; color: #9a6200; }}
  .banner.fatal {{ background: #fdecea; border: 2px solid #c0392b; color: #8e2418;
                   font-size: 16px; padding: 18px 20px; }}
  header {{ margin-bottom: 20px; }}
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
  .models {{ display: flex; gap: 12px; margin: 14px 0; flex-wrap: wrap; }}
  .model-card {{
    background: #fff; border: 1px solid #e0e3e8; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; min-width: 220px;
  }}
  .model-card b {{ font-size: 13px; }}
  .model-card div {{ color: #555; margin-top: 2px; }}
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
    padding: 8px 12px; text-align: right; font-size: 12px; font-weight: 500; white-space: nowrap;
  }}
  th:first-child {{ text-align: center; width: 40px; }}
  th:nth-child(2) {{ text-align: left; }}
  th:nth-child(3) {{ text-align: left; }}
  td {{
    padding: 6px 12px; font-size: 12px; border-bottom: 1px solid #eef0f4;
    text-align: right; white-space: nowrap;
  }}
  td:first-child {{ text-align: center; color: #888; font-size: 11px; width: 40px; }}
  td:nth-child(2) {{ text-align: left; font-family: "SF Mono", "Cascadia Code", "Consolas", monospace; }}
  td:nth-child(3) {{ text-align: left; }}
  tbody tr:hover {{ background: #f7f8fb; }}
  .score-pos {{ color: #0a8f4a; font-weight: 600; }}
  .score-neg {{ color: #c0392b; font-weight: 600; }}
  .muted {{ color: #999; }}
  .score-bar {{
    display: inline-block; height: 7px; border-radius: 3px;
    vertical-align: middle; margin-right: 4px; min-width: 2px;
  }}
  .hidden {{ display: none; }}
  footer {{ margin-top: 24px; font-size: 11px; color: #aaa; text-align: center; }}
</style>
</head>
<body>
<div class="container">
{banner}
  <header>
    <h1>Dual-Regression Forecast (v8 blend)</h1>
    <div class="meta">
      <span>Pred Date: {prediction_date}</span>
      {live_badge}
      <span>Generated: {generated_at}</span>
      <span>Sell zero: {sell_zero:+.5f}</span>
      <span>Pool: {pool_name}</span>
    </div>
    {live_notes_html}
    <div class="stats">
      <div class="stat-box"><div class="val">{n_stocks}</div><div class="lbl">Total</div></div>
      <div class="stat-box pos"><div class="val">{above_count}</div><div class="lbl">&gt; sell zero</div></div>
      <div class="stat-box neg"><div class="val">{below_count}</div><div class="lbl">&le; sell zero</div></div>
    </div>
    <div class="models">{model_cards_html}</div>
  </header>
  <div class="toolbar">
    <input type="text" id="filter" placeholder="Search code or name..." oninput="doFilter()">
    <span class="hint">Ranked by blended score (descending); components in raw return units</span>
  </div>
  <table>
    <thead><tr>
      <th>#</th>
      <th>Code</th>
      <th>Name</th>
      <th>open2d</th>
      <th>6d</th>
      <th>20d</th>
      <th>Score</th>
    </tr></thead>
    <tbody id="table-body">
{table_rows}
    </tbody>
  </table>
  <footer>
    score = {w2d:.2f}&middot;open2d + {w6d:.2f}&middot;6d + {w20d:.2f}&middot;20d (renormalized if a component is missing)<br>
    sell zero = &Sigma; w&#7522;&middot;calib_median&#7522; = {med_formula} | {layer_line}<br>
    Predictions end at TEST_END (walk-forward hard cut) — report date is the latest available, not necessarily today.<br>
    Dual-regression LightGBM (L1 + output calibration) &middot; For reference only
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


def _fmt(v, fmt="{:+.4f}", na="—"):
    return na if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)


def _cell(val, neutral: float, bar: bool) -> str:
    dev = (val if pd.notna(val) else 0.0) - neutral
    cls = "score-pos" if dev > 0 else "score-neg"
    prefix = ""
    if bar and pd.notna(val):
        px = max(min(int(abs(dev) * 1500), 110), 2)
        color = "#0a8f4a" if dev > 0 else "#c0392b"
        prefix = f'<span class="score-bar" style="width:{px}px;background:{color};"></span>'
    txt = f"{val:+.4f}" if pd.notna(val) else '<span class="muted">—</span>'
    return f'<td class="{cls}">{prefix}{txt}</td>'


def build_html(df: pd.DataFrame, meta: dict) -> str:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{r['rank']}</td>"
            f"<td>{r['code']}</td>"
            f"<td>{r['name']}</td>"
            f"{_cell(r.get('open2d'), 0.0, False)}"
            f"{_cell(r.get('6d'), 0.0, False)}"
            f"{_cell(r.get('20d'), 0.0, False)}"
            f"{_cell(r['score'], meta['sell_zero'], True)}"
            f"</tr>"
        )

    if meta["degraded"]:
        banner = (
            '<div class="banner downgraded">DOWNGRADED：缺失成分 '
            + ", ".join(meta["missing"])
            + "——融合分按可用权重重归一，因子/权重语义与完整版不同，仅供参考。</div>"
        )
    else:
        banner = ""

    cards = []
    for c in meta["models"]:
        cards.append(
            f'<div class="model-card"><b>{c["model"]} (w_eff {c["w_eff"]:.2f})</b>'
            f"<div>IC {_fmt(c['ic'], '{:+.4f}')} | IR {_fmt(c['ir'], '{:.2f}')}"
            f" | calib slope {_fmt(c['slope'], '{:.2f}')}"
            f" | median {_fmt(c['median'], '{:+.5f}')}"
            f"<br>train_end {c['train_end']} | {c['n_factors']} factors</div></div>"
        )

    m = dict(meta)
    m.update({
        "banner": banner,
        "pool_name": POOL_NAME,
        "table_rows": "\n".join(rows),
        "model_cards_html": "\n".join(cards),
        "n_stocks": len(df),
        "above_count": int((df["score"] > meta["sell_zero"]).sum()),
        "below_count": int((df["score"] <= meta["sell_zero"]).sum()),
        "w2d": W2D, "w6d": W6D, "w20d": W20D,
        "med_formula": "; ".join(meta["med_parts"]) or "meta 缺 calib_median，回退 0",
        "layer_line": "防御层: " + ", ".join(meta["layer_notes"]),
        "live_badge": ('<span style="background:#e3f2e8;color:#0a8f4a;font-weight:600;">'
                       "LIVE 实时推理（冻结模型×校准，前沿日）</span>") if meta.get("live") else "",
        "live_notes_html": (('<div style="margin-top:6px;font-size:12px;color:#9a6200;'
                             "background:#fff8e6;padding:6px 10px;border-radius:4px;\">"
                             "前沿缺失披露：" + "；".join(meta.get("live_notes", []))
                             + "</div>") if meta.get("live") and meta.get("live_notes") else ""),
    })
    return HTML_TEMPLATE.format(**m)


PLACEHOLDER_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Forecast UNAVAILABLE</title>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f4f5f7;
         padding: 40px 16px; color: #1a1a2e; }}
  .box {{ max-width: 760px; margin: 0 auto; }}
  .banner {{
    background: #fdecea; border: 3px solid #c0392b; color: #8e2418;
    border-radius: 8px; padding: 24px 28px; font-size: 18px; font-weight: 700;
    text-align: center;
  }}
  .detail {{
    margin-top: 16px; background: #fff; border: 1px solid #e0e3e8; border-radius: 8px;
    padding: 16px 20px; font-size: 13px; color: #444; line-height: 1.7;
  }}
  code {{ background: #eef0f4; padding: 1px 6px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="box">
  <div class="banner">&#9888; 预测缺失 — 数据 / 训练待查</div>
  <div class="detail">
    <div>生成时间：{generated_at}</div>
    <div>原因：{reason}</div>
    <div>排查顺序：
      <ol>
        <li><code>python -m factors.update --dry-run</code> — 因子是否算到最新交易日</li>
        <li><code>python -m data.pull</code> — 行情/因子数据是否缺口</li>
        <li><code>python run_lgb.py</code> — 重训三模型（open2d / 6d / 20d）</li>
      </ol>
    </div>
    <div>本占位报告代替断流：旧三分类 lgb_multi.joblib 为带泄漏模型（2026-08-13），
    已裁定不作出数来源，故不回退。</div>
  </div>
</div>
</body>
</html>"""


def write_placeholder(reason: str) -> Path:
    # 不用 .format：reason 可能含 traceback 花括号
    html = (PLACEHOLDER_TEMPLATE
            .replace("{generated_at}",
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            .replace("{reason}", reason))
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    out = HTML_DIR / f"{datetime.date.today():%Y-%m-%d}_forecast_lgb.html"
    out.write_text(html, encoding="utf-8")
    return out


# ============================================================================
# main
# ============================================================================
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Dual-regression v8 forecast HTML")
    parser.add_argument("--date", type=str, default=None,
                        help="Target prediction date (YYYY-MM-DD). Default: latest in parquets.")
    args = parser.parse_args()

    print("Generating dual-regression (v8) forecast HTML ...")
    try:
        comps = load_components()
        if not comps:
            out = write_placeholder(
                "三个预测 parquet（open2d/6d/20d）全部缺失或不可读 — "
                f"期望路径 {get_lgb_predictions_path('<model>')}（run_lgb.py 产物）")
            print(f"  [L3] 全部成分缺失 → 占位报告: {out}")
            return

        last_dates = {m: c["series"].index.get_level_values("date").max()
                      for m, c in comps.items()}
        avail_dates: set = set()
        for c in comps.values():
            avail_dates |= set(c["series"].index.get_level_values("date").unique())

        # 因子表最新日期（前沿）：预测产物止于 TEST_END，前沿日期走实时推理
        con = duckdb.connect(str(DB_PATH), read_only=True)
        latest_factor_date = pd.Timestamp(
            con.execute("SELECT max(date) FROM factor_values").fetchone()[0])
        con.close()

        if args.date is not None:
            target = pd.Timestamp(args.date)
            if target not in avail_dates and target != latest_factor_date:
                out = write_placeholder(
                    f"指定日期 {args.date} 不在预测产物范围内且非因子表最新日；"
                    f"产物范围 {min(last_dates.values()).date()} ~ "
                    f"{max(last_dates.values()).date()}，因子最新 {latest_factor_date.date()}")
                print(f"  [L3] 指定日期无预测 → 占位报告: {out}")
                return
        else:
            target = max(max(last_dates.values()), latest_factor_date)

        # 2026-08-24 用户裁定：名称快照层退役——ST/退市判定 = 日度 IsST +
        # delist 日期（build_day_frame 内的 ②③ 层，均为时点口径）
        name_st = set()

        # 前沿日期（parquet 未覆盖）→ 冻结模型实时推理；否则走 parquet 切片
        live_notes: list[str] = []
        is_live = target not in avail_dates
        if is_live:
            print(f"  [LIVE] {target.date()} 超出预测产物范围 "
                  f"(止于 {max(last_dates.values()).date()})，冻结模型实时推理")
            day_series, live_notes = live_day_predictions(target, comps)
            built = build_day_frame(comps, target, name_st, day_series=day_series,
                                    live_notes=live_notes, is_live=True)
        else:
            built = build_day_frame(comps, target, name_st)
        if built is None:
            out = write_placeholder(f"目标日 {target.date()} 无任何预测行")
            print(f"  [L3] 当日无预测行 → 占位报告: {out}")
            return
        df, meta = built

        html = build_html(df, meta)
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        out = HTML_DIR / f"{meta['prediction_date']}_forecast_lgb.html"
        out.write_text(html, encoding="utf-8")

        level = "L2 DOWNGRADED" if meta["degraded"] else "L1 完整"
        print(f"  [{level}] date={meta['prediction_date']} stocks={len(df)} "
              f"sell_zero={meta['sell_zero']:+.5f}"
              + (f" missing={meta['missing']}" if meta["degraded"] else ""))
        print(f"  防御层: {', '.join(meta['layer_notes'])}, name-ST/退={len(name_st)}")
        print(f"\n  Top 5 by blended score:")
        for _, r in df.head(5).iterrows():
            comp = " ".join(
                f"{m}=" + (f"{r[m]:+.4f}" if pd.notna(r.get(m)) else "—")
                for m in ("open2d", "6d", "20d"))
            print(f"    {r['rank']:3d}. {r['code']:6s} {str(r['name'])[:8]:8s} "
                  f"score={r['score']:+.4f}  ({comp})")
        print(f"\n=== HTML written to: {out} ===")
    except Exception:  # noqa: BLE001 — 夜间流水线末段，任何异常都转占位不断流
        out = write_placeholder(
            "生成过程异常：<pre>" + traceback.format_exc(limit=8) + "</pre>")
        print(f"  [L3] 异常 → 占位报告: {out}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

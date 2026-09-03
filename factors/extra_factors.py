# -*- coding: utf-8 -*-
"""
Non-alpha factors computed from raw OHLCV data and supplementary tables.
"""

from datetime import date, timedelta

import polars as pl


def third_friday(y: int, m: int) -> date:
    """该月第三个周五（股指期货 IF/IH/IC/IM 交割日的日历基准）。
    注意：交割日逢法定假日顺延至下一交易日，顺延逻辑在
    compute._load_days_to_delivery（trading_calendar 感知），此处只给日历基准。"""
    first = date(y, m, 1)
    off = (4 - first.weekday()) % 7      # 周一=0..周日=6，周五=4
    return first + timedelta(days=off + 14)


def compute_non_alpha_factors(df_long: pl.DataFrame) -> pl.DataFrame:
    """
    Compute non-alpha factors from OHLCV data.

    Parameters
    ----------
    df_long : pl.DataFrame
        Long-format with columns:
        [datetime, vt_symbol, open, high, low, close, volume,
         total_mv, circ_mv, list_date, vwap]

    Returns
    -------
    pl.DataFrame with columns [datetime, vt_symbol, factor1, factor2, ...]
    """
    cs = pl.col("close")
    os = pl.col("open")
    hs = pl.col("high")
    ls = pl.col("low")
    vs = pl.col("volume")
    sym = pl.col("vt_symbol")

    # Keep all source columns for computation, will select factor columns at end
    result = df_long.sort(["datetime", "vt_symbol"])

    # ---- momentum ----
    result = result.with_columns([
        (cs / cs.shift(3).over("vt_symbol") - 1).alias("Return_3d"),
        (cs / cs.shift(5).over("vt_symbol") - 1).alias("Return_5d"),
        (cs / cs.shift(20).over("vt_symbol") - 1).alias("Return_20d"),
        (-(cs / cs.shift(60).over("vt_symbol") - 1)).alias("Reversal_60d"),
    ])

    # ---- intraday ----
    result = result.with_columns([
        ((os - cs.shift(1).over("vt_symbol")) / cs.shift(1).over("vt_symbol")).alias("Gap_pct"),
        ((hs - ls) / os).alias("Intraday_range_pct"),
        ((cs - os) / os).alias("Intraday_return"),
    ])

    # ---- returns for volatility ----
    ret1d = (cs / cs.shift(1).over("vt_symbol") - 1)
    result = result.with_columns(ret1d.alias("_ret1d"))

    # ---- volatility ----
    result = result.with_columns([
        (pl.col("_ret1d").rolling_std(3, min_samples=1).over("vt_symbol")).alias("Volatility_3d"),
        (pl.col("_ret1d").rolling_std(20, min_samples=1).over("vt_symbol")).alias("Volatility"),
        (pl.col("_ret1d").rolling_std(60, min_samples=1).over("vt_symbol")).alias("Volatility_60d"),
    ])

    # ---- ATR (Average True Range, 14-day) ----
    tr = pl.max_horizontal(
        hs - ls,
        (hs - cs.shift(1).over("vt_symbol")).abs(),
        (ls - cs.shift(1).over("vt_symbol")).abs(),
    )
    result = result.with_columns(tr.alias("_tr"))
    # Exponential smoothing for ATR (Wilder's method approximation)
    # 2026-08-22 审计修复：绝对水平 ATR 建在 qfq 上，分母含 latest_adj 未来
    # 信息；改比值形式 ATR_pct（True Range 均值 / close），尺度无关且时点干净
    atr = pl.col("_tr").rolling_mean(14, min_samples=1).over("vt_symbol")
    result = result.with_columns((atr / cs).alias("ATR_pct"))

    # ---- Bollinger Band width ----
    ma20 = cs.rolling_mean(20, min_samples=1).over("vt_symbol")
    std20 = cs.rolling_std(20, min_samples=1).over("vt_symbol")
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    result = result.with_columns(((upper - lower) / ma20).alias("Bollinger_width"))

    # ---- Price Position 252d ----
    c_min_252 = cs.rolling_min(252, min_samples=1).over("vt_symbol")
    c_max_252 = cs.rolling_max(252, min_samples=1).over("vt_symbol")
    result = result.with_columns(
        ((cs - c_min_252) / (c_max_252 - c_min_252 + 1e-10)).alias("Price_position_252d")
    )

    # ---- Price Position 5d（Price_position 家族短窗形式）----
    l_min_5 = ls.rolling_min(5, min_samples=1).over("vt_symbol")
    h_max_5 = hs.rolling_max(5, min_samples=1).over("vt_symbol")
    result = result.with_columns(
        ((cs - l_min_5) / (h_max_5 - l_min_5 + 1e-10)).alias("Price_position_5d")
    )

    # ---- Stochastic K (14d) ----
    min_low_14 = ls.rolling_min(14, min_samples=1).over("vt_symbol")
    max_high_14 = hs.rolling_max(14, min_samples=1).over("vt_symbol")
    result = result.with_columns(
        ((cs - min_low_14) / (max_high_14 - min_low_14 + 1e-10)).alias("Stochastic_K")
    )

    # ---- SMA (20d) → CloseBIAS_20d（2026-08-22 审计修复）----
    # 绝对水平 SMA 同样带 qfq latest_adj 未来信息；换乖离率 close/ma20 − 1
    result = result.with_columns((cs / ma20 - 1).alias("CloseBIAS_20d"))

    # ---- MACD histogram (12, 26, 9) ----
    # 2026-08-22 审计修复：MACD_signal 为价格水平差（qfq 未来信息），
    # 改归一化柱 (DIF − DEA)/close
    ema12 = cs.ewm_mean(span=12, min_periods=1).over("vt_symbol")
    ema26 = cs.ewm_mean(span=26, min_periods=1).over("vt_symbol")
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm_mean(span=9, min_periods=1).over("vt_symbol")
    result = result.with_columns(((macd_line - signal_line) / cs).alias("MACD_hist_pct"))

    # ---- Return Skew (20d) ----
    result = result.with_columns(
        pl.col("_ret1d").rolling_skew(20, min_samples=5).over("vt_symbol").alias("Return_skew_20d")
    )

    # ---- Trend Strength (correlation of close with linear ramp, 20d) ----
    # Approximate as ts_rsquare of close
    n = 20
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    mean_x = (n - 1) / 2
    var_x = sum_x2 / n - mean_x * mean_x
    sum_xy_expr = pl.sum_horizontal([
        (n - 1 - j) * cs.shift(j) for j in range(n)
    ])
    df_tmp = result.with_columns([
        cs.rolling_sum(n, min_samples=n).over("vt_symbol").alias("sum_y"),
        cs.rolling_var(n, min_samples=n, ddof=0).over("vt_symbol").alias("var_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy"),
    ])
    df_tmp = df_tmp.with_columns((pl.col("sum_y") / n).alias("mean_y"))
    df_tmp = df_tmp.with_columns(
        (pl.col("sum_xy") / n - mean_x * pl.col("mean_y")).alias("cov_xy")
    )
    df_tmp = df_tmp.select([
        pl.col("datetime"), pl.col("vt_symbol"),
        (pl.col("cov_xy").pow(2) / (var_x * pl.col("var_y"))).alias("Trend_strength")
    ])
    df_tmp = df_tmp.with_columns(
        pl.when(pl.col("Trend_strength").is_infinite() | pl.col("Trend_strength").is_nan())
        .then(None)
        .otherwise(pl.col("Trend_strength"))
        .alias("Trend_strength")
    )
    result = result.join(df_tmp, on=["datetime", "vt_symbol"], how="left")

    # ---- Volume Ratio ----
    result = result.with_columns(
        (vs / vs.rolling_mean(20, min_samples=1).over("vt_symbol")).alias("Volume_ratio")
    )

    # ---- Amihud Illiquidity ----
    result = result.with_columns([
        (pl.col("_ret1d").abs() / (vs + 1e-10)).alias("Amihud_illiquidity"),
        ((pl.col("_ret1d").abs() / (vs + 1e-10))
         .rolling_mean(3, min_samples=1).over("vt_symbol")).alias("Amihud_3d"),
    ])

    # ---- Avg Amount 90d (daily amount, Tushare unit: 千元) ----
    result = result.with_columns([
        pl.col("amount").rolling_mean(3, min_samples=1).over("vt_symbol").alias("AvgAmount_3d"),
        pl.col("amount").rolling_mean(90, min_samples=1).over("vt_symbol").alias("AvgAmount_90d"),
    ])

    # ---- LnMktCap / LnFloatCap ----
    result = result.with_columns([
        pl.col("total_mv").log().alias("LnMktCap"),
        pl.col("circ_mv").log().alias("LnFloatCap"),
    ])

    # ---- Turnover ----
    # amount(千元) / circ_mv(万元) / 10 = 换手率(小数)：amount*1e3/(circ_mv*1e4)
    turnover = pl.col("amount") / pl.col("circ_mv") / 10
    result = result.with_columns(turnover.alias("_turnover_1d"))
    result = result.with_columns([
        pl.col("_turnover_1d").rolling_mean(3, min_samples=1).over("vt_symbol").alias("Turnover_3d"),
    ])
    result = result.with_columns([
        (pl.col("Turnover_3d") / pl.col("Turnover_3d").rolling_mean(20, min_samples=1).over("vt_symbol")).alias("Turnover_3d_ratio"),
    ])

    # ---- Intraday shape（日线 OHLCV 衍生日内形态，bench 2026-08 spec §3.8）----
    # 一字板（high==low）的 ClosePos/OpenPos 置 NULL 而非 epsilon 兜底：
    # 该日无日内信息，与 Price_position_252d 的 +1e-10 风格不同是有意的
    body_max = pl.max_horizontal(os, cs)
    body_min = pl.min_horizontal(os, cs)
    result = result.with_columns([
        ((hs - body_max) / cs).alias("UpperShadow"),
        ((body_min - ls) / cs).alias("LowerShadow"),
        pl.when(hs > ls).then((cs - ls) / (hs - ls)).otherwise(None).alias("ClosePos"),
        pl.when(hs > ls).then((os - ls) / (hs - ls)).otherwise(None).alias("OpenPos"),
    ])
    result = result.with_columns(
        pl.when((pl.col("UpperShadow") + pl.col("LowerShadow")) > 0)
        .then(pl.col("UpperShadow") / (pl.col("UpperShadow") + pl.col("LowerShadow")))
        .otherwise(None)
        .alias("ShadowRatio")
    )
    # 单位换手的价格波动效率（零换手日置 NULL，除零由 compute 末端 cleanup 兜底）
    result = result.with_columns(
        pl.when(pl.col("_turnover_1d") > 0)
        .then(pl.col("Intraday_range_pct") / pl.col("_turnover_1d"))
        .otherwise(None)
        .alias("RangeEfficiency")
    )
    result = result.with_columns([
        pl.col("ClosePos").rolling_mean(3, min_samples=1).over("vt_symbol").alias("ClosePos_mean_3d"),
        pl.col("ClosePos").rolling_mean(20, min_samples=1).over("vt_symbol").alias("ClosePos_mean_20d"),
        pl.col("ClosePos").rolling_std(20, min_samples=1).over("vt_symbol").alias("ClosePos_std_20d"),
    ])

    # ---- Cross-sectional Rank factors (percentile (rank-0.5)/n minus 0.5, ∈ (-0.5, 0.5)) ----
    def _pct_rank(col: str) -> pl.Expr:
        r = pl.col(col)
        return ((r.rank() - 0.5) / r.count()).over("datetime") - 0.5

    result = result.with_columns([
        _pct_rank("_ret1d").alias("Return_1d_rank"),
        _pct_rank("Return_20d").alias("Return_20d_rank"),
        _pct_rank("Turnover_3d").alias("Turnover_3d_rank"),
    ])

    # ---- HighVolCrowd_5d（2026-08-28 用户口述 101 系三明治：
    #      -rank( cov( rank(high), rank(volume), 5 ) )：
    #      当日横截面秩 → 个股 5 日时序协方差 → 横截面秩取负。
    #      经济含义：高点排名与量能排名同步抬升 = 放量上攻拥挤，反着做。
    #      内层秩参考系 = 当日池内成员（与 Return_1d_rank 同语义，按池隔离）。
    #      协方差用 E[xy]−E[x]E[y] 恒等式（polars 无双列 rolling_cov，
    #      同 VolPriceCorr_20d 的 vp_cov）。min_samples=5，不足为 NaN。）----
    def _pct_rank_e(e: pl.Expr) -> pl.Expr:
        return ((e.rank() - 0.5) / e.count()).over("datetime") - 0.5

    result = result.with_columns([
        _pct_rank_e(pl.col("high")).alias("_hv_rh"),
        _pct_rank_e(pl.col("volume")).alias("_hv_rv"),
    ])
    result = result.with_columns(
        (
            (pl.col("_hv_rh") * pl.col("_hv_rv")).rolling_mean(5, min_samples=5).over("vt_symbol")
            - pl.col("_hv_rh").rolling_mean(5, min_samples=5).over("vt_symbol")
            * pl.col("_hv_rv").rolling_mean(5, min_samples=5).over("vt_symbol")
        ).alias("_hv_cov5")
    )
    result = result.with_columns((-_pct_rank_e(pl.col("_hv_cov5"))).alias("HighVolCrowd_5d"))

    # ---- HighVolHeat_10d（2026-08-28 用户口述 101 系：
    #      (-1 * rank(std(high,10))) * correlation(high, volume, 10)——
    #      横截面秩(高点10日波动烈度) × 个股10日价量时序相关（乘法交互，
    #      非"乘积再取秩"）。经济含义：波动放大且高点放量 = 过热反指；
    #      corr<0（波动伴下跌）时因子转正。corr 用 E[xy]−E[x]E[y] / (σxσy)
    #      恒等式，零分母 guard 同 VolPriceCorr_20d；min_samples=10。）----
    result = result.with_columns([
        pl.col("high").rolling_std(10, min_samples=10).over("vt_symbol").alias("_hhv_std10"),
        pl.col("volume").rolling_std(10, min_samples=10).over("vt_symbol").alias("_hhv_volstd10"),
    ])
    _cov10 = (
        (pl.col("high") * pl.col("volume")).rolling_mean(10, min_samples=10).over("vt_symbol")
        - pl.col("high").rolling_mean(10, min_samples=10).over("vt_symbol")
        * pl.col("volume").rolling_mean(10, min_samples=10).over("vt_symbol")
    )
    result = result.with_columns(
        pl.when((pl.col("_hhv_std10") > 0) & (pl.col("_hhv_volstd10") > 0))
        .then(_cov10 / (pl.col("_hhv_std10") * pl.col("_hhv_volstd10")))
        .otherwise(None)
        .alias("_hhv_corr10")
    )
    result = result.with_columns(
        (-_pct_rank_e(pl.col("_hhv_std10")) * pl.col("_hhv_corr10")).alias("HighVolHeat_10d")
    )

    # ---- LnAge (trading days since list_date) ----
    result = result.with_columns(
        pl.col("list_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_list_dt")
    )
    result = result.with_columns(
        (pl.col("datetime").cast(pl.Date) - pl.col("_list_dt")).dt.total_days().alias("_age_days")
    )
    result = result.with_columns(
        pl.when(pl.col("_age_days") > 0)
        .then(pl.col("_age_days").cast(pl.Float64).log())
        .otherwise(None)
        .alias("LnAge")
    )

    # ---- LLM 挖矿第一批幸存因子（2026-08-22，document/llm_factor_mining/ 假设库
    # 首测 16 取 7；IC/相关性实证见 factors/test_new_factors.py，口径 2020~2025-06）----
    # 涨停次数：近似口径 _ret1d>=9.5%（前复权收益率，与 labels.py 的 limit 检测同族近似）
    up_flag = (pl.col("_ret1d") >= 0.095).cast(pl.Float64)
    # 近 10 日自 5 日高点的最大回撤（深度为正→回撤浅→未来跑赢）
    hi5_now = hs.rolling_max(5, min_samples=1).over("vt_symbol")
    dd5 = cs / hi5_now - 1
    # 量价相关（20d 滚动 Pearson，rolling_sum 手动展开；分母 0 → NULL）
    _n_vp = 20
    vol_ret_xy = vs * pl.col("_ret1d")
    vp_mean_x = vs.rolling_mean(_n_vp, min_samples=_n_vp).over("vt_symbol")
    vp_mean_y = pl.col("_ret1d").rolling_mean(_n_vp, min_samples=_n_vp).over("vt_symbol")
    vp_mean_xy = vol_ret_xy.rolling_mean(_n_vp, min_samples=_n_vp).over("vt_symbol")
    vp_var_x = vs.rolling_var(_n_vp, min_samples=_n_vp).over("vt_symbol")
    vp_var_y = pl.col("_ret1d").rolling_var(_n_vp, min_samples=_n_vp).over("vt_symbol")
    vp_cov = vp_mean_xy - vp_mean_x * vp_mean_y
    vp_denom = (vp_var_x * vp_var_y).sqrt()
    result = result.with_columns([
        up_flag.rolling_sum(20, min_samples=1).over("vt_symbol").alias("LimitUpCnt_20d"),
        dd5.rolling_min(10, min_samples=1).over("vt_symbol").alias("PostHighDrawdown_10d"),
        pl.col("_ret1d").rolling_min(5, min_samples=1).over("vt_symbol").alias("MIN_5d"),
        pl.col("Intraday_return").rolling_skew(60, min_samples=5).over("vt_symbol").alias("IntradaySkew_60d"),
        pl.when(vp_denom > 0)
        .then(vp_cov / vp_denom)
        .otherwise(None)
        .alias("VolPriceCorr_20d"),
        pl.col("Gap_pct").rolling_mean(20, min_samples=1).over("vt_symbol").alias("OvernightMean_20d"),
    ])
    # ---- LLM 挖矿第二批幸存因子（2026-08-22 深夜，因子优先战略：IC 为唯一
    # 迭代指标；实证见 factors/test_new_factors.py 批次二，全池 max 相关 <0.75）----
    # 中间列先物化（polars 禁止 over 嵌套 over）：
    #   _pool_ret   池等权日收益（横截面均值广播）
    #   _lup_streak 当前连续涨停 streak（段内行号差 +1，非涨停段乘 0 置 0）
    _si_n = 20
    _chg = (up_flag != up_flag.shift(1).over("vt_symbol").fill_null(False))
    _idx = pl.int_range(pl.len())
    result = result.with_columns(
        pl.col("_ret1d").mean().over("datetime").alias("_pool_ret")
    ).with_columns(
        _chg.cum_sum().over("vt_symbol").alias("_lup_grp")
    ).with_columns(
        ((_idx - _idx.first().over(["vt_symbol", "_lup_grp"])) + 1).cast(pl.Float64).alias("_lup_streak_raw")
    ).with_columns(
        (up_flag * pl.col("_lup_streak_raw")).alias("_lup_streak")
    )
    _si_x = pl.col("_ret1d")
    _si_y = pl.col("_pool_ret")
    _si_mxy = (_si_x * _si_y).rolling_mean(_si_n, min_periods=_si_n).over("vt_symbol")
    _si_mx = _si_x.rolling_mean(_si_n, min_periods=_si_n).over("vt_symbol")
    _si_my = _si_y.rolling_mean(_si_n, min_periods=_si_n).over("vt_symbol")
    _si_vx = _si_x.rolling_var(_si_n, min_periods=_si_n).over("vt_symbol")
    _si_vy = _si_y.rolling_var(_si_n, min_periods=_si_n).over("vt_symbol")
    _si_den = (_si_vx * _si_vy).sqrt()
    a2_sum = (pl.col("amount") ** 2).rolling_sum(20, min_periods=1).over("vt_symbol")
    a_sum = pl.col("amount").rolling_sum(20, min_periods=1).over("vt_symbol")
    # 首次放量：量 > 2×20日均量 且近 5 日（含当日）放量日数 <=1
    vol_ma20 = vs.rolling_mean(20, min_periods=1).over("vt_symbol")
    spike = (vs > 2.0 * vol_ma20).cast(pl.Float64)
    spike5 = spike.rolling_sum(5, min_periods=1).over("vt_symbol")
    neg_ret = pl.when(pl.col("_ret1d") < 0).then(pl.col("_ret1d")).otherwise(None)
    op_ratio = pl.when(hs > ls).then((os - ls) / (hs - ls)).otherwise(None)
    result = result.with_columns([
        pl.when(_si_den > 0)
        .then((_si_mxy - _si_mx * _si_my) / _si_den)
        .otherwise(None).alias("StockIndexCorr_20d"),
        (pl.col("amount").rolling_mean(5, min_periods=1).over("vt_symbol")
         / pl.col("amount").rolling_mean(60, min_periods=1).over("vt_symbol")).alias("AmountShrink_5_60"),
        pl.when((spike == 1) & (spike5 <= 1)).then(spike).otherwise(None).alias("FirstVolumeSpike_5d"),
        (a2_sum / (a_sum * a_sum)).alias("AmountConc_20d"),
        op_ratio.rolling_mean(20, min_periods=10).over("vt_symbol").alias("OpenPos_mean_20d"),
        pl.col("_lup_streak").rolling_max(60, min_periods=1).over("vt_symbol").alias("LimitUpStreakMax_60d"),
        neg_ret.rolling_std(20, min_periods=5).over("vt_symbol").alias("DownsideVol_20d"),
    ])

    # LogClose 已删（2026-08-22 审计）：qfq 水平因子带 latest_adj 未来信息，
    # 且与 SMA 相关 0.93 准冗余；DB 旧列成孤儿不再计算

    # DaysToDelivery / DaysToNextTrading 不在此计算：日历感知版在
    # compute.py 的 _load_days_to_delivery / _load_days_to_next_trading，
    # 与市场特征同路径按日期合并（compute_panel 全量/增量共路径）

    # ---- drop intermediate columns and keep only factor columns ----
    intermediate_cols = ["_ret1d", "_tr", "_turnover_1d", "_list_dt", "_age_days",
                         "_pool_ret", "_lup_grp", "_lup_streak_raw", "_lup_streak",
                         "_hv_rh", "_hv_rv", "_hv_cov5",
                         "_hhv_std10", "_hhv_volstd10", "_hhv_corr10"]
    source_cols = ["open", "high", "low", "close", "volume", "amount", "vwap",
                   "total_mv", "circ_mv", "cap", "list_date", "pct_chg"]

    factor_cols = [c for c in result.columns
                   if c not in ("datetime", "vt_symbol") and
                   c not in source_cols and
                   c not in intermediate_cols]

    result = result.select(["datetime", "vt_symbol"] + factor_cols)
    result = result.drop([c for c in intermediate_cols if c in result.columns])

    return result

# -*- coding: utf-8 -*-
"""
Alpha 101 factor definitions from WorldQuant 101 Formulaic Alphas.
Expressions ported from vnpy alpha dataset.
All 101 factors are registered: 77 standard + 19 previously skipped
(now enabled with cap from total_mv and IndNeutralize post-processing).
"""

import polars as pl

from .utility import calculate_by_expression

IND_NEUTRALIZE_ALPHAS = {
    "alpha48", "alpha58", "alpha59", "alpha63", "alpha67",
    "alpha69", "alpha70", "alpha76", "alpha79", "alpha80",
    "alpha82", "alpha87", "alpha89", "alpha90", "alpha91",
    "alpha93", "alpha97", "alpha100",
}

RETURNS_EXPR = "ret"

ALPHA_EXPRESSIONS: dict[str, str] = {}


def _register(name: str, expr: str):
    ALPHA_EXPRESSIONS[name] = expr


def build_all_expressions():
    r = RETURNS_EXPR

    _register("alpha1", f"(cs_rank(ts_argmax(pow1(quesval(0, {r}, close, ts_std({r}, 20)), 2.0), 5)) - 0.5)")
    _register("alpha2", "(-1) * ts_corr(cs_rank(ts_delta(log(volume), 2)), cs_rank((close - open) / open), 6)")
    _register("alpha3", "ts_corr(cs_rank(open), cs_rank(volume), 10) * -1")
    _register("alpha4", "-1 * ts_rank(cs_rank(low), 9)")
    _register("alpha5", "cs_rank((open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(cs_rank((close - vwap))))")
    _register("alpha6", "(-1) * ts_corr(open, volume, 10)")
    _register("alpha7", f"quesval2(ts_mean(volume, 20), volume, (-1 * ts_rank(abs(close - ts_delay(close, 7)), 60)) * sign(ts_delta(close, 7)), -1)")
    _register("alpha8", f"-1 * cs_rank(((ts_sum(open, 5) * ts_sum({r}, 5)) - ts_delay((ts_sum(open, 5) * ts_sum({r}, 5)), 10)))")
    _register("alpha9", "quesval(0, ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 5), (-1 * ts_delta(close, 1)), ts_delta(close, 1)))")
    _register("alpha10", "cs_rank(quesval(0, ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 4), (-1 * ts_delta(close, 1)), ts_delta(close, 1))))")
    _register("alpha11", "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(volume, 3))")
    _register("alpha12", "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))")
    _register("alpha13", "-1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))")
    _register("alpha14", f"(-1 * cs_rank(({r}) - ts_delay({r}, 3))) * ts_corr(open, volume, 10)")
    _register("alpha15", "-1 * ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), 3)), 3)")
    _register("alpha16", "-1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))")
    _register("alpha17", "(-1 * cs_rank(ts_rank(close, 10))) * cs_rank(close - 2 * ts_delay(close, 1) + ts_delay(close, 2)) * cs_rank(ts_rank(volume / ts_mean(volume, 20), 5))")
    _register("alpha18", "-1 * cs_rank((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))")
    _register("alpha19", f"(-1 * sign(ts_delta(close, 7) + (close - ts_delay(close, 7)))) * (cs_rank(ts_sum({r}, 250) + 1) + 1)")
    _register("alpha20", "(-1 * cs_rank(open - ts_delay(high, 1))) * cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))")
    _register("alpha21", "quesval2((ts_mean(close, 8) + ts_std(close, 8)), ts_mean(close, 2), -1, quesval2(ts_mean(close, 2), (ts_mean(close, 8) - ts_std(close, 8)), 1, quesval(1, (volume / ts_mean(volume, 20)), 1, -1)))")
    _register("alpha22", "-1 * ts_delta(ts_corr(high, volume, 5), 5) * cs_rank(ts_std(close, 20))")
    _register("alpha23", "quesval2(ts_mean(high, 20), high, -1 * ts_delta(high, 2), 0)")
    _register("alpha24", "quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), (-1 * ts_delta(close, 3)), (-1 * (close - ts_min(close, 100))))")
    _register("alpha25", f"cs_rank( (-1 * {r}) * ts_mean(volume, 20) * vwap * (high - close) )")
    _register("alpha26", "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)")
    _register("alpha27", "quesval(0.5, cs_rank(ts_mean(ts_corr(cs_rank(volume), cs_rank(vwap), 6), 2)), -1, 1)")
    _register("alpha28", "cs_scale(ts_corr(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)")
    _register("alpha29", f"ts_min(ts_product(cs_rank(cs_rank(cs_scale(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(ts_delay((-1 * {r}), 6), 5)")
    _register("alpha30", "((cs_rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3))) * -1 + 1) * ts_sum(volume, 5)) / ts_sum(volume, 20)")
    _register("alpha31", "(cs_rank(cs_rank(cs_rank(ts_decay_linear((-1) * cs_rank(cs_rank(ts_delta(close, 10))), 10)))) + cs_rank((-1) * ts_delta(close, 3))) + sign(cs_scale(ts_corr(ts_mean(volume, 20), low, 12)))")
    _register("alpha32", "cs_scale((ts_sum(close, 7) / 7 - close)) + (20 * cs_scale(ts_corr(vwap, ts_delay(close, 5), 230)))")
    _register("alpha33", "cs_rank((-1) * (open / close * -1 + 1))")
    _register("alpha34", f"cs_rank((cs_rank(ts_std({r}, 2) / ts_std({r}, 5)) * -1 + 1) + (cs_rank(ts_delta(close, 1)) * -1 + 1))")
    _register("alpha35", f"(ts_rank(volume, 32) * (ts_rank((close + high - low), 16) * -1 + 1)) * (ts_rank({r}, 32) * -1 + 1)")
    _register("alpha36", f"((((2.21 * cs_rank(ts_corr((close - open), ts_delay(volume, 1), 15))) + (0.7 * cs_rank((open - close)))) + (0.73 * cs_rank(ts_rank(ts_delay((-1) * {r}, 6), 5)))) + cs_rank(abs(ts_corr(vwap, ts_mean(volume, 20), 6)))) + (0.6 * cs_rank(((ts_sum(close, 200) / 200 - open) * (close - open))))")
    _register("alpha37", "cs_rank(ts_corr(ts_delay((open - close), 1), close, 200)) + cs_rank((open - close))")
    _register("alpha38", "((-1) * cs_rank(ts_rank(close, 10))) * cs_rank((close / open))")
    _register("alpha39", f"((-1) * cs_rank((ts_delta(close, 7) * (cs_rank(ts_decay_linear((volume / ts_mean(volume, 20)), 9)) * -1 + 1)))) * (cs_rank(ts_sum({r}, 250)) + 1)")
    _register("alpha40", "((-1) * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)")
    _register("alpha41", "pow1((high * low), 0.5) - vwap")
    _register("alpha42", "cs_rank((vwap - close)) / cs_rank((vwap + close))")
    _register("alpha43", "ts_rank((volume / ts_mean(volume, 20)), 20) * ts_rank((-1) * ts_delta(close, 7), 8)")
    _register("alpha44", "(-1) * ts_corr(high, cs_rank(volume), 5)")
    _register("alpha45", "(-1) * cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_corr(close, volume, 2) * cs_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))")
    _register("alpha46", "quesval(0.25, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), -1, quesval(0, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1))")
    _register("alpha47", "((cs_rank(pow1(close, -1)) * volume / ts_mean(volume, 20)) * (high * cs_rank(high - close)) / (ts_sum(high, 5) / 5)) - cs_rank(vwap - ts_delay(vwap, 5))")
    _register("alpha48", "(ts_corr(ts_delta(close, 1), ts_delta(ts_delay(close, 1), 1), 250) * ts_delta(close, 1)) / close / ts_sum(pow1((ts_delta(close, 1) / ts_delay(close, 1)), 2), 250)")
    _register("alpha49", "quesval(-0.1, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)")
    _register("alpha50", "(-1) * ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank(vwap), 5)), 5)")
    _register("alpha51", "quesval(-0.05, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)")
    _register("alpha52", f"(((-1) * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * cs_rank((ts_sum({r}, 240) - ts_sum({r}, 20)) / 220) * ts_rank(volume, 5)")
    _register("alpha53", "(-1) * ts_delta(((close - low) - (high - close)) / (close - low), 9)")
    _register("alpha54", "((-1) * ((low - close) * pow1(open, 5))) / ((low - high) * pow1(close, 5))")
    _register("alpha55", "(-1) * ts_corr(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)")
    _register("alpha56", f"(0 - (1 * (cs_rank(ts_sum({r}, 10) / ts_sum(ts_sum({r}, 2), 3)) * cs_rank({r} * cap))))")
    _register("alpha57", "-1 * ((close - vwap) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2))")
    _register("alpha58", "(-1) * ts_rank(ts_decay_linear(ts_corr(vwap, volume, 4), 8), 6)")
    _register("alpha59", "(-1) * ts_rank(ts_decay_linear(ts_corr(((vwap * 0.728317) + (vwap * (1 - 0.728317))), volume, 4), 16), 8)")
    _register("alpha60", "- 1 * ((2 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low)) * volume))) - cs_scale(cs_rank(ts_argmax(close, 10))))")
    _register("alpha61", "quesval2(cs_rank(vwap - ts_min(vwap, 16)), cs_rank(ts_corr(vwap, ts_mean(volume, 180), 18)), 1, 0)")
    _register("alpha62", "(cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, 20), 22), 10)) < cs_rank((cs_rank(open) + cs_rank(open)) < (cs_rank((high + low) / 2) + cs_rank(high)))) * -1")
    _register("alpha63", "(cs_rank(ts_decay_linear(ts_delta(close, 2), 8)) - cs_rank(ts_decay_linear(ts_corr(vwap * 0.318108 + open * 0.681892, ts_sum(ts_mean(volume, 180), 37), 14), 12))) * -1")
    _register("alpha64", "(cs_rank(ts_corr(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < cs_rank(ts_delta((((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1")
    _register("alpha65", "(cs_rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(ts_mean(volume, 60), 9), 6)) < cs_rank(open - ts_min(open, 14))) * -1")
    _register("alpha66", "(cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11), 7)) * -1")
    _register("alpha67", "pow2(cs_rank(high - ts_min(high, 2)), cs_rank(ts_corr(vwap, ts_mean(volume, 20), 6))) * -1")
    _register("alpha68", "(ts_rank(ts_corr(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9), 14) < cs_rank(ts_delta((close * 0.518371 + low * (1 - 0.518371)), 1))) * -1")
    _register("alpha69", "pow2(cs_rank(ts_max(ts_delta(vwap, 3), 5)), ts_rank(ts_corr(close * 0.490655 + vwap * 0.509345, ts_mean(volume, 20), 5), 9)) * -1")
    _register("alpha70", "pow2(cs_rank(ts_delta(vwap, 1)), ts_rank(ts_corr(close, ts_mean(volume, 50), 18), 18)) * -1")
    _register("alpha71", "ts_greater(ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(ts_decay_linear(pow1(cs_rank((low + open) - (vwap + vwap)), 2), 16), 4))")
    _register("alpha72", "cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))")
    _register("alpha73", "ts_greater(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), ts_rank(ts_decay_linear((ts_delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845)) * -1, 3), 17)) * -1")
    _register("alpha74", "quesval2(cs_rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15)), cs_rank(ts_corr(cs_rank(high * 0.0261661 + vwap * 0.9738339), cs_rank(volume), 11)), 1, 0) * -1")
    _register("alpha75", "quesval2(cs_rank(ts_corr(vwap, volume, 4)), cs_rank(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12)), 1, 0)")
    _register("alpha76", "ts_greater(cs_rank(ts_decay_linear(ts_delta(vwap, 1), 12)), ts_rank(ts_decay_linear(ts_rank(ts_corr(low, ts_mean(volume, 81), 8), 20), 17), 19)) * -1")
    _register("alpha77", "ts_less(cs_rank(ts_decay_linear((((high + low) / 2 + high) - (vwap + high)), 20)), cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 3), 6)))")
    _register("alpha78", "pow2(cs_rank(ts_corr(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 20), ts_sum(ts_mean(volume, 40), 20), 7)), cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 6)))")
    _register("alpha79", "quesval2(cs_rank(ts_delta(close * 0.60733 + open * 0.39267, 1)), cs_rank(ts_corr(ts_rank(vwap, 4), ts_rank(ts_mean(volume, 150), 9), 15)), 1, 0)")
    _register("alpha80", "pow2(cs_rank(sign(ts_delta(open * 0.868128 + high * 0.131872, 4))), ts_rank(ts_corr(high, ts_mean(volume, 10), 5), 6)) * -1")
    _register("alpha81", "quesval2(cs_rank(log(ts_product(cs_rank(pow1(cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, 10), 50), 8)), 4)), 15))), cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 5)), 1, 0) * -1")
    _register("alpha82", "ts_less(cs_rank(ts_decay_linear(ts_delta(open, 1), 15)), ts_rank(ts_decay_linear(ts_corr(volume, open, 17), 7), 13)) * -1")
    _register("alpha83", "(cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))")
    _register("alpha84", "pow2(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))")
    _register("alpha85", "pow2(cs_rank(ts_corr(high * 0.876703 + close * 0.123297, ts_mean(volume, 30), 10)), cs_rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))")
    _register("alpha86", "quesval2(ts_rank(ts_corr(close, ts_sum(ts_mean(volume, 20), 15), 6), 20), cs_rank((open + close) - (vwap + open)), 1, 0) * -1")
    _register("alpha87", "ts_greater(cs_rank(ts_decay_linear(ts_delta(close * 0.369701 + vwap * 0.630299, 2), 3)), ts_rank(ts_decay_linear(abs(ts_corr(ts_mean(volume, 81), close, 13)), 5), 14)) * -1")
    _register("alpha88", "ts_less(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))")
    _register("alpha89", "(ts_rank(ts_decay_linear(ts_corr(low, ts_mean(volume, 10), 7), 6), 4) - ts_rank(ts_decay_linear(ts_delta(vwap, 3), 10), 15))")
    _register("alpha90", "pow2(cs_rank(close - ts_max(close, 5)), ts_rank(ts_corr(ts_mean(volume, 40), low, 5), 3)) * -1")
    _register("alpha91", "(ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(close, volume, 10), 16), 4), 5) - cs_rank(ts_decay_linear(ts_corr(vwap, ts_mean(volume, 30), 4), 3))) * -1")
    _register("alpha92", "ts_less(ts_rank(ts_decay_linear(quesval2(((high + low) / 2 + close), (low + open), 1, 0), 15), 19), ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 30)), 8), 7), 7))")
    _register("alpha93", "ts_rank(ts_decay_linear(ts_corr(vwap, ts_mean(volume, 81), 17), 20), 8) / cs_rank(ts_decay_linear(ts_delta(close * 0.524434 + vwap * 0.475566, 3), 16))")
    _register("alpha94", "pow2(cs_rank(vwap - ts_min(vwap, 12)), ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)) * -1")
    _register("alpha95", "quesval2(cs_rank(open - ts_min(open, 12)), ts_rank(pow1(cs_rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5), 12), 1, 0)")
    _register("alpha96", "ts_greater(ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(volume), 4), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)) * -1")
    _register("alpha97", "(cs_rank(ts_decay_linear(ts_delta(low * 0.721001 + vwap * 0.278999, 3), 20)) - ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5), 19), 16), 7)) * -1")
    _register("alpha98", "cs_rank(ts_decay_linear(ts_corr(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))")
    _register("alpha99", "quesval2(cs_rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)), cs_rank(ts_corr(low, volume, 6)), 1, 0) * -1")
    _register("alpha100", "-1 * ((1.5 * cs_scale(cs_rank(((close - low) - (high - close)) / (high - low) * volume))) - cs_scale(ts_corr(close, cs_rank(ts_mean(volume, 20)), 5) - cs_rank(ts_argmin(close, 30)))) * (volume / ts_mean(volume, 20))")
    _register("alpha101", "((close - open) / ((high - low) + 0.001))")

    # ---- Alternative versions (old raw formulas from legacy factors.py) ----
    _register("alpha1_v0", "cs_rank(ts_argmax(pow1(ts_corr(close, volume, 5), 2.0), 5) - 0.5)")
    _register("alpha18_v0", "-1 * ((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))")
    _register("alpha50_v0", "cs_rank(-1 * ts_corr(ts_rank(close, 10), ts_rank(volume, 10), 10))")
    _register("alpha60_v0", "-1 * (ts_rank(ts_std(close, 20), 10) - ts_rank(ts_std(close, 5), 10))")


build_all_expressions()

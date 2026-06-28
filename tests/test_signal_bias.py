"""Tests for the trend-gated signal decision pipeline (signals/bias.py).

Covers the refactor's required cases (Step 13):
- Strong bullish / bearish trend → only the trend direction allowed
- EMA pullback (the original counter-trend bug) → bias does NOT flip to BUY
- EMA crossover → bias flips
- Range market → both directions allowed, MACD may pick
- MACD bullish in a bearish trend → rejected / not BUY (counter-trend rejection)
- MACD bearish in a bullish trend → rejected / not SELL
- Price below/above EMA200 filter
- Distance filter: far from EMA200 → trend-only, never counter-trend
- decide_direction end-to-end (strict vs non-strict)
- classify_setup pullback/extension/neutral
- momentum_confirms
"""
import unittest
from types import SimpleNamespace

from config.settings import SignalType
from signals.bias import (
    DISTANCE_ATR_TREND_ONLY,
    MarketBias,
    MarketRegime,
    allowed_directions,
    classify_setup,
    decide_direction,
    detect_regime,
    get_market_bias,
    momentum_confirms,
    momentum_pick,
)


def feat(
    ema20, ema50, ema200, *, close=None, rsi=50.0, macd_line=0.0, macd_signal=0.0,
    macd_histogram=0.0, atr=100.0, atr_ratio=1.0, dist_ema200=None, momentum_score=0.0,
    trend_strength=0.0,
):
    """Build a lightweight feature bag for decide_direction."""
    if close is None:
        close = ema20
    if dist_ema200 is None:
        dist_ema200 = close - (ema200 if ema200 is not None else close)
    return SimpleNamespace(
        ema20=ema20, ema50=ema50, ema200=ema200, close=close, rsi14=rsi,
        macd_line=macd_line, macd_signal=macd_signal, macd_histogram=macd_histogram,
        atr14=atr, atr_ratio=atr_ratio, dist_ema200=dist_ema200,
        momentum_score=momentum_score, trend_strength=trend_strength,
    )


class MarketBiasTests(unittest.TestCase):
    def test_strong_bullish_stack_is_buy_bias(self):
        self.assertEqual(get_market_bias(110, 105, 100), MarketBias.BUY_BIAS)

    def test_strong_bearish_stack_is_sell_bias(self):
        self.assertEqual(get_market_bias(90, 95, 100), MarketBias.SELL_BIAS)

    def test_pullback_above_ema20_does_not_flip_bearish_bias(self):
        # The original bug: bearish EMA stack + price pulled above EMA20.
        # Bias must stay SELL — price location does not change trend.
        self.assertEqual(get_market_bias(92, 95, 100), MarketBias.SELL_BIAS)

    def test_pullback_below_ema20_does_not_flip_bullish_bias(self):
        self.assertEqual(get_market_bias(108, 105, 100), MarketBias.BUY_BIAS)

    def test_two_ema_fallback_when_ema200_missing(self):
        self.assertEqual(get_market_bias(110, 105, None), MarketBias.BUY_BIAS)
        self.assertEqual(get_market_bias(90, 95, None), MarketBias.SELL_BIAS)

    def test_flat_emas_are_range(self):
        self.assertEqual(get_market_bias(100, 100, 100), MarketBias.RANGE)

    def test_crossover_flips_bias(self):
        self.assertEqual(get_market_bias(105, 100, 95), MarketBias.BUY_BIAS)
        self.assertEqual(get_market_bias(95, 100, 105), MarketBias.SELL_BIAS)


class RegimeTests(unittest.TestCase):
    def test_trending_up(self):
        self.assertEqual(detect_regime(110, 105, 100, 112, 100, 1.2), MarketRegime.TRENDING_UP)

    def test_trending_down(self):
        self.assertEqual(detect_regime(90, 95, 100, 88, 100, 1.2), MarketRegime.TRENDING_DOWN)

    def test_flat_emas_are_ranging(self):
        self.assertEqual(detect_regime(100, 100.05, 100, 100, 50, 1.2), MarketRegime.RANGING)

    def test_compressed_atr_is_ranging(self):
        # EMAs stacked up but ATR compressed → ranging (volatility gone).
        self.assertEqual(detect_regime(110, 105, 100, 112, 100, 0.5), MarketRegime.RANGING)


class AllowedDirectionsTests(unittest.TestCase):
    def test_bullish_bias_allows_buy_forbids_sell(self):
        allowed, _ = allowed_directions(MarketBias.BUY_BIAS, MarketRegime.TRENDING_UP, 110, 100, 50)
        self.assertEqual(allowed, {SignalType.BUY})

    def test_bearish_bias_allows_sell_forbids_buy(self):
        allowed, _ = allowed_directions(MarketBias.SELL_BIAS, MarketRegime.TRENDING_DOWN, 90, 100, 50)
        self.assertEqual(allowed, {SignalType.SELL})

    def test_ema200_filter_blocks_buy_below_ema200(self):
        # Uptrend bias but price slipped below EMA200 → no BUY.
        allowed, _ = allowed_directions(MarketBias.BUY_BIAS, MarketRegime.TRENDING_UP, 99, 100, 50)
        self.assertNotIn(SignalType.BUY, allowed)

    def test_ema200_filter_blocks_sell_above_ema200(self):
        allowed, _ = allowed_directions(MarketBias.SELL_BIAS, MarketRegime.TRENDING_DOWN, 101, 100, 50)
        self.assertNotIn(SignalType.SELL, allowed)

    def test_distance_filter_forces_trend_only_far_below(self):
        # Price 3 ATR below EMA200 → only SELL, never BUY.
        allowed, dist = allowed_directions(MarketBias.SELL_BIAS, MarketRegime.TRENDING_DOWN, 70, 100, 10)
        self.assertEqual(allowed, {SignalType.SELL})
        self.assertGreaterEqual(dist, DISTANCE_ATR_TREND_ONLY)

    def test_distance_filter_forces_trend_only_far_above(self):
        allowed, _ = allowed_directions(MarketBias.BUY_BIAS, MarketRegime.TRENDING_UP, 130, 100, 10)
        self.assertEqual(allowed, {SignalType.BUY})

    def test_ranging_allows_both_directions(self):
        allowed, _ = allowed_directions(MarketBias.RANGE, MarketRegime.RANGING, 100, 100, 50)
        self.assertEqual(allowed, {SignalType.BUY, SignalType.SELL})


class DecideDirectionTests(unittest.TestCase):
    def test_strict_bullish_trend_returns_buy(self):
        f = feat(110, 105, 100, close=112, atr=50)
        d, log = decide_direction(f, strict=True)
        self.assertEqual(d, SignalType.BUY)

    def test_strict_bearish_trend_returns_sell(self):
        f = feat(90, 95, 100, close=88, atr=50)
        d, log = decide_direction(f, strict=True)
        self.assertEqual(d, SignalType.SELL)

    def test_counter_trend_buy_in_bearish_trend_rejected(self):
        # The original bug: bearish stack + MACD bullish bounce. Must NOT be BUY.
        f = feat(92, 95, 100, close=93, macd_histogram=50, macd_line=10, macd_signal=-10, atr=50)
        d, log = decide_direction(f, strict=True)
        self.assertNotEqual(d, SignalType.BUY)
        self.assertEqual(d, SignalType.SELL)

    def test_counter_trend_sell_in_bullish_trend_rejected(self):
        f = feat(108, 105, 100, close=107, macd_histogram=-50, macd_line=-10, macd_signal=10, atr=50)
        d, log = decide_direction(f, strict=True)
        self.assertNotEqual(d, SignalType.SELL)
        self.assertEqual(d, SignalType.BUY)

    def test_macd_picks_direction_only_in_range(self):
        # Flat EMAs + bullish MACD → BUY is allowed (RANGE tie-break by MACD).
        f = feat(100, 100, 100, close=100, macd_histogram=20, macd_line=5, macd_signal=-5, atr=50, atr_ratio=1.2)
        d, log = decide_direction(f, strict=True)
        self.assertEqual(d, SignalType.BUY)

    def test_macd_cannot_pick_counter_trend_when_distance_far(self):
        # Ranging bias but price 3 ATR below EMA200 → distance filter forces SELL
        # even though MACD is bullish.
        f = feat(70, 70, 100, close=70, macd_histogram=50, macd_line=20, macd_signal=10, atr=10, atr_ratio=1.2)
        d, log = decide_direction(f, strict=True)
        self.assertEqual(d, SignalType.SELL)

    def test_non_strict_returns_raw_bias(self):
        # H1 confirmation path: strict=False skips EMA200 filter.
        # Bullish stack but price below EMA200 → raw bias BUY (filter not applied).
        f = feat(110, 105, 100, close=99, atr=50)
        d, log = decide_direction(f, strict=False)
        self.assertEqual(d, SignalType.BUY)

    def test_no_trade_when_nothing_allowed(self):
        # BUY_BIAS but price below EMA200 and not far enough for distance filter,
        # and no SELL allowed → empty → NO_TRADE.
        f = feat(101, 100.5, 100, close=99.5, atr=50)  # close < ema200, bias BUY
        d, log = decide_direction(f, strict=True)
        self.assertEqual(d, SignalType.NO_TRADE)


class SetupAndMomentumTests(unittest.TestCase):
    def test_pullback_buy(self):
        self.assertEqual(classify_setup(98, 100, 50, SignalType.BUY), "pullback")

    def test_pullback_sell(self):
        self.assertEqual(classify_setup(102, 100, 50, SignalType.SELL), "pullback")

    def test_extension_buy(self):
        # Price 2 ATR above EMA20 in a BUY → over-extended (late entry).
        self.assertEqual(classify_setup(200, 100, 50, SignalType.BUY), "extension")

    def test_extension_sell(self):
        self.assertEqual(classify_setup(0, 100, 50, SignalType.SELL), "extension")

    def test_neutral(self):
        self.assertEqual(classify_setup(101, 100, 50, SignalType.BUY), "neutral")

    def test_momentum_confirms_buy(self):
        self.assertTrue(momentum_confirms(10, 5, -5, SignalType.BUY))
        self.assertFalse(momentum_confirms(-10, -5, 5, SignalType.BUY))

    def test_momentum_confirms_sell(self):
        self.assertTrue(momentum_confirms(-10, -5, 5, SignalType.SELL))
        self.assertFalse(momentum_confirms(10, 5, -5, SignalType.SELL))

    def test_momentum_pick_bullish(self):
        self.assertEqual(momentum_pick(20, 5, -5), SignalType.BUY)

    def test_momentum_pick_bearish(self):
        self.assertEqual(momentum_pick(-20, -5, 5), SignalType.SELL)

    def test_momentum_pick_neutral_is_no_trade(self):
        self.assertEqual(momentum_pick(0, 0, 0), SignalType.NO_TRADE)


if __name__ == "__main__":
    unittest.main()
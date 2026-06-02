"""Trade Strength Score: evaluates if trend is strong enough to hold a position.

When a trade reaches 50-70% of TP, we need to decide:
- Keep trade open if trend remains strong
- Close immediately if trend weakens

Score components (0-100 scale):
- EMA20 trend direction (0-25): Price relative to EMA20, EMA20 slope
- EMA20 slope strength (0-20): Rate and consistency of slope
- RSI momentum (0-25): RSI level and direction
- Volume relative to average (0-15): Volume surge/decline
- ATR expansion/contraction (0-15): Volatility regime

Rules:
- Score >= 70: Hold trade (trend strong)
- Score 50-69: Monitor (trend weakening but OK)
- Score < 50: Exit trade (trend weak)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrengthScoreComponents:
    """Individual components that make up the strength score."""
    ema_trend_direction: float = 0.0  # 0-25
    ema_slope_strength: float = 0.0  # 0-20
    rsi_momentum: float = 0.0  # 0-25
    volume_ratio: float = 0.0  # 0-15
    atr_expansion: float = 0.0  # 0-15

    def to_dict(self) -> dict:
        return {
            "ema_trend_direction": self.ema_trend_direction,
            "ema_slope_strength": self.ema_slope_strength,
            "rsi_momentum": self.rsi_momentum,
            "volume_ratio": self.volume_ratio,
            "atr_expansion": self.atr_expansion,
        }


@dataclass
class StrengthScoreResult:
    """Result of strength score calculation."""
    score: float  # 0-100
    components: StrengthScoreComponents
    decision: str  # "hold", "monitor", "exit"
    reason: str

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "components": self.components.to_dict(),
            "decision": self.decision,
            "reason": self.reason,
        }


class TradeStrengthScore:
    """Calculates trade strength score to decide whether to hold or exit.

    Usage:
        scorer = TradeStrengthScore()
        result = scorer.evaluate(df, position_direction="BUY")
        if result.decision == "exit":
            close_position()
    """

    def __init__(self):
        self.volume_window = 20
        self.atr_window = 14
        self.slope_window = 5

    def evaluate(
        self,
        df: pd.DataFrame,
        position_direction: str,
    ) -> StrengthScoreResult:
        """Evaluate trade strength for a position.

        Args:
            df: DataFrame with OHLCV data and indicators (ema20, rsi14, atr14, etc.)
            position_direction: "BUY" or "SELL"

        Returns:
            StrengthScoreResult with score and decision
        """
        if len(df) < max(self.volume_window, self.atr_window, self.slope_window):
            return StrengthScoreResult(
                score=50.0,
                components=StrengthScoreComponents(),
                decision="monitor",
                reason="Insufficient data for evaluation",
            )

        latest = df.iloc[-1]

        # Calculate components
        components = StrengthScoreComponents(
            ema_trend_direction=self._ema_trend_direction(latest, position_direction),
            ema_slope_strength=self._ema_slope_strength(df, position_direction),
            rsi_momentum=self._rsi_momentum(latest, position_direction),
            volume_ratio=self._volume_ratio(df, latest, position_direction),
            atr_expansion=self._atr_expansion(df, position_direction),
        )

        # Sum components
        score = (
            components.ema_trend_direction +
            components.ema_slope_strength +
            components.rsi_momentum +
            components.volume_ratio +
            components.atr_expansion
        )
        score = min(score, 100.0)  # Cap at 100

        # Make decision
        decision, reason = self._make_decision(score, components, position_direction)

        return StrengthScoreResult(
            score=score,
            components=components,
            decision=decision,
            reason=reason,
        )

    def _ema_trend_direction(self, row: pd.Series, direction: str) -> float:
        """Score EMA20 trend direction (0-25 points).

        For BUY:
        - Price above EMA20: +10
        - EMA20 above EMA50: +10
        - EMA20 above EMA200: +5

        For SELL: reverse logic.
        """
        score = 0.0
        ema20 = row.get("ema20")
        ema50 = row.get("ema50")
        ema200 = row.get("ema200")
        close = row.get("close")

        if pd.isna(ema20) or pd.isna(close):
            return 0.0

        is_buy = direction.upper() == "BUY"

        # Price vs EMA20 (most important)
        if is_buy:
            if close > ema20:
                score += 10.0
            # Partial credit if close is near EMA20
            elif close > ema20 * 0.995:
                score += 5.0
        else:
            if close < ema20:
                score += 10.0
            elif close < ema20 * 1.005:
                score += 5.0

        # EMA20 vs EMA50
        if not pd.isna(ema50):
            if is_buy and ema20 > ema50:
                score += 10.0
            elif not is_buy and ema20 < ema50:
                score += 10.0
            # Partial credit for near-equal
            elif abs(ema20 - ema50) / ema20 < 0.001:
                score += 3.0

        # EMA20 vs EMA200 (big picture)
        if not pd.isna(ema200):
            if is_buy and ema20 > ema200:
                score += 5.0
            elif not is_buy and ema20 < ema200:
                score += 5.0

        return score

    def _ema_slope_strength(self, df: pd.DataFrame, direction: str) -> float:
        """Score EMA20 slope strength (0-20 points).

        Calculates:
        - Slope magnitude (how fast EMA20 is moving)
        - Slope consistency (is it accelerating or decelerating)

        For BUY: positive slope is good
        For SELL: negative slope is good
        """
        if len(df) < self.slope_window:
            return 0.0

        recent = df.tail(self.slope_window)
        ema20 = recent["ema20"].values

        if pd.isna(ema20).any():
            return 0.0

        # Calculate slope using linear regression
        x = np.arange(len(ema20))
        try:
            slope = np.polyfit(x, ema20, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

        # Normalize slope relative to price level
        avg_ema = np.mean(ema20)
        if avg_ema == 0:
            return 0.0

        normalized_slope = (slope / avg_ema) * 100  # Percentage change per candle

        is_buy = direction.upper() == "BUY"
        target_sign = 1 if is_buy else -1

        # Score based on magnitude in correct direction
        if normalized_slope * target_sign > 0:
            # Positive slope in right direction
            magnitude = abs(normalized_slope)
            # 0.05% per candle = full points, scale accordingly
            points = min(magnitude / 0.05 * 20, 20.0)
        else:
            # Wrong direction - penalize
            points = 0.0

        return points

    def _rsi_momentum(self, row: pd.Series, direction: str) -> float:
        """Score RSI momentum (0-25 points).

        For BUY:
        - RSI 50-70: bullish momentum (+15)
        - RSI 40-50: neutral (+5)
        - RSI > 70: overbought (penalize)
        - RSI < 40: bearish (0 points)

        For SELL: reverse logic.
        """
        rsi = row.get("rsi14")
        if pd.isna(rsi):
            return 0.0

        is_buy = direction.upper() == "BUY"
        score = 0.0

        if is_buy:
            if 50 <= rsi <= 70:
                # Sweet spot for bullish momentum
                score = 15.0
                # Bonus for strong but not overbought
                if 55 <= rsi <= 65:
                    score += 5.0
            elif 40 <= rsi < 50:
                # Neutral/bearish - some credit
                score = 5.0
            elif 70 < rsi <= 75:
                # Getting overbought - reduced credit
                score = 5.0
            else:
                # RSI < 40 or > 75 - weak or overbought
                score = 0.0
        else:
            if 30 <= rsi <= 50:
                # Sweet spot for bearish momentum
                score = 15.0
                if 35 <= rsi <= 45:
                    score += 5.0
            elif 50 < rsi <= 60:
                # Neutral/bullish - some credit
                score = 5.0
            elif 25 <= rsi < 30:
                # Getting oversold - reduced credit
                score = 5.0
            else:
                # RSI > 60 or < 25 - weak or oversold
                score = 0.0

        return score

    def _volume_ratio(self, df: pd.DataFrame, row: pd.Series, direction: str) -> float:
        """Score volume relative to average (0-15 points).

        Higher volume supporting the trend is good.
        Volume dropping off is bad (trend weakening).

        For BUY: high volume on up bars is good
        For SELL: high volume on down bars is good
        """
        recent = df.tail(self.volume_window)
        volumes = recent["volume"].values

        if pd.isna(volumes).any() or len(volumes) < 5:
            return 7.5  # Neutral if can't calculate

        current_vol = row.get("volume", 0)
        avg_volume = np.mean(volumes[volumes > 0])

        if avg_volume == 0 or current_vol == 0:
            return 7.5

        volume_ratio = current_vol / avg_volume

        # Check if volume is expanding (good) or contracting (bad)
        is_buy = direction.upper() == "BUY"

        # Recent trend in volume (last 5 vs previous)
        recent_vol = np.mean(volumes[-5:])
        prev_vol = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else avg_volume

        score = 7.5  # Start at neutral

        if is_buy:
            # For BUY: want expanding volume
            if volume_ratio > 1.2:
                score += 5.0
            elif volume_ratio > 1.5:
                score += 7.5
            if recent_vol > prev_vol * 1.1:
                score += 2.5
            elif recent_vol < prev_vol * 0.8:
                score -= 2.5
        else:
            # Same logic for SELL - volume should support move
            if volume_ratio > 1.2:
                score += 5.0
            elif volume_ratio > 1.5:
                score += 7.5
            if recent_vol > prev_vol * 1.1:
                score += 2.5
            elif recent_vol < prev_vol * 0.8:
                score -= 2.5

        return max(0.0, min(score, 15.0))

    def _atr_expansion(self, df: pd.DataFrame, direction: str) -> float:
        """Score ATR expansion/contraction (0-15 points).

        Expanding ATR = increasing volatility = trend strengthening
        Contracting ATR = consolidation = trend weakening

        However, too much expansion can mean exhaustion/reversal risk.
        """
        atr_col = "atr14"
        if atr_col not in df.columns:
            return 7.5

        recent = df.tail(self.atr_window)
        atrs = recent[atr_col].dropna()

        if len(atrs) < 5:
            return 7.5

        current_atr = atrs.iloc[-1]
        avg_atr = atrs.mean()

        if avg_atr == 0:
            return 7.5

        atr_ratio = current_atr / avg_atr

        # Calculate trend in ATR
        recent_atr = atrs.tail(5).mean()
        prev_atr = atrs.iloc[-10:-5].mean() if len(atrs) >= 10 else avg_atr

        score = 7.5  # Neutral

        # Moderate expansion is good
        if 1.0 < atr_ratio <= 1.3:
            score += 5.0
        elif 1.3 < atr_ratio <= 1.5:
            score += 7.5
        elif atr_ratio > 1.5:
            # Too much expansion - exhaustion risk
            score = 5.0
        elif atr_ratio < 0.8:
            # Contraction - consolidation
            score = 5.0

        # ATR trend
        if recent_atr > prev_atr * 1.1:
            score += 2.5
        elif recent_atr < prev_atr * 0.9:
            score -= 2.5

        return max(0.0, min(score, 15.0))

    def _make_decision(
        self,
        score: float,
        components: StrengthScoreComponents,
        direction: str,
    ) -> tuple[str, str]:
        """Make hold/monitor/exit decision based on score.

        Rules:
        - Score >= 70: Hold
        - Score 50-69: Monitor
        - Score < 50: Exit

        Additional checks for edge cases.
        """
        if score >= 70:
            return "hold", f"Strong trend (score {score:.1f})"
        elif score >= 50:
            # Check if critical components are weak
            if components.ema_trend_direction < 10:
                return "exit", f"Price crossed EMA20 (score {score:.1f})"
            if components.rsi_momentum < 5:
                return "monitor", f"Momentum fading (score {score:.1f})"
            return "monitor", f"Trend weakening (score {score:.1f})"
        else:
            return "exit", f"Weak trend (score {score:.1f})"


def calculate_strength_score(
    df: pd.DataFrame,
    position_direction: str,
) -> StrengthScoreResult:
    """Convenience function to calculate strength score."""
    scorer = TradeStrengthScore()
    return scorer.evaluate(df, position_direction)

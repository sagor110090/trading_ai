"""Debug script to show why signals are NOT being generated.

This displays the actual indicator values and decision points
for each symbol so you can see what's blocking signals.
"""

import logging
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import SETTINGS, Symbol, Timeframe, TradeMode, SignalType
from data.database import get_session
from data.models import Candle
from indicators.features import compute_all_features, extract_feature_set_with_lags
from signals.scoring import compute_signal_scores

console = Console()


def load_candles(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Load candles from database."""
    with get_session() as session:
        candles = (
            session.query(Candle)
            .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
            .order_by(Candle.timestamp.desc())
            .limit(limit)
            .all()
        )

        if not candles:
            return pd.DataFrame()

        # Extract values within session context
        rows = [{
            "timestamp": c.timestamp,
            "open": c.open_price,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "spread": c.spread,
        } for c in reversed(candles)]

    return pd.DataFrame(rows)


def analyze_symbol(symbol_str: str):
    """Analyze a single symbol and show why no signal."""
    console.print()
    console.print(Panel(
        f"[bold cyan]{symbol_str}[/]",
        title=f"Signal Analysis",
        border_style="cyan",
    ))

    try:
        # Load candles
        df_h4 = load_candles(symbol_str, "4h", 500)
        df_h1 = load_candles(symbol_str, "1h", 500)
        df_m5 = load_candles(symbol_str, "5m", 500)

        if df_h4.empty:
            console.print("[red]✗ No H4 data[/]")
            return
        if df_h1.empty:
            console.print("[red]✗ No H1 data[/]")
            return

        # Compute features
        h4_features = compute_all_features(df_h4)
        h1_features = compute_all_features(df_h1)
        m5_features = compute_all_features(df_m5)

        h4_last = h4_features.iloc[-1]
        h1_last = h1_features.iloc[-1]
        m5_last = m5_features.iloc[-1] if not m5_features.empty else None

        # Get direction for each timeframe
        def get_direction(df, last_row):
            """Get direction and scores for a timeframe."""
            features = extract_feature_set_with_lags(df, len(df) - 1)

            scores, direction = compute_signal_scores(
                ema20=features.ema20 or 0,
                ema50=features.ema50 or 0,
                ema200=features.ema200 or 0,
                close=last_row["close"],
                rsi=features.rsi14 or 50,
                macd_line=features.macd_line or 0,
                macd_signal=features.macd_signal or 0,
                macd_histogram=features.macd_histogram or 0,
                atr=features.atr14 or 0,
                candle_body_pct=features.candle_body_pct or 0,
                upper_wick_pct=features.upper_wick_pct or 0,
                lower_wick_pct=features.lower_wick_pct or 0,
                momentum_score=features.momentum_score or 0,
                trend_strength=features.trend_strength or 0,
            )
            return scores, direction, features

        # Analyze each timeframe
        h4_scores, h4_dir, h4_feat = get_direction(df_h4, h4_last)
        h1_scores, h1_dir, h1_feat = get_direction(df_h1, h1_last)

        # M5 quick check
        m5_dir = SignalType.NO_TRADE
        m5_momentum = 0.0
        if m5_last is not None and not pd.isna(m5_last.get("ema20")):
            m5_momentum = float(m5_last.get("momentum_score", 0))
            if m5_last["close"] > m5_last["ema20"] and m5_momentum > 0:
                m5_dir = SignalType.BUY
            elif m5_last["close"] < m5_last["ema20"] and m5_momentum < 0:
                m5_dir = SignalType.SELL

        # Display results
        console.print()

        # H4 Table
        h4_table = Table(title="[bold]H4 (Big Picture)[/]", show_header=True)
        h4_table.add_column("Metric", style="cyan")
        h4_table.add_column("Value", justify="right")
        h4_dir_str = h4_dir.value if hasattr(h4_dir, 'value') else str(h4_dir)
        h4_table.add_row("Direction", f"[{'green' if h4_dir != SignalType.NO_TRADE else 'red'}]{h4_dir_str}[/]")
        h4_table.add_row("Close", f"${h4_last['close']:.2f}")
        h4_table.add_row("EMA20", f"${h4_feat.ema20 or 0:.2f}")
        h4_table.add_row("EMA50", f"${h4_feat.ema50 or 0:.2f}")
        h4_table.add_row("EMA200", f"${h4_feat.ema200 or 0:.2f}")
        h4_table.add_row("RSI14", f"{h4_feat.rsi14 or 50:.1f}")
        h4_table.add_row("MACD Hist", f"{h4_feat.macd_histogram or 0:.2f}")
        h4_table.add_row("Confidence", f"[bold]{h4_scores.total:.1f}%[/]")
        console.print(h4_table)

        # H1 Table
        h1_table = Table(title="[bold]H1 (Trend)[/]", show_header=True)
        h1_table.add_column("Metric", style="cyan")
        h1_table.add_column("Value", justify="right")
        h1_dir_str = h1_dir.value if hasattr(h1_dir, 'value') else str(h1_dir)
        h1_table.add_row("Direction", f"[{'green' if h1_dir != SignalType.NO_TRADE else 'red'}]{h1_dir_str}[/]")
        h1_table.add_row("Close", f"${h1_last['close']:.2f}")
        h1_table.add_row("EMA20", f"${h1_feat.ema20 or 0:.2f}")
        h1_table.add_row("EMA50", f"${h1_feat.ema50 or 0:.2f}")
        h1_table.add_row("EMA200", f"${h1_feat.ema200 or 0:.2f}")
        h1_table.add_row("RSI14", f"{h1_feat.rsi14 or 50:.1f}")
        h1_table.add_row("MACD Hist", f"{h1_feat.macd_histogram or 0:.2f}")
        h1_table.add_row("Confidence", f"[bold]{h1_scores.total:.1f}%[/]")
        console.print(h1_table)

        # M5 Table
        m5_table = Table(title="[bold]M5 (Entry Trigger)[/]", show_header=True)
        m5_table.add_column("Metric", style="cyan")
        m5_table.add_column("Value", justify="right")
        m5_dir_str = m5_dir.value if hasattr(m5_dir, 'value') else str(m5_dir)
        m5_table.add_row("Direction", f"[{'green' if m5_dir != SignalType.NO_TRADE else 'yellow'}]{m5_dir_str}[/]")
        if m5_last is not None:
            m5_table.add_row("Close", f"${m5_last['close']:.2f}")
            m5_table.add_row("EMA20", f"${m5_last.get('ema20', 0) or 0:.2f}")
            m5_table.add_row("Momentum", f"{m5_momentum:.1f}")
        console.print(m5_table)

        console.print()

        # Decision analysis for current trade mode
        mode = SETTINGS.trade_mode
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        console.print(Panel(
            f"[bold]Trade Mode: {mode_str.upper()}[/]",
            title="Decision Analysis",
            border_style="yellow",
        ))

        if mode == TradeMode.DAY_TRADING:
            # Day trading rules
            issues = []

            # Check H1 direction
            if h1_dir == SignalType.NO_TRADE:
                issues.append("[red]✗ H1 has NO direction (EMAs are flat/mixed)[/]")
            else:
                console.print(f"[green]✓ H1 has direction: {h1_dir_str}[/]")

            # Check H4 opposition
            if h4_dir != SignalType.NO_TRADE and h4_dir != h1_dir:
                issues.append(f"[red]✗ H4 ({h4_dir_str}) opposes H1 ({h1_dir_str})[/]")
            elif h4_dir == h1_dir:
                console.print(f"[green]✓ H4 agrees with H1: {h4_dir_str}[/]")
            else:
                console.print("[yellow]○ H4 is neutral (no conflict)[/]")

            # Check M5 alignment
            if m5_dir != SignalType.NO_TRADE and m5_dir != h1_dir:
                console.print(f"[yellow]⚠ M5 ({m5_dir_str}) opposes H1 ({h1_dir_str}) - confidence penalty[/]")
            elif m5_dir == h1_dir:
                console.print(f"[green]✓ M5 agrees with H1: {m5_dir_str}[/]")

            # Calculate confidence
            base_confidence = min(h1_scores.total / SETTINGS.signal_weights.total * 100.0, 100.0)
            threshold = SETTINGS.day_trading.confidence_threshold

            console.print()
            console.print(f"Base Confidence: {base_confidence:.1f}% | Threshold: {threshold:.1f}%")

            if base_confidence < threshold:
                issues.append(f"[red]✗ Confidence ({base_confidence:.1f}%) below threshold ({threshold:.1f}%)[/]")
            else:
                console.print(f"[green]✓ Confidence meets threshold[/]")

            # Summary
            console.print()
            if issues:
                console.print(Panel(
                    "\n".join(issues),
                    title="[bold red]Signal Blocked[/]",
                    border_style="red",
                ))
            else:
                console.print(Panel(
                    f"[bold green]✓ Signal should generate: {h1_dir_str}[/]",
                    border_style="green",
                ))

    except Exception as e:
        import traceback
        console.print(f"[red]Error: {e}[/]")
        console.print(f"[dim]{traceback.format_exc()}[/]")


def main():
    """Analyze all configured symbols."""
    console.print("[bold cyan]Signal Diagnostics[/]")
    console.print("[dim]Showing indicator values and decision points...[/]")

    for symbol in SETTINGS.symbols:
        try:
            analyze_symbol(symbol.value)
        except Exception as e:
            console.print(f"[red]Error analyzing {symbol.value}: {e}[/]")


if __name__ == "__main__":
    main()

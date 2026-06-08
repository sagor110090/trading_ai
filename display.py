"""Rich terminal display for the trading AI system.

Provides formatted, color-coded output for each cycle:
signal panels, ML predictions, trade results, and summaries.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from config.settings import SETTINGS
from ml.predictor import Prediction
from signals.engine import TradeSignal
from trade.executor import OrderResult

# Shared console instance
console = Console()

logger = logging.getLogger(__name__)

# ── Cycle Header ──────────────────────────────────────────────────────────


def display_cycle_header(symbol: str) -> None:
    """Print a prominent cycle header for a symbol."""
    console.print()
    console.print(Rule(f"[bold cyan]{symbol}[/]", style="cyan"))
    console.print()


# ── Candle Fetch ──────────────────────────────────────────────────────────


def display_candle_fetch(results: dict[str, int], loaded: Optional[dict[str, int]] = None) -> None:
    """Show candle fetch results.

    Args:
        results: {timeframe: new_count_inserted}
        loaded: Optional {timeframe: total_count_loaded_from_db}
    """
    saved_parts = []
    loaded_parts = []

    for tf, count in results.items():
        if count > 0:
            saved_parts.append(f"[green]+{count}[/] [dim]{tf}[/]")
        else:
            saved_parts.append(f"[dim]0 {tf}[/]")

        if loaded is not None:
            loaded_count = loaded.get(tf, 0)
            if loaded_count > 0:
                loaded_parts.append(f"[cyan]{loaded_count}[/] [dim]{tf}[/]")
            else:
                loaded_parts.append(f"[red]0[/] [dim]{tf}[/]")

    console.print(f"  [dim]New candles[/]  {' · '.join(saved_parts)}")
    if loaded is not None:
        console.print(f"  [dim]Loaded candles[/]  {' · '.join(loaded_parts)}")


def display_missing_confluence_data(symbol: str, h4_count: int, h1_count: int, m5_count: int) -> None:
    """Show that signal generation was skipped because candle data is missing."""
    console.print(
        f"  [yellow]Missing candle data for {symbol}:[/] "
        f"[dim]H4={h4_count} H1={h1_count} M5={m5_count}[/]"
    )


# ── Verification ──────────────────────────────────────────────────────────


def display_verification_summary(verified: int, pending: int) -> None:
    """Show signal verification summary."""
    if verified > 0:
        console.print(
            f"  [dim]Verified[/]  [green]{verified}[/] signal{'s' if verified != 1 else ''} "
            f"[dim]· {pending} still pending[/]"
        )
    elif pending > 0:
        console.print(
            f"  [dim]Verified[/]  0 signals · [dim]{pending} pending (need more data)[/]"
        )


# ── Signal Display ────────────────────────────────────────────────────────


def display_signal(signal: TradeSignal) -> None:
    """Show a generated trade signal in a styled panel."""
    direction = signal.signal.value
    dir_style = "[bold green]BUY[/]" if direction == "BUY" else "[bold red]SELL[/]"
    dir_emoji = "📈" if direction == "BUY" else "📉"

    risk = abs(signal.entry - signal.stop_loss)
    rr = abs(signal.take_profit - signal.entry) / risk if risk > 0 else 0

    content = (
        f"  {dir_emoji} {dir_style}  [white]{signal.symbol}[/]  "
        f"[dim]confidence=[/]{signal.confidence:.1f}%\n"
        f"  [dim]Entry[/]  {signal.entry:,.2f}    [dim]SL[/]  {signal.stop_loss:,.2f}\n"
        f"  [dim]TP[/]     {signal.take_profit:,.2f}    [dim]R:R[/] 1:{rr:.1f}\n"
        f"  [dim]Mode[/]    {signal.trade_mode}  [{signal.confluence}]"
    )

    console.print(Panel(content, title="[bold]Signal[/]", border_style="cyan", padding=(0, 2)))


def display_no_signal(symbol: str) -> None:
    """Show that no signal was generated."""
    console.print(f"  [dim]No signal for {symbol} this cycle[/]")


# ── ML Prediction ─────────────────────────────────────────────────────────


def display_ml_prediction(prediction: Prediction, threshold: float, blocked: bool) -> None:
    """Show ML prediction result with filter decision."""
    # ML-adjusted confidence = rule confidence × ML probability
    ml_adj_conf = prediction.confidence

    if blocked:
        content = (
            f"  [dim]{prediction.model_type}_v{prediction.model_version}[/]  ·  "
            f"ML prob [bold red]{prediction.probability:.1%}[/]  →  "
            f"[dim]adj. confidence [bold red]{ml_adj_conf:.1f}%[/]\n"
            f"  [bold red]❌ BLOCKED[/]  below {threshold:.0%} threshold"
        )
        console.print(Panel(content, title="[bold]ML Prediction[/]", border_style="red", padding=(0, 2)))
    else:
        content = (
            f"  [dim]{prediction.model_type}_v{prediction.model_version}[/]  ·  "
            f"ML prob [bold green]{prediction.probability:.1%}[/]  →  "
            f"[dim]adj. confidence [bold green]{ml_adj_conf:.1f}%[/]\n"
            f"  [bold green]✅ APPROVED[/]  above {threshold:.0%} threshold"
        )
        console.print(Panel(content, title="[bold]ML Prediction[/]", border_style="green", padding=(0, 2)))


def display_no_ml_model() -> None:
    """Show that no ML model is available (rule-based only)."""
    console.print(f"  [dim]ML[/]  No model — using rule-based confidence only")


# ── Trade Result ───────────────────────────────────────────────────────────


def display_trade_placed(result: OrderResult) -> None:
    """Show a successful trade placement."""
    dir_style = "[bold green]BUY[/]" if result.direction == "BUY" else "[bold red]SELL[/]"
    content = (
        f"  ✅ {dir_style}  [white]{result.epic}[/]  [dim]size=[/]{result.size}\n"
        f"  [dim]SL[/]  {result.stop_level:,.2f}  [dim]TP[/]  {result.profit_level:,.2f}\n"
        f"  [dim]Ref[/]  {result.deal_reference}"
    )
    console.print(Panel(content, title="[bold green]Trade Placed[/]", border_style="green", padding=(0, 2)))


def display_trade_blocked(reason: str) -> None:
    """Show that a trade was blocked."""
    console.print(f"  [yellow]⏭  Trade blocked:[/] {reason}")


def display_trade_failed(result: OrderResult) -> None:
    """Show a failed trade."""
    content = f"  [bold red]❌ FAILED[/]  {result.direction} {result.epic}\n  [dim]{result.error}[/]"
    console.print(Panel(content, title="[bold red]Trade Error[/]", border_style="red", padding=(0, 2)))


def display_trade_skipped_existing(symbol: str) -> None:
    """Show that a trade was skipped because a position already exists."""
    console.print(f"  [yellow]⏭  Skipped: open position already exists for {symbol}[/]")


def display_trade_skipped_cooldown(symbol: str, remaining_min: float) -> None:
    """Show that a trade was skipped because a cooldown is active."""
    console.print(
        f"  [yellow]⏭  Cooldown: {remaining_min:.0f} min remaining for {symbol}[/]"
    )


def display_signal_skipped_cooldown(symbol: str, remaining_min: float) -> None:
    """Show that signal generation was skipped due to cooldown."""
    console.print(
        f"  [yellow]⏭  Signal skipped: loss cooldown for {symbol} "
        f"({remaining_min:.0f} min remaining)[/]"
    )


def display_position_reversed(closed_info: str, new_direction: str, symbol: str) -> None:
    """Show that an existing position was closed to reverse direction."""
    content = (
        f"  [yellow]🔄 REVERSED[/]  Closed {closed_info}\n"
        f"  [bold]{'[green]BUY[/]' if new_direction == 'BUY' else '[red]SELL[/]'}[/] "
        f"{symbol} — opposite signal detected"
    )
    console.print(Panel(content, title="[bold yellow]Position Reversed[/]", border_style="yellow", padding=(0, 2)))


def display_duplicate_signal(symbol: str, direction: str) -> None:
    """Show that a signal was not stored because it's a duplicate."""
    console.print(
        f"  [dim]⏭  Duplicate {direction} {symbol} signal — not stored (same direction within dedup window)[/]"
    )


# ── Auto-Train ────────────────────────────────────────────────────────────


def display_auto_train_waiting(verified: int, needed: int) -> None:
    """Show auto-train waiting for more data."""
    console.print(
        f"  [dim]🔄 Auto-train[/]  {verified}/{needed} verified — "
        f"[dim]need {needed - verified} more[/]"
    )


def display_auto_train_skipped(verified: int, last_trained: int) -> None:
    """Show auto-train skipping because already trained on this many signals."""
    console.print(
        f"  [dim]🔄 Auto-train[/]  {verified} verified · last trained on {last_trained} — [dim]skipping[/]"
    )


def display_auto_train_starting(verified: int) -> None:
    """Show auto-train starting."""
    console.print(f"  [bold cyan]🔄 Auto-train[/]  {verified} verified — training...")


def display_auto_train_result(model_type: str, symbol: str, accuracy: float, auc: float) -> None:
    """Show a single model training result."""
    auc_str = f"{auc:.4f}" if auc and auc > 0 else "N/A"
    console.print(f"  [dim]  {model_type:10s}[/]  acc={accuracy:.1%}  auc={auc_str}")


# ── Cycle Summary ─────────────────────────────────────────────────────────


def display_position_strength_exit(
    symbol: str,
    direction: str,
    tp_progress: float,
    strength_score: float,
    pnl: float,
    reason: str,
) -> None:
    """Show that a position was closed due to weak trend strength."""
    dir_style = "[bold green]BUY[/]" if direction == "BUY" else "[bold red]SELL[/]"
    content = (
        f"  [red]🚪 CLOSED[/]  {dir_style} [white]{symbol}[/]\n"
        f"  [dim]TP Progress[/]  {tp_progress:.1%}\n"
        f"  [dim]Strength Score[/]  {strength_score:.1f}/100\n"
        f"  [dim]P&L[/]  {pnl:+.2f}\n"
        f"  [dim]Reason[/]  {reason}"
    )
    console.print(Panel(content, title="[bold red]Strength Exit[/]", border_style="red", padding=(0, 2)))


def display_position_external_close(
    symbol: str,
    direction: str,
    cooldown_minutes: int,
    close_reason: str = "external_close",
) -> None:
    """Show that a position was closed externally (SL/TP hit at broker)."""
    dir_style = "[bold green]BUY[/]" if direction == "BUY" else "[bold red]SELL[/]"
    if close_reason == "tp_close":
        close_text = "Take profit hit at broker"
    elif close_reason == "sl_close":
        close_text = "Stop loss hit at broker"
    else:
        close_text = "Position closed externally (broker or manual close)"
    content = (
        f"  [yellow]🔔 EXTERNAL CLOSE[/]  {dir_style} [white]{symbol}[/]\n"
        f"  [dim]{close_text}[/]\n"
        f"  [dim]Cooldown started[/]  {cooldown_minutes} minutes before re-entry"
    )
    console.print(Panel(content, title="[bold yellow]External Close[/]", border_style="yellow", padding=(0, 2)))


def display_weekly_close_positions_closed(count: int, symbol: str) -> None:
    """Show that positions were closed due to weekly market close."""
    content = (
        f"  [yellow]🏁 WEEKLY CLOSE[/]  [white]{symbol}[/]\n"
        f"  [dim]Closed {count} position{'s' if count != 1 else ''} before weekly market close[/]\n"
        f"  [dim]Exness: Markets close Saturday 02:58/02:59 AM BST (UTC+6)[/]"
    )
    console.print(Panel(content, title="[bold yellow]Weekly Close[/]", border_style="yellow", padding=(0, 2)))


def display_weekly_close_status(
    symbol: str,
    hours_until: float | None,
    should_stop: bool,
    should_close: bool,
    is_weekend: bool = False,
    message: str = "",
    show_always: bool = False,
) -> None:
    """Display weekly close status for a symbol.

    Args:
        symbol: The trading symbol
        hours_until: Hours until weekly close (None if weekend)
        should_stop: Whether new trades should be stopped
        should_close: Whether positions should be closed
        is_weekend: Whether market is closed for weekend
        message: Status message to display
        show_always: If True, show status even when not in window
    """
    if is_weekend:
        # Weekend mode - market is closed
        status = "[red]WEEKEND - MARKET CLOSED[/]"
        icon = "[red]![/]"
    elif should_close:
        status = "[red]CLOSING POSITIONS[/]"
        icon = "[red]![/]"
    elif should_stop:
        status = "[yellow]STOP NEW TRADES[/]"
        icon = "[yellow]![/]"
    elif show_always and message:
        # Just show the message
        console.print(f"  [dim]i[/]  {message}")
        return
    else:
        return  # Don't show if not in window and not showing always

    console.print(
        f"  {icon}  {message} - {status}"
    )


def display_cycle_summary(stats: dict) -> None:
    """Print end-of-cycle summary panel."""
    lines = []

    # Signals
    if stats.get("signal_generated"):
        sig = stats["signal"]
        direction = sig.signal.value if hasattr(sig.signal, "value") else str(sig.signal)
        dir_emoji = "📈" if direction == "BUY" else "📉"
        lines.append(f"{dir_emoji} {direction} {sig.symbol}")
        lines.append(f"   Entry {sig.entry:,.2f}  SL {sig.stop_loss:,.2f}  TP {sig.take_profit:,.2f}")
    else:
        lines.append("[dim]No signal generated[/]")

    # ML
    if stats.get("ml_prediction"):
        pred = stats["ml_prediction"]
        if stats.get("ml_blocked"):
            lines.append(f"🤖 ML: [red]{pred.probability:.1%}[/] → BLOCKED")
        elif stats.get("ml_approved"):
            lines.append(f"🤖 ML: [green]{pred.probability:.1%}[/] → Approved")
    elif stats.get("no_ml"):
        lines.append("🤖 ML: [dim]No model[/]")

    # Trade
    if stats.get("trade_placed"):
        lines.append("✅ [green]Trade placed[/]")
    elif stats.get("trade_blocked"):
        lines.append("🔒 [yellow]Trade blocked[/]")
    elif stats.get("trade_reversed"):
        lines.append("🔄 [yellow]Position reversed[/]")
    elif stats.get("trade_cooldown"):
        lines.append("⏭  [yellow]Trade skipped (cooldown)[/]")
    elif stats.get("trade_skipped"):
        lines.append("⏭  [dim]Trade skipped[/]")
    elif stats.get("trade_failed"):
        lines.append("❌ [red]Trade failed[/]")

    # Verification
    if stats.get("verified", 0) > 0:
        lines.append(f"🔍 Verified [green]{stats['verified']}[/] signals")

    # Auto-train
    if stats.get("auto_train_started"):
        lines.append("🔄 [cyan]Model retrained[/]")

    # Timeout
    if stats.get("timeout_closed"):
        lines.append("⏰ [yellow]Position closed (timeout)[/]")

    # Strength exit
    if stats.get("strength_exit"):
        lines.append("🚪 [yellow]Position closed (weak trend)[/]")

    # Weekly close
    if stats.get("weekly_close_closed", 0) > 0:
        count = stats["weekly_close_closed"]
        lines.append(f"🏁 [yellow]{count} position{'s' if count != 1 else ''} closed (weekly close)[/]")

    content = "\n".join(lines)
    console.print()
    console.print(Panel(content, title="[bold]Cycle Summary[/]", border_style="bright_black", padding=(0, 2)))
    console.print()

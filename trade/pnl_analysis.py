"""PnL analysis module for trade performance metrics.

Provides comprehensive statistics for trading performance analysis:
- Overall PnL summary
- Win rate and loss rate
- PnL by symbol, direction, timeframe
- Daily/monthly PnL aggregation
- Risk-reward analysis
- Drawdown calculations
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

from sqlalchemy import func, case, and_, extract
from sqlalchemy.orm import Session

from data.database import get_session
from data.models import TradeLog

logger = logging.getLogger(__name__)


@dataclass
class PnLSummary:
    """Overall PnL summary statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float  # 0.0 to 1.0
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float  # gross wins / gross losses
    total_risk: float  # sum of all SL distances
    avg_risk_reward: float
    max_drawdown: float
    max_drawdown_pct: float


@dataclass
class SymbolPerformance:
    """Performance metrics per symbol."""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float


@dataclass
class DirectionPerformance:
    """Performance metrics per direction (BUY/SELL)."""
    direction: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float


def get_pnl_summary(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> PnLSummary:
    """Get overall PnL summary for completed trades.

    Args:
        start_date: Optional start date filter (UTC)
        end_date: Optional end date filter (UTC)

    Returns:
        PnLSummary with all statistics
    """
    with get_session() as session:
        # Build base query with filters
        query = session.query(
            func.count(TradeLog.id).label('total'),
            func.sum(
                case((TradeLog.final_pnl > 0, 1), else_=0)
            ).label('wins'),
            func.sum(
                case((TradeLog.final_pnl < 0, 1), else_=0)
            ).label('losses'),
            func.sum(
                case((TradeLog.final_pnl == 0, 1), else_=0)
            ).label('breakeven'),
            func.sum(TradeLog.final_pnl).label('total_pnl'),
            func.max(TradeLog.final_pnl).label('max_win'),
            func.min(TradeLog.final_pnl).label('max_loss'),
        ).filter(TradeLog.final_pnl.isnot(None))

        # Apply date filters
        if start_date:
            query = query.filter(TradeLog.closed_at >= start_date)
        if end_date:
            query = query.filter(TradeLog.closed_at <= end_date)

        result = query.first()

        if not result or result.total == 0:
            return PnLSummary(
                total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
                win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0, profit_factor=0.0,
                total_risk=0.0, avg_risk_reward=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
            )

        # Calculate average win/loss
        avg_win_q = session.query(
            func.avg(TradeLog.final_pnl)
        ).filter(
            TradeLog.final_pnl > 0,
            TradeLog.final_pnl.isnot(None)
        )
        avg_loss_q = session.query(
            func.avg(TradeLog.final_pnl)
        ).filter(
            TradeLog.final_pnl < 0,
            TradeLog.final_pnl.isnot(None)
        )

        if start_date:
            avg_win_q = avg_win_q.filter(TradeLog.closed_at >= start_date)
            avg_loss_q = avg_loss_q.filter(TradeLog.closed_at >= start_date)
        if end_date:
            avg_win_q = avg_win_q.filter(TradeLog.closed_at <= end_date)
            avg_loss_q = avg_loss_q.filter(TradeLog.closed_at <= end_date)

        avg_win = avg_win_q.scalar() or 0.0
        avg_loss = avg_loss_q.scalar() or 0.0

        # Calculate gross wins and losses for profit factor
        gross_wins = session.query(
            func.sum(TradeLog.final_pnl)
        ).filter(
            TradeLog.final_pnl > 0,
            TradeLog.final_pnl.isnot(None)
        )
        gross_losses = session.query(
            func.sum(func.abs(TradeLog.final_pnl))
        ).filter(
            TradeLog.final_pnl < 0,
            TradeLog.final_pnl.isnot(None)
        )

        if start_date:
            gross_wins = gross_wins.filter(TradeLog.closed_at >= start_date)
            gross_losses = gross_losses.filter(TradeLog.closed_at >= start_date)
        if end_date:
            gross_wins = gross_wins.filter(TradeLog.closed_at <= end_date)
            gross_losses = gross_losses.filter(TradeLog.closed_at <= end_date)

        gross_win_total = gross_wins.scalar() or 0.0
        gross_loss_total = abs(gross_losses.scalar() or 0.0)

        profit_factor = gross_win_total / gross_loss_total if gross_loss_total > 0 else 0.0

        # Calculate drawdown (need to get cumulative PnL over time)
        max_drawdown, max_drawdown_pct = _calculate_drawdown(session, start_date, end_date)

        # Calculate average risk-reward
        total_risk = session.query(
            func.sum(func.abs(TradeLog.entry_price - TradeLog.stop_loss) * TradeLog.size)
        ).filter(TradeLog.final_pnl.isnot(None))

        if start_date:
            total_risk = total_risk.filter(TradeLog.closed_at >= start_date)
        if end_date:
            total_risk = total_risk.filter(TradeLog.closed_at <= end_date)

        total_risk_val = total_risk.scalar() or 0.0
        total_realized = abs(result.total_pnl or 0)
        avg_risk_reward = total_realized / total_risk_val if total_risk_val > 0 else 0.0

        return PnLSummary(
            total_trades=result.total or 0,
            winning_trades=result.wins or 0,
            losing_trades=result.losses or 0,
            breakeven_trades=result.breakeven or 0,
            win_rate=(result.wins or 0) / result.total if result.total else 0.0,
            total_pnl=result.total_pnl or 0.0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=result.max_win or 0.0,
            largest_loss=result.max_loss or 0.0,
            profit_factor=profit_factor,
            total_risk=total_risk_val,
            avg_risk_reward=avg_risk_reward,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
        )


def _calculate_drawdown(
    session: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> tuple[float, float]:
    """Calculate maximum drawdown from peak equity.

    Returns:
        (max_drawdown_amount, max_drawdown_percent)
    """
    query = session.query(
        TradeLog.closed_at,
        TradeLog.final_pnl
    ).filter(
        TradeLog.final_pnl.isnot(None),
        TradeLog.closed_at.isnot(None)
    ).order_by(TradeLog.closed_at)

    if start_date:
        query = query.filter(TradeLog.closed_at >= start_date)
    if end_date:
        query = query.filter(TradeLog.closed_at <= end_date)

    trades = query.all()

    if not trades:
        return 0.0, 0.0

    # Calculate cumulative PnL over time
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0

    for _, pnl in trades:
        cumulative += pnl

        if cumulative > peak:
            peak = cumulative

        dd = peak - cumulative
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0

        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    return max_dd, max_dd_pct


def get_symbol_performance(top_n: int = 10) -> List[SymbolPerformance]:
    """Get performance metrics grouped by symbol.

    Args:
        top_n: Only return top N symbols by trade count

    Returns:
        List of SymbolPerformance sorted by total trades
    """
    with get_session() as session:
        results = session.query(
            TradeLog.symbol,
            func.count(TradeLog.id).label('total'),
            func.sum(
                case((TradeLog.final_pnl > 0, 1), else_=0)
            ).label('wins'),
            func.sum(
                case((TradeLog.final_pnl < 0, 1), else_=0)
            ).label('losses'),
            func.sum(TradeLog.final_pnl).label('total_pnl'),
        ).filter(
            TradeLog.final_pnl.isnot(None)
        ).group_by(
            TradeLog.symbol
        ).order_by(
            func.count(TradeLog.id).desc()
        ).limit(top_n).all()

        performances = []
        for r in results:
            total = r.total or 0
            wins = r.wins or 0
            losses = r.losses or 0
            total_pnl = r.total_pnl or 0.0

            # Calculate gross wins/losses for profit factor
            gross_wins = session.query(
                func.sum(TradeLog.final_pnl)
            ).filter(
                TradeLog.symbol == r.symbol,
                TradeLog.final_pnl > 0
            ).scalar() or 0.0

            gross_losses = abs(session.query(
                func.sum(TradeLog.final_pnl)
            ).filter(
                TradeLog.symbol == r.symbol,
                TradeLog.final_pnl < 0
            ).scalar() or 0.0)

            profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

            performances.append(SymbolPerformance(
                symbol=r.symbol,
                total_trades=total,
                wins=wins,
                losses=losses,
                win_rate=wins / total if total > 0 else 0.0,
                total_pnl=total_pnl,
                avg_pnl=total_pnl / total if total > 0 else 0.0,
                profit_factor=profit_factor,
            ))

        return performances


def get_direction_performance() -> List[DirectionPerformance]:
    """Get performance metrics grouped by direction (BUY/SELL).

    Returns:
        List of DirectionPerformance
    """
    with get_session() as session:
        results = session.query(
            TradeLog.direction,
            func.count(TradeLog.id).label('total'),
            func.sum(
                case((TradeLog.final_pnl > 0, 1), else_=0)
            ).label('wins'),
            func.sum(
                case((TradeLog.final_pnl < 0, 1), else_=0)
            ).label('losses'),
            func.sum(TradeLog.final_pnl).label('total_pnl'),
        ).filter(
            TradeLog.final_pnl.isnot(None)
        ).group_by(
            TradeLog.direction
        ).all()

        return [
            DirectionPerformance(
                direction=r.direction,
                total_trades=r.total or 0,
                wins=r.wins or 0,
                losses=r.losses or 0,
                win_rate=(r.wins or 0) / r.total if r.total else 0.0,
                total_pnl=r.total_pnl or 0.0,
                avg_pnl=(r.total_pnl or 0.0) / r.total if r.total else 0.0,
            )
            for r in results
        ]


def get_close_reason_stats() -> Dict[str, int]:
    """Get trade count grouped by close reason.

    Returns:
        Dict mapping close_reason to count
    """
    with get_session() as session:
        results = session.query(
            TradeLog.close_reason,
            func.count(TradeLog.id).label('count')
        ).filter(
            TradeLog.close_reason.isnot(None)
        ).group_by(
            TradeLog.close_reason
        ).all()

        return {r.close_reason: r.count for r in results}


def print_summary_report(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> None:
    """Print a comprehensive PnL summary report.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Overall summary
    summary = get_pnl_summary(start_date, end_date)

    console.print()
    console.print(Panel(
        f"[bold cyan]P&L Summary Report[/]\n"
        f"Period: [dim]{start_date or 'All time'} to {end_date or 'Now'}[/]",
        border_style="cyan"
    ))

    # Summary table
    table = Table(title="Overall Performance", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    pnl_color = "[green]" if summary.total_pnl >= 0 else "[red]"
    table.add_row("Total Trades", str(summary.total_trades))
    table.add_row("Win Rate", f"{summary.win_rate * 100:.1f}%")
    table.add_row("Total P&L", f"{pnl_color}{summary.total_pnl:.2f}[/]")
    table.add_row("Avg Win", f"[green]{summary.avg_win:.2f}[/]")
    table.add_row("Avg Loss", f"[red]{summary.avg_loss:.2f}[/]")
    table.add_row("Largest Win", f"[green]{summary.largest_win:.2f}[/]")
    table.add_row("Largest Loss", f"[red]{summary.largest_loss:.2f}[/]")
    table.add_row("Profit Factor", f"{summary.profit_factor:.2f}")
    table.add_row("Max Drawdown", f"[red]{summary.max_drawdown:.2f}[/]")
    table.add_row("Max Drawdown %", f"[red]{summary.max_drawdown_pct:.1f}%[/]")

    console.print(table)

    # Symbol performance
    console.print()
    symbol_perf = get_symbol_performance()

    sym_table = Table(title="Performance by Symbol", show_header=True)
    sym_table.add_column("Symbol", style="cyan")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("Win Rate", justify="right")
    sym_table.add_column("Total P&L", justify="right")
    sym_table.add_column("Avg P&L", justify="right")
    sym_table.add_column("Profit Factor", justify="right")

    for s in symbol_perf:
        pnl_color = "[green]" if s.total_pnl >= 0 else "[red]"
        sym_table.add_row(
            s.symbol,
            str(s.total_trades),
            f"{s.win_rate * 100:.1f}%",
            f"{pnl_color}{s.total_pnl:.2f}[/]",
            f"{pnl_color}{s.avg_pnl:.2f}[/]",
            f"{s.profit_factor:.2f}",
        )

    console.print(sym_table)

    # Direction performance
    console.print()
    dir_perf = get_direction_performance()

    dir_table = Table(title="Performance by Direction", show_header=True)
    dir_table.add_column("Direction", style="cyan")
    dir_table.add_column("Trades", justify="right")
    dir_table.add_column("Win Rate", justify="right")
    dir_table.add_column("Total P&L", justify="right")

    for d in dir_perf:
        pnl_color = "[green]" if d.total_pnl >= 0 else "[red]"
        dir_table.add_row(
            d.direction,
            str(d.total_trades),
            f"{d.win_rate * 100:.1f}%",
            f"{pnl_color}{d.total_pnl:.2f}[/]",
        )

    console.print(dir_table)

    # Close reason stats
    console.print()
    close_stats = get_close_reason_stats()

    reason_table = Table(title="Close Reason Distribution", show_header=True)
    reason_table.add_column("Reason", style="cyan")
    reason_table.add_column("Count", justify="right")

    for reason, count in close_stats.items():
        reason_table.add_row(reason, str(count))

    console.print(reason_table)
    console.print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze trading P&L")
    parser.add_argument("--days", type=int, default=30, help="Days to look back")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")

    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    print_summary_report(start, end)

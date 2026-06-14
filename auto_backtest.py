"""Automated backtest and ML training pipeline.

This command automatically:
1. Fetches candle data for specified days
2. Backtests to generate and verify signals
3. Trains ML models if enough verified signals (30+)

Usage:
    python auto_backtest.py                          # Auto-detect and update all stale symbols
    python auto_backtest.py --symbol XAUUSD --days 30    # Backtest XAUUSD for 30 days
    python auto_backtest.py --symbol XAUUSD --days 30 --reset    # Delete existing signals first
    python auto_backtest.py --symbols XAGUSD BTCUSD --force    # Force update specific symbols
"""

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config.settings import SETTINGS, Symbol, Timeframe, TradeMode, TradingPlatform
from config.utils import normalize_symbol
from data.collector import CapitalComClient
from data.database import get_session
from data.models import Signal, SignalVerification, MLModelMeta
from sqlalchemy import func, desc

# Import backtest functions
from backtest_historical import (
    _fetch_candles_batched, _store_candles, _load_candles_from_db,
    _CandleProxy, backfill_signals, _DUP_WINDOW
)
from backtesting.walk_forward import run_walk_forward
from display import display_auto_train_result

console = Console()

logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)


# ── Configuration ───────────────────────────────────────────────────────────

AUTO_DETECT_DAYS = True  # Automatically detect how many days of new data exist
MIN_SIGNALS_FOR_TRAIN = 30  # Minimum verified signals to trigger ML training


# ── Data Analysis ────────────────────────────────────────────────────────────

def get_symbol_status(symbol: str) -> dict:
    """Get the current status of a symbol's data and models.

    Returns:
        dict with:
        - last_signal_date: datetime of most recent signal
        - last_model_date: datetime of most recent model training
        - total_signals: count of total signals
        - total_verified: count of verified signals
        - has_active_model: bool
        - days_since_model: int
        - needs_update: bool
    """
    with get_session() as session:
        # Get last signal date
        last_signal = session.query(func.max(Signal.timestamp)).filter(
            Signal.symbol == symbol
        ).scalar()

        # Get last model training date (active model)
        last_model = session.query(
            func.max(MLModelMeta.trained_at)
        ).filter(
            MLModelMeta.symbol == symbol,
            MLModelMeta.is_active == True
        ).scalar()

        # Get counts
        total_signals = session.query(func.count(Signal.id)).filter(
            Signal.symbol == symbol
        ).scalar() or 0

        total_verified = session.query(func.count(SignalVerification.id)).join(
            Signal
        ).filter(Signal.symbol == symbol).scalar() or 0

        # Check if has active model
        has_active_model = session.query(func.count(MLModelMeta.id)).filter(
            MLModelMeta.symbol == symbol,
            MLModelMeta.is_active == True
        ).scalar() > 0

    now = datetime.now(tz=timezone.utc)

    # Calculate days since model training
    days_since_model = 0
    if last_model:
        if last_model.tzinfo is None:
            last_model = last_model.replace(tzinfo=timezone.utc)
        days_since_model = (now - last_model).days

    # Calculate available days since last signal (how many days of NEW data exist)
    available_days = 0
    if last_signal:
        if last_signal.tzinfo is None:
            last_signal = last_signal.replace(tzinfo=timezone.utc)
        available_days = (now - last_signal).days
    else:
        # No signals yet - need to fetch initial data
        available_days = 999

    # Determine if needs update (any new data available)
    needs_update = available_days >= 1

    return {
        "symbol": symbol,
        "last_signal_date": last_signal,
        "last_model_date": last_model,
        "total_signals": total_signals,
        "total_verified": total_verified,
        "has_active_model": has_active_model,
        "days_since_model": days_since_model,
        "available_days": available_days,
        "needs_update": needs_update,
    }


def analyze_all_symbols(symbols: List[Symbol]) -> List[dict]:
    """Analyze status for all symbols and return those needing updates."""
    statuses = []

    console.print("\n[bold cyan]📊 Analyzing symbol data status...[/]\n")

    for symbol in symbols:
        status = get_symbol_status(symbol.value)
        statuses.append(status)

    # Sort by needs_update first, then by available_days (most stale first)
    statuses.sort(key=lambda x: (-x["needs_update"], -x.get("available_days", 0)))

    # Display status table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="cyan")
    table.add_column("Signals", justify="right")
    table.add_column("Verified", justify="right")
    table.add_column("New Days", justify="right")
    table.add_column("Status", justify="center")

    for s in statuses:
        new_days = s.get("available_days", 0)
        days_text = f"{new_days}d" if new_days < 999 else "NEW"
        status_color = "[yellow]UPDATE[/]" if s['needs_update'] else "[dim]OK[/]"

        table.add_row(
            s["symbol"],
            str(s["total_signals"]),
            str(s["total_verified"]),
            days_text,
            status_color
        )

    console.print(table)

    return statuses


# ── Candle Fetching ──────────────────────────────────────────────────────────

def fetch_new_candles_for_symbol(
    client,
    symbol: Symbol,
    last_date: Optional[datetime] = None,
    weeks: int = 2,
) -> dict:
    """Fetch new candles for a symbol since last update.

    Returns:
        dict with counts of new candles per timeframe
    """
    if last_date:
        # Start from the day after last update
        from_date = last_date + timedelta(days=1)
    else:
        # No existing data - fetch full history
        from_date = datetime.now(tz=timezone.utc) - timedelta(weeks=weeks)

    now = datetime.now(tz=timezone.utc)
    days_to_fetch = (now - from_date).days

    if days_to_fetch <= 0:
        return {"M5": 0, "H1": 0, "H4": 0}

    console.print(
        f"\n[dim]Fetching {symbol.value} candles from {from_date.strftime('%Y-%m-%d')} "
        f"({days_to_fetch} days)[/]"
    )

    results = {}

    for timeframe in [Timeframe.M5, Timeframe.H1, Timeframe.H4]:
        try:
            df = client.fetch_candles(
                symbol, timeframe,
                from_date=from_date,
                to_date=now,
            )

            if not df.empty:
                # Store new candles
                count = _store_candles(df, symbol.value, timeframe.value)
                results[timeframe.value] = count
                console.print(f"  [green]+{count}[/] [dim]{timeframe.value} candles[/]")
            else:
                results[timeframe.value] = 0
                console.print(f"  [dim]0 {timeframe.value} candles[/]")

        except Exception as e:
            logger.warning("Fetch failed for %s %s: %s", symbol.value, timeframe.value, e)
            results[timeframe.value] = 0

    return results


# ── Signal Generation ────────────────────────────────────────────────────────

def generate_new_signals(
    symbol: Symbol,
    mode: TradeMode,
    from_date: datetime,
    skip_duplicates: bool = True,
) -> dict:
    """Generate and verify signals only for the new data period.

    This replays the signal generation only on candles AFTER from_date,
    using historical data only for indicator calculation (not for signal generation).

    Args:
        symbol: Trading symbol
        mode: Trade mode
        from_date: Start of new data period

    Returns:
        dict with signal stats
    """
    console.print(f"\n[bold cyan]🔄 Generating signals for NEW data ({symbol.value})[/]")

    # DB-loaded timestamps are tz-naive (SQLite strips timezone info),
    # so strip tz from from_date to ensure consistent comparisons.
    if from_date.tzinfo is not None:
        from_date = from_date.replace(tzinfo=None)

    # Load all candles (historical + new)
    df_m5 = _load_candles_from_db(symbol.value, Timeframe.M5.value, limit=20000)
    df_h1 = _load_candles_from_db(symbol.value, Timeframe.H1.value, limit=5000)
    df_h4 = _load_candles_from_db(symbol.value, Timeframe.H4.value, limit=2000)

    if df_m5.empty or df_h1.empty:
        console.print("[yellow]⚠ No candle data available[/]")
        return {"error": "no_data"}

    # Find the index where new data starts
    new_data_start_idx = 0
    for i, ts in enumerate(df_m5["timestamp"]):
        if pd.Timestamp(ts) >= pd.Timestamp(from_date):
            new_data_start_idx = i
            break

    console.print(f"  [dim]Total M5 candles[/]  {len(df_m5)}")
    console.print(f"  [dim]New data starts at index[/]  {new_data_start_idx}")

    if new_data_start_idx < 50:
        console.print("[yellow]⚠ Not enough historical data for indicators[/]")
        return {"error": "insufficient_hist"}

    # Settings
    lookahead = SETTINGS.verification.lookahead_candles
    min_lookahead = SETTINGS.verification.min_lookahead_candles
    dup_window = _DUP_WINDOW.get(mode, 60)

    total_m5 = len(df_m5)
    end_idx = total_m5 - min_lookahead

    if end_idx <= new_data_start_idx + 50:
        console.print("[yellow]⚠ Not enough new M5 data for replay[/]")
        return {"error": "insufficient_new"}

    console.print(
        f"  [dim]Replay range[/]  index {new_data_start_idx} → {end_idx} "
        f"({end_idx - new_data_start_idx} new candles)"
    )

    stats = {
        "checked": 0,
        "signals": 0,
        "wins": 0,
        "losses": 0,
        "breakevens": 0,
        "no_trade": 0,
        "verified": 0,
        "skipped_duplicate": 0,
    }

    # Pre-convert to arrays
    m5_timestamps = df_m5["timestamp"].values
    m5_highs = df_m5["high"].values
    m5_lows = df_m5["low"].values
    m5_closes = df_m5["close"].values

    step = 12  # Check every hour for efficiency

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("• {task.fields[info]}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Replaying...", total=end_idx - new_data_start_idx, info="starting"
        )

        with get_session() as session:
            for i in range(new_data_start_idx, end_idx, step):
                m5_ts = pd.Timestamp(m5_timestamps[i])

                # Only process if this is in the "new" period
                if i < new_data_start_idx:
                    continue

                # Slice data up to this point
                df_m5_slice = df_m5.iloc[:i + 1].copy()
                df_h1_slice = df_h1[df_h1["timestamp"] <= m5_ts].copy()
                df_h4_slice = df_h4[df_h4["timestamp"] <= m5_ts].copy()

                if len(df_h1_slice) < 200:
                    progress.update(task, advance=step, info="warming up")
                    continue
                if mode != TradeMode.SCALPING and len(df_h4_slice) < 5:
                    progress.update(task, advance=step, info="warming up")
                    continue

                stats["checked"] += 1

                # Import generate_signal
                from signals.engine import generate_signal
                from verification.verifier import verify_signal
                from data.models import Signal, SignalVerification

                try:
                    trade_signal = generate_signal(
                        df_h4_slice, df_h1_slice, df_m5_slice, symbol, mode, prefix="auto_"
                    )
                except Exception as e:
                    logger.debug("Signal error at %s: %s", m5_ts, e)
                    progress.update(task, advance=step, info="error")
                    continue

                if not trade_signal or trade_signal.signal.value == "NO_TRADE":
                    stats["no_trade"] += 1
                    progress.update(task, advance=step, info="no trade")
                    continue

                stats["signals"] += 1

                # Duplicate check (skip when reset — replaying all signals from scratch)
                if skip_duplicates:
                    dup_cutoff = m5_ts - pd.Timedelta(minutes=dup_window)
                    recent_dup = session.query(Signal).filter(
                        Signal.symbol == symbol.value,
                        Signal.signal == trade_signal.signal.value,
                        Signal.entry >= trade_signal.entry * 0.9995,
                        Signal.entry <= trade_signal.entry * 1.0005,
                        Signal.timestamp >= dup_cutoff,
                    ).first()

                    if recent_dup:
                        stats["skipped_duplicate"] += 1
                        progress.update(task, advance=step, info="duplicate")
                        continue

                # Store signal
                features = trade_signal.features.to_dict()
                db_signal = Signal(
                    signal_id=trade_signal.signal_id,
                    symbol=trade_signal.symbol,
                    timeframe=trade_signal.timeframe,
                    signal=trade_signal.signal,
                    confidence=trade_signal.confidence,
                    entry=trade_signal.entry,
                    stop_loss=trade_signal.stop_loss,
                    take_profit=trade_signal.take_profit,
                    timestamp=trade_signal.timestamp,
                    ema20=features.get("ema20"),
                    ema50=features.get("ema50"),
                    ema200=features.get("ema200"),
                    rsi14=features.get("rsi14"),
                    macd_line=features.get("macd_line"),
                    macd_signal=features.get("macd_signal"),
                    macd_histogram=features.get("macd_histogram"),
                    atr14=features.get("atr14"),
                    candle_body_pct=features.get("candle_body_pct"),
                    upper_wick_pct=features.get("upper_wick_pct"),
                    lower_wick_pct=features.get("lower_wick_pct"),
                    dist_ema20=features.get("dist_ema20"),
                    dist_ema50=features.get("dist_ema50"),
                    dist_ema200=features.get("dist_ema200"),
                    trend_strength=features.get("trend_strength"),
                    momentum_score=features.get("momentum_score"),
                    session=features.get("session"),
                    spread=features.get("spread"),
                )
                session.add(db_signal)
                session.flush()

                # Verify using future candles
                future_start = i + 1
                future_count = total_m5 - future_start

                if future_count < min_lookahead:
                    progress.update(task, advance=step, info="end of data")
                    continue

                check_count = min(future_count, lookahead)
                future_candles = [
                    _CandleProxy(
                        high=float(m5_highs[j]),
                        low=float(m5_lows[j]),
                        close=float(m5_closes[j]),
                    )
                    for j in range(future_start, future_start + check_count)
                ]

                result = verify_signal(db_signal, future_candles)

                if result:
                    verification = SignalVerification(
                        signal_id=db_signal.id,
                        result=result.result,
                        profit_points=result.profit_points,
                        risk_reward=result.risk_reward,
                        holding_candles=result.holding_candles,
                        max_favorable_excursion=result.max_favorable_excursion,
                        max_adverse_excursion=result.max_adverse_excursion,
                    )
                    session.add(verification)
                    stats["verified"] += 1

                    if result.result.value == "WIN":
                        stats["wins"] += 1
                        emoji = "📈"
                    elif result.result.value == "LOSS":
                        stats["losses"] += 1
                        emoji = "📉"
                    else:
                        stats["breakevens"] += 1
                        emoji = "➖"

                    progress.update(
                        task, advance=step,
                        info=f"{emoji} {trade_signal.signal.value} "
                             f"W{stats['wins']} L{stats['losses']}"
                    )
                else:
                    progress.update(task, advance=step, info="unverified")

                session.commit()

    return stats


# ── ML Training ───────────────────────────────────────────────────────────────

def train_if_needed(symbol: str, new_verified_count: int) -> bool:
    """Train ML models if enough new verified signals exist.

    Returns:
        True if training was performed
    """
    if new_verified_count < MIN_SIGNALS_FOR_TRAIN:
        console.print(
            f"\n[dim]Only {new_verified_count} new verified signals "
            f"(need {MIN_SIGNALS_FOR_TRAIN}) — skipping ML training[/]"
        )
        return False

    console.print(
        f"\n[bold cyan]🤖 Training ML models for {symbol}[/] "
        f"({new_verified_count} new signals)"
    )

    try:
        reports = run_walk_forward(symbol, timeframe=None)

        for report in reports:
            display_auto_train_result(
                report.model_type, report.symbol,
                report.mean_accuracy, report.mean_roc_auc,
            )

        console.print("[green]✓ ML models trained and activated![/]")
        return True

    except Exception as e:
        console.print(f"[red]✗ ML training failed: {e}[/]")
        logger.error("ML training failed for %s: %s", symbol, e)
        return False


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_auto_backtest(
    symbols: Optional[List[Symbol]] = None,
    force: bool = False,
    mode: TradeMode = None,
    days: Optional[int] = None,
    reset: bool = False,
) -> dict:
    """Run the full automated backtest and training pipeline.

    Args:
        symbols: List of symbols to process (None = all configured symbols)
        force: Force update all symbols regardless of staleness
        mode: Trade mode (None = from settings)
        days: Number of days to backtest (None = auto-detect)
        reset: Delete existing signals before backtesting

    Returns:
        dict with overall results
    """
    if symbols is None:
        symbols = list(SETTINGS.symbols)

    if mode is None:
        mode = SETTINGS.trade_mode

    display_days = f"{days} days" if days else "auto-detect"
    display_reset = "ON (will delete existing signals)" if reset else "OFF"

    console.print(Panel(
        f"[bold]Automated Backtest & ML Pipeline[/]\n"
        f"  [dim]Mode[/]  {mode.value}\n"
        f"  [dim]Symbols[/]  {', '.join(s.value for s in symbols)}\n"
        f"  [dim]Days[/]  {display_days}\n"
        f"  [dim]Reset[/]  {display_reset}\n"
        f"  [dim]Force[/]  {force}",
        border_style="cyan",
    ))

    # ── Reset: Delete existing signals if requested ────────────────────────────────
    if reset:
        console.print("\n[bold yellow]🗑️  Reset mode: deleting existing signals...[/]")
        with get_session() as session:
            for symbol in symbols:
                # Delete verifications first (foreign key constraint)
                # SQLAlchemy doesn't allow .delete() on a query with .join(),
                # so we fetch signal IDs first and delete by those IDs.
                signal_ids = [
                    row[0] for row in
                    session.query(Signal.id)
                    .filter(Signal.symbol == symbol.value)
                    .all()
                ]
                deleted_verifications = (
                    session.query(SignalVerification)
                    .filter(SignalVerification.signal_id.in_(signal_ids))
                    .delete(synchronize_session=False)
                ) if signal_ids else 0
                # Delete signals
                deleted_signals = (
                    session.query(Signal)
                    .filter(Signal.symbol == symbol.value)
                    .delete(synchronize_session=False)
                )
                # Delete ML models
                deleted_models = (
                    session.query(MLModelMeta)
                    .filter(MLModelMeta.symbol == symbol.value)
                    .delete(synchronize_session=False)
                )
                # Delete ML model files on disk (.joblib + _features.json + _calibrated.joblib)
                model_dir = SETTINGS.ml.model_dir
                deleted_files = 0
                if os.path.isdir(model_dir):
                    for fname in os.listdir(model_dir):
                        if symbol.value in fname:
                            fpath = os.path.join(model_dir, fname)
                            try:
                                os.remove(fpath)
                                deleted_files += 1
                            except OSError as e:
                                logger.warning("Failed to delete %s: %s", fpath, e)
                # Delete tuned params files on disk
                deleted_params = 0
                if os.path.isdir(model_dir):
                    for fname in os.listdir(model_dir):
                        if symbol.value in fname and fname.endswith("_tuned_params.json"):
                            fpath = os.path.join(model_dir, fname)
                            try:
                                os.remove(fpath)
                                deleted_params += 1
                            except OSError as e:
                                logger.warning("Failed to delete %s: %s", fpath, e)
                session.commit()
                console.print(
                    f"  [yellow]Deleted[/] {deleted_signals} signals, "
                    f"{deleted_verifications} verifications, "
                    f"{deleted_models} ML models, "
                    f"{deleted_files} model files for {symbol.value}"
                )
        console.print("[green]✓ Reset complete[/]\n")

    # ── Phase 1: Analyze ────────────────────────────────────────────────────────
    statuses = analyze_all_symbols(symbols)

    # Filter symbols that need processing (any with new data available)
    if force:
        to_process = statuses
    else:
        to_process = [s for s in statuses if s["needs_update"]]

    if not to_process:
        console.print("\n[green]✓ All symbols up to date![/]")
        return {"status": "up_to_date"}

    console.print(
        f"\n[dim]Processing {len(to_process)} symbols[/]"
    )

    # ── Phase 2: Connect ────────────────────────────────────────────────────────
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        from data.mt5_client import MT5Client
        client = MT5Client()
        if not client.connect():
            console.print("[bold red]✗ MT5 connection failed[/]")
            return {"error": "mt5_failed"}
    else:
        client = CapitalComClient()
        if not client.authenticate():
            console.print("[bold red]✗ Authentication failed[/]")
            return {"error": "auth_failed"}

    # ── Phase 3: Process each symbol ────────────────────────────────────────────
    results = {
        "processed": [],
        "skipped": [],
        "errors": [],
    }

    for status in to_process:
        symbol = Symbol(status["symbol"])
        console.print(f"\n{'═' * 50}")
        console.print(f"[bold cyan]═══ {symbol.value} ═══[/]")
        console.print(f"{'═' * 50}")

        try:
            # If days specified, use full backfill (like backtest_historical.py)
            if days is not None:
                weeks = max(1, days // 7)  # Convert days to weeks
                console.print(f"\n[bold]🔄 Full backfill for {days} days ({weeks} weeks)[/]")
                signal_results = backfill_signals(symbol, mode, weeks=weeks, step=12)

                if "error" in signal_results:
                    console.print(f"[yellow]⚠ Backfill error: {signal_results.get('error')}[/]")
                    results["errors"].append((symbol.value, signal_results.get('error')))
                    continue

                # Extract results from backfill
                new_signals = signal_results.get("signals", 0)
                new_verified = signal_results.get("verified", 0)
                new_wins = signal_results.get("wins", 0)
                new_losses = signal_results.get("losses", 0)
                new_be = signal_results.get("breakevens", 0)
                total_new_candles = "N/A"

            else:
                # 1. Fetch new candles
                candle_results = fetch_new_candles_for_symbol(
                    client,
                    symbol,
                    status["last_signal_date"],
                    weeks=4,
                )

                total_new_candles = sum(candle_results.values())
                if total_new_candles == 0:
                    console.print("[dim]No new candles — skipping[/]")
                    results["skipped"].append(symbol.value)
                    continue

                # 2. Generate signals for new period
                from_date = status["last_signal_date"] or (datetime.now(tz=timezone.utc) - timedelta(days=7))
                signal_results = generate_new_signals(symbol, mode, from_date, skip_duplicates=not reset)

                if "error" in signal_results:
                    console.print(f"[yellow]⚠ Signal generation error: {signal_results['error']}[/]")
                    results["errors"].append((symbol.value, signal_results['error']))
                    continue

                # Extract results
                new_verified = signal_results["verified"]
                new_wins = signal_results["wins"]
                new_losses = signal_results["losses"]
                new_be = signal_results["breakevens"]
                new_signals = signal_results["signals"]

            # 3. Display signal results
            console.print(
                f"\n[dim]Signals[/]  {new_signals} generated, "
                f"{new_verified} verified\n"
                f"  [green]Wins[/]  {new_wins}  [red]Losses[/]  {new_losses}  "
                f"[yellow]BE[/]  {new_be}"
            )

            # 4. Train ML if needed
            trained = train_if_needed(symbol.value, new_verified)

            results["processed"].append({
                "symbol": symbol.value,
                "new_candles": total_new_candles,
                "new_signals": new_signals,
                "new_verified": new_verified,
                "new_wins": new_wins,
                "new_losses": new_losses,
                "ml_trained": trained,
            })

        except Exception as e:
            logger.error("Error processing %s: %s", symbol.value, e)
            console.print(f"[red]✗ Error: {e}[/]")
            results["errors"].append((symbol.value, str(e)))

    # ── Summary ────────────────────────────────────────────────────────────────
    console.print("\n")
    console.print(Panel(
        f"  [dim]Symbols processed[/]  {len(results['processed'])}\n"
        f"  [dim]Skipped[/]  {len(results['skipped'])}\n"
        f"  [dim]Errors[/]  {len(results['errors'])}",
        title="[bold]Pipeline Summary[/]",
        border_style="cyan",
    ))

    if results["processed"]:
        console.print("\n[bold]Processed Results:[/]")
        for r in results["processed"]:
            train_status = "[green]✓ Trained[/]" if r["ml_trained"] else "[dim]No train[/]"
            console.print(
                f"  {r['symbol']}: +{r['new_signals']} sigs, "
                f"+{r['new_verified']} verified ({r['new_wins']}W/{r['new_losses']}L) "
                f"{train_status}"
            )

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated backtest and ML training pipeline"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Single symbol to backtest (e.g., XAUUSD)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Multiple symbols to process (default: all configured symbols)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Number of days to backtest (e.g., 30). If not specified, auto-detects new data only"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing signals for the symbol(s) before backtesting"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force update all symbols regardless of staleness"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["scalping", "day_trading", "swing"],
        help="Trade mode (default: from .env)"
    )

    args = parser.parse_args()

    mode = TradeMode(args.mode) if args.mode else None

    # Handle --symbol (single) vs --symbols (multiple)
    if args.symbol:
        symbols = [Symbol(args.symbol)]
    elif args.symbols:
        symbols = [Symbol(s) for s in args.symbols]
    else:
        symbols = None

    run_auto_backtest(
        symbols=symbols,
        force=args.force,
        mode=mode,
        days=args.days,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()

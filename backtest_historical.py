"""Backfill: replay historical candles to generate and verify signals for ML training.

Instead of waiting days for 30+ verified signals, this script:
1. Fetches weeks of historical candle data in batches
2. Replays it through the signal engine at each M5 checkpoint
3. Verifies signals immediately using known future candles
4. Trains ML models on the accumulated verified data

No lookahead bias: signals are generated using only data available at that
point in time, and verified using only future candles.

Usage:
    python3 backtest_historical.py                            # Auto-detect & update all stale symbols, then train
    python3 backtest_historical.py --weeks 4 --step 1         # Full backfill for 4 weeks, then train
    python3 backtest_historical.py --mode day_trading --symbol XAUUSD
    python3 backtest_historical.py --auto --force            # Force-update all symbols regardless of staleness
    python3 backtest_historical.py --reset                    # Delete signals/models then full backfill
    python3 backtest_historical.py --train --symbol XAUUSD    # Train ML only (no backfill)
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

from config.settings import SETTINGS, Symbol, Timeframe, TradeMode
from config.utils import normalize_symbol
from data.database import create_tables, get_session
from data.models import Candle, Signal, SignalVerification, MLModelMeta
from sqlalchemy import func, desc
from signals.engine import generate_signal
from verification.verifier import verify_signal

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

MIN_SIGNALS_FOR_TRAIN = 30  # Minimum verified signals to trigger ML training

# ── Candle storage / loading (same as main.py) ──────────────────────────


def _store_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """Store fetched candles into the database. Returns count of new candles."""
    count = 0
    normalized_symbol = normalize_symbol(symbol)  # Strip 'm' suffix for DB consistency

    with get_session() as session:
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = pd.Timestamp(ts, tz="UTC")

            exists = (
                session.query(Candle)
                .filter(
                    Candle.symbol == normalized_symbol,
                    Candle.timeframe == timeframe,
                    Candle.timestamp == ts,
                )
                .first()
            )
            if exists:
                continue

            candle = Candle(
                symbol=normalized_symbol,
                timeframe=timeframe,
                timestamp=ts,
                open_price=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0) or 0),
                spread=float(row.get("spread", 0) or 0),
            )
            session.add(candle)
            count += 1

    return count


def _load_candles_from_db(symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    """Load candles from the database into a DataFrame."""
    normalized_symbol = normalize_symbol(symbol)  # Strip 'm' suffix for DB consistency
    with get_session() as session:
        candles = (
            session.query(Candle)
            .filter(Candle.symbol == normalized_symbol, Candle.timeframe == timeframe)
            .order_by(Candle.timestamp.desc())
            .limit(limit)
            .all()
        )

        if not candles:
            return pd.DataFrame()

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


# ── Multi-batch fetching ──────────────────────────────────────────────────


def _fetch_candles_batched(
    client,
    symbol: Symbol,
    timeframe: Timeframe,
    weeks: int = 4,
) -> pd.DataFrame:
    """Fetch historical candles in batches to get weeks of data.

    Fetches data via MetaTrader 5, which can fetch larger ranges using copy_rates_range.

    For M5: 12 candles/hour, 288/day, 2016/week
    For H1: 1 candle/hour, 24/day, 168/week
    For H4: 6 candles/day, 42/week

    Args:
        client: Authenticated MT5Client.
        symbol: Trading symbol.
        timeframe: Candle timeframe.
        weeks: Number of weeks of historical data to fetch.

    Returns:
        Concatenated DataFrame with all fetched candles.
    """
    now = datetime.now(tz=timezone.utc)
    all_dfs = []

    # Calculate how many candles we need per timeframe
    candles_per_week = {
        Timeframe.M5: 288 * 7,   # 2016
        Timeframe.H1: 24 * 7,    # 168
        Timeframe.H4: 6 * 7,     # 42
    }
    target_candles = candles_per_week.get(timeframe, 2000) * weeks

    # MT5 can fetch larger date ranges at once
    # Calculate start date based on weeks requested
    from_date = now - timedelta(weeks=weeks)

    console.print(
        f"  [dim]MT5 fetch[/]  from {from_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"
    )

    try:
        df = client.fetch_candles(
            symbol, timeframe,
            from_date=from_date,
            to_date=now,
        )
        if not df.empty:
            all_dfs.append(df)
            console.print(f"    [green]+{len(df)}[/] [dim]{timeframe.value} candles[/]")
        else:
            console.print(f"    [dim]0 {timeframe.value} candles[/]")
    except Exception as e:
        logger.warning("MT5 fetch failed for %s %s: %s",
                     symbol.value, timeframe.value, e)
        console.print(f"    [red]Error: {e}[/]")

    if not all_dfs:
        return pd.DataFrame()

    # Concatenate and deduplicate
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ── Candle proxy for verification ────────────────────────────────────────


class _CandleProxy:
    """Lightweight candle object that the verifier can use (duck-typed)."""
    __slots__ = ("high", "low", "close")

    def __init__(self, high: float, low: float, close: float):
        self.high = high
        self.low = low
        self.close = close


# ── Main backfill logic ──────────────────────────────────────────────────


# Duplicate detection window (minutes) per trade mode
_DUP_WINDOW = {
    TradeMode.SCALPING: 15,
    TradeMode.DAY_TRADING: 60,
    TradeMode.SWING: 240,
}


def backfill_signals(
    symbol: Symbol,
    mode: TradeMode,
    weeks: int = 4,
    step: int = 1,
    from_date: Optional[datetime] = None,
) -> dict:
    """Replay historical candle data and generate + verify signals.

    Steps through M5 candles, generates confluence signals at each point,
    and immediately verifies them using subsequent candles.

    Args:
        symbol: Trading symbol.
        mode: Trade mode (scalping/day_trading/swing).
        weeks: Weeks of historical data to fetch.
        step: Step size in M5 candles (1 = every 5 min, 12 = every hour).
        from_date: Optional start date for incremental replay. When set, only
            candles at or after this date generate new signals (historical
            data before it is still loaded for indicator calculation).

    Returns:
        Stats dict with counts.
    """
    # ── 1. Fetch historical candles ────────────────────────────────────
    console.print(f"\n[bold cyan]📥 Fetching {weeks} weeks of data for {symbol.value}...[/]")

    # Create MT5 client
    from data.mt5_client import MT5Client
    client = MT5Client()
    if not client.connect():
        console.print("[bold red]✗ MT5 connection failed[/]")
        return {"error": "mt5_failed"}
    console.print("[green]✓ Connected to MT5[/]")

    df_m5 = _fetch_candles_batched(client, symbol, Timeframe.M5, weeks=weeks)
    df_h1 = _fetch_candles_batched(client, symbol, Timeframe.H1, weeks=weeks)
    df_h4 = _fetch_candles_batched(client, symbol, Timeframe.H4, weeks=weeks)

    if df_m5.empty or df_h1.empty or df_h4.empty:
        console.print("[bold red]✗ No candle data available[/]")
        return {"error": "no_data"}

    # Store candles in DB
    nm5 = _store_candles(df_m5, symbol.value, Timeframe.M5.value)
    nh1 = _store_candles(df_h1, symbol.value, Timeframe.H1.value)
    nh4 = _store_candles(df_h4, symbol.value, Timeframe.H4.value)

    console.print(
        f"  [dim]Stored[/]  M5 [green]+{nm5}[/]  H1 [green]+{nh1}[/]  H4 [green]+{nh4}[/]"
    )

    # Reload from DB for clean data
    df_m5 = _load_candles_from_db(symbol.value, Timeframe.M5.value, limit=10000)
    df_h1 = _load_candles_from_db(symbol.value, Timeframe.H1.value, limit=10000)
    df_h4 = _load_candles_from_db(symbol.value, Timeframe.H4.value, limit=10000)

    console.print(
        f"  [dim]Total[/]   M5 {len(df_m5)}  H1 {len(df_h1)}  H4 {len(df_h4)}"
    )

    # ── 2. Check minimum data requirements ──────────────────────────────
    # Signal engine needs at least 200 H1 candles for EMA200 + scoring
    if len(df_h1) < 200:
        console.print(
            f"[bold red]✗ Only {len(df_h1)} H1 candles (need 200+)[/]\n"
            f"[dim]Try increasing --weeks or check API access[/]"
        )
        return {"error": "insufficient_h1"}

    # ── 3. Replay through M5 candles ───────────────────────────────────
    lookahead = SETTINGS.verification.lookahead_candles
    min_lookahead = SETTINGS.verification.min_lookahead_candles
    total_m5 = len(df_m5)

    # Determine the replay start index.
    # Default: start after enough M5 candles for indicators + M5 direction (50+).
    # Incremental mode: skip to the first candle at or after from_date.
    start_idx = 50
    if from_date is not None:
        # DB timestamps are tz-naive (SQLite strips tz), so compare naive.
        cmp_from_date = from_date.replace(tzinfo=None) if from_date.tzinfo else from_date
        for i, ts in enumerate(df_m5["timestamp"]):
            if pd.Timestamp(ts) >= pd.Timestamp(cmp_from_date):
                start_idx = max(start_idx, i)
                break
        console.print(f"  [dim]Incremental replay from[/]  {cmp_from_date} (index {start_idx})")

    # End before the last few M5 candles (need future data for verification)
    end_idx = total_m5 - min_lookahead

    if end_idx <= start_idx:
        console.print(f"[bold yellow]⚠ Not enough new M5 data ({total_m5}) for replay in this window[/]")
        return {"error": "insufficient_m5"}

    dup_window = _DUP_WINDOW.get(mode, 60)

    console.print(
        f"\n[bold]🔄 Replaying {symbol.value} in {mode.value} mode[/]\n"
        f"  [dim]M5 range[/]  {df_m5.iloc[start_idx]['timestamp']} → {df_m5.iloc[end_idx]['timestamp']}\n"
        f"  [dim]Step[/]  every {step} M5 candles ({step * 5} min)\n"
        f"  [dim]Dup window[/]  {dup_window} min\n"
        f"  [dim]Lookahead[/]  {lookahead} candles"
    )

    stats = {
        "checked": 0,
        "signals": 0,
        "wins": 0,
        "losses": 0,
        "breakevens": 0,
        "no_trade": 0,
        "stored": 0,
        "verified": 0,
        "skipped_duplicate": 0,
        "skipped_short_data": 0,
    }

    # Pre-convert M5 data to arrays for faster slicing
    m5_timestamps = df_m5["timestamp"].values
    m5_highs = df_m5["high"].values
    m5_lows = df_m5["low"].values
    m5_closes = df_m5["close"].values

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("• {task.fields[info]}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Replaying...", total=end_idx - start_idx, info="starting"
        )

        with get_session() as session:
            for i in range(start_idx, end_idx, step):
                # Get the M5 timestamp at this point
                m5_ts = pd.Timestamp(m5_timestamps[i])

                # Slice data up to this point (NO future data!)
                df_m5_slice = df_m5.iloc[:i + 1].copy()
                df_h1_slice = df_h1[df_h1["timestamp"] <= m5_ts].copy()
                df_h4_slice = df_h4[df_h4["timestamp"] <= m5_ts].copy()

                # Need enough H1/H4 data for the engine
                if len(df_h1_slice) < 200:
                    progress.update(task, advance=step, info="H1 warming up")
                    continue
                if mode != TradeMode.SCALPING and len(df_h4_slice) < 5:
                    progress.update(task, advance=step, info="H4 warming up")
                    continue

                stats["checked"] += 1

                # ── Generate signal at this historical point ──
                try:
                    trade_signal = generate_signal(
                        df_h4_slice, df_h1_slice, df_m5_slice, symbol, mode, prefix="backtest_"
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

                # ── Duplicate check ──
                dup_cutoff = m5_ts - pd.Timedelta(minutes=dup_window)
                normalized_symbol = normalize_symbol(symbol.value)
                recent_dup = session.query(Signal).filter(
                    Signal.symbol == normalized_symbol,
                    Signal.signal == trade_signal.signal.value,
                    Signal.entry >= trade_signal.entry * 0.9995,
                    Signal.entry <= trade_signal.entry * 1.0005,
                    Signal.timestamp >= dup_cutoff,
                ).first()

                if recent_dup:
                    stats["skipped_duplicate"] += 1
                    progress.update(task, advance=step, info="duplicate")
                    continue

                # ── Store signal ──
                features = trade_signal.features.to_dict()
                db_signal = Signal(
                    signal_id=trade_signal.signal_id,
                    symbol=normalized_symbol,  # Store normalized symbol
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
                session.flush()  # Get the auto-incremented ID
                stats["stored"] += 1
                logger.info(
                    "Stored signal %s: %s %s conf=%.1f entry=%.2f (id=%s)",
                    trade_signal.signal_id, trade_signal.symbol, trade_signal.signal,
                    trade_signal.confidence, trade_signal.entry, db_signal.id,
                )

                # ── Verify signal using future M5 candles ──
                future_start = i + 1
                future_count = total_m5 - future_start

                if future_count < min_lookahead:
                    stats["skipped_short_data"] += 1
                    progress.update(task, advance=step, info="short data")
                    continue

                # Build candle proxies (duck-typed for verify_signal)
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
                    elif result.result.value == "LOSS":
                        stats["losses"] += 1
                    else:
                        stats["breakevens"] += 1

                # Commit every signal to persist immediately
                session.commit()

                direction = trade_signal.signal.value
                emoji = "📈" if direction == "BUY" else "📉"
                progress.update(
                    task, advance=step,
                    info=f"{emoji} {direction} conf={trade_signal.confidence:.0f}%  "
                         f"W{stats['wins']} L{stats['losses']} BE{stats['breakevens']}"
                )

            # Session auto-commits when exiting 'with' block

    # Verify signals were persisted
    normalized_symbol = normalize_symbol(symbol.value)
    with get_session() as check_session:
        db_count = check_session.query(Signal).filter(Signal.symbol == normalized_symbol).count()
        ver_count = check_session.query(SignalVerification).join(Signal).filter(Signal.symbol == normalized_symbol).count()
        logger.info(
            "DB check after backfill: %d signals for %s, %d verifications",
            db_count, normalized_symbol, ver_count,
        )
        console.print(f"  [dim]DB verify:[/] {db_count} signals, {ver_count} verifications for {normalized_symbol}")

    return stats


# ── Auto-detect: symbol staleness & incremental updates ──────────────────


def get_symbol_status(symbol: str) -> dict:
    """Get the current status of a symbol's data and models.

    Returns a dict describing how stale the symbol's signals/models are and
    whether it needs an update.
    """
    with get_session() as session:
        last_signal = session.query(func.max(Signal.timestamp)).filter(
            Signal.symbol == symbol
        ).scalar()

        last_model = session.query(
            func.max(MLModelMeta.trained_at)
        ).filter(
            MLModelMeta.symbol == symbol,
            MLModelMeta.is_active == True
        ).scalar()

        total_signals = session.query(func.count(Signal.id)).filter(
            Signal.symbol == symbol
        ).scalar() or 0

        total_verified = session.query(func.count(SignalVerification.id)).join(
            Signal
        ).filter(Signal.symbol == symbol).scalar() or 0

        has_active_model = session.query(func.count(MLModelMeta.id)).filter(
            MLModelMeta.symbol == symbol,
            MLModelMeta.is_active == True
        ).scalar() > 0

    now = datetime.now(tz=timezone.utc)

    days_since_model = 0
    if last_model:
        if last_model.tzinfo is None:
            last_model = last_model.replace(tzinfo=timezone.utc)
        days_since_model = (now - last_model).days

    available_days = 0
    if last_signal:
        if last_signal.tzinfo is None:
            last_signal = last_signal.replace(tzinfo=timezone.utc)
        available_days = (now - last_signal).days
    else:
        available_days = 999  # No signals yet — need full fetch

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

    statuses.sort(key=lambda x: (-x["needs_update"], -x.get("available_days", 0)))

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
        table.add_row(s["symbol"], str(s["total_signals"]), str(s["total_verified"]),
                      days_text, status_color)

    console.print(table)
    return statuses


def fetch_new_candles_for_symbol(
    client,
    symbol: Symbol,
    last_date: Optional[datetime] = None,
    weeks: int = 4,
) -> dict:
    """Fetch new candles for a symbol since the last update.

    Returns a dict of {timeframe_value: new_candle_count}.
    """
    if last_date:
        from_date = last_date + timedelta(days=1)
    else:
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
            df = client.fetch_candles(symbol, timeframe, from_date=from_date, to_date=now)
            if not df.empty:
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


def reset_symbol_data(symbols: List[Symbol]) -> None:
    """Delete existing signals, verifications, ML models, and model files for symbols."""
    console.print("\n[bold yellow]🗑️  Reset mode: deleting existing signals...[/]")
    with get_session() as session:
        for symbol in symbols:
            signal_ids = [
                row[0] for row in
                session.query(Signal.id).filter(Signal.symbol == symbol.value).all()
            ]
            deleted_verifications = (
                session.query(SignalVerification)
                .filter(SignalVerification.signal_id.in_(signal_ids))
                .delete(synchronize_session=False)
            ) if signal_ids else 0
            deleted_signals = (
                session.query(Signal)
                .filter(Signal.symbol == symbol.value)
                .delete(synchronize_session=False)
            )
            deleted_models = (
                session.query(MLModelMeta)
                .filter(MLModelMeta.symbol == symbol.value)
                .delete(synchronize_session=False)
            )
            # Delete model files on disk (.joblib, _features.json, _calibrated.joblib, _tuned_params.json)
            deleted_files = 0
            model_dir = SETTINGS.ml.model_dir
            if os.path.isdir(model_dir):
                for fname in os.listdir(model_dir):
                    if symbol.value in fname:
                        fpath = os.path.join(model_dir, fname)
                        try:
                            os.remove(fpath)
                            deleted_files += 1
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


def train_if_needed(symbol: str, new_verified_count: int) -> bool:
    """Train ML models via walk-forward if enough new verified signals exist.

    Returns True if training was performed.
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
        from backtesting.walk_forward import run_walk_forward
        from display import display_auto_train_result
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


def run_backfill(
    symbols: list[Symbol] = None,
    mode: TradeMode = None,
    weeks: int = 4,
    step: int = 1,
    reset: bool = False,
) -> None:
    """Run the full backfill process: fetch → replay → verify → train.

    This is the manual/explicit mode: fetches `weeks` of history for every
    symbol and replays the full range, then trains.
    """
    if symbols is None:
        symbols = SETTINGS.symbols
    if mode is None:
        mode = SETTINGS.trade_mode

    console.print(Panel(
        f"[bold]Historical Backfill[/]\n"
        f"  [dim]Mode[/]    {mode.value}\n"
        f"  [dim]Symbols[/]  {', '.join(s.value for s in symbols)}\n"
        f"  [dim]Weeks[/]    {weeks}\n"
        f"  [dim]Step[/]     every {step} M5 candle ({step * 5} min)\n"
        f"  [dim]Timeout[/]  {SETTINGS.get_max_hold_minutes()} min"
        + ("\n  [yellow]⚠ Reset mode — clearing existing signals[/]" if reset else ""),
        border_style="cyan",
    ))

    create_tables()

    if reset:
        reset_symbol_data(list(symbols))

    total_stats = {
        "checked": 0, "signals": 0, "wins": 0, "losses": 0,
        "breakevens": 0, "stored": 0, "verified": 0, "no_trade": 0,
        "skipped_duplicate": 0, "skipped_short_data": 0,
    }

    for symbol in symbols:
        console.print(f"\n[bold cyan]═══ {symbol.value} ═══[/]")
        stats = backfill_signals(symbol, mode, weeks=weeks, step=step)
        if "error" in stats:
            continue
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    _print_backfill_summary(total_stats)

    # ── Train ML if enough data ──
    if total_stats["verified"] >= SETTINGS.walk_forward.min_signals:
        console.print(
            f"\n[bold cyan]🔄 Training ML models on "
            f"{total_stats['verified']} verified signals...[/]"
        )
        for sym in symbols:
            train_if_needed(sym.value, total_stats["verified"])
    else:
        needed = SETTINGS.walk_forward.min_signals
        have = total_stats["verified"]
        console.print(
            f"\n[yellow]⚠ Only {have} verified signals "
            f"(need {needed}) — ML training deferred[/]\n"
            f"[dim]Try increasing --weeks or lowering --step[/]"
        )

    console.print()


def run_auto_backfill(
    symbols: Optional[List[Symbol]] = None,
    force: bool = False,
    mode: TradeMode = None,
    days: Optional[int] = None,
    reset: bool = False,
) -> dict:
    """Auto-detect stale symbols, fetch new data incrementally, backtest, and train.

    Args:
        symbols: Symbols to process (None = all configured symbols).
        force: Force update all symbols regardless of staleness.
        mode: Trade mode (None = from settings).
        days: Number of days to backtest (None = auto-detect new data only).
        reset: Delete existing signals/models before backtesting.

    Returns:
        dict with overall pipeline results.
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

    create_tables()

    if reset:
        reset_symbol_data(symbols)

    # ── Phase 1: Analyze ────────────────────────────────────────────────────────
    statuses = analyze_all_symbols(symbols)

    if force or reset:
        to_process = statuses
    else:
        to_process = [s for s in statuses if s["needs_update"]]

    if not to_process:
        console.print("\n[green]✓ All symbols up to date![/]")
        return {"status": "up_to_date"}

    console.print(f"\n[dim]Processing {len(to_process)} symbols[/]")

    # ── Phase 2: Connect ────────────────────────────────────────────────────────
    from data.mt5_client import MT5Client
    client = MT5Client()
    if not client.connect():
        console.print("[bold red]✗ MT5 connection failed[/]")
        return {"error": "mt5_failed"}

    # ── Phase 3: Process each symbol ────────────────────────────────────────────
    results = {"processed": [], "skipped": [], "errors": []}

    for status in to_process:
        symbol = Symbol(status["symbol"])
        console.print(f"\n{'═' * 50}")
        console.print(f"[bold cyan]═══ {symbol.value} ═══[/]")
        console.print(f"{'═' * 50}")

        try:
            # When days is specified or reset mode, do a full backfill over that range.
            # Otherwise use the incremental path (fetch only new candles, replay only the new window).
            effective_days = days or (28 if reset else None)

            if effective_days is not None:
                weeks = max(1, effective_days // 7)
                console.print(f"\n[bold]🔄 Full backfill for {effective_days} days ({weeks} weeks)[/]")
                signal_results = backfill_signals(symbol, mode, weeks=weeks, step=12)

                if "error" in signal_results:
                    console.print(f"[yellow]⚠ Backfill error: {signal_results.get('error')}[/]")
                    results["errors"].append((symbol.value, signal_results.get('error')))
                    continue

                new_signals = signal_results.get("signals", 0)
                new_verified = signal_results.get("verified", 0)
                new_wins = signal_results.get("wins", 0)
                new_losses = signal_results.get("losses", 0)
                new_be = signal_results.get("breakevens", 0)
                total_new_candles = "N/A"
            else:
                # Incremental: fetch only new candles, then replay only the new window.
                candle_results = fetch_new_candles_for_symbol(
                    client, symbol, status["last_signal_date"], weeks=4,
                )
                total_new_candles = sum(candle_results.values())

                if total_new_candles == 0:
                    console.print("[dim]No new candles — skipping[/]")
                    results["skipped"].append(symbol.value)
                    continue

                from_date = status["last_signal_date"] or (
                    datetime.now(tz=timezone.utc) - timedelta(days=7)
                )
                # Use a short weeks window so backfill_signals loads enough history
                # for indicators, but only replays from from_date onward.
                signal_results = backfill_signals(
                    symbol, mode, weeks=4, step=12, from_date=from_date,
                )

                if "error" in signal_results:
                    console.print(f"[yellow]⚠ Signal generation error: {signal_results['error']}[/]")
                    results["errors"].append((symbol.value, signal_results['error']))
                    continue

                new_signals = signal_results.get("signals", 0)
                new_verified = signal_results.get("verified", 0)
                new_wins = signal_results.get("wins", 0)
                new_losses = signal_results.get("losses", 0)
                new_be = signal_results.get("breakevens", 0)

            console.print(
                f"\n[dim]Signals[/]  {new_signals} generated, {new_verified} verified\n"
                f"  [green]Wins[/]  {new_wins}  [red]Losses[/]  {new_losses}  "
                f"[yellow]BE[/]  {new_be}"
            )

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


def run_train_only(symbols: List[Symbol], mode: TradeMode = None) -> None:
    """Train ML models from existing verified signals (no backfill)."""
    if mode is None:
        mode = SETTINGS.trade_mode

    console.print(Panel(
        f"[bold]ML Training Only[/]\n"
        f"  [dim]Mode[/]    {mode.value}\n"
        f"  [dim]Symbols[/]  {', '.join(s.value for s in symbols)}",
        border_style="cyan",
    ))

    create_tables()

    for symbol in symbols:
        status = get_symbol_status(symbol.value)
        console.print(
            f"\n[dim]{symbol.value}[/]  {status['total_verified']} verified signals, "
            f"model active: {status['has_active_model']}"
        )
        train_if_needed(symbol.value, status["total_verified"])

    console.print()


def _print_backfill_summary(total_stats: dict) -> None:
    """Print the backfill summary panel."""
    wins = total_stats["wins"]
    losses = total_stats["losses"]
    be = total_stats["breakevens"]
    total_resolved = wins + losses + be
    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0

    border = "green" if win_rate > 40 else "yellow" if win_rate > 25 else "red"

    console.print()
    console.print(Panel(
        f"  [dim]Candles checked[/]      {total_stats['checked']}\n"
        f"  [dim]Signals generated[/]    {total_stats['signals']}\n"
        f"  [dim]Duplicates skipped[/]  {total_stats['skipped_duplicate']}\n"
        f"  [dim]Short data skipped[/]   {total_stats['skipped_short_data']}\n"
        f"  [dim]Verified[/]             {total_stats['verified']}\n"
        f"  [dim]No-trade[/]             {total_stats['no_trade']}\n"
        f"\n"
        f"  [green]Wins[/]  {wins}  [red]Losses[/]  {losses}  [yellow]Breakeven[/]  {be}\n"
        f"  [bold]Win rate[/]  {win_rate:.1f}%",
        title="[bold]Backfill Summary[/]",
        border_style=border,
    ))


def main() -> None:
    """CLI entry point — single backtest & ML training command.

    Modes:
      default       Auto-detect stale symbols, incrementally backtest & train.
      --weeks N     Full backfill of N weeks for all configured symbols, then train.
      --train        Train ML only from existing verified signals (no backfill).
      --reset        Delete existing signals/models first (use with --weeks or default).
    """
    parser = argparse.ArgumentParser(
        description="Backtest & ML training: backfill signals, verify, and train models"
    )
    parser.add_argument(
        "--weeks", type=int, default=None,
        help="Weeks of history to backfill (default: auto-detect new data only)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Days of history to backfill (alternative to --weeks; e.g. 30)"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Step size in M5 candles: 1=5min, 12=1hr (default: 1, auto mode uses 12)"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["scalping", "day_trading", "swing"],
        help="Trade mode (default: from .env TRADE_MODE)"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        choices=["XAUUSD"],
        help="Single symbol to process (default: from .env SYMBOLS)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Multiple symbols to process (default: all configured symbols)"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-detect stale symbols and incrementally update (default behavior)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force update all symbols regardless of staleness (use with --auto)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing signals, verifications, and ML models before backfilling"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train ML models only from existing verified signals (no backfill)"
    )
    args = parser.parse_args()

    mode = TradeMode(args.mode) if args.mode else SETTINGS.trade_mode

    if args.symbol:
        symbols = [Symbol(args.symbol)]
    elif args.symbols:
        symbols = [Symbol(s) for s in args.symbols]
    else:
        symbols = list(SETTINGS.symbols)

    # --train: ML training only, no backfill
    if args.train:
        run_train_only(symbols, mode=mode)
        return

    # --weeks or --days: explicit full backfill over that range
    if args.weeks is not None or args.days is not None:
        weeks = args.weeks if args.weeks is not None else max(1, args.days // 7)
        run_backfill(symbols=symbols, mode=mode, weeks=weeks, step=args.step, reset=args.reset)
        return

    # Default & --auto: auto-detect stale symbols and incrementally update
    run_auto_backfill(
        symbols=symbols,
        force=args.force,
        mode=mode,
        days=args.days,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()

"""Backfill: replay historical candles to generate and verify signals for ML training.

Instead of waiting days for 30+ verified signals, this script:
1. Fetches weeks of historical candle data in batches
2. Replays it through the signal engine at each M5 checkpoint
3. Verifies signals immediately using known future candles
4. Trains ML models on the accumulated verified data

No lookahead bias: signals are generated using only data available at that
point in time, and verified using only future candles.

Usage:
    python3 -m trading_ai.backtest_historical
    python3 -m trading_ai.backtest_historical --weeks 4 --step 1
    python3 -m trading_ai.backtest_historical --mode scalping --symbol XAUUSD
"""

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from config.settings import SETTINGS, Symbol, Timeframe, TradeMode, TradingPlatform
from data.collector import CapitalComClient
from data.database import create_tables, get_session
from data.models import Candle, Signal, SignalVerification
from signals.engine import generate_signal, TradeSignal
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

# ── Candle storage / loading (same as main.py) ──────────────────────────


def _store_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """Store fetched candles into the database. Returns count of new candles."""
    count = 0

    with get_session() as session:
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = pd.Timestamp(ts, tz="UTC")

            exists = (
                session.query(Candle)
                .filter(
                    Candle.symbol == symbol,
                    Candle.timeframe == timeframe,
                    Candle.timestamp == ts,
                )
                .first()
            )
            if exists:
                continue

            candle = Candle(
                symbol=symbol,
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

    Supports both Capital.com and MT5 clients.
    - Capital.com: API returns max 1000 candles, paginate with from_date
    - MT5: Can fetch larger ranges, uses copy_rates_range

    For M5: 12 candles/hour, 288/day, 2016/week
    For H1: 1 candle/hour, 24/day, 168/week
    For H4: 6 candles/day, 42/week

    Args:
        client: Authenticated client (CapitalComClient or MT5Client).
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

    # Check if using MT5 client
    is_mt5 = hasattr(client, '_connected') or hasattr(client, '_mt5')

    if is_mt5:
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
    else:
        # Capital.com: batch fetching with 1000 candle limit
        from_date = None
        batch = 0

        while len(all_dfs) == 0 or sum(len(df) for df in all_dfs) < target_candles:
            batch += 1

            if from_date:
                console.print(
                    f"  [dim]Batch {batch}[/]  "
                    f"[dim]from {from_date[:10]}[/]", highlight=False
                )
            else:
                console.print(
                    f"  [dim]Batch {batch}[/]  "
                    f"[dim]latest[/]", highlight=False
                )

            try:
                df = client.fetch_candles(
                    symbol, timeframe,
                    limit=1000,
                    from_date=from_date,
                )
                if not df.empty:
                    all_dfs.append(df)
                    console.print(f"    [green]+{len(df)}[/] [dim]{timeframe.value} candles[/]")

                    # Next batch starts from the earliest candle we have
                    earliest_ts = df["timestamp"].min()
                    from_date = earliest_ts.strftime("%Y-%m-%dT%H:%M:%S")

                    # If we got fewer than 1000, we've reached the API's data limit
                    if len(df) < 1000:
                        console.print(f"    [dim]API data limit reached[/]")
                        break
                else:
                    console.print(f"    [dim]0 {timeframe.value} candles — stopping[/]")
                    break
            except Exception as e:
                logger.warning("Batch fetch failed for %s %s: %s",
                             symbol.value, timeframe.value, e)
                console.print(f"    [red]Error: {e}[/]")
                break

            # Safety: max 10 batches to avoid infinite loops
            if batch >= 10:
                console.print(f"    [dim]Max batches reached[/]")
                break

            # Small delay to avoid rate limiting
            time.sleep(0.5)

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
) -> dict:
    """Replay historical candle data and generate + verify signals.

    Steps through M5 candles, generates confluence signals at each point,
    and immediately verifies them using subsequent candles.

    Args:
        symbol: Trading symbol.
        mode: Trade mode (scalping/day_trading/swing).
        weeks: Weeks of historical data to fetch.
        step: Step size in M5 candles (1 = every 5 min, 12 = every hour).

    Returns:
        Stats dict with counts.
    """
    # ── 1. Fetch historical candles ────────────────────────────────────
    console.print(f"\n[bold cyan]📥 Fetching {weeks} weeks of data for {symbol.value}...[/]")

    # Create appropriate client based on platform
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        from data.mt5_client import MT5Client
        client = MT5Client()
        if not client.connect():
            console.print("[bold red]✗ MT5 connection failed[/]")
            return {"error": "mt5_failed"}
        console.print("[green]✓ Connected to MT5[/]")
    else:
        client = CapitalComClient()
        if not client.authenticate():
            console.print("[bold red]✗ Authentication failed[/]")
            return {"error": "auth_failed"}
        console.print("[green]✓ Connected to Capital.com[/]")

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

    # Start after enough M5 candles for indicators + M5 direction (50+)
    start_idx = 50
    # End before the last few M5 candles (need future data for verification)
    end_idx = total_m5 - min_lookahead

    if end_idx <= start_idx:
        console.print(f"[bold red]✗ Not enough M5 data ({total_m5}) for replay[/]")
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

                # ── Store signal ──
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
    with get_session() as check_session:
        db_count = check_session.query(Signal).filter(Signal.symbol == symbol.value).count()
        ver_count = check_session.query(SignalVerification).join(Signal).filter(Signal.symbol == symbol.value).count()
        logger.info(
            "DB check after backfill: %d signals for %s, %d verifications",
            db_count, symbol.value, ver_count,
        )
        console.print(f"  [dim]DB verify:[/] {db_count} signals, {ver_count} verifications for {symbol.value}")

    return stats


def run_backfill(
    symbols: list[Symbol] = None,
    mode: TradeMode = None,
    weeks: int = 4,
    step: int = 1,
    reset: bool = False,
) -> None:
    """Run the full backfill process: fetch → replay → verify → train."""
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

    # ── Optionally reset signals/verifications ──
    if reset:
        with get_session() as session:
            for sym in symbols:
                # Delete verifications for this symbol's signals
                sig_ids = [s.id for s in session.query(Signal).filter(Signal.symbol == sym.value).all()]
                if sig_ids:
                    session.query(SignalVerification).filter(
                        SignalVerification.signal_id.in_(sig_ids)
                    ).delete(synchronize_session=False)
                    session.query(Signal).filter(Signal.symbol == sym.value).delete(synchronize_session=False)
                    console.print(f"  [yellow]Cleared signals & verifications for {sym.value}[/]")
                # Also delete ML model meta for this symbol
                from data.models import MLModelMeta
                session.query(MLModelMeta).filter(MLModelMeta.symbol == sym.value).delete(synchronize_session=False)
        console.print()

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

    # ── Summary ──
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

    # ── Train ML if enough data ──
    if total_stats["verified"] >= SETTINGS.walk_forward.min_signals:
        console.print(
            f"\n[bold cyan]🔄 Training ML models on "
            f"{total_stats['verified']} verified signals...[/]"
        )
        from backtesting.walk_forward import run_walk_forward
        from display import display_auto_train_result

        for sym in symbols:
            try:
                reports = run_walk_forward(sym.value, timeframe=None)
                for report in reports:
                    display_auto_train_result(
                        report.model_type, report.symbol,
                        report.mean_accuracy, report.mean_roc_auc,
                    )
            except Exception as e:
                console.print(f"  [red]ML training failed for {sym.value}: {e}[/]")
                logger.error("ML training failed for %s: %s", sym.value, e)

        console.print("[bold green]✓ ML models trained![/]")
    else:
        needed = SETTINGS.walk_forward.min_signals
        have = total_stats["verified"]
        console.print(
            f"\n[yellow]⚠ Only {have} verified signals "
            f"(need {needed}) — ML training deferred[/]\n"
            f"[dim]Try increasing --weeks or lowering --step[/]"
        )

    console.print()


def main() -> None:
    """CLI entry point for backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill: generate and verify signals from historical data"
    )
    parser.add_argument(
        "--weeks", type=int, default=4,
        help="Weeks of historical data to fetch (default: 4)"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Step size in M5 candles: 1=5min, 12=1hr (default: 1)"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["scalping", "day_trading", "swing"],
        help="Trade mode (default: from .env TRADE_MODE)"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        choices=["XAUUSD", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "XAGUSD", "USDJPY", "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "AUDUSD", "NZDUSD", "USDCAD"],
        help="Symbol (default: from .env SYMBOLS)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear existing signals and ML models before backfilling"
    )
    args = parser.parse_args()

    mode = TradeMode(args.mode) if args.mode else SETTINGS.trade_mode
    symbols = [Symbol(args.symbol)] if args.symbol else list(SETTINGS.symbols)

    run_backfill(symbols=symbols, mode=mode, weeks=args.weeks, step=args.step, reset=args.reset)


if __name__ == "__main__":
    main()
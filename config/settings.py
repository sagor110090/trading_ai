"""Central configuration for the trading prediction system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class Symbol(str, Enum):
    # Crypto
    BTCUSD = "BTCUSD"
    ETHUSD = "ETHUSD"
    SOLUSD = "SOLUSD"
    LTCUSD = "LTCUSD"
    BCHUSD = "BCHUSD"
    XRPUSD = "XRPUSD"
    BTCEUR = "BTCEUR"
    ETHGBP = "ETHGBP"

    # Metals
    XAUUSD = "XAUUSD"
    XAGUSD = "XAGUSD"

    # Indices
    US30 = "US30"
    US500 = "US500"
    NAS100 = "NAS100"

    # Forex Majors
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDJPY = "USDJPY"
    USDCHF = "USDCHF"

    # Forex Minors
    EURGBP = "EURGBP"
    EURJPY = "EURJPY"
    GBPJPY = "GBPJPY"
    AUDUSD = "AUDUSD"
    NZDUSD = "NZDUSD"
    USDCAD = "USDCAD"
    EURAUD = "EURAUD"
    GBPCHF = "GBPCHF"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class SignalResult(str, Enum):
    PENDING = "PENDING"
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


class Session(str, Enum):
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"


class TradeMode(str, Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING = "swing"


class TradingPlatform(str, Enum):
    CAPITAL = "CAPITAL"
    MT5 = "MT5"


@dataclass(frozen=True)
class FastMarketConfig:
    """Configuration for fast market cooldown extension.

    When ATR (Average True Range) exceeds a threshold relative to its average,
    the market is considered 'fast' (high volatility). In fast markets, cooldowns
    are multiplied to reduce overtrading during chaotic conditions.

    ATR ratio = current ATR / average ATR over the lookback window.
    """
    # ATR ratio threshold above which the market is considered "fast"
    # 1.5 = current ATR is 50% above its recent average
    atr_threshold: float = 1.5
    # Multiplier applied to cooldowns when market is fast
    # 2.0 = cooldowns last twice as long
    cooldown_multiplier: float = 2.0


@dataclass(frozen=True)
class CapitalComConfig:
    api_key: str = ""
    identifier: str = ""
    password: str = ""
    use_demo: bool = True
    base_url: str = "https://api-capital.backend-capital.com"
    demo_url: str = "https://demo-api-capital.backend-capital.com"

    @property
    def active_url(self) -> str:
        return self.demo_url if self.use_demo else self.base_url


@dataclass(frozen=True)
class MT5Config:
    login: int = 0
    password: str = ""
    server: str = ""
    timeout: int = 60000  # Connection timeout in milliseconds


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = "localhost"
    port: int = 3307
    database: str = "trading_ai"
    username: str = "root"
    password: str = "root"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass(frozen=True)
class SignalWeights:
    ema_trend: float = 30.0
    rsi: float = 15.0
    macd: float = 20.0
    atr: float = 10.0
    structure: float = 15.0
    momentum: float = 10.0
    total: float = 100.0


@dataclass(frozen=True)
class SignalThresholds:
    buy_threshold: float = 40.0
    sell_threshold: float = 40.0
    confidence_min: float = 0.0
    confidence_max: float = 100.0


@dataclass(frozen=True)
class ScalpingConfig:
    """Scalping: fast in, fast out. M5 entry, H1 trend only."""
    sl_atr_mult: float = 0.5
    tp_atr_mult: float = 0.8
    confidence_threshold: float = 20.0  # Lowered for more signals
    trend_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    confluence_label: str = "scalping"
    # Max time a scalping trade stays open (minutes)
    max_hold_minutes: int = 30


@dataclass(frozen=True)
class DayTradingConfig:
    """Day trading: close same day. M5 entry, H1 trend, H4 big picture."""
    sl_atr_mult: float = 0.7
    tp_atr_mult: float = 1.2
    confidence_threshold: float = 20.0  # Lowered from 35% for more signals
    trend_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    big_picture_timeframe: str = "4h"
    confluence_label: str = "day_trading"
    # Max time a day trade stays open (minutes)
    max_hold_minutes: int = 240


@dataclass(frozen=True)
class SwingConfig:
    """Swing: hold hours to days. H4+H1 must agree, M5 entry trigger."""
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    confidence_threshold: float = 25.0  # Lowered from 40% for more swing trades
    trend_timeframe: str = "4h"
    confirmation_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    confluence_label: str = "swing"
    # Max time a swing trade stays open (minutes)
    max_hold_minutes: int = 1440


@dataclass(frozen=True)
class MLConfig:
    xgboost: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "eval_metric": "logloss",
    })
    lightgbm: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
    })
    catboost: dict = field(default_factory=lambda: {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "eval_metric": "Logloss",
        "verbose": 0,
        # Auto class weights to handle imbalance (70% losses, 30% wins)
        "auto_class_weights": "Balanced",
    })
    model_dir: str = ""  # Set dynamically in load_settings() to absolute path
    # ML prediction thresholds per symbol (read from .env)
    # Default threshold and per-symbol overrides
    prediction_thresholds: dict = field(default_factory=lambda: {"DEFAULT": 0.3})
    # Minimum verified signals before ML predictions are used for filtering
    # Below this count, ML predictions are logged but don't block trades
    min_verified_for_filter: int = 50
    # Auto-train models during live trading (recommended: false - train via backtest instead)
    auto_train: bool = False
    # Drift detection settings
    drift_detection_enabled: bool = True
    min_signals_for_drift: int = 20
    drift_threshold: float = 0.15


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 6
    test_months: int = 1
    min_signals: int = 20


@dataclass(frozen=True)
class VerificationConfig:
    # Max candles to check for SL/TP hit before declaring BREAKEVEN
    # If SL/TP is hit within fewer candles, it's verified immediately.
    # Dynamic calculation: max_hold_minutes / 5 (for M5 candles)
    # - Scalping (30 min) → 6 candles
    # - Day trading (120 min) → 24 candles
    # - Swing (480 min) → 96 candles
    # This value is used as fallback when dynamic calculation is not available.
    lookahead_candles: int = 100
    # Minimum candles before we try to verify. If fewer candles exist after
    # the signal, we skip it (not enough data yet). Set low so early
    # SL/TP hits get caught quickly.
    min_lookahead_candles: int = 5
    breakeven_threshold_pips: float = 1.0
    # Enable dynamic lookahead based on trade mode (recommended)
    dynamic_lookahead: bool = True


@dataclass(frozen=True)
class WeeklyCloseConfig:
    """Configuration for weekly market close handling.

    Uses Bangladesh Time (Asia/Dhaka UTC+6) for Exness market hours:
    - Forex (EURUSD, GBPUSD, etc.): Close Saturday 02:59 AM BST, Open Monday 03:05 AM BST
    - Gold (XAUUSD): Close Saturday 02:58 AM BST, Open Monday 04:05 AM BST

    Time values are in decimal hours: 0.5 = 30 minutes, 2.0 = 2 hours.

    Note: close_time_utc and close_day are kept for config compatibility but
    are no longer used (actual close times are hardcoded in weekly_close.py).
    """
    # Weekly close time (UTC) - DEPRECATED, now uses Bangladesh time in weekly_close.py
    close_time_utc: str = "22:00"
    # Day of week (4 = Friday, where Monday=0, Sunday=6) - DEPRECATED
    close_day: int = 4
    # Hours before close to stop new trades
    stop_new_trades_hours: float = 2.0
    # Hours before close to close existing positions
    close_positions_hours: float = 0.5
    # Enable/disable weekly close handling
    enabled: bool = True


@dataclass(frozen=True)
class Settings:
    capital_com: CapitalComConfig = field(default_factory=CapitalComConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    trading_platform: TradingPlatform = TradingPlatform.CAPITAL
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    signal_weights: SignalWeights = field(default_factory=SignalWeights)
    signal_thresholds: SignalThresholds = field(default_factory=SignalThresholds)
    scalping: ScalpingConfig = field(default_factory=ScalpingConfig)
    day_trading: DayTradingConfig = field(default_factory=DayTradingConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    weekly_close: WeeklyCloseConfig = field(default_factory=WeeklyCloseConfig)
    fast_market: FastMarketConfig = field(default_factory=FastMarketConfig)
    # Max positions per correlation group (prevents overexposure)
    max_positions_per_group: dict = field(default_factory=lambda: {
        "CRYPTO": 1,
        "METALS": 1,
        "FOREX_MAJOR": 2,
        "FOREX_MINOR": 2,
        "INDICES": 1,
    })
    trade_mode: TradeMode = TradeMode.SWING
    auto_trade: bool = False
    trade_sizes: dict = field(default_factory=lambda: {"DEFAULT": 0.01})
    broker_spread_pips: float = 0.0  # Broker execution speed/spread buffer in pips
    # Per-symbol SL/TP multipliers (fallback to trade mode defaults)
    sl_multipliers: dict = field(default_factory=lambda: {})
    tp_multipliers: dict = field(default_factory=lambda: {})

    def get_trade_size(self, symbol: str) -> float:
        """Get position size for a symbol. Falls back to DEFAULT."""
        return self.trade_sizes.get(symbol, self.trade_sizes.get("DEFAULT", 0.01))

    def get_sl_tp_multipliers(self, symbol: str) -> tuple[float, float]:
        """Get SL and TP multipliers for a symbol.

        Returns:
            (sl_multiplier, tp_multiplier)

        Falls back to trade mode defaults if not set per-symbol.
        """
        sl_mult = self.sl_multipliers.get(symbol)
        tp_mult = self.tp_multipliers.get(symbol)

        if sl_mult is None or tp_mult is None:
            # Use trade mode defaults
            if self.trade_mode == TradeMode.SCALPING:
                default_sl = self.scalping.sl_atr_mult
                default_tp = self.scalping.tp_atr_mult
            elif self.trade_mode == TradeMode.DAY_TRADING:
                default_sl = self.day_trading.sl_atr_mult
                default_tp = self.day_trading.tp_atr_mult
            else:  # SWING
                default_sl = self.swing.sl_atr_mult
                default_tp = self.swing.tp_atr_mult

            return (sl_mult if sl_mult is not None else default_sl,
                    tp_mult if tp_mult is not None else default_tp)

        return sl_mult, tp_mult

    def get_prediction_threshold(self, symbol: str) -> float:
        """Get ML prediction threshold for a symbol. Falls back to DEFAULT."""
        return self.ml.prediction_thresholds.get(symbol, self.ml.prediction_thresholds.get("DEFAULT", 0.3))

    def get_max_hold_minutes(self) -> int:
        """Get max hold time in minutes for the current trade mode."""
        if self.trade_mode == TradeMode.SCALPING:
            return self.scalping.max_hold_minutes
        elif self.trade_mode == TradeMode.DAY_TRADING:
            return self.day_trading.max_hold_minutes
        else:
            return self.swing.max_hold_minutes

    def get_dynamic_lookahead(self) -> int:
        """Get dynamic verification lookahead based on trade mode.

        Uses M5 candles for verification (finest granularity).
        Formula: max_hold_minutes / 5 (for 5-minute candles)

        Examples:
        - Scalping (30 min) → 6 candles
        - Day trading (120 min) → 24 candles
        - Swing (480 min) → 96 candles

        Returns:
            Number of M5 candles to check for SL/TP hit.
        """
        if not self.verification.dynamic_lookahead:
            return self.verification.lookahead_candles

        max_hold = self.get_max_hold_minutes()
        # Divide by 5 for M5 candles, minimum of 5
        dynamic_candles = max(max_hold // 5, self.verification.min_lookahead_candles)

        # Cap at reasonable maximum (swing should still be reasonable)
        return min(dynamic_candles, 200)

    symbols: tuple[Symbol, ...] = (Symbol.BTCUSD,)
    timeframes: tuple[Timeframe, ...] = (Timeframe.M5, Timeframe.H1, Timeframe.H4)
    position_monitor_interval: int = 30  # Seconds between position monitor checks
    log_level: str = "INFO"


def load_settings() -> Settings:
    """Load settings from .env file and environment variables."""
    import os
    from dotenv import load_dotenv

    # Find .env file: check project root, then current dir
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env_path = os.path.join(_project_root, ".env")
    if not os.path.exists(_env_path):
        _env_path = os.path.join(os.getcwd(), ".env")

    load_dotenv(_env_path, override=False)

    # Database
    db_cfg = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "trading_ai"),
        username=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
    )

    # Capital.com
    cap_cfg = CapitalComConfig(
        api_key=os.getenv("CAPITAL_API_KEY", ""),
        identifier=os.getenv("CAPITAL_IDENTIFIER", ""),
        password=os.getenv("CAPITAL_PASSWORD", ""),
        use_demo=os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true",
        base_url=os.getenv("CAPITAL_BASE_URL", CapitalComConfig.base_url),
        demo_url=os.getenv("CAPITAL_DEMO_URL", CapitalComConfig.demo_url),
    )

    # MT5
    mt5_cfg = MT5Config(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
    )

    # Trading platform
    platform_str = os.getenv("TRADING_PLATFORM", "CAPITAL").upper()
    platform_map = {"CAPITAL": TradingPlatform.CAPITAL, "MT5": TradingPlatform.MT5}
    trading_platform = platform_map.get(platform_str, TradingPlatform.CAPITAL)

    # Trade mode
    mode_str = os.getenv("TRADE_MODE", "swing").lower()
    mode_map = {"scalping": TradeMode.SCALPING, "day_trading": TradeMode.DAY_TRADING, "swing": TradeMode.SWING}
    trade_mode = mode_map.get(mode_str, TradeMode.SWING)

    # Symbols
    symbols_str = os.getenv("SYMBOLS", "BTCUSD").upper()
    symbol_map = {s.value: s for s in Symbol}
    symbols = tuple(symbol_map[s.strip()] for s in symbols_str.split(",") if s.strip() in symbol_map)
    if not symbols:
        symbols = (Symbol.BTCUSD,)

    auto_trade = os.getenv("AUTO_TRADE", "false").lower() == "true"

    # Read all TRADE_SIZE_<SYMBOL> from env
    trade_sizes = {"DEFAULT": float(os.getenv("TRADE_SIZE_DEFAULT", "0.01"))}
    for key, value in os.environ.items():
        if key.startswith("TRADE_SIZE_") and key != "TRADE_SIZE_DEFAULT":
            symbol_name = key.replace("TRADE_SIZE_", "")
            try:
                trade_sizes[symbol_name] = float(value)
            except ValueError:
                pass

    # Read all SL_MULTIPLIER_<SYMBOL> and TP_MULTIPLIER_<SYMBOL> from env
    sl_multipliers = {}
    tp_multipliers = {}
    for key, value in os.environ.items():
        if key.startswith("SL_MULTIPLIER_"):
            symbol_name = key.replace("SL_MULTIPLIER_", "")
            try:
                sl_multipliers[symbol_name] = float(value)
            except ValueError:
                pass
        elif key.startswith("TP_MULTIPLIER_"):
            symbol_name = key.replace("TP_MULTIPLIER_", "")
            try:
                tp_multipliers[symbol_name] = float(value)
            except ValueError:
                pass

    # Model directory — absolute path so it works regardless of CWD
    model_dir = os.path.join(_project_root, "trading_ai", "ml", "models")
    auto_train = os.getenv("ML_AUTO_TRAIN", "false").lower() == "true"

    # Drift detection settings
    drift_detection_enabled = os.getenv("ML_DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    min_signals_for_drift = int(os.getenv("ML_MIN_SIGNALS_FOR_DRIFT", "20"))
    drift_threshold = float(os.getenv("ML_DRIFT_THRESHOLD", "0.15"))

    # Weekly close settings
    weekly_close_enabled = os.getenv("WEEKLY_CLOSE_ENABLED", "true").lower() == "true"
    weekly_close_stop_new_trades = float(os.getenv("WEEKLY_CLOSE_STOP_NEW_TRADES_HOURS", "2.0"))
    weekly_close_close_positions = float(os.getenv("WEEKLY_CLOSE_CLOSE_POSITIONS_HOURS", "0.5"))

    # Verification settings
    dynamic_lookahead = os.getenv("DYNAMIC_LOOKAHEAD", "true").lower() == "true"

    # Fast market cooldown settings
    fast_market_atr_threshold = float(os.getenv("FAST_MARKET_ATR_THRESHOLD", "1.5"))
    fast_market_cooldown_multiplier = float(os.getenv("FAST_MARKET_COOLDOWN_MULTIPLIER", "2.0"))

    # Correlation limit settings
    max_positions_per_group = {
        "CRYPTO": int(os.getenv("MAX_POSITIONS_CRYPTO", "1")),
        "METALS": int(os.getenv("MAX_POSITIONS_METALS", "1")),
        "FOREX_MAJOR": int(os.getenv("MAX_POSITIONS_FOREX_MAJOR", "2")),
        "FOREX_MINOR": int(os.getenv("MAX_POSITIONS_FOREX_MINOR", "2")),
        "INDICES": int(os.getenv("MAX_POSITIONS_INDICES", "1")),
    }

    # Read all ML_PREDICTION_THRESHOLD_<SYMBOL> from env
    prediction_thresholds = {"DEFAULT": float(os.getenv("ML_PREDICTION_THRESHOLD", "0.3"))}
    for key, value in os.environ.items():
        if key.startswith("ML_PREDICTION_THRESHOLD_") and key != "ML_PREDICTION_THRESHOLD_DEFAULT":
            symbol_name = key.replace("ML_PREDICTION_THRESHOLD_", "")
            try:
                prediction_thresholds[symbol_name] = float(value)
            except ValueError:
                pass

    return Settings(
        capital_com=cap_cfg,
        mt5=mt5_cfg,
        trading_platform=trading_platform,
        database=db_cfg,
        ml=MLConfig(
            model_dir=model_dir,
            prediction_thresholds=prediction_thresholds,
            auto_train=auto_train,
            drift_detection_enabled=drift_detection_enabled,
            min_signals_for_drift=min_signals_for_drift,
            drift_threshold=drift_threshold,
        ),
        weekly_close=WeeklyCloseConfig(
            enabled=weekly_close_enabled,
            stop_new_trades_hours=weekly_close_stop_new_trades,
            close_positions_hours=weekly_close_close_positions,
        ),
        verification=VerificationConfig(dynamic_lookahead=dynamic_lookahead),
        fast_market=FastMarketConfig(
            atr_threshold=fast_market_atr_threshold,
            cooldown_multiplier=fast_market_cooldown_multiplier,
        ),
        max_positions_per_group=max_positions_per_group,
        trade_mode=trade_mode,
        auto_trade=auto_trade,
        trade_sizes=trade_sizes,
        sl_multipliers=sl_multipliers,
        tp_multipliers=tp_multipliers,
        symbols=symbols,
        position_monitor_interval=int(os.getenv("POSITION_MONITOR_INTERVAL", "30")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        broker_spread_pips=float(os.getenv("BROKER_SPREAD_PIPS", "0.0")),
    )


SETTINGS: Final = load_settings()

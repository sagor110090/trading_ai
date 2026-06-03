"""Central configuration for the trading prediction system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class Symbol(str, Enum):
    # Crypto
    BTCUSD = "BTCUSD"
    ETHUSD = "ETHUSD"
    LTCUSD = "LTCUSD"
    BCHUSD = "BCHUSD"
    XRPUSD = "XRPUSD"
    BTCEUR = "BTCEUR"
    ETHGBP = "ETHGBP"

    # Metals
    XAUUSD = "XAUUSD"
    XAGUSD = "XAGUSD"

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
    confidence_threshold: float = 30.0
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
    confidence_threshold: float = 35.0
    trend_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    big_picture_timeframe: str = "4h"
    confluence_label: str = "day_trading"
    # Max time a day trade stays open (minutes)
    max_hold_minutes: int = 120


@dataclass(frozen=True)
class SwingConfig:
    """Swing: hold hours to days. H4+H1 must agree, M5 entry trigger."""
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    confidence_threshold: float = 40.0
    trend_timeframe: str = "4h"
    confirmation_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    confluence_label: str = "swing"
    # Max time a swing trade stays open (minutes)
    max_hold_minutes: int = 480


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
    # ML prediction threshold: only take trades where ML probability >= this value
    # 0.0 = take all trades (ML disabled), 0.3 = moderate filter, 0.5 = strict
    prediction_threshold: float = 0.3
    # Minimum verified signals before ML predictions are used for filtering
    # Below this count, ML predictions are logged but don't block trades
    min_verified_for_filter: int = 50
    # Auto-train models during live trading (recommended: false - train via backtest instead)
    auto_train: bool = False


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 6
    test_months: int = 1
    min_signals: int = 20


@dataclass(frozen=True)
class VerificationConfig:
    # Max candles to check for SL/TP hit before declaring BREAKEVEN
    # Scalping: trades resolve in 30-60min → 24 candles
    # Day trading: hours → 100 candles (8+ hours)
    # Swing: days → 200 candles
    # If SL/TP is hit within fewer candles, it's verified immediately.
    lookahead_candles: int = 100
    # Minimum candles before we try to verify. If fewer candles exist after
    # the signal, we skip it (not enough data yet). Set low so early
    # SL/TP hits get caught quickly.
    min_lookahead_candles: int = 5
    breakeven_threshold_pips: float = 1.0


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
    trade_mode: TradeMode = TradeMode.SWING
    auto_trade: bool = False
    trade_sizes: dict = field(default_factory=lambda: {"DEFAULT": 0.01})

    def get_trade_size(self, symbol: str) -> float:
        """Get position size for a symbol. Falls back to DEFAULT."""
        return self.trade_sizes.get(symbol, self.trade_sizes.get("DEFAULT", 0.01))

    def get_max_hold_minutes(self) -> int:
        """Get max hold time in minutes for the current trade mode."""
        if self.trade_mode == TradeMode.SCALPING:
            return self.scalping.max_hold_minutes
        elif self.trade_mode == TradeMode.DAY_TRADING:
            return self.day_trading.max_hold_minutes
        else:
            return self.swing.max_hold_minutes

    symbols: tuple[Symbol, ...] = (Symbol.BTCUSD,)
    timeframes: tuple[Timeframe, ...] = (Timeframe.M5, Timeframe.H1, Timeframe.H4)
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

    # Model directory — absolute path so it works regardless of CWD
    model_dir = os.path.join(_project_root, "trading_ai", "ml", "models")
    prediction_threshold = float(os.getenv("ML_PREDICTION_THRESHOLD", "0.3"))
    auto_train = os.getenv("ML_AUTO_TRAIN", "false").lower() == "true"

    return Settings(
        capital_com=cap_cfg,
        mt5=mt5_cfg,
        trading_platform=trading_platform,
        database=db_cfg,
        ml=MLConfig(model_dir=model_dir, prediction_threshold=prediction_threshold, auto_train=auto_train),
        trade_mode=trade_mode,
        auto_trade=auto_trade,
        trade_sizes=trade_sizes,
        symbols=symbols,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


SETTINGS: Final = load_settings()

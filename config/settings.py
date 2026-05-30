"""Central configuration for the trading prediction system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class Symbol(str, Enum):
    XAUUSD = "XAUUSD"
    BTCUSD = "BTCUSD"


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


@dataclass(frozen=True)
class DayTradingConfig:
    """Day trading: close same day. M5 entry, H1 trend, H4 big picture."""
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 1.5
    confidence_threshold: float = 35.0
    trend_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    big_picture_timeframe: str = "4h"
    confluence_label: str = "day_trading"


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
    })
    model_dir: str = "trading_ai/ml/models"
    prediction_threshold: float = 0.5


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 6
    test_months: int = 1
    min_signals: int = 30


@dataclass(frozen=True)
class VerificationConfig:
    lookahead_candles: int = 50
    breakeven_threshold_pips: float = 1.0


@dataclass(frozen=True)
class Settings:
    capital_com: CapitalComConfig = field(default_factory=CapitalComConfig)
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

    return Settings(
        capital_com=cap_cfg,
        database=db_cfg,
        trade_mode=trade_mode,
        symbols=symbols,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


SETTINGS: Final = load_settings()
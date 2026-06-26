"""Central configuration for the trading prediction system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class Symbol(str, Enum):
    # Metals
    XAUUSD = "XAUUSD"


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
class FastMarketConfig:
    """Configuration for fast market handling.

    When ATR (Average True Range) exceeds a threshold relative to its average,
    the market is considered 'fast' (high volatility). In fast markets:
    - Cooldowns are multiplied to reduce overtrading
    - SL/TP multipliers are reduced to limit risk per trade

    ATR ratio = current ATR / average ATR over the lookback window.
    """
    # ATR ratio threshold above which the market is considered "fast"
    # 1.5 = current ATR is 50% above its recent average
    atr_threshold: float = 1.5
    # Multiplier applied to cooldowns when market is fast
    # 2.0 = cooldowns last twice as long
    cooldown_multiplier: float = 2.0
    # Factor to reduce SL/TP distances in fast markets
    # 0.5 = halve SL/TP distances when ATR ratio >= threshold
    # Applied before the max distance cap.
    sl_tp_factor: float = 0.5


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
    confidence_threshold: float = 30.0  # Raised from 20 for higher quality signals
    trend_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    confluence_label: str = "scalping"
    # Max time a scalping trade stays open (minutes)
    max_hold_minutes: int = 30


@dataclass(frozen=True)
class DayTradingConfig:
    """Day trading: close same day. M5 entry, H1 trend, M15 confirmation."""
    sl_atr_mult: float = 0.7
    tp_atr_mult: float = 1.2
    confidence_threshold: float = 30.0  # Raised from 20 for higher quality signals
    trend_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    confirmation_timeframe: str = "15m"  # M15 replaces H4 for day trading
    confluence_label: str = "day_trading"
    # Max time a day trade stays open (minutes)
    max_hold_minutes: int = 240


@dataclass(frozen=True)
class SwingConfig:
    """Swing: hold hours to days. H4+H1 must agree, M5 entry trigger."""
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    confidence_threshold: float = 35.0  # Raised from 25 for higher quality signals
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
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
    })
    histgbrt: dict = field(default_factory=lambda: {
        "max_iter": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_bins": 255,
        "early_stopping": True,
        "n_iter_no_change": 50,
        "scoring": "roc_auc",
        "verbose": 0,
    })
    catboost: dict = field(default_factory=lambda: {
        "iterations": 500,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "bootstrap_type": "Bernoulli",
        "eval_metric": "Logloss",
        "verbose": 0,
    })
    model_dir: str = ""  # Set dynamically in load_settings() to absolute path
    # ML prediction thresholds per symbol (read from .env)
    # Default threshold and per-symbol overrides
    prediction_thresholds: dict = field(default_factory=lambda: {"DEFAULT": 0.5})
    # Hard block threshold: trades with ML probability below this are always blocked.
    # This catches trades where the model strongly disagrees (e.g., prob=0.15).
    hard_block_threshold: float = 0.35
    # Minimum verified signals before ML predictions are used for filtering
    # Below this count, ML predictions are logged but don't block trades
    min_verified_for_filter: int = 200
    # Auto-train models during live trading (recommended: false - train via backtest instead)
    auto_train: bool = False
    # Drift detection settings
    drift_detection_enabled: bool = True
    min_signals_for_drift: int = 50
    drift_threshold: float = 0.15
    # SMOTE oversampling: generate synthetic minority class samples to
    # balance the dataset (typically ~70% LOSS / ~30% WIN).
    # Applied only to training data, never validation.
    # When use_smote=True, model-level class weights are DISABLED to avoid
    # double-balancing (SMOTE + scale_pos_weight over-corrects toward WIN).
    use_smote: bool = True
    # Hyperparameter optimization with Optuna (Bayesian search).
    # Finds better params than hardcoded defaults. ~50 trials per model.
    use_tuning: bool = True
    # Probability calibration via Platt scaling.
    # Makes predict_proba outputs more accurate (e.g., 0.3 actually
    # means 30% chance of winning, not an arbitrary score).
    use_calibration: bool = True
    # Feature selection: drop features with <1% importance and retrain.
    # Removes noise from uninformative features (session, mode, spread).
    use_feature_selection: bool = True


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 6
    test_months: int = 1
    min_signals: int = 200


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
    mt5: MT5Config = field(default_factory=MT5Config)
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
        "METALS": 1,
    })
    trade_mode: TradeMode = TradeMode.SWING
    auto_trade: bool = False
    # Position timeout: force-close trades that exceed max hold time
    position_timeout_enabled: bool = True
    # Override max hold minutes (0 = use trade mode default: scalping=30, day=240, swing=1440)
    position_timeout_minutes: int = 0
    trade_sizes: dict = field(default_factory=lambda: {"DEFAULT": 0.01})
    broker_spread_pips: float = 0.0  # Broker execution speed/spread buffer in pips
    # Per-symbol SL/TP multipliers (fallback to trade mode defaults)
    sl_multipliers: dict = field(default_factory=lambda: {})
    tp_multipliers: dict = field(default_factory=lambda: {})
    # Per-symbol max SL/TP distance caps in price units (None = uncapped).
    # Prevents runaway risk when ATR spikes in volatile markets.
    # E.g. {"XAUUSD": 30.0} means gold SL never exceeds $30 from entry.
    max_sl_distance: dict = field(default_factory=lambda: {})
    max_tp_distance: dict = field(default_factory=lambda: {})

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

    def get_max_sl_tp(self, symbol: str) -> tuple[float | None, float | None]:
        """Get max SL and TP distance caps for a symbol in price units.

        Returns (max_sl, max_tp) where None means uncapped.
        """
        return (
            self.max_sl_distance.get(symbol),
            self.max_tp_distance.get(symbol),
        )

    def get_prediction_threshold(self, symbol: str) -> float:
        """Get ML prediction threshold for a symbol. Falls back to DEFAULT."""
        return self.ml.prediction_thresholds.get(symbol, self.ml.prediction_thresholds.get("DEFAULT", 0.3))

    def get_max_hold_minutes(self) -> int:
        """Get max hold time in minutes for the current trade mode.

        If POSITION_TIMEOUT_MINUTES is set (>0) in .env, it overrides the
        trade-mode default. Otherwise falls back to the mode-specific value.
        """
        if self.position_timeout_minutes > 0:
            return self.position_timeout_minutes
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

    symbols: tuple[Symbol, ...] = (Symbol.XAUUSD,)
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

    # MT5
    mt5_cfg = MT5Config(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
    )

    # Trade mode
    mode_str = os.getenv("TRADE_MODE", "swing").lower()
    mode_map = {"scalping": TradeMode.SCALPING, "day_trading": TradeMode.DAY_TRADING, "swing": TradeMode.SWING}
    trade_mode = mode_map.get(mode_str, TradeMode.SWING)

    # Symbols
    symbols_str = os.getenv("SYMBOLS", "XAUUSD").upper()
    symbol_map = {s.value: s for s in Symbol}
    symbols = tuple(symbol_map[s.strip()] for s in symbols_str.split(",") if s.strip() in symbol_map)
    if not symbols:
        symbols = (Symbol.XAUUSD,)

    auto_trade = os.getenv("AUTO_TRADE", "false").lower() == "true"

    # Position timeout settings
    position_timeout_enabled = os.getenv("POSITION_TIMEOUT_ENABLED", "true").lower() == "true"
    position_timeout_minutes = int(os.getenv("POSITION_TIMEOUT_MINUTES", "0"))

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

    # Read max SL/TP distance caps per symbol (in price units)
    max_sl_distance = {}
    max_tp_distance = {}
    for key, value in os.environ.items():
        if key.startswith("MAX_SL_DISTANCE_"):
            symbol_name = key.replace("MAX_SL_DISTANCE_", "")
            try:
                max_sl_distance[symbol_name] = float(value)
            except ValueError:
                pass
        elif key.startswith("MAX_TP_DISTANCE_"):
            symbol_name = key.replace("MAX_TP_DISTANCE_", "")
            try:
                max_tp_distance[symbol_name] = float(value)
            except ValueError:
                pass

    # Model directory — absolute path so it works regardless of CWD
    model_dir = os.path.join(_project_root, "trading_ai", "ml", "models")
    auto_train = os.getenv("ML_AUTO_TRAIN", "false").lower() == "true"

    # Drift detection settings
    drift_detection_enabled = os.getenv("ML_DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    min_signals_for_drift = int(os.getenv("ML_MIN_SIGNALS_FOR_DRIFT", "20"))
    drift_threshold = float(os.getenv("ML_DRIFT_THRESHOLD", "0.15"))

    # ML pipeline enhancements
    use_smote = os.getenv("ML_USE_SMOTE", "true").lower() == "true"
    use_tuning = os.getenv("ML_USE_TUNING", "true").lower() == "true"
    use_calibration = os.getenv("ML_USE_CALIBRATION", "true").lower() == "true"
    use_feature_selection = os.getenv("ML_USE_FEATURE_SELECTION", "true").lower() == "true"

    # Weekly close settings
    weekly_close_enabled = os.getenv("WEEKLY_CLOSE_ENABLED", "true").lower() == "true"
    weekly_close_stop_new_trades = float(os.getenv("WEEKLY_CLOSE_STOP_NEW_TRADES_HOURS", "2.0"))
    weekly_close_close_positions = float(os.getenv("WEEKLY_CLOSE_CLOSE_POSITIONS_HOURS", "0.5"))

    # Verification settings
    dynamic_lookahead = os.getenv("DYNAMIC_LOOKAHEAD", "true").lower() == "true"

    # Fast market cooldown settings
    fast_market_atr_threshold = float(os.getenv("FAST_MARKET_ATR_THRESHOLD", "1.5"))
    fast_market_cooldown_multiplier = float(os.getenv("FAST_MARKET_COOLDOWN_MULTIPLIER", "2.0"))
    fast_market_sl_tp_factor = float(os.getenv("FAST_MARKET_SL_TP_FACTOR", "0.5"))

    # Correlation limit settings
    max_positions_per_group = {
        "METALS": int(os.getenv("MAX_POSITIONS_METALS", "1")),
    }

    # Read all ML_PREDICTION_THRESHOLD_<SYMBOL> from env
    prediction_thresholds = {"DEFAULT": float(os.getenv("ML_PREDICTION_THRESHOLD", "0.5"))}
    for key, value in os.environ.items():
        if key.startswith("ML_PREDICTION_THRESHOLD_") and key != "ML_PREDICTION_THRESHOLD_DEFAULT":
            symbol_name = key.replace("ML_PREDICTION_THRESHOLD_", "")
            try:
                prediction_thresholds[symbol_name] = float(value)
            except ValueError:
                pass

    return Settings(
        mt5=mt5_cfg,
        database=db_cfg,
        ml=MLConfig(
            model_dir=model_dir,
            prediction_thresholds=prediction_thresholds,
            hard_block_threshold=float(os.getenv("ML_HARD_BLOCK_THRESHOLD", "0.35")),
            auto_train=auto_train,
            drift_detection_enabled=drift_detection_enabled,
            min_signals_for_drift=min_signals_for_drift,
            drift_threshold=drift_threshold,
            use_smote=use_smote,
            use_tuning=use_tuning,
            use_calibration=use_calibration,
            use_feature_selection=use_feature_selection,
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
            sl_tp_factor=fast_market_sl_tp_factor,
        ),
        max_positions_per_group=max_positions_per_group,
        trade_mode=trade_mode,
        auto_trade=auto_trade,
        position_timeout_enabled=position_timeout_enabled,
        position_timeout_minutes=position_timeout_minutes,
        trade_sizes=trade_sizes,
        sl_multipliers=sl_multipliers,
        tp_multipliers=tp_multipliers,
        max_sl_distance=max_sl_distance,
        max_tp_distance=max_tp_distance,
        symbols=symbols,
        position_monitor_interval=int(os.getenv("POSITION_MONITOR_INTERVAL", "30")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        broker_spread_pips=float(os.getenv("BROKER_SPREAD_PIPS", "0.0")),
    )


SETTINGS: Final = load_settings()

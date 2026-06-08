"""SQLAlchemy ORM models for candles, signals, and verification results."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open_price: Mapped[float] = mapped_column("open", Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle_symbol_tf_ts"),
        Index("ix_candle_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"Candle({self.symbol} {self.timeframe} "
            f"{self.timestamp} O={self.open_price} H={self.high} "
            f"L={self.low} C={self.close})"
        )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False)
    signal: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    entry: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Feature snapshot at signal time
    ema20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema200: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_histogram: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atr14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    candle_body_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper_wick_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lower_wick_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dist_ema20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dist_ema50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dist_ema200: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trend_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    momentum_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    session: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Verification relationship
    verification: Mapped[Optional["SignalVerification"]] = relationship(
        "SignalVerification", back_populates="signal", uselist=False
    )

    __table_args__ = (
        Index("ix_signal_symbol_tf", "symbol", "timeframe"),
        Index("ix_signal_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"Signal({self.signal_id} {self.signal} {self.symbol} conf={self.confidence})"


class SignalVerification(Base):
    __tablename__ = "signal_verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("signals.id"), nullable=False, unique=True
    )
    result: Mapped[str] = mapped_column(String(20), nullable=False)
    profit_points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    holding_candles: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_favorable_excursion: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_adverse_excursion: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    signal: Mapped["Signal"] = relationship("Signal", back_populates="verification")

    __table_args__ = (
        Index("ix_verification_result", "result"),
    )

    def __repr__(self) -> str:
        return f"SignalVerification(signal_id={self.signal_id} result={self.result})"


class MLModelMeta(Base):
    __tablename__ = "ml_model_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_signal_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index("ix_ml_model_active", "is_active"),
        UniqueConstraint("model_type", "symbol", "timeframe", "version", name="uq_model_version"),
    )

    def __repr__(self) -> str:
        return f"MLModelMeta({self.model_type} v{self.version} {self.symbol})"


class TradeLog(Base):
    """Comprehensive trade execution log for analysis.

    Tracks every trade from entry to exit with:
    - Entry and exit details
    - MFE/MAE (best/worst prices seen)
    - TP progress tracking
    - Trade strength score history
    - Final PnL and outcome
    """
    __tablename__ = "trade_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deal_id: Mapped[str] = mapped_column(String(100), nullable=True)  # Broker deal reference

    # Position details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY/SELL
    size: Mapped[float] = mapped_column(Float, nullable=False)

    # Entry and exit
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)  # Current SL (may be trailing-adjusted)
    original_stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Initial SL before trailing
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Excursion metrics
    mfe_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Max favorable price
    mae_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Max adverse price
    mfe_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Best unrealized PnL
    mae_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Worst unrealized PnL

    # TP Progress
    highest_tp_progress: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.0 to 1.0+

    # Strength score history (JSON string)
    strength_scores: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Outcome
    final_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # close_reason values: "tp_hit", "sl_hit", "manual", "timeout", "strength_exit", "reversal"

    # Signal reference
    signal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_trade_log_symbol", "symbol"),
        Index("ix_trade_log_opened_at", "opened_at"),
        Index("ix_trade_log_closed_at", "closed_at"),
        Index("ix_trade_log_close_reason", "close_reason"),
    )

    def __repr__(self) -> str:
        return (
            f"TradeLog({self.direction} {self.symbol} "
            f"entry={self.entry_price} exit={self.exit_price} "
            f"pnl={self.final_pnl})"
        )
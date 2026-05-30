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
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
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
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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
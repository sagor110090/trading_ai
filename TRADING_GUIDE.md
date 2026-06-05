# Trading AI - User Guide

## Table of Contents
- [Weekly Close Control](#weekly-close-control)
- [Auto Backtest](#auto-backtest)
- [Manual Backtest](#manual-backtest)
- [Live Trading](#live-trading)

---

## Weekly Close Control

The system automatically handles weekly market close for forex, metals, and indices.

### What It Does

- **Stops new trades** 2 hours before weekly close (Friday 20:00 UTC / 3 PM EST)
- **Closes existing positions** 30 minutes before close (Friday 21:30 UTC / 4:30 PM EST)
- **Crypto markets are unaffected** (BTCUSD, ETHUSD, etc. trade 24/7)

### Timeline

| Time (UTC) | Time (EST) | Action |
|------------|------------|--------|
| Friday 20:00 | Friday 3 PM | Stop new trades |
| Friday 21:30 | Friday 4:30 PM | Close existing positions |
| Friday 22:00 | Friday 5 PM | Market closes |

### Configuration (.env)

```bash
# Weekly Close Settings
WEEKLY_CLOSE_ENABLED=true                    # Enable/disable feature
WEEKLY_CLOSE_STOP_NEW_TRADES_HOURS=2.0      # Hours before close to stop new trades
WEEKLY_CLOSE_CLOSE_POSITIONS_HOURS=0.5      # Hours before close to close positions
```

### Asset Classes

| Asset Class | Symbols | Weekly Close |
|-------------|---------|--------------|
| Crypto | BTCUSD, ETHUSD, SOLUSD, LTCUSD, etc. | ❌ No (24/7) |
| Forex | EURUSD, GBPUSD, USDJPY, AUDUSD, etc. | ✅ Yes |
| Metals | XAUUSD, XAGUSD | ✅ Yes |
| Indices | US30, US500, NAS100 | ✅ Yes |

---

## Auto Backtest

Automated backtesting and ML training pipeline.

### Basic Usage

```bash
# Auto-detect and update all stale symbols
python auto_backtest.py

# Backtest specific symbol for 30 days
python auto_backtest.py --symbol XAUUSD --days 30

# Delete existing data, then backtest fresh
python auto_backtest.py --symbol XAUUSD --days 60 --reset

# Specify trade mode
python auto_backtest.py --symbol XAUUSD --days 30 --mode day_trading
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--symbol` | Single symbol to backtest | `--symbol XAUUSD` |
| `--symbols` | Multiple symbols | `--symbols XAGUSD BTCUSD` |
| `--days` | Days to backtest | `--days 30` |
| `--reset` | Delete existing signals first | `--reset` |
| `--force` | Process all symbols regardless | `--force` |
| `--mode` | Trade mode | `--mode day_trading` |

### Trade Modes

- `scalping` - Fast in/out (5 min entry, 1H trend)
- `day_trading` - Close same day (5 min entry, 1H trend, 4H big picture)
- `swing` - Hold hours/days (4H+H1 must agree)

### Examples

```bash
# Fresh 60-day backtest for XAUUSD (deletes old data first)
python auto_backtest.py --symbol XAUUSD --days 60 --reset --mode day_trading

# Update multiple specific symbols
python auto_backtest.py --symbols XAGUSD BTCUSD --force

# Quick test with 7 days
python auto_backtest.py --symbol AUDUSD --days 7
```

---

## Manual Backtest

Full control over backtesting parameters.

### Usage

```bash
python backtest_historical.py --weeks 8 --step 1 --mode day_trading --symbol XAUUSD
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weeks` | Weeks of historical data | 4 |
| `--step` | Step size in M5 candles (1=5min, 12=1hour) | 1 |
| `--mode` | Trade mode | from .env |
| `--symbol` | Symbol to backtest | all symbols |
| `--reset` | Delete existing signals first | false |

### Step Values

| Step | Meaning | Use Case |
|------|---------|----------|
| `--step 1` | Every 5 minutes | Most signals, accurate (slow) |
| `--step 12` | Every hour | Fewer signals, faster |
| `--step 24` | Every 2 hours | Quick testing |

### Examples

```bash
# Thorough backtest (every 5 minutes)
python backtest_historical.py --weeks 8 --step 1 --mode day_trading --symbol XAUUSD

# Faster backtest (every hour)
python backtest_historical.py --weeks 8 --step 12 --mode swing --symbol BTCUSD

# Reset and backtest fresh
python backtest_historical.py --weeks 4 --step 1 --mode day_trading --symbol XAUUSD --reset
```

---

## Live Trading

Run the trading bot in continuous or single-cycle mode.

### Usage

```bash
# Single cycle (for external loops)
python -m trading_ai.main

# Continuous loop
python -m trading_ai.main --loop --interval 300
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--loop` | Run continuously | false |
| `--interval` | Seconds between cycles | 300 |

### Configuration (.env)

```bash
# Trading Settings
TRADING_PLATFORM=MT5                    # CAPITAL or MT5
TRADE_MODE=day_trading                 # scalping, day_trading, swing
AUTO_TRADE=true                        # true = place orders, false = signals only

# Symbols
SYMBOLS=BTCUSD,XAUUSD,XAGUSD,AUDUSD

# Position Sizes
TRADE_SIZE_DEFAULT=0.01
TRADE_SIZE_XAUUSD=0.01
TRADE_SIZE_BTCUSD=0.01

# ML Settings
ML_PREDICTION_THRESHOLD=0.6
ML_AUTO_TRAIN=false

# Intervals
RUN_INTERVAL=300                       # Seconds between trading cycles
POSITION_MONITOR_INTERVAL=10           # Seconds between position checks
```

### Examples

```bash
# Run in loop mode (every 5 minutes)
python -m trading_ai.main --loop --interval 300

# Single cycle for testing
python -m trading_ai.main
```

---

## Quick Reference

### Common Workflows

**Fresh ML Training:**
```bash
# Delete old signals, backtest 60 days, train new models
python auto_backtest.py --symbol XAUUSD --days 60 --reset --mode day_trading
```

**Update All Symbols:**
```bash
# Auto-detect new data and update all symbols
python auto_backtest.py --force
```

**Test New Settings:**
```bash
# Quick 7-day test
python auto_backtest.py --symbol XAUUSD --days 7 --mode scalping --reset
```

**Start Live Trading:**
```bash
# Run continuous loop
python -m trading_ai.main --loop --interval 300
```

### File Summary

| File | Purpose |
|------|---------|
| `auto_backtest.py` | Automated backtest + ML training |
| `backtest_historical.py` | Manual backtest with full control |
| `main.py` | Live trading bot |
| `trade/weekly_close.py` | Weekly close handling logic |

### Troubleshooting

**Not enough signals for ML training:**
- Increase `--days` to backtest more data
- Lower `--step` to generate more signals (use `--step 1`)

**Weekly close blocking trades:**
- Set `WEEKLY_CLOSE_ENABLED=false` in .env
- Crypto symbols are never affected by weekly close

**MT5 connection issues:**
- Ensure MT5 terminal is running
- Check MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env

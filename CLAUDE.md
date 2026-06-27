# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-learning trading system that combines rule-based multi-timeframe confluence signals with ML filtering (XGBoost / HistGradientBoosting / CatBoost). Supports MetaTrader 5 and Capital.com APIs. Runs on a configurable loop (default 5 min) fetching candles, generating signals, filtering with ML, and executing trades.

## Commands

### Run
```bash
# Single cycle
python -m trading_ai.main

# Continuous loop
python -m trading_ai.main --loop --interval 300

# Via shell wrapper (sets PYTHONPATH)
./run.sh
```

### Backfill & Retrain
```bash
# Backtest & ML training (single entry point)
python backtest_historical.py                           # Auto-detect stale symbols, incrementally update & train
python backtest_historical.py --weeks 4 --step 1        # Full backfill for 4 weeks, then train
python backtest_historical.py --symbol XAUUSD --days 30 --reset   # Fresh 30-day backfill
python backtest_historical.py --train --symbol XAUUSD   # Train ML only from existing signals
```

### Test
```bash
python -m pytest tests/test_cooldown_logic.py
# or
python tests/test_cooldown_logic.py
```

## Architecture

### Main Loop Cycle (`main.py:run_signal_cycle`)

```
Fetch Candles (M5/H1/H4) → Store in DB
  → Verify Pending Signals (WIN/LOSS labels)
  → Weekly Close Check → Position Monitor Sync
  → Compute Features (17 indicators + session)
  → Generate Signal (day trading via confluence)
  → Score Signal (weighted multi-factor, 0-100)
  → ML Predict (blocks trades below threshold)
  → Position Manager (cooldowns, correlation limits)
  → Execute Trade (Capital.com or MT5)
  → Execute Trade (MetaTrader 5)
  → Track Drift
```

### ML Self-Learning Loop

```
Signal → Verification (WIN/LOSS) → Dataset → Walk-Forward CV → New Model → Better Filter
```

Models saved as `.joblib` + `_features.json` sidecars in `trading_ai/ml/models/`. Metadata tracked in `ml_model_meta` DB table. Predictor is model-agnostic — loads whichever active model has highest ROC AUC.

### Key Component Connections

| Component | Entry | Connects To |
|-----------|-------|-------------|
| `signals/engine.py` | `generate_signal()` → day trading signal | Reads H4/H1/M5 candles, produces `TradeSignal` |
| `signals/scoring.py` | `compute_signal_scores()` | Weights: EMA trend (30), RSI (15), MACD (20), ATR (10), structure (15), momentum (10) |
| `ml/predictor.py` | `predict_signal()` | Loads model by highest AUC, builds feature vector, returns probability |
| `ml/trainer.py` | `train_models()` | Trains all model types, saves `.joblib` + `_features.json`, stores metadata in DB |
| `trade/position_manager.py` | `evaluate_trade()` | Checks cooldowns, correlation groups (`trade/correlation.py`), dedup → "place"/"skip"/"reverse" |
| `trade/executor.py` | `place_trade_unified()` | Places orders on MetaTrader 5 |
| `ml/drift_detection.py` | `DriftDetector` | Tracks predictions vs outcomes per best-AUC model |

### Trade Mode

| Mode | SL | TP | Min Conf | Max Hold |
|------|----|----|----------|----------|
| Day Trading | 0.7× ATR | 1.2× ATR | 20% | 240 min |

The application uses day trading exclusively (`generate_signal_day_trading()`). The `TRADE_MODE` env var is kept for backward compatibility but ignored.

## Configuration

All settings in `config/settings.py`, loaded from `.env` via `python-dotenv`. Settings are frozen dataclasses — `SETTINGS: Final`. Key env vars: `TRADING_PLATFORM` (MT5/CAPITAL), `USE_SQLITE` (true/false), `AUTO_TRADE`, `SYMBOLS`, `ML_PREDICTION_THRESHOLD`, `POSITION_TIMEOUT_ENABLED` (on/off), `POSITION_TIMEOUT_MINUTES` (override hold time, 0=use day_trading default of 240 min). See `.env.example` for full reference.

## Database

SQLAlchemy 2.0+ ORM. SQLite by default (`USE_SQLITE=true`), MySQL via PyMySQL as alternative. Five tables: `candles`, `signals`, `signal_verifications`, `ml_model_meta`, `trade_log`. Auto-creates tables on startup. Schema migrations are manual (column additions in `database.py:create_tables()`).

## Dual Package Structure

The project runs as both a flat script collection (root `.py` files like `backtest_historical.py`) and a sub-package (`python -m trading_ai.main`). The `run.sh` sets `PYTHONPATH` to the parent directory to make both work. When adding new entry points as scripts, place them at root; when adding library modules, place them under the appropriate sub-package directory.

## Git Workflow

When asked to commit, always:
1. Create a new branch from the current branch (descriptive kebab-case name)
2. Stage and commit with a clear message that includes the branch it was created from
3. The commit message should note what was changed and why

Example:
```
git checkout -b <branch-name>
git add -A
git commit -m "<what changed> — branched from <parent-branch>"
```

## Cooldown System

`trade/position_manager.py` manages trade cooldowns with tiered durations by close reason (breakeven < loss < verified-loss). Fast market detection (ATR ≥ 1.5× average) doubles all cooldowns. Verified results anchor cooldowns to actual trade close time, not signal generation time. Deal-ID dedup prevents duplicate cooldowns from the same broker deal.

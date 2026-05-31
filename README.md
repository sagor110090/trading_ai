# Trading AI

Self-learning trading prediction system for **XAUUSD** and **BTCUSD** using the Capital.com API. Combines rule-based multi-timeframe confluence signals with ML filtering (XGBoost / LightGBM / CatBoost) that improves itself over time.

---

## How It Works

```
                         ┌──────────────────────────────────────────┐
                         │           Every Cycle (5 min)            │
                         └──────────────────────────────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              Fetch Candles        Verify Pending        Check Timeouts
              M5 / H1 / H4         Signals               (close expired)
                     │                    │                    │
                     └────────┬───────────┘                    │
                              ▼                                │
                    ┌──────────────────┐                        │
                    │  Signal Engine   │                        │
                    │  H4 → H1 → M5   │                        │
                    │  Confluence      │                        │
                    └────────┬─────────┘                        │
                             │                                  │
                      ┌──────┴──────┐                           │
                      ▼             ▼                           │
                   BUY/SELL      NO TRADE ─────────────────────┘
                      │
                      ▼
              ┌──────────────┐     ┌──────────────┐
              │ ML Predictor │────▶│  ≥ 30% prob?  │
              │ (catboost,   │     └──────┬───────┘
              │  xgboost,    │            │
              │  lightgbm)   │     ┌──────┴───────┐
              └──────────────┘     ▼              ▼
                               APPROVED       BLOCKED
                                  │
                                  ▼
                          ┌──────────────┐
                          │  Auto Trade  │──── Place Order
                          │  (if on)     │──── Set SL / TP
                          └──────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   Auto-Train     │
                        │ (new verified    │
                        │  > last trained?) │
                        └──────────────────┘
```

### Signal Flow

1. **Fetch** — Pull latest M5, H1, H4 candles from Capital.com API
2. **Verify** — Check if any pending signals hit SL/TP in recent candles → label WIN/LOSS
3. **Timeout** — Close positions that exceeded max hold time
4. **Confluence** — H4 sets the big trend, H1 confirms, M5 triggers entry
5. **ML Filter** — Pre-trained model predicts probability of success; block below 30%
6. **Trade** — Place order on Capital.com (if `AUTO_TRADE=true`)
7. **Auto-Train** — Re-train ML when new verified signals accumulate

### ML Self-Learning Loop

```
  Signal → Verification → Dataset → Walk-Forward CV → New Model → Better Filter
    ▲                                                        │
    └──────────────────── Next Cycle ◄───────────────────────┘
```

The ML does **not** generate signals. It only **filters** rule-based signals — blocking low-probability ones and approving high-probability ones. As more signals are verified (WIN/LOSS labels from real outcomes), the model re-trains and improves.

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/sagor/Code/trading_ai
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` with your Capital.com credentials and preferences:

| Key | Description | Example |
|-----|-------------|---------|
| `CAPITAL_API_KEY` | Capital.com API key | `b3XJpRBS...` |
| `CAPITAL_IDENTIFIER` | Login email | `you@email.com` |
| `CAPITAL_PASSWORD` | Login password | `@YourPassword` |
| `CAPITAL_USE_DEMO` | Use demo account | `true` |
| `DB_HOST` | MySQL host | `127.0.0.1` |
| `DB_PORT` | MySQL port | `3307` |
| `DB_USER` | MySQL user | `root` |
| `DB_PASSWORD` | MySQL password | `root` |
| `TRADE_MODE` | `scalping` / `day_trading` / `swing` | `day_trading` |
| `AUTO_TRADE` | Place real orders | `true` or `false` |
| `SYMBOLS` | Comma-separated | `BTCUSD` or `XAUUSD,BTCUSD` |
| `ML_PREDICTION_THRESHOLD` | Min ML probability to trade | `0.3` |
| `TRADE_SIZE_DEFAULT` | Default position size | `0.01` |
| `TRADE_SIZE_XAUUSD` | Gold position size | `0.01` |
| `TRADE_SIZE_BTCUSD` | Bitcoin position size | `0.1` |
| `RUN_INTERVAL` | Seconds between cycles | `300` |

### 3. Set Up Database

MySQL must be running. The system creates tables automatically on first run.

### 4. Backfill Historical Data (Recommended First Step)

Train ML models immediately from historical data instead of waiting days:

```bash
./run_backfill.sh --weeks 4 --step 1 --reset
```

This fetches weeks of candle data, replays it through the signal engine, verifies signals, and trains ML models. Takes ~2-5 minutes.

### 5. Run Live

```bash
./run.sh
```

Runs continuously, cycling every `RUN_INTERVAL` seconds.

---

## Trade Modes

| Mode | H4 Role | H1 Role | M5 Role | SL | TP | Min Conf | Max Hold | Run Interval |
|------|---------|---------|---------|-----|-----|----------|----------|-------------|
| **Scalping** | — | Trend | Entry | 0.5× ATR | 0.8× ATR | 30% | 30 min | 60–120s |
| **Day Trading** | Must not oppose | Trend | Entry | 1.0× ATR | 1.5× ATR | 35% | 2 hr | 300s |
| **Swing** | Boss trend | Must confirm | Support | 2.0× ATR | 3.0× ATR | 40% | 8 hr | 900s |

### Confluence Logic

**Scalping** — H1 provides direction. M5 momentum must agree (or be neutral). Tight stops, quick in/out.

**Day Trading** — H1 provides direction. H4 must not oppose (opposite H4 = blocked). M5 agreement gives bonus. Medium stops, close same day.

**Swing** — H4 is the boss. H1 must confirm H4 direction. M5 supports entry. Wide stops, hold hours to days.

### Position Timeout

Open positions that exceed the mode's max hold time are automatically closed:

- Scalping: close after 30 minutes
- Day Trading: close after 2 hours
- Swing: close after 8 hours

---

## ML System

### Models

Three gradient boosting classifiers are trained in parallel:

| Model | Hyperparams | File Size |
|-------|-------------|-----------|
| XGBoost | 500 trees, depth 6, lr 0.05, subsample 0.8 | ~355 KB |
| LightGBM | 500 trees, depth 6, lr 0.05, binary objective | ~15 KB |
| CatBoost | 500 iterations, depth 6, lr 0.05, Logloss | ~47 KB |

### Features (18 columns)

| Feature | Description |
|---------|-------------|
| `ema20` / `ema50` / `ema200` | Exponential moving averages |
| `rsi14` | Relative Strength Index (14-period) |
| `macd_line` / `macd_signal` / `macd_histogram` | MACD (12/26/9) |
| `atr14` | Average True Range (14-period) |
| `momentum_score` | Composite: RSI + MACD + price vs EMA20 (−100 to +100) |
| `trend_strength` | EMA stacking + price position (−100 to +100) |
| `dist_ema20` / `dist_ema50` / `dist_ema200` | Distance from price to EMA (% of EMA) |
| `candle_body_pct` | Body as % of total range (positive = bullish) |
| `upper_wick_pct` / `lower_wick_pct` | Wick percentages |
| `spread` | Bid-ask spread |
| `session_enc` | 0=Asian, 1=London, 2=New York |

### Target

Binary: `1 = WIN`, `0 = LOSS`. BREAKEVEN signals are excluded from training.

### Walk-Forward Validation

Three strategies, used in cascade:

1. **Month-based folds** — 6-month expanding train, 1-month test. No future data leakage.
2. **Percentage split** — 70/30 stratified time-blocked split for short datasets.
3. **Oversampled split** — sklearn `resample` on minority class for tiny/imbalanced datasets.

### Prediction Filter

- ML model predicts probability that a rule-based signal will succeed
- **Below threshold** (`ML_PREDICTION_THRESHOLD=0.3`): trade is BLOCKED
- **Above threshold**: trade is APPROVED
- Before 50 verified signals: ML predictions are logged but don't block trades (not enough data to be reliable)

---

## Signal Scoring

Six weighted factors (total 100 points):

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| EMA Trend | 30 pts | EMA20/50/200 stacking + price position |
| MACD | 20 pts | Histogram magnitude + crossover proximity |
| Structure | 15 pts | Candle body dominance (directional strength) |
| RSI | 15 pts | Overbought/oversold zones |
| Momentum | 10 pts | Momentum × trend alignment |
| ATR | 10 pts | Volatility sweet spot (not too quiet, not extreme) |

Direction (BUY/SELL/NO_TRADE) is determined by majority vote across EMA, RSI, and MACD factors. Ties → NO_TRADE.

---

## Verification

Every generated signal is tracked against future M5 candles:

| Outcome | Condition |
|---------|-----------|
| **WIN** | Price hits take-profit before stop-loss |
| **LOSS** | Price hits stop-loss before take-profit |
| **BREAKEVEN** | Neither hit within 50 candles; unrealized P&L within ±1 pip |

Tracks **MFE** (Max Favorable Excursion) and **MAE** (Max Adverse Excursion) for each signal.

---

## CLI Commands

### Live Trading

```bash
# Single cycle
python3 -m trading_ai.main

# Continuous loop (reads RUN_INTERVAL from .env)
./run.sh
```

### Historical Backfill

```bash
# Default: 4 weeks, every M5 candle
python3 -m trading_ai.backtest_historical

# Options
--weeks 4          # Weeks of historical data
--step 1           # M5 candles per step (1=5min, 12=1hr)
--mode day_trading # Override trade mode
--symbol BTCUSD    # Specific symbol
--reset            # Clear existing signals & ML models first

# Examples
./run_backfill.sh --weeks 2 --step 1 --reset
./run_backfill.sh --symbol XAUUSD --mode scalping --reset
```

### View Signals

```bash
# Last 20 signals
python3 -m trading_ai.signals_cmd

# Last 50 signals
python3 -m trading_ai.signals_cmd 50
```

---

## Project Structure

```
trading_ai/
├── main.py                     # Main orchestrator / prediction loop
├── display.py                   # Rich terminal UI (panels, colors, emojis)
├── backtest_historical.py       # Historical backfill script
├── signals_cmd.py               # CLI to view signals from MySQL
├── .env                         # Configuration (not committed)
├── .env.example                 # Template
├── requirements.txt             # 13 Python dependencies
├── run.sh                       # Main loop runner
├── run_backfill.sh              # Backfill runner
│
├── config/
│   └── settings.py              # Enums, dataclasses, .env loader
│
├── data/
│   ├── collector.py             # Capital.com REST API client
│   ├── database.py              # SQLAlchemy engine & sessions
│   └── models.py                # ORM: Candle, Signal, SignalVerification, MLModelMeta
│
├── indicators/
│   ├── ema.py                   # EMA20, EMA50, EMA200
│   ├── rsi.py                   # RSI14 (Wilder's smoothing)
│   ├── macd.py                  # MACD 12/26/9
│   ├── atr.py                   # ATR14 (Wilder's smoothing)
│   └── features.py              # FeatureSet + compute_all_features assembler
│
├── signals/
│   ├── scoring.py               # 6-factor weighted confidence scoring
│   └── engine.py                # Multi-TF confluence engine (3 trade modes)
│
├── ml/
│   ├── dataset.py                # Build training dataset from verified signals
│   ├── trainer.py                # Train XGBoost/LightGBM/CatBoost
│   ├── predictor.py             # Live ML prediction / filtering
│   └── models/                  # Saved .joblib model files + _features.json
│
├── backtesting/
│   └── walk_forward.py          # Walk-forward validation (3 fallback strategies)
│
├── trade/
│   └── executor.py              # Capital.com order placement & position management
│
└── verification/
    └── verifier.py              # Signal outcome verification via future candles
```

---

## Database

MySQL on `127.0.0.1:3307` (database: `trading_ai`).

### Tables

**`candles`** — Stored market data

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| symbol | String(20) | XAUUSD / BTCUSD |
| timeframe | String(10) | 5m / 1h / 4h |
| timestamp | DateTime | UTC |
| open | Float | Mapped from `open_price` |
| high / low / close | Float | |
| volume / spread | Float | Nullable |

Unique constraint: `(symbol, timeframe, timestamp)`

**`signals`** — Generated trade signals

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| signal_id | String(64) | UUID |
| symbol / timeframe / signal | String | BUY / SELL / NO_TRADE |
| confidence | Float | 0–100% |
| entry / stop_loss / take_profit | Float | Price levels |
| timestamp | DateTime | When signal was generated |
| 17 feature columns | Float | Snapshot at signal time |

**`signal_verifications`** — Signal outcomes

| Column | Type | Notes |
|--------|------|-------|
| signal_id | Integer FK | → signals.id (unique) |
| result | String(20) | WIN / LOSS / BREAKEVEN |
| profit_points | Float | P&L in price units |
| risk_reward | Float | R:R ratio |
| holding_candles | Integer | M5 candles held |
| max_favorable_excursion | Float | Best unrealized gain |
| max_adverse_excursion | Float | Worst unrealized loss |

**`ml_model_meta`** — Model metadata

| Column | Type | Notes |
|--------|------|-------|
| model_type | String(50) | xgboost / lightgbm / catboost |
| symbol / timeframe | String | |
| version | Integer | Auto-incremented |
| filepath | Text | Path to .joblib |
| accuracy / precision / recall / f1 / roc_auc | Float | Walk-forward metrics |
| train_signal_count | Integer | How many signals trained on |
| is_active | Boolean | Only active model is used for prediction |

---

## Capital.com API

### Endpoints Used

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/session` | Authenticate |
| GET | `/api/v1/prices/{epic}` | Fetch candle data |
| GET | `/api/v1/markets?searchTerm=` | Search market instruments |
| GET | `/api/v1/markets/{epic}` | Latest bid/ask price |
| GET | `/api/v1/positions` | List open positions |
| POST | `/api/v1/positions` | Place market order |
| DELETE | `/api/v1/positions/{dealId}` | Close position |
| GET | `/api/v1/confirms/{dealRef}` | Confirm deal status |

### Symbol → Epic Mapping

| Symbol | Epic |
|--------|------|
| XAUUSD | `GOLD` |
| BTCUSD | `BTCUSD` |

### Timeframe → Resolution

| Timeframe | API Code |
|-----------|----------|
| M1 | `MINUTE` |
| M5 | `MINUTE_5` |
| M15 | `MINUTE_15` |
| H1 | `HOUR` |
| H4 | `HOUR_4` |
| D1 | `DAY` |

### Date Format

The `from` parameter uses ISO format: `2026-05-24T00:00:00`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.1 | DataFrames |
| numpy | ≥1.26 | Numerical operations |
| ta | ≥0.11 | Technical analysis (available, custom indicators used) |
| xgboost | ≥2.0 | XGBoost classifier |
| lightgbm | ≥4.1 | LightGBM classifier |
| catboost | ≥1.2 | CatBoost classifier |
| scikit-learn | ≥1.3 | Metrics, resampling |
| sqlalchemy | ≥2.0 | ORM & database |
| pymysql | ≥1.1 | MySQL driver |
| joblib | ≥1.3 | Model serialization |
| requests | ≥2.31 | API calls |
| python-dotenv | ≥1.0 | .env loading |
| rich | ≥13.0 | Terminal formatting |

---

## Example Output

```
──────────────────────────────────── BTCUSD ────────────────────────────────────

  Candles  +1 5m · 0 1h · 0 4h
  Verified  0 signals · 4 pending (need more data)

╭─────────────────────────── Signal ───────────────────────────╮
│  📈 BUY  BTCUSD  confidence=50.2%                           │
│  Entry  73,877.10    SL  73,681.10                          │
│  TP     74,171.09    R:R 1:1.5                              │
│  Mode    day_trading                                         │
╰──────────────────────────────────────────────────────────────╯

╭─────────────────────── ML Prediction ────────────────────────╮
│  catboost_v1  ·  probability 56.4%                          │
│  ✅ APPROVED  above 30% threshold                           │
╰──────────────────────────────────────────────────────────────╯

  🔄 Auto-train  28 verified · last trained on 28 — skipping

╭─────────────────────── Cycle Summary ───────────────────────╮
│  📈 BUY BTCUSD                                              │
│     Entry 73,877.10  SL 73,681.10  TP 74,171.09            │
│  🤖 ML: 56.4% → Approved                                    │
│  ⏭  Trade skipped                                           │
╰──────────────────────────────────────────────────────────────╯
```

---

## Backfill Results

First run on BTCUSD with 3.5 days of M5 data:

```
╭────────────────────── Backfill Summary ──────────────────────╮
│  Candles checked      946                                    │
│  Signals generated    335                                    │
│  Duplicates skipped  303                                     │
│  Verified             28                                     │
│  No-trade             611                                    │
│                                                             │
│  Wins  17  Losses  11  Breakeven  0                         │
│  Win rate  60.7%                                            │
╰──────────────────────────────────────────────────────────────╯

🔄 Training ML models on 28 verified signals...
    xgboost     acc=64.3%  auc=0.5000
    lightgbm    acc=64.3%  auc=0.5000
    catboost    acc=64.3%  auc=0.5333
✓ ML models trained!
```

**28 verified signals in 2 minutes** — instead of waiting days for real-time accumulation.

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| ML filters, doesn't generate | Rule-based signals are interpretable; ML only blocks bad ones |
| M5 candles for all verification | Finest granularity → most accurate SL/TP detection |
| Walk-forward validation | Strict temporal ordering, no future data leakage |
| 3 fallback split strategies | Handles any dataset size from 20 to 10,000+ signals |
| Duplicate position check | Prevents opening two trades on the same symbol |
| Trade timeout per mode | Scalping shouldn't hold for hours; swing needs days |
| `train_signal_count` in meta | Prevents re-training every cycle when no new data |
| Rich terminal output | Color-coded panels make decisions instantly visible |
| Historical backfill | Train ML in minutes from old data instead of waiting weeks |

---

## Common Workflows

### Fresh Start

```bash
# 1. Backfill historical data and train ML
./run_backfill.sh --weeks 4 --step 1 --reset

# 2. Run live trading
./run.sh
```

### Check Current State

```bash
# View recent signals
python3 -m trading_ai.signals_cmd

# Check ML models
python3 -c "
from trading_ai.data.database import get_session
from trading_ai.data.models import MLModelMeta
s = next(get_session())
for m in s.query(MLModelMeta).filter(MLModelMeta.is_active == True).all():
    print(f'{m.model_type} v{m.version} {m.symbol} auc={m.roc_auc}')
"
```

### Switch Trade Mode

```bash
# Edit .env
TRADE_MODE=scalping   # or day_trading, swing

# Re-backfill with new mode and retrain
./run_backfill.sh --mode scalping --reset

# Run live
./run.sh
```

### Add a New Symbol

```bash
# 1. Edit .env
SYMBOLS=XAUUSD,BTCUSD
TRADE_SIZE_XAUUSD=0.01

# 2. Backfill the new symbol
./run_backfill.sh --symbol XAUUSD --reset

# 3. Run live (both symbols)
./run.sh
```

### Reset Everything

```bash
./run_backfill.sh --reset --weeks 4
```

This clears all signals, verifications, and ML models, then rebuilds from scratch.

---

## License

Private project. All rights reserved.
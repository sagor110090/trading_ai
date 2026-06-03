# Trading AI

Self-learning trading prediction system using **multi-symbol support** with both **MetaTrader 5** and **Capital.com** APIs. Combines rule-based multi-timeframe confluence signals with ML filtering (XGBoost / LightGBM / CatBoost) that improves itself over time.

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
              │ ML Predictor │────▶│  ≥ Threshold? │
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

1. **Fetch** — Pull latest M5, H1, H4 candles from MT5 or Capital.com API
2. **Verify** — Check if any pending signals hit SL/TP in recent candles → label WIN/LOSS
3. **Timeout** — Close positions that exceeded max hold time
4. **Confluence** — H4 sets the big trend, H1 confirms, M5 triggers entry
5. **ML Filter** — Pre-trained model predicts probability of success; blocks below threshold
6. **Trade** — Place order on MT5 or Capital.com (if `AUTO_TRADE=true`)
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
cd C:/Users/mehed/Herd/trading_ai
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install MetaTrader5  # For MT5 support
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` with your credentials and preferences:

#### Platform Selection
| Key | Description | Example |
|-----|-------------|---------|
| `TRADING_PLATFORM` | `CAPITAL` or `MT5` | `MT5` |

#### MT5 Settings (if TRADING_PLATFORM=MT5)
| Key | Description | Example |
|-----|-------------|---------|
| `MT5_LOGIN` | MT5 account number | `370275` |
| `MT5_PASSWORD` | MT5 password | `YourPassword` |
| `MT5_SERVER` | Broker server name | `FusionMarkets-Demo` |

#### Capital.com Settings (if TRADING_PLATFORM=CAPITAL)
| Key | Description | Example |
|-----|-------------|---------|
| `CAPITAL_API_KEY` | Capital.com API key | `b3XJpRBS...` |
| `CAPITAL_IDENTIFIER` | Login email | `you@email.com` |
| `CAPITAL_PASSWORD` | Login password | `@YourPassword` |
| `CAPITAL_USE_DEMO` | Use demo account | `true` |

#### Trading Settings
| Key | Description | Example |
|-----|-------------|---------|
| `USE_SQLITE` | Use SQLite instead of MySQL | `true` |
| `TRADE_MODE` | `scalping` / `day_trading` / `swing` | `day_trading` |
| `AUTO_TRADE` | Place real orders | `true` or `false` |
| `TRAILING_STOP_ENABLED` | Move SL at profit milestones | `true` |
| `SYMBOLS` | Comma-separated symbols | `BTCUSD,XAUUSD,ETHUSD,EURUSD,GBPUSD` |
| `ML_PREDICTION_THRESHOLD` | Min ML probability to trade (0.0-1.0) | `0.3` |
| `TRADE_SIZE_DEFAULT` | Default position size | `0.01` |
| `TRADE_SIZE_*` | Symbol-specific size | `TRADE_SIZE_XAUUSD=0.01` |
| `RUN_INTERVAL` | Seconds between cycles | `300` |

### 3. Supported Symbols

**Crypto:** BTCUSD, ETHUSD, LTCUSD, BCHUSD, XRPUSD, BTCEUR, ETHGBP

**Metals:** XAUUSD, XAGUSD

**Forex Majors:** EURUSD, GBPUSD, USDJPY, USDCHF

**Forex Minors:** EURGBP, EURJPY, GBPJPY, AUDUSD, NZDUSD, USDCAD, EURAUD, GBPCHF

### 4. Set Up Database

SQLite is used by default (`USE_SQLITE=true`). The system creates tables automatically on first run.

For MySQL, set `USE_SQLITE=false` and configure:
```env
DB_HOST=127.0.0.1
DB_PORT=3307
DB_NAME=trading_ai
DB_USER=root
DB_PASSWORD=root
```

### 5. Backfill Historical Data (Recommended First Step)

Train ML models immediately from historical data instead of waiting days:

```bash
# MT5 or Capital.com (auto-detected from TRADING_PLATFORM)
python -m trading_ai.backtest_historical --weeks 4 --step 1

# Specific symbol
python -m trading_ai.backtest_historical --symbol ETHUSD --weeks 4

# Reset existing data first
python -m trading_ai.backtest_historical --weeks 4 --reset
```

This fetches weeks of candle data, replays it through the signal engine, verifies signals, and trains ML models. Takes ~2-5 minutes per symbol.

### 6. Run Live

```bash
# Single cycle
python -m trading_ai.main

# Continuous loop
python -m trading_ai.main --loop

# With custom interval (seconds)
python -m trading_ai.main --loop --interval 300
```

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

- ML model predicts **probability** that a rule-based signal will succeed
- **Below threshold** (`ML_PREDICTION_THRESHOLD`): trade is BLOCKED
- **Above threshold**: trade is APPROVED
- Before 50 verified signals: ML predictions are logged but don't block trades (not enough data to be reliable)

**Note:** The threshold compares against **ML probability** (not adjusted confidence). The adjusted confidence displayed is: `rule_confidence × ML_probability`

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
python main.py

# Continuous loop
python main.py --loop

# Custom interval (seconds)
python main.py --loop --interval 300
```

### Historical Backfill

```bash
# Default: 4 weeks, every M5 candle
python backtest_historical.py

# Options
--weeks 4               # Weeks of historical data
--step 1                # M5 candles per step (1=5min, 12=1hr)
--mode day_trading      # Override trade mode
--symbol ETHUSD         # Specific symbol
--reset                 # Clear existing signals & ML models first

# Examples
python backtest_historical.py --weeks 4 --symbol ETHUSD
python backtest_historical.py --symbol EURUSD --mode scalping --reset
```

### ML Model Management

```bash
# Retrain all models manually (with class weights for imbalance)
python retrain_ml.py

# Retrain specific symbol/model
python retrain_ml.py --symbols BTCUSD XAUUSD --models catboost

# Check database statistics
python check_db_stats.py
```

### View Signals

```bash
# Last 20 signals
python signals_cmd.py

# Last 50 signals
python signals_cmd.py 50
```

---

## Project Structure

```
trading_ai/
├── main.py                     # Main orchestrator / prediction loop
├── display.py                  # Rich terminal UI (panels, colors, emojis)
├── backtest_historical.py      # Historical backfill script (MT5 + Capital.com)
├── signals_cmd.py              # CLI to view signals from database
├── .env                        # Configuration (not committed)
├── .env.example                # Template
├── requirements.txt            # Python dependencies
├── run.sh                      # Main loop runner
├── run_backfill.sh             # Backfill runner
│
├── config/
│   └── settings.py             # Enums, dataclasses, .env loader
│
├── data/
│   ├── collector.py            # Capital.com REST API client
│   ├── mt5_client.py           # MetaTrader 5 client
│   ├── database.py             # SQLAlchemy engine & sessions
│   └── models.py               # ORM: Candle, Signal, SignalVerification, MLModelMeta
│
├── indicators/
│   ├── ema.py                  # EMA20, EMA50, EMA200
│   ├── rsi.py                  # RSI14 (Wilder's smoothing)
│   ├── macd.py                 # MACD 12/26/9
│   ├── atr.py                  # ATR14 (Wilder's smoothing)
│   └── features.py             # FeatureSet + compute_all_features assembler
│
├── signals/
│   ├── scoring.py              # 6-factor weighted confidence scoring
│   └── engine.py               # Multi-TF confluence engine (3 trade modes)
│
├── ml/
│   ├── dataset.py              # Build training dataset from verified signals
│   ├── trainer.py              # Train XGBoost/LightGBM/CatBoost
│   ├── predictor.py            # Live ML prediction / filtering
│   └── models/                 # Saved .joblib model files + _features.json
│
├── backtesting/
│   └── walk_forward.py         # Walk-forward validation (3 fallback strategies)
│
├── trade/
│   └── executor.py             # Unified order placement (MT5 + Capital.com)
│
└── verification/
    └── verifier.py             # Signal outcome verification via future candles
```

---

## Database

SQLite (`trading_ai.db`) by default, or MySQL on `127.0.0.1:3307`.

### Tables

**`candles`** — Stored market data

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| symbol | String(20) | XAUUSD / BTCUSD / ETHUSD / etc. |
| timeframe | String(10) | 5m / 1h / 4h |
| timestamp | DateTime | UTC |
| open / high / low / close | Float | |
| volume / spread | Float | Nullable |

Unique constraint: `(symbol, timeframe, timestamp)`

**`signals`** — Generated trade signals

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| signal_id | String(64) | UUID |
| symbol / timeframe / signal | String | BUY / SELL / NO_TRADE |
| confidence | Float | 0–100% (rule-based) |
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

## Platform APIs

### MetaTrader 5

**Requirements:**
- MT5 terminal installed and running
- Account logged in
- `pip install MetaTrader5`

**Symbol Mapping:** Direct symbol names (XAUUSD, BTCUSD, ETHUSD, etc.)

**Timeframe Mapping:**
| Timeframe | MT5 Constant |
|-----------|--------------|
| M5 | `TIMEFRAME_M5` |
| H1 | `TIMEFRAME_H1` |
| H4 | `TIMEFRAME_H4` |

### Capital.com

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
| ETHUSD | `ETHUSD` |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.1 | DataFrames |
| numpy | ≥1.26 | Numerical operations |
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
| MetaTrader5 | ≥5.0 | MT5 support |

---

## Example Output

```
──────────────────────────────────── BTCUSD ────────────────────────────────────

  New candles  +1 5m · 0 1h · 0 4h
  Loaded candles  500 5m · 500 1h · 500 4h
  Verified  0 signals · 4 pending (need more data)

╭─────────────────────────── Signal ───────────────────────────╮
│  📈 BUY  BTCUSD  confidence=81.6%                          │
│  Entry  73,098.00    SL  73,380.38                          │
│  TP     72,674.43    R:R 1:1.5                              │
│  Mode    day_trading                                         │
╰──────────────────────────────────────────────────────────────╯

╭─────────────────────── ML Prediction ────────────────────────╮
│  catboost_v3  ·  ML prob 66.1%  →  adj. confidence 54.0%    │
│  ✅ APPROVED  above 30% threshold                           │
╰──────────────────────────────────────────────────────────────╯

╭─────────────────────── Trade Placed ────────────────────────╮
│  ✅ BUY  BTCUSD  size=0.01                                  │
│  SL  73,363.38  TP  72,657.43                               │
│  Ref  407975262                                             │
╰──────────────────────────────────────────────────────────────╯

  🔄 Auto-train  69 verified · last trained on 69 — skipping

╭─────────────────────── Cycle Summary ───────────────────────╮
│  📈 BUY BTCUSD                                              │
│     Entry 73,098.00  SL 73,380.38  TP 72,674.43            │
│  ✅ Trade placed                                            │
╰──────────────────────────────────────────────────────────────╯
```

---

## Backfill Results

First run on BTCUSD with 4 weeks of M5 data:

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
| Dual platform support | Works with both MT5 and Capital.com |
| Multi-symbol support | Trade crypto, metals, and forex pairs |

---

## Common Workflows

### Fresh Start

```bash
# 1. Backfill historical data and train ML
python -m trading_ai.backtest_historical --weeks 4 --step 1 --reset

# 2. Run live trading
python -m trading_ai.main --loop
```

### Add a New Symbol

```bash
# 1. Edit .env
SYMBOLS=BTCUSD,XAUUSD,ETHUSD

# 2. Backfill the new symbol
python -m trading_ai.backtest_historical --symbol ETHUSD --weeks 4

# 3. Run live (all symbols)
python -m trading_ai.main --loop
```

### Switch Trade Mode

```bash
# Edit .env
TRADE_MODE=scalping   # or day_trading, swing

# Re-backfill with new mode and retrain
python -m trading_ai.backtest_historical --mode scalping --reset

# Run live
python -m trading_ai.main --loop
```

### Switch Between MT5 and Capital.com

```bash
# Edit .env
TRADING_PLATFORM=MT5        # or CAPITAL

# Edit symbols if needed
SYMBOLS=BTCUSD,ETHUSD       # MT5 supports more symbols

# Backfill and run
python -m trading_ai.backtest_historical --weeks 4
python -m trading_ai.main --loop
```

### Reset Everything

```bash
python -m trading_ai.backtest_historical --reset --weeks 4
```

This clears all signals, verifications, and ML models, then rebuilds from scratch.

---

## License

Private project. All rights reserved.

# Position Monitor

A background monitoring system that continuously evaluates open positions and automatically closes them when trend strength weakens.

---

## How It Works

### 1. Initialization & Sync

- `PositionMonitor` is created in `main()` with the trading client
- On every check cycle, `sync_positions()` queries the broker for all open positions:
  - **New positions** are added to the `TPProgressTracker` (keyed by broker `deal_id`)
  - **Externally closed positions** (TP hit, SL hit, manual close) are removed from tracking and logged to the `TradeLog` database

### 2. Background Monitoring Loop

- A daemon thread (`"position-monitor"`) runs every **60 seconds**
- Calls `check_all_positions()` which iterates every open position

### 3. Per-Position Check (`_check_position`)

For each open position:

1. Fetches current market price
2. Updates the `TPProgressTracker` with current price (updates MFE, MAE, progress)
3. Reads broker's unrealized P&L
4. Decides whether to evaluate trade strength

### 4. Strength Evaluation Triggers

Strength is evaluated when **any** of these conditions are met:

| Trigger | Condition | Reason |
|---------|-----------|--------|
| **40% TP reached** | TP progress >= 40% for the first time | First check-in on a maturing trade |
| **Peak already above 40%** | Position is already >= 40% TP on first check with no prior strength score | Catches positions that were already mid-move when tracking started |
| **10% drop from peak** | TP progress drops 10% from the highest progress seen (requires peak >= 50%) | Trend is reversing; re-evaluate whether to stay in |

### 5. Strength Scoring (`TradeStrengthScore`)

When triggered, the system evaluates 5 sub-scores on M5 OHLCV data:

| Component | Max Points | What It Measures |
|-----------|-----------|-----------------|
| **EMA Trend Direction** | 25 | Price vs EMA20 (+10), EMA20 vs EMA50 (+10), EMA20 vs EMA200 (+5) |
| **EMA Slope Strength** | 20 | Linear regression slope of EMA20; magnitude and direction consistency |
| **RSI Momentum** | 25 | RSI level in direction-appropriate zone (BUY: 50-70 sweet spot, SELL: 30-50 sweet spot) |
| **Volume Ratio** | 15 | Current volume vs 20-period average; expanding volume supports trend |
| **ATR Expansion** | 15 | ATR ratio vs average; moderate expansion = strong trend, extreme = exhaustion risk |

**Total: 0-100 points**

### 6. Decision Rules

| Total Score | Decision | Action |
|-------------|----------|--------|
| **>= 70** | **Hold** | Trend is strong; keep position open |
| **50-69** | **Monitor** | Trend is weakening; keep open but watch closely. **Exception:** if EMA trend direction < 10 (price crossed EMA20), exit immediately |
| **< 50** | **Exit** | Trend is weak; close position now |

### 7. Position Closure

When the decision is **exit**:

1. Position is closed via `close_position_unified()`
2. Final stats are retrieved from the `TPProgressTracker` (MFE, MAE, highest TP progress, strength score history)
3. Trade outcome is logged to the `TradeLog` database table with all statistics

### 8. Display

The monitor renders a Rich table every cycle showing:

| Column | Content |
|--------|---------|
| Symbol | Instrument name |
| Direction | BUY / SELL |
| TP Progress | Current progress toward take profit (0-100%+) |
| P&L | Unrealized profit/loss from broker |
| Strength | Strength score (0-100) if evaluated |
| Status | HOLD / MONITOR / WEAK / CLOSED / ERROR |

Closed positions also trigger a dedicated red "Strength Exit" panel showing exit details.

---

## Architecture Diagram

```
main.py
  |
  +-- position-monitor thread (60s interval)
  |     |
  |     +-- _check_position_strength()
  |           |
  |           +-- PositionMonitor.check_all_positions()
  |                 |
  |                 +-- sync_positions()  -->  broker API  /  TradeLog DB
  |                 |
  |                 +-- _check_position()  (per position)
  |                       |
  |                       +-- TPProgressTracker.update_position()
  |                       +-- TPProgressTracker.should_evaluate_strength()
  |                       |
  |                       +-- _evaluate_strength()  (if triggered)
  |                       |     |
  |                       |     +-- TradeStrengthScore.evaluate()
  |                       |         |-- _ema_trend_direction()   (0-25)
  |                       |         |-- _ema_slope_strength()   (0-20)
  |                       |         |-- _rsi_momentum()         (0-25)
  |                       |         |-- _volume_ratio()        (0-15)
  |                       |         |-- _atr_expansion()       (0-15)
  |                       |         +-- _make_decision()       --> hold/monitor/exit
  |                       |
  |                       +-- _execute_close()  (if exit)
  |                             |
  |                             +-- close_position_unified()  -->  broker API
  |                             +-- TPProgressTracker.close_position()
  |                             +-- _log_trade()  -->  TradeLog DB
```

---

## Key Files

| File | Purpose |
|------|---------|
| `trade/position_monitor.py` | `PositionMonitor` class - orchestrates syncing, checking, and closing |
| `trade/tp_tracker.py` | `TPProgressTracker` / `TPProgress` - tracks TP progress, MFE/MAE, strength triggers |
| `trade/strength_score.py` | `TradeStrengthScore` - 5-component strength scoring engine (0-100) |
| `data/models.py` | `TradeLog` - SQLAlchemy model for persisting closed trade statistics |
| `display.py` | `display_position_strength_exit()` - Rich panel for exit events |
| `main.py` | Wires everything together; runs the background monitor thread every 60s |

---

## Data Tracked Per Position (`TPProgress`)

| Field | Description |
|-------|-------------|
| `highest_tp_progress` | Best TP progress ratio seen (0.0-1.0+) |
| `mfe_price` / `mfe_pnl` | Maximum Favorable Excursion - best price and PnL reached |
| `mae_price` / `mae_pnl` | Maximum Adverse Excursion - worst price and PnL reached |
| `strength_scores` | List of all strength evaluations with timestamps, scores, decisions |
| `close_reason` | Reason for closing (e.g. `strength_exit:...`, `external_close`) |

---

## TradeLog Database Fields

| Field | Description |
|-------|-------------|
| `deal_id` | Broker deal/ticket ID |
| `symbol`, `direction`, `size` | Position identifiers |
| `entry_price`, `exit_price` | Entry and exit prices |
| `stop_loss`, `take_profit` | SL/TP levels |
| `opened_at`, `closed_at` | Timestamps |
| `mfe_price`, `mfe_pnl` | Maximum favorable excursion |
| `mae_price`, `mae_pnl` | Maximum adverse excursion |
| `highest_tp_progress` | Best TP progress reached |
| `strength_scores` | JSON text of all strength evaluations |
| `final_pnl` | Final realized P&L |
| `close_reason` | Why the position was closed |
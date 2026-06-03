"""Fix imports: remove trading_ai. prefix from all imports."""
import os
import re

files_to_fix = [
    'ml/predictor.py',
    'backtest_historical.py',
    'main.py',
    'signals_cmd.py',
    'check_ml.py',
    'ml/trainer.py',
    'verification/verifier.py',
    'trade/position_manager.py',
    'trade/position_monitor.py',
    'data/database.py',
    'migrate_to_mysql.py',
    'data/mt5_client.py',
    'display.py',
    'trade/executor.py',
    'signals/engine.py',
    'run_main_wrapper.py',
    'run_backfill_wrapper.py',
    'backtesting/walk_forward.py',
    'ml/dataset.py',
    'signals/scoring.py',
    'indicators/features.py',
    'data/collector.py',
]

fixed_count = 0
for filepath in files_to_fix:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Replace all trading_ai imports
    content = re.sub(r'from trading_ai\.config\.settings', 'from config.settings', content)
    content = re.sub(r'from trading_ai\.data\.database', 'from data.database', content)
    content = re.sub(r'from trading_ai\.data\.models', 'from data.models', content)
    content = re.sub(r'from trading_ai\.data\.collector', 'from data.collector', content)
    content = re.sub(r'from trading_ai\.ml\.dataset', 'from ml.dataset', content)
    content = re.sub(r'from trading_ai\.ml\.trainer', 'from ml.trainer', content)
    content = re.sub(r'from trading_ai\.ml\.predictor', 'from ml.predictor', content)
    content = re.sub(r'from trading_ai\.indicators\.features', 'from indicators.features', content)
    content = re.sub(r'from trading_ai\.signals\.engine', 'from signals.engine', content)
    content = re.sub(r'from trading_ai\.signals\.scoring', 'from signals.scoring', content)
    content = re.sub(r'from trading_ai\.verification\.verifier', 'from verification.verifier', content)
    content = re.sub(r'from trading_ai\.trade\.executor', 'from trade.executor', content)
    content = re.sub(r'from trading_ai\.trade\.position_manager', 'from trade.position_manager', content)
    content = re.sub(r'from trading_ai\.trade\.position_monitor', 'from trade.position_monitor', content)
    content = re.sub(r'from trading_ai\.display', 'from display', content)
    content = re.sub(r'from trading_ai\.backtesting\.walk_forward', 'from backtesting.walk_forward', content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed_count += 1
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes: {filepath}")

print(f"\nFixed {fixed_count} files")

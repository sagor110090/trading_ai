"""Database statistics checker for backtest results."""

import sqlite3
from datetime import datetime

DB_PATH = "trading_ai.db"


def print_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 60)
    print("BACKTEST DATABASE STATISTICS")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Table counts
    print("=== TABLE ROW COUNTS ===")
    tables = [
        ('signals', 'Signals'),
        ('signal_verifications', 'Signal Verifications'),
        ('trade_log', 'Trade Logs'),
        ('candles', 'Candles'),
        ('ml_model_meta', 'ML Models')
    ]

    for table, name in tables:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f'{name}: {count}')
        except sqlite3.OperationalError:
            print(f'{name}: Table not found')

    print()

    # Signal stats
    print("=== SIGNAL STATS ===")
    try:
        cursor.execute('SELECT signal, COUNT(*) FROM signals GROUP BY signal')
        for row in cursor.fetchall():
            print(f'{row[0]}: {row[1]}')
    except sqlite3.OperationalError:
        print("No signals found")
    print()

    # Verification results
    print("=== VERIFICATION RESULTS ===")
    try:
        cursor.execute('SELECT result, COUNT(*) FROM signal_verifications GROUP BY result')
        results = dict(cursor.fetchall())
        wins = results.get('WIN', 0)
        losses = results.get('LOSS', 0)
        be = results.get('BREAKEVEN', 0)
        total = wins + losses + be

        print(f'WIN: {wins}')
        print(f'LOSS: {losses}')
        print(f'BREAKEVEN: {be}')
        if total > 0:
            print(f'Win Rate: {wins/total*100:.2f}%')
    except sqlite3.OperationalError:
        print("No verification data found")
    print()

    # ML models
    print("=== ML MODELS ===")
    try:
        cursor.execute('''
            SELECT id, model_type, symbol, timeframe, version,
                   accuracy, precision, recall, f1, roc_auc,
                   train_signal_count, is_active
            FROM ml_model_meta
            ORDER BY trained_at DESC
        ''')
        print(f"{'ID':<4} {'Type':<10} {'Symbol':<8} {'TF':<6} {'Ver':<4} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUC':<6} {'Active':<7}")
        print("-" * 80)
        for row in cursor.fetchall():
            id, model_type, symbol, tf, ver, acc, prec, rec, f1, auc, sig_count, active = row
            active_str = 'YES' if active else 'NO'
            acc_val = acc if acc else 0
            prec_val = prec if prec else 0
            rec_val = rec if rec else 0
            f1_val = f1 if f1 else 0
            auc_val = auc if auc else 0
            print(f'{id:<4} {model_type:<10} {symbol:<8} {tf:<6} {ver:<4} {acc_val:.4f} {prec_val:.4f} {rec_val:.4f} {f1_val:.4f} {auc_val:.4f} {active_str:<7}')
    except sqlite3.OperationalError:
        print("No ML models found")

    print()
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    print_stats()

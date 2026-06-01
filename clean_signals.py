"""Clean old signals to prepare for fresh backtest with new parameters.

This deletes:
- All signal records
- All signal verification records
- ML model metadata from database
- ALL ML model files (complete fresh start)

This keeps:
- Candle data (historical price data)
"""

import sqlite3
from pathlib import Path

def clean_signals(db_path: str = "trading_ai.db"):
    """Delete all signals and verifications to prepare for fresh backtest."""

    db_file = Path(db_path)
    if not db_file.exists():
        print(f"❌ Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Count before deletion
        cursor.execute("SELECT COUNT(*) FROM signals")
        signal_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM signal_verifications")
        verification_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ml_model_meta")
        model_count = cursor.fetchone()[0]

        model_dir = Path("trading_ai/ml/models")
        ml_files = list(model_dir.glob("*.joblib")) if model_dir.exists() else []

        print(f"\n📊 Current data:")
        print(f"  Signals: {signal_count}")
        print(f"  Verifications: {verification_count}")
        print(f"  ML models (DB): {model_count}")
        print(f"  ML model files: {len(ml_files)}")

        if signal_count == 0 and verification_count == 0 and model_count == 0 and len(ml_files) == 0:
            print("\n✅ Database is already clean!")
            return

        print(f"\n⚠️  This will delete EVERYTHING for a FRESH START:")
        print(f"  - All signals ({signal_count})")
        print(f"  - All verifications ({verification_count})")
        print(f"  - All ML model metadata ({model_count})")
        print(f"  - All ML model files ({len(ml_files)})")
        print(f"\n✅ This will KEEP:")
        print(f"  - Candle data (price history)")
        print(f"\nType 'yes' to confirm: ", end="")

        confirm = input()

        if confirm.lower() != 'yes':
            print("❌ Cancelled.")
            return

        # Delete verifications first (foreign key constraint)
        cursor.execute("DELETE FROM signal_verifications")
        verif_deleted = cursor.rowcount

        # Delete signals
        cursor.execute("DELETE FROM signals")
        signals_deleted = cursor.rowcount

        # Delete ML model metadata
        cursor.execute("DELETE FROM ml_model_meta")
        models_deleted = cursor.rowcount

        conn.commit()

        print(f"\n✅ Database cleanup complete:")
        print(f"  Signals deleted: {signals_deleted}")
        print(f"  Verifications deleted: {verif_deleted}")
        print(f"  ML metadata deleted: {models_deleted}")

        # Delete ALL ML model files
        _clean_all_models()

        print(f"\n💡 Ready for fresh backtest with new parameters!")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error: {e}")
        raise
    finally:
        conn.close()


def _clean_all_models():
    """Delete ALL ML model files for a fresh start."""

    model_dir = Path("trading_ai/ml/models")
    if not model_dir.exists():
        print(f"\n  📁 No model directory found")
        return

    # Get all model files
    all_files = list(model_dir.glob("*"))
    removed_count = 0

    if all_files:
        print(f"\n  Removing ALL ML models ({len(all_files)} files)...")
        for file in all_files:
            try:
                file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"    ⚠️  Could not delete {file.name}: {e}")

        print(f"\n  ✅ Removed {removed_count} files")
        print(f"  📁 Model directory is now empty")
    else:
        print(f"\n  📁 No ML model files found")


if __name__ == "__main__":
    print("=" * 50)
    print("🧹 Complete Reset - Fresh Start")
    print("=" * 50)
    clean_signals()

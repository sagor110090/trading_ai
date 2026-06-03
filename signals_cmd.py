"""Quick CLI to view trading signals from the database."""

import sys
import os
os.environ.setdefault("USE_SQLITE", "true")

from data.database import get_session
from data.models import Signal, SignalVerification


def show_signals(limit: int = 20):
    with get_session() as session:
        signals = (
            session.query(Signal)
            .order_by(Signal.timestamp.desc())
            .limit(limit)
            .all()
        )

        if not signals:
            print("No signals yet. Run the system to generate signals.")
            return

        print(f"{'ID':<18} {'Symbol':<8} {'TF':<12} {'Signal':<6} {'Conf':>5} {'Entry':>10} {'SL':>10} {'TP':>10} {'Result':<8} {'Time':<20}")
        print("-" * 130)

        for s in signals:
            result = "PENDING"
            if s.verification:
                result = s.verification.result

            print(
                f"{s.signal_id:<18} "
                f"{s.symbol:<8} "
                f"{s.timeframe:<12} "
                f"{s.signal:<6} "
                f"{s.confidence:>5.1f} "
                f"{s.entry:>10.2f} "
                f"{s.stop_loss:>10.2f} "
                f"{s.take_profit:>10.2f} "
                f"{result:<8} "
                f"{str(s.timestamp):<20}"
            )

        total = session.query(Signal).count()
        wins = session.query(SignalVerification).filter(SignalVerification.result == "WIN").count()
        losses = session.query(SignalVerification).filter(SignalVerification.result == "LOSS").count()
        pending = total - wins - losses

        print()
        print(f"Total: {total}  |  Wins: {wins}  |  Losses: {losses}  |  Pending: {pending}", end="")
        if wins + losses > 0:
            print(f"  |  Win Rate: {wins / (wins + losses) * 100:.1f}%")
        else:
            print()


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    show_signals(limit)
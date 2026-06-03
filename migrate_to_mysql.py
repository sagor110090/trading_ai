"""Migrate data from SQLite to MySQL."""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from trading_ai.config.settings import SETTINGS
from trading_ai.data.database import create_tables

SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_ai.db")

def migrate():
    cfg = SETTINGS.database
    password_part = f":{cfg.password}" if cfg.password else ""
    mysql_url = (
        f"mysql+pymysql://{cfg.username}{password_part}"
        f"@{cfg.host}:{cfg.port}/{cfg.database}?charset=utf8mb4"
    )

    print(f"Connecting to MySQL: {cfg.host}:{cfg.port}/{cfg.database}")
    mysql_engine = create_engine(mysql_url, pool_pre_ping=True)

    print("Creating tables in MySQL...")
    create_tables()

    print(f"Reading from SQLite: {SQLITE_PATH}")
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sc = sqlite_conn.cursor()

    mysql_conn = mysql_engine.connect()

    # Migrate candles
    sc.execute("SELECT COUNT(*) FROM candles")
    count = sc.fetchone()[0]
    print(f"Migrating {count} candles...")
    sc.execute("SELECT * FROM candles")
    rows = sc.fetchall()
    if rows:
        cols = rows[0].keys()
        placeholders = ", ".join([f":{c}" for c in cols])
        insert_sql = text(f"INSERT IGNORE INTO candles ({', '.join(cols)}) VALUES ({placeholders})")
        for i, row in enumerate(rows):
            d = dict(zip(cols, row))
            mysql_conn.execute(insert_sql, d)
            if (i + 1) % 500 == 0:
                mysql_conn.commit()
                print(f"  {i+1}/{count} candles...")
        mysql_conn.commit()
    print(f"  Done: {len(rows)} candles migrated")

    # Migrate signals
    sc.execute("SELECT COUNT(*) FROM signals")
    count = sc.fetchone()[0]
    print(f"Migrating {count} signals...")
    sc.execute("SELECT * FROM signals")
    rows = sc.fetchall()
    if rows:
        cols = rows[0].keys()
        placeholders = ", ".join([f":{c}" for c in cols])
        insert_sql = text(f"INSERT IGNORE INTO signals ({', '.join(cols)}) VALUES ({placeholders})")
        for i, row in enumerate(rows):
            d = dict(zip(cols, row))
            d_clean = {k: (None if v == '' else v) for k, v in d.items()}
            mysql_conn.execute(insert_sql, d_clean)
            if (i + 1) % 200 == 0:
                mysql_conn.commit()
                print(f"  {i+1}/{count} signals...")
        mysql_conn.commit()
    print(f"  Done: {len(rows)} signals migrated")

    # Migrate signal_verifications
    sc.execute("SELECT COUNT(*) FROM signal_verifications")
    count = sc.fetchone()[0]
    print(f"Migrating {count} signal_verifications...")
    sc.execute("SELECT * FROM signal_verifications")
    rows = sc.fetchall()
    if rows:
        cols = rows[0].keys()
        placeholders = ", ".join([f":{c}" for c in cols])
        insert_sql = text(f"INSERT IGNORE INTO signal_verifications ({', '.join(cols)}) VALUES ({placeholders})")
        for i, row in enumerate(rows):
            d = dict(zip(cols, row))
            d_clean = {k: (None if v == '' else v) for k, v in d.items()}
            mysql_conn.execute(insert_sql, d_clean)
            if (i + 1) % 200 == 0:
                mysql_conn.commit()
                print(f"  {i+1}/{count} verifications...")
        mysql_conn.commit()
    print(f"  Done: {len(rows)} verifications migrated")

    # Migrate ml_model_meta
    sc.execute("SELECT COUNT(*) FROM ml_model_meta")
    count = sc.fetchone()[0]
    print(f"Migrating {count} ml_model_meta...")
    sc.execute("SELECT * FROM ml_model_meta")
    rows = sc.fetchall()
    if rows:
        cols = rows[0].keys()
        placeholders = ", ".join([f":{c}" for c in cols])
        insert_sql = text(f"INSERT IGNORE INTO ml_model_meta ({', '.join(cols)}) VALUES ({placeholders})")
        for i, row in enumerate(rows):
            d = dict(zip(cols, row))
            d_clean = {k: (None if v == '' else v) for k, v in d.items()}
            mysql_conn.execute(insert_sql, d_clean)
        mysql_conn.commit()
    print(f"  Done: {len(rows)} model metadata migrated")

    # Migrate trade_log
    sc.execute("SELECT COUNT(*) FROM trade_log")
    count = sc.fetchone()[0]
    print(f"Migrating {count} trade_log entries...")
    sc.execute("SELECT * FROM trade_log")
    rows = sc.fetchall()
    if rows:
        cols = rows[0].keys()
        placeholders = ", ".join([f":{c}" for c in cols])
        insert_sql = text(f"INSERT IGNORE INTO trade_log ({', '.join(cols)}) VALUES ({placeholders})")
        for i, row in enumerate(rows):
            d = dict(zip(cols, row))
            d_clean = {k: (None if v == '' else v) for k, v in d.items()}
            mysql_conn.execute(insert_sql, d_clean)
        mysql_conn.commit()
    print(f"  Done: {len(rows)} trade logs migrated")

    mysql_conn.close()
    sqlite_conn.close()

    # Verify
    print("\nVerification:")
    mysql_conn = mysql_engine.connect()
    for table in ["candles", "signals", "signal_verifications", "ml_model_meta", "trade_log"]:
        result = mysql_conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count = result.fetchone()[0]
        print(f"  {table}: {count} rows")
    mysql_conn.close()

    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
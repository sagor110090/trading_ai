"""Create SQLite tables and import data from MySQL backup."""

import re
import sqlite3
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import create_tables
from data.models import Base

sql_file = "trading_ai_backup.sql"
db_path = "trading_ai.db"

def extract_insert_statements(sql_content):
    """Extract INSERT statements from SQL file."""
    inserts = []
    lines = sql_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.upper().startswith('INSERT INTO'):
            # Collect the full INSERT statement (may span multiple lines)
            full_insert = line
            i += 1
            while i < len(lines) and ';' not in lines[i-1]:
                full_insert += lines[i]
                i += 1
            inserts.append(full_insert)
        else:
            i += 1

    return inserts

def convert_insert_to_sqlite(insert_stmt):
    """Convert MySQL INSERT to SQLite format."""
    # Remove backticks
    insert = insert_stmt.replace('`', '')
    # Fix escaped quotes
    insert = insert.replace("\\'", "''")
    insert = insert.replace('\\"', '"')
    return insert

def main():
    print("Creating SQLite database schema...")

    # Remove existing database if any
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create tables using SQLAlchemy
    create_tables()
    print(f"Tables created in {db_path}")

    # Read MySQL backup
    if not os.path.exists(sql_file):
        print(f"Warning: {sql_file} not found. Database created with empty tables.")
        return

    print(f"\nImporting data from {sql_file}...")

    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    # Extract and process INSERT statements
    inserts = extract_insert_statements(sql_content)
    print(f"Found {len(inserts)} INSERT statements")

    # Connect to SQLite and import
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    imported = 0
    for insert_stmt in inserts:
        try:
            sqlite_insert = convert_insert_to_sqlite(insert_stmt)
            cursor.execute(sqlite_insert)
            imported += 1
            if imported % 100 == 0:
                print(f"  Imported {imported} statements...")
        except Exception as e:
            print(f"  Warning: {str(e)[:80]}")

    conn.commit()

    # Show table stats
    print("\nDatabase statistics:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count} rows")
        except Exception as e:
            print(f"  {table_name}: Error - {e}")

    conn.close()
    print(f"\nDone! SQLite database: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    main()

"""Wrapper script to run main trading loop with correct Python path."""
import sys
import os

# Get the current directory (trading_ai package)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory (Herd) - this allows "from trading_ai.config" to work
parent_dir = os.path.dirname(current_dir)

# Add parent to path FIRST so Python can find the trading_ai package
sys.path.insert(0, parent_dir)

# Now imports like "from trading_ai.config.settings" will work
# We need to change to current_dir so any relative paths in the script work
os.chdir(current_dir)

# Import main module - now it's part of trading_ai package
from trading_ai import main

if __name__ == "__main__":
    main()

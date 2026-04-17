"""
seed_data.py
------------
Run this script ON THE HOST (not inside Docker) to pre-fetch real stock data
from Yahoo Finance and save it to ./data/stocks.csv

Docker cannot reach Yahoo Finance (blocked by their anti-bot system),
but your host machine can. This script bridges that gap.

Usage:
    python seed_data.py

The saved CSV is then mounted into the Docker container via a volume,
and the app loads it automatically instead of calling yfinance directly.
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import pandas as pd
from data_collector import fetch_all_stocks, DEFAULT_SYMBOLS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stocks.csv")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🌐 Fetching real stock data from Yahoo Finance (runs on host)...")
    df = fetch_all_stocks(DEFAULT_SYMBOLS, period_days=365)

    if df.empty:
        print("❌ No data fetched! Check your internet connection or yfinance.")
        sys.exit(1)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(df)} rows to {OUTPUT_FILE}")
    print("   Now rebuild and restart your Docker container.")
    print("   The app will automatically load from this file.")

if __name__ == "__main__":
    main()

"""
data_collector.py
-----------------
External data ingestion engine.

Manages robust fetching of historical stock data specifically structured for Yahoo Finance (NSE).
Features a priority-based fallback architecture evaluating local pre-fetched seeding,
TLS impersonation algorithms (curl_cffi) to bypass CDN blocks, and standard yfinance pooling.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    cffi_requests = None


# Path where host-fetched data is mounted into the container.
# Run `python seed_data.py` on the host to populate this.
CSV_SEED_PATH = os.environ.get("STOCK_CSV_PATH", "/app/data/stocks.csv")

# A curated list of popular Indian stocks (NSE)
# .NS suffix tells yfinance to pull from NSE exchange
DEFAULT_SYMBOLS = [
    "INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS",
    "WIPRO.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS",
    "ADANIENT.NS", "MARUTI.NS"
]

# Human-readable names mapped to their ticker symbols
SYMBOL_NAMES = {
    "INFY.NS":       "Infosys",
    "TCS.NS":        "Tata Consultancy Services",
    "RELIANCE.NS":   "Reliance Industries",
    "HDFCBANK.NS":   "HDFC Bank",
    "WIPRO.NS":      "Wipro",
    "ICICIBANK.NS":  "ICICI Bank",
    "SBIN.NS":       "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS":   "Adani Enterprises",
    "MARUTI.NS":     "Maruti Suzuki",
}


def fetch_stock_data(symbol: str, period_days: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV historical time-series data for a single equity symbol.

    Executes a direct chronological query against Yahoo Finance. First attempts TLS impersonation
    via `curl_cffi` to mitigate IP/Bot blocking. If that connection module is unavailable or fails,
    defaults to the standard `yfinance` history engine.

    Data is forcibly padded up to 365 days regardless of the default requested limit
    to calculate long-term trailing metrics like the 52-week High/Low.

    Args:
        symbol (str): Target stock ticker string mapped to NSE convention (e.g. 'INFY.NS').
        period_days (int): Required backward-looking day boundary. Defaults to 365.

    Returns:
        pd.DataFrame: Sliced tabular OHLCV data. Returned empty if completely unresponsive.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=period_days)
    
    # Try fetching with curl_cffi (Robust for Docker IP blocks)
    if cffi_requests is not None:
        try:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={period_days}d&interval=1d"
            resp = cffi_requests.get(url, impersonate="chrome110", timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("chart", {}).get("result", [])
                if data:
                    res = data[0]
                    timestamps = res.get("timestamp", [])
                    quote = res.get("indicators", {}).get("quote", [{}])[0]
                    
                    df = pd.DataFrame({
                        "Date": [datetime.fromtimestamp(ts).date() for ts in timestamps],
                        "Open": quote.get("open", []),
                        "High": quote.get("high", []),
                        "Low": quote.get("low", []),
                        "Close": quote.get("close", []),
                        "Volume": quote.get("volume", [])
                    })
                    df["Symbol"] = symbol
                    return df.dropna()
        except Exception as e:
            print(f"  ⚠️ curl_cffi failed for {symbol}: {e}. Falling back to yfinance.")

    # Fallback to yfinance if curl_cffi fails or is unavailable
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"))

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()                      # Move Date from index → column
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Symbol"] = symbol                      # Tag each row with its ticker
    df["Date"]   = pd.to_datetime(df["Date"]).dt.date  # Strip timezone info

    return df


def load_from_csv(path: str = CSV_SEED_PATH) -> pd.DataFrame:
    """
    Ingest pre-fetched historical telemetry mapped from an isolated local CSV.

    Functionally critical when acting inside Docker ecosystems where external
    CDN/WAF boundaries block runtime TCP connections from dynamic IPs. Resolves
    against a volume-mounted static data blob structured by `seed_data.py`.

    Args:
        path (str): Target path vector for the CSV store. Contextually overridden.

    Returns:
        pd.DataFrame: Tabular extraction converted into datetime-based Pandas rows.
                      Yields an empty DataFrame if the host volume lacks the file.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    print(f"  📂 Loading pre-fetched data from {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    print(f"  ✅ Loaded {len(df)} rows from CSV ({df['Symbol'].nunique()} symbols)")
    return df


def fetch_all_stocks(symbols: list = None, period_days: int = 365) -> pd.DataFrame:
    """
    Construct a consolidated DataFrame aggregating historical patterns across multiple equities.

    Invokes a strategy-priority architectural pattern:
        1. Explores and validates structural data off the static local CSV mount.
        2. Escalates to dynamic live yfinance polling iterating sequentially across all symbols.

    Args:
        symbols (list, optional): Array of equity identifier strings. Yields defaults if empty.
        period_days (int): Span of the query lookup. Defaults to 365.

    Returns:
        pd.DataFrame: Aggregated multi-index compatible table combining all requested equities.
    """
    symbols = symbols or DEFAULT_SYMBOLS

    # ── Priority 1: Host-mounted CSV (Docker-safe) ────────────────────────────
    csv_df = load_from_csv()
    if not csv_df.empty:
        # Filter to only requested symbols
        if symbols:
            csv_df = csv_df[csv_df["Symbol"].isin(symbols)]
        return csv_df

    # ── Priority 2: Live yfinance (works on host, blocked in Docker) ──────────
    print("  ℹ️  No CSV seed found — attempting live yfinance fetch…")
    frames  = []

    for sym in symbols:
        try:
            df = fetch_stock_data(sym, period_days)
            if not df.empty:
                frames.append(df)
                print(f"  ✅ Fetched {sym} — {len(df)} rows")
            else:
                print(f"  ⚠️  No data for {sym}")
        except Exception as e:
            print(f"  ❌ Error fetching {sym}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
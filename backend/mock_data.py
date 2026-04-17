"""
mock_data.py
------------
Procedural synthetic data generator sandbox.

Provides a terminal failsafe data ingestion pathway mathematically simulating OHLCV 
metrics. Guarantees 100% frontend dashboard uptime in deeply disconnected or actively 
blocked orchestration environments wherein live APIs and local static volumes both fail.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def _make_symbol_series(symbol: str, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate an isolated synthetic numeric sequence depicting constrained random walk valuations.

    Mimics organic asset price fluctuations by anchoring the closing price against a randomized 
    normal distribution base, interpolating Open/High/Low bands deterministically.
    
    Args:
        symbol (str): Target stock ticker format (e.g. 'TCS.NS').
        n_days (int): Total timeframe count needed.
        rng (np.random.Generator): State-seeded random generator guaranteeing predictable output.

    Returns:
        pd.DataFrame: Deterministic generated timeseries mapping matching chronological constraints.
    """
    end_date = datetime.today().date()
    dates = [end_date - timedelta(days=offset) for offset in range(n_days)][::-1]

    base_price = rng.uniform(100.0, 2500.0)
    returns = rng.normal(loc=0.0005, scale=0.015, size=n_days)

    close_prices = [base_price]
    for daily_return in returns[1:]:
        close_prices.append(max(1.0, close_prices[-1] * (1.0 + daily_return)))

    close = np.array(close_prices)
    open_price = close * (1 + rng.normal(0, 0.004, size=n_days))
    high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.006, size=n_days)))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.006, size=n_days)))
    volume = rng.integers(200_000, 5_000_000, size=n_days)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_price, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume.astype(int),
            "Symbol": symbol,
        }
    )


def generate_all_mock_data(symbols: list[str], n_days: int = 365) -> pd.DataFrame:
    """
    Produce a holistic dummy telemetry batch mapped across all requested identifiers.

    Operates as the terminal fallback architecture if all remote requests and local cached 
    data mounts are permanently unavailable, ensuring the application UI components render.

    Args:
        symbols (list[str]): Desired tracker keys mapping the generated assets.
        n_days (int): Span duration block tracing backward from today. Defaults to 365.

    Returns:
        pd.DataFrame: Consolidated payload structure compatible directly with downstream Pandas aggregators.
    """
    if not symbols or n_days <= 0:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"])

    rng = np.random.default_rng(42)
    frames = [_make_symbol_series(symbol, n_days, rng) for symbol in symbols]
    return pd.concat(frames, ignore_index=True)

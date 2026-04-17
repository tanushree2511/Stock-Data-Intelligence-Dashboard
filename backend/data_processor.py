"""
data_processor.py
-----------------
Core quantitative transformation suite.

Isolates all Pandas-driven algorithmic computations ensuring the downstream dataset 
is deterministically derived. Facilitates complete unit-test isolation decoupled 
from the external presentation or networking layers.

Available Metrics:
    - `daily_return`: Intraday delta percentage `(Close - Open) / Open`.
    - `ma_7`: Exponential/Rolling 7-day numerical trend smoother.
    - `week52_high/low`: Maximum and minimum tracking against the available historical bound.
    - `volatility_7d`: Standard deviation of returns reflecting risk gradients.
    - `momentum_score`: Mean reversion differential against a 30-day index.
    - `sentiment_index`: Composite numerical mood indicator clamping volatility, momentum, and returns.
"""

import pandas as pd
import numpy as np


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proportional intraday fluctuations.

    Calculates the delta between opening and closing valuations normalized against the 
    opening benchmark. Determines the directional thrust of the instrument on that specific day.

    Args:
        df (pd.DataFrame): Target frame containing 'Close' and 'Open' continuous columns.

    Returns:
        pd.DataFrame: Mutated frame containing the new 'daily_return' float column.
    """
    df["daily_return"] = (df["Close"] - df["Open"]) / df["Open"]
    return df


def add_moving_average(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Calculate the rolling sequential arithmetic mean against closing prices.

    The dataset is strictly pre-sorted chronologically to protect the integrity 
    of the sliding historical window.

    Args:
        df (pd.DataFrame): Time-series dataframe requiring temporal smoothing.
        window (int): The trailing lookback window size. Defaults to 7.

    Returns:
        pd.DataFrame: Mutated baseline frame injected with the 'ma_{window}' index.
    """
    df = df.sort_values("Date")
    df[f"ma_{window}"] = (
        df.groupby("Symbol")["Close"]
          .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    return df


def add_52week_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the expanding historical boundaries for equity valuations.

    Leverages an expanding (forward-filling) computation window instead of a hard 
    365-day rolling loop to seamlessly manage heavily filtered or incomplete yearly datasets.

    Args:
        df (pd.DataFrame): Frame organized sequentially containing stock 'Close' identifiers.

    Returns:
        pd.DataFrame: Extended frame carrying 'week52_high' and 'week52_low' ceiling and floor values.
    """
    df = df.sort_values("Date")
    df["week52_high"] = (
        df.groupby("Symbol")["Close"]
          .transform(lambda x: x.expanding().max())
    )
    df["week52_low"] = (
        df.groupby("Symbol")["Close"]
          .transform(lambda x: x.expanding().min())
    )
    return df


def add_volatility(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Assess inherent structural risks via standard deviation trajectories.

    Computes the trailing standard deviation isolated against chronological daily return percentages.
    Values are automatically annualized using the classical finance heuristic (multiplying by √252 
    trading iterations).

    Args:
        df (pd.DataFrame): Populated timeseries block; requires pre-calculated 'daily_return'.
        window (int): Lookback boundary length. Defaults to 7.

    Returns:
        pd.DataFrame: Frame integrating the 'volatility_7d' risk proxy.
    """
    df = df.sort_values("Date")
    df["volatility_7d"] = (
        df.groupby("Symbol")["daily_return"]
          .transform(lambda x: x.rolling(window=window, min_periods=1).std() * np.sqrt(252))
    )
    return df


def add_momentum_score(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Gauge medium-term directional velocity of the equity symbol.

    Formulates the deviation ratio scaling the current closing valuation against 
    its established historical baseline. Positive distributions indicate bullish 
    buy-side pressure.

    Args:
        df (pd.DataFrame): Time-series ledger.
        window (int): Length of the trailing average logic. Defaults to 30.

    Returns:
        pd.DataFrame: Augmented ledger with the discrete 'momentum_score' index.
    """
    df = df.sort_values("Date")
    ma_30 = (
        df.groupby("Symbol")["Close"]
          .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    df["momentum_score"] = (df["Close"] - ma_30) / ma_30
    return df


def add_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate an abstracted, deterministic 'market mood' indicator.

    Employs a custom fractional weighting strategy resolving multiple distinct signal components:
        - 50% Positive allocation representing prolonged numerical momentum.
        - 30% Positive allocation representing short-term daily intraday action.
        - 20% Negative allocation penalizing extreme volatile uncertainty.
    
    The resulting scalar is strictly clamped between -1.0 and 1.0.

    Args:
        df (pd.DataFrame): Fully parameterized dataset.

    Returns:
        pd.DataFrame: Frame yielding the 'sentiment_index' indicator.
    """
    # Normalize volatility to same scale as the other two signals
    vol_norm = df["volatility_7d"] / (df["volatility_7d"].max() + 1e-9)

    df["sentiment_index"] = (
        0.5 * df["momentum_score"]
      + 0.3 * df["daily_return"]
      - 0.2 * vol_norm
    ).clip(-1, 1)   # Clamp to [-1, 1] for clean display

    return df


def process_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the unified analytical pipeline sequence.

    Orchestrates sequential data cleansing, formatting routines, and comprehensive
    mathematical mutations aggregating standard OHLCV logs into high-value indicator sheets.

    Args:
        df (pd.DataFrame): Raw historical data payload.

    Returns:
        pd.DataFrame: System-ready master block structured for SQL ingestion or JSON serialization.
    """
    if df.empty:
        return df

    # --- Clean up ---
    df = df.dropna(subset=["Open", "Close", "High", "Low"])   # Drop corrupt rows
    df["Volume"] = df["Volume"].fillna(0).astype(int)
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # --- Required metrics ---
    df = add_daily_return(df)
    df = add_moving_average(df, window=7)
    df = add_52week_high_low(df)

    # --- Custom metrics ---
    df = add_volatility(df)
    df = add_momentum_score(df)
    df = add_sentiment_index(df)

    # Round floats for cleaner storage / JSON output
    float_cols = ["daily_return", "ma_7", "week52_high", "week52_low",
                  "volatility_7d", "momentum_score", "sentiment_index"]
    df[float_cols] = df[float_cols].round(6)

    return df


def get_summary(df: pd.DataFrame, symbol: str) -> dict:
    """
    Extract a high-level statistical snapshot for an isolated corporate symbol.

    Filters the master index to capture closing states alongside structural maximums,
    volatility footprints, and terminal deterministic sentiments.

    Args:
        df (pd.DataFrame): Master payload of historical rows.
        symbol (str): Targeted ticker notation.

    Returns:
        dict: Condensed associative array mapping metrics to values, alongside an
              evaluation label ('Bullish', 'Bearish', 'Neutral').
    """
    sym_df = df[df["Symbol"] == symbol].sort_values("Date")
    if sym_df.empty:
        return {}

    latest = sym_df.iloc[-1]
    return {
        "symbol":          symbol,
        "latest_close":    round(float(latest["Close"]), 2),
        "daily_return":    round(float(latest["daily_return"]) * 100, 3),   # in %
        "week52_high":     round(float(sym_df["Close"].max()), 2),
        "week52_low":      round(float(sym_df["Close"].min()), 2),
        "avg_close":       round(float(sym_df["Close"].mean()), 2),
        "volatility":      round(float(latest["volatility_7d"]), 4),
        "momentum":        round(float(latest["momentum_score"]), 4),
        "sentiment_index": round(float(latest["sentiment_index"]), 4),
        "sentiment_label": (
            "Bullish" if latest["sentiment_index"] > 0.1
            else "Bearish" if latest["sentiment_index"] < -0.1
            else "Neutral"
        ),
    }


def get_top_movers(df: pd.DataFrame, n: int = 3) -> dict:
    """
    Isolate extreme edge cases representing highest daily volatility segments.

    Calculates absolute delta rankings isolated to the final chronological instance 
    to pinpoint primary asset beneficiaries and primary asset laggards.

    Args:
        df (pd.DataFrame): Segmented historical datastore.
        n (int): Slice bounds per distribution side. Defaults to 3.

    Returns:
        dict: Top-level mapped dictionary containing {"gainers": list, "losers": list}.
    """
    latest_day = df.sort_values("Date").groupby("Symbol").tail(1)
    latest_day = latest_day.sort_values("daily_return", ascending=False)

    gainers = latest_day.head(n)[["Symbol", "Close", "daily_return"]].to_dict("records")
    losers  = latest_day.tail(n)[["Symbol", "Close", "daily_return"]].to_dict("records")

    # Convert to % and round
    for row in gainers + losers:
        row["daily_return"] = round(row["daily_return"] * 100, 3)

    return {"gainers": gainers, "losers": losers}
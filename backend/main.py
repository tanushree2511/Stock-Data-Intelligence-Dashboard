"""
main.py
-------
FastAPI application entry point.

This module houses all REST endpoints for the Stock Data Intelligence Dashboard.
It establishes the FastAPI lifecycle, defines the application routes, configures CORS,
and coordinates data ingestion, database management, and asynchronous scheduling.

Design Decisions:
    - Leveraging the lifespan context manager to hydrate the database on start-up.
    - Implementing thin controllers; all heavy data transformations and ML logic are
      delegated to downstream utility modules (data_processor.py, ml_predictor.py).
    - Permissive CORS for development; requires production lockdown.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sqlalchemy import and_
import pandas as pd
import os
from apscheduler.schedulers.background import BackgroundScheduler

from database       import init_db, upsert_stock_data, SessionLocal, StockPrice
from data_collector import fetch_all_stocks, DEFAULT_SYMBOLS, SYMBOL_NAMES
from data_processor import process_stock_data, get_summary, get_top_movers
from ml_predictor   import predict_next_n_days
from mock_data      import generate_all_mock_data


def load_stock_seed_data(period_days: int = 365) -> pd.DataFrame:
    """
    Load initial historical stock seed data from Yahoo Finance.

    Attempts to fetch data for the default tracking symbols over the specified
    period. If the external dependency is blocked or unavailable, it falls back
    to generating synthetically plausible mock data to ensure system stability.

    Args:
        period_days (int): The number of historical days to fetch. Defaults to 365.

    Returns:
        pd.DataFrame: A populated DataFrame with historical stock OHLCV data.
                      If fetching fails, returns synthetically generated tracking data.
    """
    try:
        raw_df = fetch_all_stocks(DEFAULT_SYMBOLS, period_days=period_days)
        if raw_df.empty:
            raise ValueError("Empty response from yfinance")
        print("  ✅ Live yfinance data loaded")
        return raw_df
    except Exception as e:
        print(f"  ⚠️  yfinance unavailable ({e}). Using mock data instead.")
        return generate_all_mock_data(DEFAULT_SYMBOLS, n_days=period_days)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Execute application startup and shutdown lifecycle events.

    Startup:
        - Initializes the database engine and tables.
        - Seeds the database if it is found to be empty.
        - Instantiates and registers the background APScheduler for nightly data refreshes.
    
    Shutdown:
        - Gracefully terminates the background task scheduler.

    Args:
        app (FastAPI): The ASGI application instance.
    """
    init_db()
    print("📦 DB initialised")

    with SessionLocal() as session:
        row_count = session.query(StockPrice).count()

    if row_count == 0:
        print("🌐 Fetching stock data (first run — takes ~30s)…")
        raw_df = load_stock_seed_data(period_days=365)
        processed_df = process_stock_data(raw_df)
        upsert_stock_data(processed_df)
        print(f"✅ Seeded {len(processed_df)} rows into DB")
    else:
        print(f"✅ DB already has data ({row_count} rows) — skipping fetch")

    # Start APScheduler for nightly data refresh
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_data, 'cron', hour=0, minute=0)
    scheduler.start()
    app.state.scheduler = scheduler

    yield   # Server is running
    app.state.scheduler.shutdown()
    print("👋 Shutting down")


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock Data Intelligence Dashboard",
    description="Mini financial data platform for NSE stocks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /frontend — access at http://localhost:8000/
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ── Helper ────────────────────────────────────────────────────────────────────

def db_rows_to_df(rows) -> pd.DataFrame:
    """
    Serialize synchronous SQLAlchemy ORM objects into a Pandas DataFrame.

    Useful for pushing persisted metrics back into analytical libraries that expect
    DataFrame structures (e.g., scikit-learn, chart pre-processing).

    Args:
        rows (List[StockPrice]): A collection of SQLAlchemy model instances.

    Returns:
        pd.DataFrame: A matrix matching the database schema where each dict key maps to a column.
    """
    return pd.DataFrame([{
        "Symbol":         r.symbol,
        "Date":           r.date,
        "Open":           r.open,
        "High":           r.high,
        "Low":            r.low,
        "Close":          r.close,
        "Volume":         r.volume,
        "daily_return":   r.daily_return,
        "ma_7":           r.ma_7,
        "week52_high":    r.week52_high,
        "week52_low":     r.week52_low,
        "volatility_7d":  r.volatility_7d,
        "momentum_score": r.momentum_score,
        "sentiment_index":r.sentiment_index,
    } for r in rows])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """
    Serve the static frontend entrypoint point.

    Attempts to locate and return the frontend 'index.html' file. If the file is 
    not found natively (e.g. absent directory mount), drops down to returning a generic 
    diagnostic JSON response denoting the API status.

    Returns:
        FileResponse or dict: The HTML payload or a fallback dictionary object.
    """
    index = os.path.join(frontend_path, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Stock Dashboard API running. Visit /docs for API reference."}


@app.get("/companies", tags=["Stocks"])
def get_companies():
    """
    Retrieve all available corporate tickers and their current snapshots.

    Queries the latest recorded closing metrics and computed sentiment for all default 
    tracked entries, providing a bird's-eye view of current market direction.

    Returns:
        dict: A structured dictionary mapping {"companies": list_of_dicts}
              where each dictionary contains 'symbol', 'name', 'latest_close', 
              'daily_return_pct', and a qualitative 'sentiment' string.
    """
    with SessionLocal() as session:
        # Get latest date per symbol
        results = []
        for symbol in DEFAULT_SYMBOLS:
            row = (session.query(StockPrice)
                   .filter(StockPrice.symbol == symbol)
                   .order_by(StockPrice.date.desc())
                   .first())
            if row:
                results.append({
                    "symbol":          symbol,
                    "name":            SYMBOL_NAMES.get(symbol, symbol),
                    "latest_close":    round(row.close, 2),
                    "daily_return_pct": round((row.daily_return or 0) * 100, 3),
                    "sentiment":       (
                        "Bullish" if (row.sentiment_index or 0) > 0.1
                        else "Bearish" if (row.sentiment_index or 0) < -0.1
                        else "Neutral"
                    ),
                })
    return {"companies": results}


@app.get("/data/{symbol}", tags=["Stocks"])
def get_stock_data(symbol: str, days: int = Query(default=30, ge=1, le=365)):
    """
    Retrieve granular OHLCV metrics along with downstream statistical indicators.

    Reads descending contiguous ranges from the data layer corresponding to the 
    specified ticker over the preceding `days` period.

    Args:
        symbol (str): The specific ticker identifier (e.g., 'TCS.NS').
        days (int): Period to track back from the current date. Max 365. Defaults to 30.

    Raises:
        HTTPException: If no data rows are available for the targeted symbol.

    Returns:
        dict: Embedded symbol metadata along with an array ordered from oldest to newest day.
    """
    with SessionLocal() as session:
        rows = (session.query(StockPrice)
                .filter(StockPrice.symbol == symbol)
                .order_by(StockPrice.date.desc())
                .limit(days)
                .all())

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    rows = list(reversed(rows))  # Oldest → Newest order for charts

    return {
        "symbol": symbol,
        "name":   SYMBOL_NAMES.get(symbol, symbol),
        "days":   len(rows),
        "data": [{
            "date":           str(r.date),
            "open":           r.open,
            "high":           r.high,
            "low":            r.low,
            "close":          r.close,
            "volume":         r.volume,
            "daily_return":   round((r.daily_return or 0) * 100, 3),
            "ma_7":           r.ma_7,
            "volatility_7d":  r.volatility_7d,
            "momentum_score": r.momentum_score,
            "sentiment_index":r.sentiment_index,
        } for r in rows]
    }


@app.get("/summary/{symbol}", tags=["Stocks"])
def get_stock_summary(symbol: str):
    """
    Extract aggregated summary metrics for a specific asset.

    Calculates cross-sectional properties across the fully persisted dataset for the
    requested ticker symbol, establishing 52-week boundaries, trend summaries, and sentiment averages.

    Args:
        symbol (str): The ticker symbol (e.g., 'INFY.NS').

    Raises:
        HTTPException: If the asset isn't tracked or the DB table fails constraints.

    Returns:
        dict: Condensed metrics suitable for rendering top-level KPI widgets.
    """
    with SessionLocal() as session:
        rows = (session.query(StockPrice)
                .filter(StockPrice.symbol == symbol)
                .all())

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = db_rows_to_df(rows)
    return get_summary(df, symbol)


@app.get("/compare", tags=["Stocks"])
def compare_stocks(
    symbol1: str = Query(..., example="INFY.NS"),
    symbol2: str = Query(..., example="TCS.NS"),
    days:    int = Query(default=30, ge=7, le=365),
):
    """
    Compute a statistically robust comparison between two respective assets.

    Executes a base-100 normalization on opening prices corresponding to the start of the defined
    period. Consequently, determines the Pearson correlation coefficient between the daily 
    return arrays to describe the behavioral relationship between the assets.

    Args:
        symbol1 (str): Base symbol identifier string.
        symbol2 (str): Secondary parallel symbol identifier string.
        days (int): Lookback duration constraint. Bounded [7, 365]. Defaults to 30.

    Raises:
        HTTPException: If either symbol returns empty datasets.

    Returns:
        dict: Multi-dimensional metrics containing the comparative normalized charts and core correlation scores.
    """
    results = {}
    for sym in [symbol1, symbol2]:
        with SessionLocal() as session:
            rows = (session.query(StockPrice)
                    .filter(StockPrice.symbol == sym)
                    .order_by(StockPrice.date.desc())
                    .limit(days)
                    .all())
        if not rows:
            raise HTTPException(status_code=404, detail=f"No data for {sym}")

        rows = list(reversed(rows))
        closes = [r.close for r in rows]
        base   = closes[0] if closes[0] else 1   # Normalise to 100

        results[sym] = {
            "name":   SYMBOL_NAMES.get(sym, sym),
            "dates":  [str(r.date) for r in rows],
            "closes": [round(c, 2) for c in closes],
            "normalised": [round((c / base) * 100, 3) for c in closes],
            "total_return_pct": round(((closes[-1] - closes[0]) / closes[0]) * 100, 3),
        }

    # Correlation of daily returns
    with SessionLocal() as session:
        r1 = (session.query(StockPrice)
              .filter(StockPrice.symbol == symbol1)
              .order_by(StockPrice.date.desc()).limit(days).all())
        r2 = (session.query(StockPrice)
              .filter(StockPrice.symbol == symbol2)
              .order_by(StockPrice.date.desc()).limit(days).all())

    min_len = min(len(r1), len(r2))
    ret1 = [r.daily_return for r in r1[:min_len]]
    ret2 = [r.daily_return for r in r2[:min_len]]
    correlation = round(float(pd.Series(ret1).corr(pd.Series(ret2))), 4)

    return {
        "period_days": days,
        "correlation": correlation,
        "correlation_label": (
            "Highly Correlated"  if correlation > 0.7
            else "Moderate"      if correlation > 0.3
            else "Low / Inverse"
        ),
        "stocks": results,
    }


@app.get("/movers", tags=["Insights"])
def top_movers():
    """
    Identify and extract the highest positive and negative relative movers.

    Evaluates the most chronologically robust record for all symbols to rank 
    day-over-day return percentages. Identifies top gainers and bottom losers.

    Raises:
        HTTPException: When core database arrays return fundamentally empty.

    Returns:
        dict: Output mapping corresponding arrays to their highest deviating symbols.
    """
    with SessionLocal() as session:
        rows = session.query(StockPrice).all()

    if not rows:
        raise HTTPException(status_code=404, detail="No data available")

    df = db_rows_to_df(rows)
    return get_top_movers(df)


@app.get("/predict/{symbol}", tags=["ML"])
def predict_prices(symbol: str, days: int = Query(default=7, ge=1, le=30)):
    """
    Extrapolate upcoming target asset values utilizing an ML predictor model.

    Generates forward-looking trajectory mapping (via a scikit-learn regression model)
    to yield trend estimation indices over the configured future period duration, while surfacing
    the previous month of hard values for integrated chart contextualization.

    Args:
        symbol (str): Equity identification string.
        days (int): Future prediction bounds. Confined [1, 30]. Defaults to 7.

    Raises:
        HTTPException: Invoked upon lookup failure or missing base symbol values.

    Returns:
        dict: Trailed actual metrics against extrapolated predictive time-series arrays.
    """
    with SessionLocal() as session:
        rows = (session.query(StockPrice)
                .filter(StockPrice.symbol == symbol)
                .order_by(StockPrice.date.asc())
                .all())

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = db_rows_to_df(rows)
    predictions = predict_next_n_days(symbol, df, n_days=days)

    # Also return last 30 actual closes for chart context
    recent = df.tail(30)
    actuals = [{"date": str(r["Date"]), "close": round(r["Close"], 2)}
               for _, r in recent.iterrows()]

    return {
        "symbol":      symbol,
        "name":        SYMBOL_NAMES.get(symbol, symbol),
        "actuals":     actuals,
        "predictions": predictions,
    }


@app.post("/refresh", tags=["Admin"])
def refresh_data():
    """
    Force a manual asynchronous rebuild pipeline.

    Invokes downstream polling structures to re-fetch upstream datasets, compute 
    extended indicator dimensions, and merge changes via UPSERT logic against the active database.

    Returns:
        dict: Execution status accompanied by overall mutation row counts.
    """
    print("🔄 Refreshing stock data…")
    raw_df = load_stock_seed_data(period_days=365)
    processed_df = process_stock_data(raw_df)
    upsert_stock_data(processed_df)
    return {"status": "ok", "rows_updated": len(processed_df)}
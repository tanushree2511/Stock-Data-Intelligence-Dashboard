"""
database.py
-----------
Persistence layer orchestration and SQLAlchemy configuration.

Manages connection pooling, relational declarative schemas, and transaction controls.
Designed fundamentally around SQLite for zero-configuration rapid local development,
seamlessly substitutable into robust DBMS environments (PostgreSQL/MySQL) by simply 
exchanging the underlying `DATABASE_URL` runtime environment variable.
"""

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Date, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./stocks.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class StockPrice(Base):
    """
    Core relational entity representing a unified snapshot of daily equity indicators.

    Corresponds to the `stock_prices` table. Captures primitive OHLCV boundaries alongside 
    derived, computationally heavy metrics (moving averages, bounded sentiment indices) 
    to prevent latency during runtime dashboard requests.

    Attributes:
        id (int): Primary clustering index.
        symbol (str): Associated equity identifying string.
        date (datetime.date): Chronological record mapping point.
        open (float): Opening valuation.
        high (float): Daily maximum boundary.
        low (float): Daily minimum boundary.
        close (float): Absolute closing termination.
        volume (int): Transacted scale count.
        daily_return (float): Interday delta percentage.
        ma_7 (float): Smoothing rolling numerical average.
        week52_high (float): Ceiling proxy boundary.
        week52_low (float): Floor proxy boundary.
        volatility_7d (float): Local risk standard deviation metric.
        momentum_score (float): Target offset against an established MA index.
        sentiment_index (float): Abstract clamped mood weighting index.
    """
    __tablename__ = "stock_prices"

    id             = Column(Integer, primary_key=True, index=True)
    symbol         = Column(String,  nullable=False)
    date           = Column(Date,    nullable=False)
    open           = Column(Float)
    high           = Column(Float)
    low            = Column(Float)
    close          = Column(Float)
    volume         = Column(Integer)

    # Computed metrics
    daily_return   = Column(Float)
    ma_7           = Column(Float)
    week52_high    = Column(Float)
    week52_low     = Column(Float)
    volatility_7d  = Column(Float)
    momentum_score = Column(Float)
    sentiment_index= Column(Float)

    # Composite index on (symbol, date) for fast per-symbol range queries
    __table_args__ = (
        Index("ix_symbol_date", "symbol", "date"),
    )


def init_db():
    """
    Synchronously initialize connection engines and reflect metadata layouts.

    Triggers iterative DDL CREATE TABLE executions representing the `Base` declarative states. 
    Typically hooked into application startup lifecycle routines.
    """
    Base.metadata.create_all(bind=engine)


def upsert_stock_data(df):
    """
    Execute block collision-resistant bulk insert routines into the local datastore.

    Resolves duplication vectors natively associated with repetitive scheduler fetches. 
    Operates sequentially by identifying distinct overlapping constraints (`symbol` AND `date`),
    terminating conflicting rows prior to bulk transaction commit phases. Functionally equivalent 
    to complex DBMS MERGE operations handled natively via Python abstractions.

    Args:
        df (pd.DataFrame): System-ready master payload block requiring persistence.
    """
    from sqlalchemy import and_

    with SessionLocal() as session:
        for _, row in df.iterrows():
            # Delete existing row for this symbol+date if present
            session.query(StockPrice).filter(
                and_(
                    StockPrice.symbol == row["Symbol"],
                    StockPrice.date   == row["Date"]
                )
            ).delete()

            record = StockPrice(
                symbol         = row["Symbol"],
                date           = row["Date"],
                open           = row["Open"],
                high           = row["High"],
                low            = row["Low"],
                close          = row["Close"],
                volume         = int(row["Volume"]),
                daily_return   = row["daily_return"],
                ma_7           = row["ma_7"],
                week52_high    = row["week52_high"],
                week52_low     = row["week52_low"],
                volatility_7d  = row["volatility_7d"],
                momentum_score = row["momentum_score"],
                sentiment_index= row["sentiment_index"],
            )
            session.add(record)

        session.commit()


@contextmanager
def get_db():
    """
    Yield an isolated logical transaction envelope spanning a distinct request boundary.

    Operates as an integrated Dependency Injection target serving HTTP endpoints,
    gracefully closing transient pipelines natively upon the context exit block 
    ensuring thread pool health and preventing systemic lock conditions.

    Yields:
        Session: Bound SQLAlchemy session logic block.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"""
ml_predictor.py
---------------
Predictive modeling pipeline utilizing Scikit-Learn.

Implements a lightweight, rapid, interpretative Linear Regression trend mapping 
algorithm designed specifically to forecast upcoming 7-day trajectories based 
on lagged feature matrices.

Design Rationale:
    Leverages foundational mathematical regression rather than deep recurrent matrices 
    (LSTM) or rigorous time-series libraries (Prophet) to prioritize CPU execution 
    speed and stateless deployment over absolute predictive accuracy.

Core Feature Matrix:
    - Chronological index offset
    - Rolling 7-day average valuation
    - Day-Minus-1 (Lag 1) daily relative return
    - Medium-term scalar momentum score
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

MODEL_DIR = "models_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_cache_model(symbol: str, df: pd.DataFrame):
    """
    Train and serialize a localized regression model mapped to a specific equity.

    Isolates lagging features from the historical dataset, scales the absolute 
    closing metric via a generic MinMaxScaler to optimize coefficient distribution, 
    and establishes a fitted analytical model. Output binaries are dumped locally 
    to prevent repetitive execution drag on the server.

    Args:
        symbol (str): Identification notation for the equity instrument.
        df (pd.DataFrame): Time-series master frame. Must exceed 30 entries.

    Returns:
        tuple[LinearRegression, MinMaxScaler] | tuple[None, None]:
            Returns the initialized model instance and scaling artifact, or
            None if data volume evaluates beneath the minimal training bounds.
    """
    if len(df) < 30:
        return None, None

    df = df.copy()
    df["day_index"]    = np.arange(len(df))
    df["return_lag1"]  = df["daily_return"].shift(1).fillna(0)
    df["ma_7_filled"]  = df["ma_7"].fillna(df["Close"])

    feature_cols = ["day_index", "ma_7_filled", "return_lag1", "momentum_score"]
    
    scaler = MinMaxScaler()
    df["close_scaled"] = scaler.fit_transform(df[["Close"]])

    X = df[feature_cols].values
    y = df["close_scaled"].values

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{symbol}_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib"))

    return model, scaler

def predict_next_n_days(symbol: str, symbol_df: pd.DataFrame, n_days: int = 7) -> list[dict]:
    """
    Extrapolate sequential predictions executing forward loop passes.

    Reconstructs the model from serialized cache files (or triggers a localized
    training instance if not found), initializing a dynamic prediction matrix 
    derived from the instrument's final chronological data block. Iteratively 
    solves the deterministic linear function bounded by `n_days`.

    Args:
        symbol (str): Active ticker symbol.
        symbol_df (pd.DataFrame): Primary metric history dataset.
        n_days (int): Future temporal bounds indicating the extrapolation limit. defaults to 7.

    Returns:
        list[dict]: Associated matrix list defining predicted dates against output closing prices.
    """
    df = symbol_df.sort_values("Date").copy()

    if len(df) < 30:
        return []

    # Try loading cached models
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        model, scaler = train_and_cache_model(symbol, df)
        if model is None:
            return []

    # ---- Build future feature rows ----
    df["day_index"]    = np.arange(len(df))                          
    df["return_lag1"]  = df["daily_return"].shift(1).fillna(0)       
    df["ma_7_filled"]  = df["ma_7"].fillna(df["Close"])

    last_idx     = int(df["day_index"].iloc[-1])
    last_ma7     = float(df["ma_7_filled"].iloc[-1])
    last_return  = float(df["daily_return"].iloc[-1])
    last_momentum= float(df["momentum_score"].iloc[-1])

    predictions  = []
    last_date    = pd.Timestamp(df["Date"].iloc[-1])

    for i in range(1, n_days + 1):
        future_idx = last_idx + i
        X_future = np.array([[future_idx, last_ma7, last_return, last_momentum]])
        y_pred_scaled = model.predict(X_future)[0]
        price = float(scaler.inverse_transform([[y_pred_scaled]])[0][0])
        next_date = last_date + pd.tseries.offsets.BDay(i)

        predictions.append({
            "date":            next_date.strftime("%Y-%m-%d"),
            "predicted_close": round(price, 2),
        })

    return predictions
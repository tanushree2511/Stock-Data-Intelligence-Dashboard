import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from data_processor import add_daily_return, add_moving_average, process_stock_data

def test_add_daily_return():
    df = pd.DataFrame({
        "Symbol": ["TCS.NS", "TCS.NS"],
        "Date": ["2024-01-01", "2024-01-02"],
        "Open": [100.0, 110.0],
        "Close": [105.0, 108.0]
    })
    
    result = add_daily_return(df)
    
    assert "daily_return" in result.columns
    assert np.isclose(result["daily_return"].iloc[0], 0.05) # (105-100)/100
    assert np.isclose(result["daily_return"].iloc[1], -0.01818181818) # (108-110)/110

def test_add_moving_average():
    df = pd.DataFrame({
        "Symbol": ["INFY", "INFY", "INFY"],
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Close": [10.0, 20.0, 30.0]
    })
    
    result = add_moving_average(df, window=2)
    assert "ma_2" in result.columns
    assert np.isnan(result["ma_2"].iloc[0]) or result["ma_2"].iloc[0] == 10.0  # min_periods=1 makes it 10
    assert result["ma_2"].iloc[1] == 15.0 # (10+20)/2
    assert result["ma_2"].iloc[2] == 25.0 # (20+30)/2

def test_process_stock_data_empty():
    df = pd.DataFrame()
    res = process_stock_data(df)
    assert res.empty

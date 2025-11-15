import sys
import types
# Prevent importing the real pandas_ta (causes ImportError in this env). Provide a minimal stub.
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))

import pandas as pd
import numpy as np
import math
from numpy.testing import assert_allclose, assert_array_equal
import pytest

# try package-style import (when running from repo root) then fallback to local import (when running from server/)
try:
    from server.swing_buy_recommender import synthesize_features
except Exception:
    from swing_buy_recommender import synthesize_features

def test_numeric_cast_and_basic_fields():
    df = pd.DataFrame([{
        "ADX": "12",
        "RSI": "50",
        "EMA20": "10",
        "EMA50": "9",
        "SMA200": "8",
        "TodayPrice": "12",
        "SignalClassifier_Rules": "1",
        "SignalClassifier_ML": "0",
        "LastTrendDays": "5",
        "Volume": "200000",
        "DITrend": "Bullish"
    }])
    out = synthesize_features(df)
    # numeric casts (accept numpy integer/floating types)
    assert isinstance(out.loc[0, "ADX"], (int, float, np.integer, np.floating))
    assert_allclose(float(out.loc[0, "ADX"]), 12.0)
    assert_allclose(float(out.loc[0, "RSI"]), 50.0)
    # ema_struct should be 2 (10>9>8)
    assert out.loc[0, "ema_struct"] == 2
    # price_over_ema20 = 12 / 10 = 1.2
    assert_allclose(float(out.loc[0, "price_over_ema20"]), 1.2)
    # classifier_consensus = 1 + 0 = 1
    assert out.loc[0, "classifier_consensus"] == 1
    # LastTrendDays -> int-like
    assert isinstance(out.loc[0, "LastTrendDays"], (int, np.integer))
    assert int(out.loc[0, "LastTrendDays"]) == 5
    # Volume numeric
    assert_allclose(float(out.loc[0, "Volume"]), 200000.0)
    # DITrend_is_bull True (accept numpy bool_)
    assert bool(out.loc[0, "DITrend_is_bull"]) is True

def test_ema_struct_assignments():
    df = pd.DataFrame([
        {"EMA20": 30, "EMA50": 20, "SMA200": 10},   # EMA20>EMA50>SMA200 => 2
        {"EMA20": 30, "EMA50": 20, "SMA200": 40},   # EMA20>EMA50 but not >SMA200 => 1
        {"EMA20": 20, "EMA50": 30, "SMA200": 10},   # EMA50>EMA20 => -1
    ])
    out = synthesize_features(df)
    assert_array_equal(out["ema_struct"].values, np.array([2.0, 1.0, -1.0]))

def test_boolean_is_bull_flags():
    df = pd.DataFrame([
        {"DITrend": "Bullish rally", "MA_Trend": "uptrend", "MADI_Trend": "positive", "Primary": "long-term bull"},
        {"DITrend": "bearish", "MA_Trend": "", "Primary": None}
    ])
    out = synthesize_features(df)
    # Row 0 should have all *_is_bull True
    assert bool(out.loc[0, "DITrend_is_bull"]) is True
    assert bool(out.loc[0, "MA_Trend_is_bull"]) is True
    assert bool(out.loc[0, "MADI_Trend_is_bull"]) is True
    assert bool(out.loc[0, "Primary_is_bull"]) is True
    # Row 1 should have DITrend_is_bull False and Primary_is_bull False
    assert bool(out.loc[1, "DITrend_is_bull"]) is False
    assert bool(out.loc[1, "Primary_is_bull"]) is False

def test_missing_columns_defaults():
    # Provide DataFrame with no numeric columns
    df = pd.DataFrame([{"Symbol": "X"}])
    out = synthesize_features(df)
    # ema_struct exists and should be NaN
    assert "ema_struct" in out.columns
    assert math.isnan(out.loc[0, "ema_struct"])
    # price_over_ema20 exists and should be NaN
    assert "price_over_ema20" in out.columns
    assert (pd.isna(out.loc[0, "price_over_ema20"]))
    # classifier_consensus default 0
    assert out.loc[0, "classifier_consensus"] == 0
    # LastTrendDays default 0
    assert out.loc[0, "LastTrendDays"] == 0
    # Volume default 0
    assert out.loc[0, "Volume"] == 0

if __name__ == "__main__":
    pytest.main([__file__])
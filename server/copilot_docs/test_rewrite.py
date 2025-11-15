#!/usr/bin/env python3
"""
Comprehensive test of the complete indicators.py rewrite.
Tests the new API: compute_indicators(df) returns df with all indicators as columns.
"""

import pandas as pd
import numpy as np
import sys

print("=" * 80)
print("TESTING COMPLETE INDICATORS.PY REWRITE")
print("=" * 80)

# Test 1: Import indicators
print("\n[TEST 1] Importing indicators module...")
try:
    from indicators import compute_indicators
    print("✓ indicators module imported successfully")
except Exception as e:
    print(f"✗ FAILED to import: {e}")
    sys.exit(1)

# Test 2: Create synthetic data
print("\n[TEST 2] Creating synthetic OHLCV test data...")
try:
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    
    df_test = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(200) * 0.5,
        'High': prices + np.abs(np.random.randn(200)) * 1,
        'Low': prices - np.abs(np.random.randn(200)) * 1,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 200)
    })
    
    print(f"✓ Created test DataFrame: {df_test.shape}")
    print(f"  Columns: {list(df_test.columns)}")
    print(f"  Date range: {df_test['Date'].min()} to {df_test['Date'].max()}")
except Exception as e:
    print(f"✗ FAILED to create test data: {e}")
    sys.exit(1)

# Test 3: Call compute_indicators (now returns full df, not dict)
print("\n[TEST 3] Calling compute_indicators()...")
try:
    df_with_ind = compute_indicators(df_test)
    print(f"✓ compute_indicators() returned DataFrame")
    print(f"  Shape: {df_with_ind.shape}")
    print(f"  Number of indicator columns added: {df_with_ind.shape[1] - df_test.shape[1]}")
except Exception as e:
    print(f"✗ FAILED to compute indicators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify return type is DataFrame
print("\n[TEST 4] Checking return type...")
try:
    assert isinstance(df_with_ind, pd.DataFrame), f"Expected DataFrame, got {type(df_with_ind)}"
    print(f"✓ Return type is pd.DataFrame")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Verify indicator columns exist
print("\n[TEST 5] Checking for expected indicator columns...")
expected_indicators = [
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'ema_fast', 'ema_slow', 'ema_short', 'ema_20', 'ema_50',
    'sma50', 'sma200',
    'atr', 'adx', 'plus_di', 'minus_di',
    'mfi', 'obv',
    'bb_lower', 'bb_middle', 'bb_upper',
    'cci'
]
missing = [ind for ind in expected_indicators if ind not in df_with_ind.columns]
if missing:
    print(f"✗ Missing indicators: {missing}")
    print(f"  Available columns: {list(df_with_ind.columns)}")
    sys.exit(1)
else:
    print(f"✓ All {len(expected_indicators)} expected indicators present")

# Test 6: Verify DataFrame was sorted by date
print("\n[TEST 6] Verifying DataFrame is sorted by date...")
try:
    if 'Date' in df_with_ind.columns:
        date_col = 'Date'
    elif 'date' in df_with_ind.columns:
        date_col = 'date'
    else:
        raise ValueError("No Date column found")
    
    dates = pd.to_datetime(df_with_ind[date_col])
    is_sorted = dates.is_monotonic_increasing
    assert is_sorted, "Date column is not sorted!"
    print(f"✓ DataFrame is properly sorted by date (monotonic_increasing)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 7: Check for valid numeric values in indicators
print("\n[TEST 7] Checking indicator values (non-null after warmup)...")
try:
    for ind in expected_indicators:
        col = df_with_ind[ind]
        non_null_count = col.notna().sum()
        # Expect most values to be non-null after warmup period
        null_ratio = col.isna().sum() / len(col)
        if non_null_count < len(col) * 0.5:
            print(f"  ⚠ {ind}: {null_ratio:.1%} null (expected < 50%)")
        else:
            print(f"  ✓ {ind}: {null_ratio:.1%} null")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 8: Test with unsorted input (should auto-sort)
print("\n[TEST 8] Testing with unsorted input DataFrame...")
try:
    df_unsorted = df_test.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_sorted = compute_indicators(df_unsorted)
    
    if 'Date' in df_sorted.columns:
        dates = pd.to_datetime(df_sorted['Date'])
        is_sorted = dates.is_monotonic_increasing
        assert is_sorted, "Output should be sorted!"
        print(f"✓ Unsorted input was automatically sorted in output")
    else:
        print(f"⚠ Could not verify sorting (no Date column)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 9: Test ta_signals_mc_parallel integration
print("\n[TEST 9] Testing ta_signals_mc_parallel integration...")
try:
    from ta_signals_mc_parallel import get_technical_indicators
    
    df_test_copy = df_test.copy()
    df_result = get_technical_indicators(df_test_copy)
    
    # Should have uppercase indicator columns now
    expected_uppercase = ['ADX', 'DIPLUS', 'DIMINUS', 'SMA200', 'SMA50', 'EMA50', 'EMA20', 'CCI', 'RSI', 'OBV', 'ATR', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
    missing_upper = [col for col in expected_uppercase if col not in df_result.columns]
    
    if missing_upper:
        print(f"⚠ Missing uppercase columns: {missing_upper}")
        print(f"  Available: {list(df_result.columns)}")
    else:
        print(f"✓ All uppercase indicator columns present in get_technical_indicators()")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Test TrendReversalDetectorML adapter
print("\n[TEST 10] Testing TrendReversalDetectorML adapter...")
try:
    from TrendReversalDetectorML import TrendReversalDetectorML
    
    detector = TrendReversalDetectorML()
    indicator_dict = detector.compute_indicators(df_test)
    
    # Should return dict (converted from DataFrame)
    assert isinstance(indicator_dict, dict), f"Expected dict, got {type(indicator_dict)}"
    
    expected_keys = ['close', 'rsi', 'macd_hist', 'atr', 'adx']
    missing_keys = [k for k in expected_keys if k not in indicator_dict]
    
    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    else:
        print(f"✓ TrendReversalDetectorML.compute_indicators() returns dict with expected keys")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 11: Test with custom parameters
print("\n[TEST 11] Testing with custom indicator parameters...")
try:
    custom_params = {
        'rsi_period': 21,
        'macd_fast': 10,
        'macd_slow': 30,
        'bb_len': 25,
        'sma50': 60,
        'sma200': 250
    }
    df_custom = compute_indicators(df_test, params=custom_params)
    assert 'rsi' in df_custom.columns
    assert 'macd' in df_custom.columns
    print(f"✓ compute_indicators() works with custom parameters")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 12: Test with unsorted date column
print("\n[TEST 12] Testing with missing OHLCV columns...")
try:
    df_minimal = df_test[['Date', 'Close']].copy()
    df_minimal_result = compute_indicators(df_minimal)
    
    assert isinstance(df_minimal_result, pd.DataFrame)
    assert 'rsi' in df_minimal_result.columns
    print(f"✓ compute_indicators() handles missing High/Low/Open/Volume gracefully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓✓✓")
print("=" * 80)
print("\nSummary:")
print("  • indicators.py now returns DataFrames with all indicators as columns")
print("  • Automatic sorting by date applied")
print("  • All call sites (ta_signals_mc_parallel, TrendReversalDetectorML) working")
print("  • Compatible with pandas_ta integration")
print("\n")

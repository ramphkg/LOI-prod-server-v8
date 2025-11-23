#!/usr/bin/env python3
"""
Test script for ML_Predict_Price.py
Tests with both synthetic realistic data and actual database data
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Test 1: Synthetic realistic data
print("=" * 70)
print("TEST 1: Synthetic Realistic Price Data")
print("=" * 70)

# Create more realistic synthetic data with trend and volatility
np.random.seed(42)
dates = pd.date_range(end=pd.Timestamp.now(), periods=150, freq='D')

# Generate realistic price movement with random walk
base_price = 100.0
returns = np.random.normal(0.001, 0.02, 150)  # ~0.1% daily return, 2% volatility
prices = base_price * np.exp(np.cumsum(returns))

# Create OHLCV data
opens = prices * np.random.uniform(0.99, 1.01, 150)
highs = np.maximum(opens, prices) * np.random.uniform(1.0, 1.02, 150)
lows = np.minimum(opens, prices) * np.random.uniform(0.98, 1.0, 150)
closes = prices
volumes = np.random.randint(1000000, 10000000, 150)

synthetic_df = pd.DataFrame({
    'Date': dates,
    'Open': opens,
    'High': highs,
    'Low': lows,
    'Close': closes,
    'Volume': volumes
})

print(f"Data shape: {synthetic_df.shape}")
print(f"Date range: {synthetic_df['Date'].min().date()} to {synthetic_df['Date'].max().date()}")
print(f"\nPrice statistics:")
print(f"  Start price: ${synthetic_df['Close'].iloc[0]:.2f}")
print(f"  End price: ${synthetic_df['Close'].iloc[-1]:.2f}")
print(f"  Min price: ${synthetic_df['Close'].min():.2f}")
print(f"  Max price: ${synthetic_df['Close'].max():.2f}")
print(f"  Avg daily volume: {synthetic_df['Volume'].mean():,.0f}")

# Test prediction
from ML_Predict_Price import predict_next_close

print("\n" + "-" * 70)
print("Running prediction...")
print("-" * 70)

predicted = predict_next_close(synthetic_df, confidence_threshold=0.5)

if predicted is not None:
    current = synthetic_df['Close'].iloc[-1]
    change = predicted - current
    change_pct = (change / current) * 100
    print(f"✓ SUCCESS!")
    print(f"  Current close: ${current:.2f}")
    print(f"  Predicted close: ${predicted:.2f}")
    print(f"  Expected change: ${change:+.2f} ({change_pct:+.2f}%)")
else:
    print("✗ FAILED - Prediction returned None")

# Test 2: Edge cases
print("\n" + "=" * 70)
print("TEST 2: Edge Cases")
print("=" * 70)

# Test with insufficient data
print("\n2a. Insufficient data (20 days - below minimum 30):")
small_df = synthetic_df.tail(20).copy()
result = predict_next_close(small_df)
print(f"   Result: {result} (Expected: None)")

# Test with exactly minimum data
print("\n2b. Minimum data (30 days):")
min_df = synthetic_df.tail(30).copy()
result = predict_next_close(min_df, confidence_threshold=0.4)
if result is not None:
    print(f"   Result: ${result:.2f} (Prediction successful)")
else:
    print(f"   Result: None (Model confidence too low)")

# Test with optimal data
print("\n2c. Optimal data (100+ days):")
optimal_df = synthetic_df.tail(120).copy()
result = predict_next_close(optimal_df, confidence_threshold=0.5)
if result is not None:
    print(f"   Result: ${result:.2f} (Prediction successful)")
else:
    print(f"   Result: None (Model confidence too low)")

# Test 3: Database data (if available)
print("\n" + "=" * 70)
print("TEST 3: Real Database Data")
print("=" * 70)

try:
    from app_imports import getDbConnection
    from sqlalchemy import text
    
    # Try to get real data from database
    with getDbConnection() as con:
        # First check which table has data
        tables_to_check = ['finnhub_stock_prices', 'eod_stock_prices']
        
        for table in tables_to_check:
            try:
                # Get a symbol with good data
                q = text(f"""
                    SELECT symbol, COUNT(*) as cnt 
                    FROM {table} 
                    GROUP BY symbol 
                    HAVING cnt >= 100 
                    LIMIT 1
                """)
                result = con.execute(q)
                row = result.fetchone()
                
                if row:
                    test_symbol = row[0]
                    print(f"\nTesting with symbol '{test_symbol}' from {table}")
                    
                    # Get price data
                    q = text(f"""
                        SELECT date AS Date, open AS Open, high AS High, 
                               low AS Low, close AS Close, volume AS Volume
                        FROM {table}
                        WHERE symbol = :symbol
                        ORDER BY date ASC
                    """)
                    df = pd.read_sql(q, con=con, params={"symbol": test_symbol})
                    
                    print(f"  Rows: {len(df)}")
                    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
                    print(f"  Current price: ${df['Close'].iloc[-1]:.2f}")
                    
                    # Make prediction
                    predicted = predict_next_close(df, confidence_threshold=0.5)
                    
                    if predicted is not None:
                        current = df['Close'].iloc[-1]
                        change_pct = ((predicted - current) / current) * 100
                        print(f"  ✓ Predicted price: ${predicted:.2f} ({change_pct:+.2f}%)")
                    else:
                        print(f"  ✗ Prediction returned None")
                    
                    break
            except Exception as e:
                print(f"  Could not test {table}: {e}")
                continue
        else:
            print("  No suitable test data found in database")
            
except Exception as e:
    print(f"  Could not access database: {e}")

# Test 4: Column name variations
print("\n" + "=" * 70)
print("TEST 4: Column Name Variations (case-insensitive)")
print("=" * 70)

# Test with lowercase column names
print("\n4a. Lowercase column names:")
lowercase_df = synthetic_df.copy()
lowercase_df.columns = [col.lower() for col in lowercase_df.columns]
result = predict_next_close(lowercase_df, confidence_threshold=0.5)
if result is not None:
    print(f"   ✓ Works with lowercase: ${result:.2f}")
else:
    print(f"   ✗ Failed with lowercase")

# Test with uppercase column names
print("\n4b. Uppercase column names:")
uppercase_df = synthetic_df.copy()
uppercase_df.columns = [col.upper() for col in uppercase_df.columns]
result = predict_next_close(uppercase_df, confidence_threshold=0.5)
if result is not None:
    print(f"   ✓ Works with uppercase: ${result:.2f}")
else:
    print(f"   ✗ Failed with uppercase")

print("\n" + "=" * 70)
print("TESTS COMPLETE")
print("=" * 70)

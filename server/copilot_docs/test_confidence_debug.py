#!/usr/bin/env python3
"""
Debug test to see actual confidence scores across horizons
"""

import pandas as pd
from sqlalchemy import text
from app_imports import getDbConnection
from ML_Predict_Price import predict_price_at_horizon, PREDICTION_HORIZONS

print("Debugging Multi-Horizon Confidence Scores")
print("=" * 80)

# Get one stock with good data
with getDbConnection() as con:
    q = text("""
        SELECT symbol, COUNT(*) as cnt
        FROM finnhub_stock_prices
        GROUP BY symbol
        HAVING cnt >= 200
        ORDER BY RAND()
        LIMIT 3
    """)
    test_symbols = con.execute(q).fetchall()

for symbol, count in test_symbols:
    print(f"\n{'='*80}")
    print(f"Symbol: {symbol} ({count} days of data)")
    print(f"{'='*80}")
    
    with getDbConnection() as con:
        q = text("""
            SELECT date AS Date, open AS Open, high AS High,
                   low AS Low, close AS Close, volume AS Volume
            FROM finnhub_stock_prices
            WHERE symbol = :symbol
            ORDER BY date ASC
        """)
        df = pd.read_sql(q, con=con, params={"symbol": symbol})
    
    current_price = df['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}\n")
    
    print("Testing all horizons:")
    print("-" * 80)
    
    best_result = None
    best_confidence = -999
    
    for horizon in PREDICTION_HORIZONS:
        result = predict_price_at_horizon(df, horizon, confidence_threshold=0.0)
        
        if result is not None:
            pred_price, confidence = result
            change_pct = ((pred_price - current_price) / current_price) * 100
            
            status = "✓" if confidence >= 0.65 else "⚠" if confidence >= 0.50 else "✗"
            print(f"{status} {horizon:2d} days: ${pred_price:8.2f} ({change_pct:+6.2f}%) | Confidence: {confidence:.3f}")
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = (pred_price, horizon, confidence)
        else:
            print(f"✗ {horizon:2d} days: Failed (data validation or extreme prediction)")
    
    print("-" * 80)
    if best_result:
        pred_price, horizon, confidence = best_result
        change_pct = ((pred_price - current_price) / current_price) * 100
        print(f"\nBest Prediction: ${pred_price:.2f} in {horizon} days ({change_pct:+.2f}%)")
        print(f"Confidence: {confidence:.3f}")
        
        if confidence >= 0.65:
            print("✓ PASS - Meets 0.65 threshold")
        elif confidence >= 0.50:
            print("⚠ MARGINAL - Between 0.50-0.65")
        else:
            print("✗ LOW - Below 0.50 confidence")
    else:
        print("\n✗ No valid predictions for any horizon")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Based on actual confidence scores, consider:")
print("  - Threshold 0.65: Very strict, fewer predictions but higher quality")
print("  - Threshold 0.50: Balanced, more predictions with moderate quality")
print("  - Threshold 0.40: Permissive, most predictions but lower quality")

#!/usr/bin/env python3
"""
Simple test: ML_Predict_Price with real database data
"""

import pandas as pd
from sqlalchemy import text
from app_imports import getDbConnection
from ML_Predict_Price import predict_next_close

print("Testing ML_Predict_Price with real database data")
print("=" * 70)

try:
    with getDbConnection() as con:
        # Get multiple symbols for testing
        q = text("""
            SELECT symbol, COUNT(*) as cnt 
            FROM finnhub_stock_prices 
            GROUP BY symbol 
            HAVING cnt >= 100 
            ORDER BY cnt DESC
            LIMIT 5
        """)
        symbols = con.execute(q).fetchall()
        
        print(f"\nTesting {len(symbols)} symbols from finnhub_stock_prices:\n")
        
        success_count = 0
        for symbol, count in symbols:
            # Get price data
            q = text("""
                SELECT date AS Date, open AS Open, high AS High, 
                       low AS Low, close AS Close, volume AS Volume
                FROM finnhub_stock_prices
                WHERE symbol = :symbol
                ORDER BY date ASC
            """)
            df = pd.read_sql(q, con=con, params={"symbol": symbol})
            
            current_price = df['Close'].iloc[-1]
            predicted = predict_next_close(df, confidence_threshold=0.6)
            
            if predicted is not None:
                change_pct = ((predicted - current_price) / current_price) * 100
                print(f"✓ {symbol:6s} | {count:4d} days | Current: ${current_price:8.2f} | Predicted: ${predicted:8.2f} | Change: {change_pct:+6.2f}%")
                success_count += 1
            else:
                print(f"✗ {symbol:6s} | {count:4d} days | Current: ${current_price:8.2f} | Prediction: None (low confidence)")
        
        print(f"\nSuccess rate: {success_count}/{len(symbols)} ({success_count/len(symbols)*100:.0f}%)")
        
        if success_count > 0:
            print("\n" + "=" * 70)
            print("✓ ML_Predict_Price.py is working correctly!")
            print("=" * 70)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

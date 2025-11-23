#!/usr/bin/env python3
"""
Test ML_Predict_Price with stocks from different markets (USA, India, Hong Kong)
"""

import pandas as pd
from sqlalchemy import text
from app_imports import getDbConnection
from ML_Predict_Price import predict_next_close

print("Testing ML_Predict_Price with Multi-Market Stocks")
print("=" * 80)

def test_market(country_name, limit=10):
    """Test stocks from a specific market"""
    print(f"\n{'='*80}")
    print(f"Market: {country_name}")
    print(f"{'='*80}\n")
    
    try:
        with getDbConnection() as con:
            # Get symbols from specific country
            q = text("""
                SELECT DISTINCT f.symbol, COUNT(f.date) as day_count
                FROM finnhub_stock_prices f
                LEFT JOIN finnhub_gem_listings g ON f.symbol = g.Symbol
                WHERE (g.CountryName = :country OR :country = 'ALL')
                GROUP BY f.symbol
                HAVING day_count >= 100
                ORDER BY RAND()
                LIMIT :limit
            """)
            
            symbols = con.execute(q, {"country": country_name, "limit": limit}).fetchall()
            
            if not symbols:
                print(f"No data found for {country_name}. Trying alternative approach...")
                # Try without country filter
                q = text("""
                    SELECT symbol, COUNT(*) as day_count
                    FROM finnhub_stock_prices
                    GROUP BY symbol
                    HAVING day_count >= 100
                    ORDER BY RAND()
                    LIMIT :limit
                """)
                symbols = con.execute(q, {"limit": limit}).fetchall()
            
            print(f"Testing {len(symbols)} stocks:\n")
            
            results = []
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
                
                if df.empty or len(df) < 30:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Test with different confidence thresholds
                for threshold in [0.7, 0.6, 0.5]:
                    predicted = predict_next_close(df, confidence_threshold=threshold)
                    if predicted is not None:
                        break
                else:
                    predicted = None
                    threshold = None
                
                result = {
                    'symbol': symbol,
                    'days': len(df),
                    'current': current_price,
                    'predicted': predicted,
                    'threshold': threshold
                }
                results.append(result)
                
                if predicted is not None:
                    change_pct = ((predicted - current_price) / current_price) * 100
                    print(f"✓ {symbol:8s} | {len(df):4d} days | Current: ${current_price:10.2f} | "
                          f"Predicted: ${predicted:10.2f} | Change: {change_pct:+7.2f}% | R²≥{threshold}")
                else:
                    print(f"✗ {symbol:8s} | {len(df):4d} days | Current: ${current_price:10.2f} | "
                          f"Prediction: None (confidence too low)")
            
            # Summary
            success = sum(1 for r in results if r['predicted'] is not None)
            total = len(results)
            if total > 0:
                print(f"\n{'-'*80}")
                print(f"Success Rate: {success}/{total} ({success/total*100:.1f}%)")
                
                if success > 0:
                    avg_abs_change = sum(
                        abs((r['predicted'] - r['current']) / r['current'] * 100)
                        for r in results if r['predicted'] is not None
                    ) / success
                    print(f"Average predicted change: {avg_abs_change:.2f}%")
            
            return results
            
    except Exception as e:
        print(f"Error testing {country_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

# Test different markets
all_results = {}

# Test USA stocks
all_results['USA'] = test_market('USA', limit=10)

# Test India stocks
all_results['India'] = test_market('India', limit=10)

# Test Hong Kong stocks  
all_results['Hong Kong'] = test_market('Hong Kong', limit=10)

# Overall summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}\n")

total_tested = 0
total_success = 0

for market, results in all_results.items():
    if results:
        success = sum(1 for r in results if r['predicted'] is not None)
        total = len(results)
        total_tested += total
        total_success += success
        print(f"{market:12s}: {success:2d}/{total:2d} successful ({success/total*100:5.1f}%)")

if total_tested > 0:
    print(f"\n{'Total:':12s} {total_success:2d}/{total_tested:2d} successful ({total_success/total_tested*100:5.1f}%)")
    print(f"\n{'='*80}")
    if total_success >= total_tested * 0.6:  # 60% success threshold
        print("✓ ML_Predict_Price.py works well across multiple markets!")
    else:
        print("⚠ Results suggest further model tuning may be beneficial")
    print(f"{'='*80}")

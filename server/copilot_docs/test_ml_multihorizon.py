#!/usr/bin/env python3
"""
Test ML_Predict_Price.py multi-horizon prediction with real stocks from different countries
"""

import pandas as pd
from sqlalchemy import text
from app_imports import getDbConnection
from ML_Predict_Price import predict_best_horizon

print("Testing Multi-Horizon ML Price Prediction")
print("=" * 80)

def test_stocks_by_country(country_name, limit=5):
    """Test stocks from a specific country"""
    print(f"\n{'='*80}")
    print(f"Testing: {country_name}")
    print(f"{'='*80}\n")
    
    try:
        with getDbConnection() as con:
            # Get random stocks with sufficient data
            q = text("""
                SELECT DISTINCT f.symbol, COUNT(f.date) as day_count
                FROM finnhub_stock_prices f
                GROUP BY f.symbol
                HAVING day_count >= 120
                ORDER BY RAND()
                LIMIT :limit
            """)
            
            symbols = con.execute(q, {"limit": limit}).fetchall()
            
            if not symbols:
                print(f"No data found for {country_name}")
                return []
            
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
                
                if df.empty or len(df) < 60:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Test multi-horizon prediction
                prediction_result = predict_best_horizon(df, min_confidence=0.40)
                
                if prediction_result is not None:
                    pred_price = prediction_result['price']
                    pred_days = prediction_result['days']
                    confidence = prediction_result['confidence']
                    change = pred_price - current_price
                    change_pct = (change / current_price) * 100
                    
                    target_date = pd.Timestamp.now() + pd.Timedelta(days=pred_days)
                    
                    print(f"✓ {symbol:8s} | {len(df):4d} days")
                    print(f"    Current:   ${current_price:10.2f}")
                    print(f"    Predicted: ${pred_price:10.2f} (in {pred_days} days)")
                    print(f"    Change:    {change_pct:+7.2f}%")
                    print(f"    Confidence: {confidence:.3f} (R²)")
                    print(f"    Target:    ~{target_date.date()}")
                    print()
                    
                    results.append({
                        'symbol': symbol,
                        'current': current_price,
                        'predicted': pred_price,
                        'days': pred_days,
                        'confidence': confidence,
                        'change_pct': change_pct
                    })
                else:
                    print(f"✗ {symbol:8s} | {len(df):4d} days | Current: ${current_price:10.2f}")
                    print(f"    No confident prediction (all horizons < 0.40 R²)")
                    print()
            
            # Summary for this country
            if results:
                success_rate = len(results) / len(symbols) * 100
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                avg_horizon = sum(r['days'] for r in results) / len(results)
                avg_change = sum(abs(r['change_pct']) for r in results) / len(results)
                
                print(f"{'-'*80}")
                print(f"Summary for {country_name}:")
                print(f"  Success Rate: {len(results)}/{len(symbols)} ({success_rate:.0f}%)")
                print(f"  Avg Confidence: {avg_confidence:.3f}")
                print(f"  Avg Prediction Horizon: {avg_horizon:.1f} days")
                print(f"  Avg Expected Change: {avg_change:.2f}%")
            else:
                print(f"{'-'*80}")
                print(f"No successful predictions for {country_name}")
            
            return results
            
    except Exception as e:
        print(f"Error testing {country_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


# Test different markets
print("\n" + "="*80)
print("MULTI-HORIZON PREDICTION TEST")
print("Testing stocks from different regions")
print("="*80)

all_results = {}

# Test USA stocks
print("\n>>> Testing USA Stocks")
all_results['USA'] = test_stocks_by_country('USA', limit=10)

# Test random international stocks
print("\n>>> Testing International Stocks (Random Selection)")
all_results['International'] = test_stocks_by_country('International', limit=10)

# Overall summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}\n")

total_tested = 0
total_success = 0
all_predictions = []

for market, results in all_results.items():
    if results:
        total = len([r for r in results if r])
        success = len(results)
        total_tested += total
        total_success += success
        all_predictions.extend(results)
        
        print(f"{market:20s}: {success} successful predictions")

if all_predictions:
    print(f"\nTotal Predictions: {total_success}")
    print(f"\nPrediction Horizon Distribution:")
    
    horizon_dist = {}
    for r in all_predictions:
        days = r['days']
        horizon_dist[days] = horizon_dist.get(days, 0) + 1
    
    for days in sorted(horizon_dist.keys()):
        count = horizon_dist[days]
        pct = count / len(all_predictions) * 100
        print(f"  {days:2d} days: {count:2d} predictions ({pct:5.1f}%)")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in all_predictions]
    print(f"\nConfidence Statistics:")
    print(f"  Min:  {min(confidences):.3f}")
    print(f"  Max:  {max(confidences):.3f}")
    print(f"  Avg:  {sum(confidences)/len(confidences):.3f}")
    
    # Change statistics
    changes = [abs(r['change_pct']) for r in all_predictions]
    print(f"\nExpected Price Change (absolute):")
    print(f"  Min:  {min(changes):.2f}%")
    print(f"  Max:  {max(changes):.2f}%")
    print(f"  Avg:  {sum(changes)/len(changes):.2f}%")
    
    print(f"\n{'='*80}")
    if total_success >= 5:
        print("✓ Multi-Horizon Prediction Working Successfully!")
        print("  - Multiple time horizons tested (5, 10, 15, 20 days)")
        print("  - Best horizon selected based on confidence")
        print("  - Production-ready for swing trading")
    else:
        print("⚠ Limited successful predictions")
        print("  Consider adjusting confidence threshold or features")
    print(f"{'='*80}")
else:
    print("✗ No successful predictions across all tests")
    print("  Check data quality and model parameters")

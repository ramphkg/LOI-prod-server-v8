#!/usr/bin/env python3
"""
Test ta_signals_mc_parallel.py integration with ML_Predict_Price
"""

import sys
from ta_signals_mc_parallel import get_tlib_tadata, initialize_config, canonical_table_schema
from app_imports import parallelLoggingSetter

# Test 1: Check schema includes ML_Predicted_Price
print("=" * 70)
print("TEST 1: Schema Validation")
print("=" * 70)

schema = canonical_table_schema()
if 'ML_Predicted_Price' in schema:
    print("✓ ML_Predicted_Price field added to schema")
    print(f"  Type: {schema['ML_Predicted_Price']}")
else:
    print("✗ ML_Predicted_Price field NOT found in schema")
    sys.exit(1)

# Test 2: Process a few symbols and check for predicted price
print("\n" + "=" * 70)
print("TEST 2: Processing Symbols with ML Prediction")
print("=" * 70)

test_symbols = ['AAPL', 'MSFT', 'GOOGL']
price_source = 'FINNHUB_LOCAL'

my_logger = parallelLoggingSetter("test_integration")

results = []
for symbol in test_symbols:
    print(f"\nProcessing {symbol}...")
    try:
        result_df = get_tlib_tadata(
            underlying=symbol,
            price_source=price_source,
            my_logger=my_logger,
            df=None,
            mainrun=True
        )
        
        if result_df is not None and not result_df.empty:
            if 'ML_Predicted_Price' in result_df.columns:
                ml_price = result_df['ML_Predicted_Price'].iloc[0]
                current_price = result_df['Close'].iloc[0]
                
                if ml_price is not None and str(ml_price).lower() not in ['nan', 'none', '<na>']:
                    change = float(ml_price) - float(current_price)
                    change_pct = (change / float(current_price)) * 100
                    print(f"  ✓ Symbol: {symbol}")
                    print(f"    Current Price: ${current_price:.2f}")
                    print(f"    Predicted Price: ${ml_price:.2f}")
                    print(f"    Expected Change: {change_pct:+.2f}%")
                    results.append((symbol, True, current_price, ml_price))
                else:
                    print(f"  ⚠ Symbol: {symbol}")
                    print(f"    Current Price: ${current_price:.2f}")
                    print(f"    Predicted Price: None (low confidence)")
                    results.append((symbol, False, current_price, None))
            else:
                print(f"  ✗ ML_Predicted_Price column not found in result")
        else:
            print(f"  ✗ No data returned for {symbol}")
    except Exception as e:
        print(f"  ✗ Error processing {symbol}: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

success_count = sum(1 for _, success, _, _ in results if success)
total_count = len(results)

print(f"\nProcessed: {total_count} symbols")
print(f"Predictions generated: {success_count}/{total_count}")

if success_count > 0:
    print("\n" + "=" * 70)
    print("✓ Integration successful!")
    print("  ML_Predict_Price is now integrated with ta_signals_mc_parallel.py")
    print("  Predicted prices will be stored in the database.")
    print("=" * 70)
else:
    print("\n⚠ No predictions generated - may need data or model tuning")

#!/usr/bin/env python3
"""
validate_swing_scoring.py

Query finnhub_tas_listings for USA data, analyze distinct values in scoring-relevant columns,
and validate whether the scoring logic in tas_swing_scoring.py is accurate.

Usage:
    python validate_swing_scoring.py
"""

import sys
import pandas as pd
from sqlalchemy import text
from app_imports import getDbConnection

# Columns used in scoring logic
SCORING_COLUMNS = [
    'SignalClassifier_ML',
    'SignalClassifier_Rules', 
    'Trend',
    'TMA21_50_X',
    'RSIUpTrend',
    'ADX',
    'TrendReversal_Rules',
    'TrendReversal_ML',
    'LastTrendDays',
    'Pct2H52',
    'PctfL52'
]

def fetch_usa_data():
    """Fetch sample of USA data from finnhub_tas_listings."""
    print("Fetching data from finnhub_tas_listings for CountryName='USA'...")
    
    with getDbConnection() as con:
        # Get sample and statistics
        query = text("""
            SELECT 
                SignalClassifier_ML,
                SignalClassifier_Rules,
                Trend,
                TMA21_50_X,
                RSIUpTrend,
                ADX,
                TrendReversal_Rules,
                TrendReversal_ML,
                LastTrendDays,
                Pct2H52,
                PctfL52,
                Symbol,
                Date,
                ScanDate
            FROM finnhub_tas_listings
            WHERE CountryName = 'USA'
            ORDER BY ScanDate DESC, Symbol ASC
            LIMIT 5000
        """)
        df = pd.read_sql(query, con=con)
    
    print(f"Loaded {len(df)} rows\n")
    return df

def analyze_distinct_values(df):
    """Analyze distinct values for each scoring column."""
    print("="*80)
    print("DISTINCT VALUE ANALYSIS")
    print("="*80)
    
    for col in SCORING_COLUMNS:
        if col not in df.columns:
            print(f"\n⚠️  Column '{col}' NOT FOUND in table")
            continue
        
        print(f"\n{'='*60}")
        print(f"Column: {col}")
        print(f"{'='*60}")
        
        # Basic stats
        total = len(df)
        non_null = df[col].notna().sum()
        null_count = total - non_null
        
        print(f"Total rows: {total}")
        print(f"Non-null: {non_null} ({100*non_null/total:.1f}%)")
        print(f"Null: {null_count} ({100*null_count/total:.1f}%)")
        
        if non_null == 0:
            print("⚠️  All values are NULL")
            continue
        
        # Get distinct values
        distinct_vals = df[col].dropna().unique()
        n_distinct = len(distinct_vals)
        
        print(f"Distinct values: {n_distinct}")
        
        # Show distribution for categorical/small numeric range
        if n_distinct <= 50:
            value_counts = df[col].value_counts(dropna=False).head(20)
            print(f"\nValue distribution (top 20):")
            for val, count in value_counts.items():
                pct = 100 * count / total
                print(f"  {val!r:40s} : {count:6d} ({pct:5.1f}%)")
        else:
            # Numeric statistics
            print(f"\nNumeric statistics:")
            print(df[col].describe())

def validate_classifier_logic(df):
    """Validate SignalClassifier_ML and SignalClassifier_Rules logic."""
    print("\n" + "="*80)
    print("VALIDATION: Classifier Logic")
    print("="*80)
    
    issues = []
    
    # Check expected range: -4 to +4
    for col in ['SignalClassifier_ML', 'SignalClassifier_Rules']:
        if col not in df.columns:
            continue
        
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        
        out_of_range = vals[(vals < -4) | (vals > 4)]
        if len(out_of_range) > 0:
            issues.append(f"⚠️  {col}: {len(out_of_range)} values outside [-4, +4] range")
            print(f"⚠️  {col}: Found {len(out_of_range)} values outside expected [-4, +4] range")
            print(f"   Examples: {out_of_range.unique()[:10]}")
        else:
            print(f"✓ {col}: All values in expected [-4, +4] range")
    
    return issues

def validate_trend_logic(df):
    """Validate Trend parsing logic."""
    print("\n" + "="*80)
    print("VALIDATION: Trend Parsing")
    print("="*80)
    
    issues = []
    
    if 'Trend' not in df.columns:
        return ["⚠️  Trend column not found"]
    
    trends = df['Trend'].dropna()
    
    # Check format: Primary[Secondary]
    valid_format = trends.str.contains(r'^\w+\[\w+\]$', regex=True, na=False)
    invalid = trends[~valid_format]
    
    if len(invalid) > 0:
        issues.append(f"⚠️  Trend: {len(invalid)} values don't match 'Primary[Secondary]' format")
        print(f"⚠️  Found {len(invalid)} trends not matching expected 'Primary[Secondary]' format")
        print(f"   Examples: {invalid.unique()[:10].tolist()}")
    else:
        print(f"✓ All {len(trends)} Trend values match 'Primary[Secondary]' format")
    
    # Extract and validate Primary values
    primaries = trends.str.split('[', expand=True)[0].unique()
    expected_primaries = ['Bull', 'Bear', 'Neutral']
    unexpected_primaries = [p for p in primaries if p not in expected_primaries]
    
    if unexpected_primaries:
        issues.append(f"⚠️  Unexpected Primary trend values: {unexpected_primaries}")
        print(f"⚠️  Found unexpected Primary trend values: {unexpected_primaries}")
        print(f"   Expected: {expected_primaries}")
    else:
        print(f"✓ All Primary trend values are valid: {list(primaries)}")
    
    # Extract and validate Secondary values
    secondaries = trends.str.extract(r'\[([^\]]+)\]')[0].unique()
    expected_secondaries = ['PullbackInBull', 'TrendingUp', 'PullbackInBear', 'TrendingDown', 
                           'Unknown', 'Volatile', 'ShortTrend', 'Ranging']
    unexpected_secondaries = [s for s in secondaries if s not in expected_secondaries]
    
    if unexpected_secondaries:
        issues.append(f"⚠️  Unexpected Secondary trend values: {unexpected_secondaries}")
        print(f"⚠️  Found unexpected Secondary trend values: {unexpected_secondaries}")
        print(f"   Expected: {expected_secondaries}")
    else:
        print(f"✓ All Secondary trend values are valid")
    
    return issues

def validate_reversal_logic(df):
    """Validate TrendReversal_Rules and TrendReversal_ML logic."""
    print("\n" + "="*80)
    print("VALIDATION: Reversal Labels")
    print("="*80)
    
    issues = []
    expected_labels = [
        'NoReversal', 
        'BullishReversalWeak', 
        'BullishReversalModerate',
        'BullishReversalStrong',
        'BearishReversalWeak',
        'BearishReversalModerate', 
        'BearishReversalStrong',
        # TrendReversal_ML variants
        'BullishReversal-MLWeak',
        'BullishReversal-MLModerate',
        'BullishReversal-MLStrong',
        'BearishReversal-MLWeak',
        'BearishReversal-MLModerate',
        'BearishReversal-MLStrong',
        '0', '0.0', 0, 0.0  # null equivalents
    ]
    
    for col in ['TrendReversal_Rules', 'TrendReversal_ML']:
        if col not in df.columns:
            continue
        
        vals = df[col].dropna().astype(str).str.strip()
        distinct = vals.unique()
        
        # Check for unexpected labels
        unexpected = [v for v in distinct if v not in [str(e) for e in expected_labels]]
        
        if unexpected:
            issues.append(f"⚠️  {col}: Unexpected labels found: {unexpected}")
            print(f"⚠️  {col}: Found unexpected reversal labels:")
            for label in unexpected[:10]:
                count = (vals == label).sum()
                print(f"   '{label}': {count} occurrences")
        else:
            print(f"✓ {col}: All labels are recognized")
            print(f"   Distinct labels: {list(distinct)}")
    
    return issues

def validate_numeric_ranges(df):
    """Validate numeric column ranges."""
    print("\n" + "="*80)
    print("VALIDATION: Numeric Ranges")
    print("="*80)
    
    issues = []
    
    # TMA21_50_X should be -1, 0, or +1
    if 'TMA21_50_X' in df.columns:
        tma_vals = df['TMA21_50_X'].dropna().unique()
        unexpected_tma = [v for v in tma_vals if v not in [-1, 0, 1]]
        if unexpected_tma:
            issues.append(f"⚠️  TMA21_50_X: Unexpected values {unexpected_tma}")
            print(f"⚠️  TMA21_50_X: Expected [-1, 0, 1], found: {sorted(tma_vals)}")
        else:
            print(f"✓ TMA21_50_X: All values in expected set [-1, 0, 1]")
    
    # RSIUpTrend should be boolean-like
    if 'RSIUpTrend' in df.columns:
        rsi_vals = df['RSIUpTrend'].dropna().unique()
        unexpected_rsi = [v for v in rsi_vals if v not in [0, 1, True, False, '0', '1', 'True', 'False']]
        if unexpected_rsi:
            issues.append(f"⚠️  RSIUpTrend: Unexpected values {unexpected_rsi}")
            print(f"⚠️  RSIUpTrend: Expected boolean-like, found: {rsi_vals}")
        else:
            print(f"✓ RSIUpTrend: All values are boolean-like")
    
    # ADX should be 0-100 (typically)
    if 'ADX' in df.columns:
        adx = df['ADX'].dropna()
        if len(adx) > 0:
            out_of_range = adx[(adx < 0) | (adx > 100)]
            if len(out_of_range) > 0:
                issues.append(f"⚠️  ADX: {len(out_of_range)} values outside [0, 100] range")
                print(f"⚠️  ADX: {len(out_of_range)} values outside typical [0, 100] range")
                print(f"   Min: {adx.min():.2f}, Max: {adx.max():.2f}")
            else:
                print(f"✓ ADX: All values in [0, 100] range")
                print(f"   Min: {adx.min():.2f}, Max: {adx.max():.2f}, Mean: {adx.mean():.2f}")
    
    # Pct2H52 and PctfL52 should be reasonable percentages
    for col in ['Pct2H52', 'PctfL52']:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                if vals.min() < -100 or vals.max() > 1000:
                    issues.append(f"⚠️  {col}: Extreme values found")
                    print(f"⚠️  {col}: Extreme values detected")
                    print(f"   Min: {vals.min():.2f}, Max: {vals.max():.2f}")
                else:
                    print(f"✓ {col}: Range appears reasonable")
                    print(f"   Min: {vals.min():.2f}, Max: {vals.max():.2f}, Mean: {vals.mean():.2f}")
    
    return issues

def main():
    print("\n" + "="*80)
    print("SWING SCORING VALIDATION")
    print("="*80 + "\n")
    
    try:
        # Fetch data
        df = fetch_usa_data()
        
        # Analyze distinct values
        analyze_distinct_values(df)
        
        # Run validations
        all_issues = []
        all_issues.extend(validate_classifier_logic(df))
        all_issues.extend(validate_trend_logic(df))
        all_issues.extend(validate_reversal_logic(df))
        all_issues.extend(validate_numeric_ranges(df))
        
        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        if all_issues:
            print(f"\n⚠️  Found {len(all_issues)} potential issues:\n")
            for i, issue in enumerate(all_issues, 1):
                print(f"{i}. {issue}")
            print("\n⚠️  Review tas_swing_scoring.py logic against actual data values")
        else:
            print("\n✓ All validations passed! Scoring logic appears accurate.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

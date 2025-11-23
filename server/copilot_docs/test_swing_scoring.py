#!/usr/bin/env python3
# test_swing_scoring.py
# Quick validation script for the swing ranking system

import pandas as pd
import numpy as np
from tas_swing_scoring import (
    detect_price_pattern,
    score_sma_distance,
    score_rsi_oversold,
    score_pct2h52,
    score_gem_rank,
    compute_composite_score,
    generate_signal
)

def test_pattern_detection():
    """Test pattern detection algorithm"""
    print("=" * 60)
    print("TEST 1: Pattern Detection")
    print("=" * 60)
    
    # Test case 1: Clear pattern (5 down, 3 up)
    prices1 = pd.Series([100, 95, 90, 85, 80, 75, 78, 81, 84])
    result1 = detect_price_pattern(prices1)
    print(f"\nTest 1.1 - Clear pattern (5 down, 3 up)")
    print(f"  Detected: {result1['detected']}")
    print(f"  d1 (recent up): {result1['d1_recent_up']}")
    print(f"  d2 (prior down): {result1['d2_prior_down']}")
    print(f"  Score: {result1['pattern_score']}")
    assert result1['detected'] == True, "Should detect pattern"
    assert result1['d1_recent_up'] == 3, "Should detect 3 up days"
    assert result1['d2_prior_down'] == 5, "Should detect 5 down days"
    
    # Test case 2: No pattern (continuous up)
    prices2 = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116])
    result2 = detect_price_pattern(prices2)
    print(f"\nTest 1.2 - No pattern (continuous up)")
    print(f"  Detected: {result2['detected']}")
    assert result2['detected'] == False, "Should not detect pattern"
    
    # Test case 3: Insufficient prior down
    prices3 = pd.Series([100, 98, 96, 98, 100, 102])
    result3 = detect_price_pattern(prices3)
    print(f"\nTest 1.3 - Insufficient prior down")
    print(f"  Detected: {result3['detected']}")
    assert result3['detected'] == False, "Should not qualify (d2 too small)"
    
    print("\n✓ Pattern detection tests passed")

def test_technical_scoring():
    """Test technical analysis scoring components"""
    print("\n" + "=" * 60)
    print("TEST 2: Technical Scoring")
    print("=" * 60)
    
    # SMA Distance
    print("\nTest 2.1 - SMA Distance Scoring")
    score1 = score_sma_distance(100, 99)  # ~1% distance
    score2 = score_sma_distance(100, 90)  # ~11% distance
    print(f"  1% distance: {score1} points (expect 25)")
    print(f"  11% distance: {score2} points (expect 10)")
    assert score1 == 25.0, "Should score 25 for close EMAs"
    assert score2 == 10.0, "Should score 10 for moderate distance"
    
    # RSI Oversold
    print("\nTest 2.2 - RSI Oversold Scoring")
    score1 = score_rsi_oversold(28)  # Deep oversold
    score2 = score_rsi_oversold(42)  # Mild
    score3 = score_rsi_oversold(55)  # Not oversold
    print(f"  RSI 28: {score1} points (expect 25)")
    print(f"  RSI 42: {score2} points (expect 10)")
    print(f"  RSI 55: {score3} points (expect 0)")
    assert score1 == 25.0, "Deep oversold should score 25"
    assert score2 == 10.0, "Mild oversold should score 10"
    assert score3 == 0.0, "Not oversold should score 0"
    
    # Pct2H52
    print("\nTest 2.3 - 52-Week High Distance Scoring")
    score1 = score_pct2h52(45)  # Far from high
    score2 = score_pct2h52(15)  # Moderate
    score3 = score_pct2h52(2)   # Near high
    print(f"  45% from high: {score1} points (expect 25)")
    print(f"  15% from high: {score2} points (expect 10)")
    print(f"  2% from high: {score3} points (expect 0)")
    assert score1 == 25.0, "Far from high should score 25"
    assert score2 == 10.0, "Moderate should score 10"
    assert score3 == 0.0, "Near high should score 0"
    
    # GEM Rank
    print("\nTest 2.4 - GEM Rank Scoring")
    score1 = score_gem_rank(300)   # Top quality
    score2 = score_gem_rank(1200)  # Good
    score3 = score_gem_rank(3500)  # Poor
    print(f"  GEM 300: {score1} points (expect 25)")
    print(f"  GEM 1200: {score2} points (expect 15)")
    print(f"  GEM 3500: {score3} points (expect 0)")
    assert score1 == 25.0, "Top GEM should score 25"
    assert score2 == 15.0, "Good GEM should score 15"
    assert score3 == 0.0, "Poor GEM should score 0"
    
    print("\n✓ Technical scoring tests passed")

def test_composite_ranking():
    """Test composite scoring and signal generation"""
    print("\n" + "=" * 60)
    print("TEST 3: Composite Ranking")
    print("=" * 60)
    
    # Composite scoring
    print("\nTest 3.1 - Composite Score Calculation")
    composite1 = compute_composite_score(90, 80)  # Excellent pattern, good tech
    composite2 = compute_composite_score(50, 60)  # Average both
    print(f"  Pattern 90, Tech 80: {composite1} (expect ~84.5)")
    print(f"  Pattern 50, Tech 60: {composite2} (expect ~55.5)")
    assert 84 <= composite1 <= 85, "Composite should be ~84.5"
    assert 55 <= composite2 <= 56, "Composite should be ~55.5"
    
    # Signal generation
    print("\nTest 3.2 - Signal Generation")
    sig1 = generate_signal(85, True)
    sig2 = generate_signal(68, True)
    sig3 = generate_signal(52, True)
    sig4 = generate_signal(90, False)  # High score but no pattern
    print(f"  Score 85, pattern detected: {sig1} (expect Strong_Buy)")
    print(f"  Score 68, pattern detected: {sig2} (expect Buy)")
    print(f"  Score 52, pattern detected: {sig3} (expect Weak_Buy)")
    print(f"  Score 90, no pattern: {sig4} (expect Neutral)")
    assert sig1 == "Strong_Buy", "Should be Strong_Buy"
    assert sig2 == "Buy", "Should be Buy"
    assert sig3 == "Weak_Buy", "Should be Weak_Buy"
    assert sig4 == "Neutral", "Should be Neutral without pattern"
    
    print("\n✓ Composite ranking tests passed")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)
    
    # Empty series
    print("\nTest 4.1 - Empty price series")
    result = detect_price_pattern(pd.Series([]))
    print(f"  Detected: {result['detected']} (expect False)")
    assert result['detected'] == False, "Empty series should not detect pattern"
    
    # NaN values in technical scoring
    print("\nTest 4.2 - NaN handling")
    score1 = score_rsi_oversold(np.nan)
    score2 = score_pct2h52(np.nan)
    score3 = score_gem_rank(np.nan)
    print(f"  RSI NaN: {score1} (expect 0)")
    print(f"  Pct2H52 NaN: {score2} (expect 0)")
    print(f"  GEM NaN: {score3} (expect 0)")
    assert score1 == 0.0, "NaN should return 0"
    assert score2 == 0.0, "NaN should return 0"
    assert score3 == 0.0, "NaN should return 0"
    
    # Very short series
    print("\nTest 4.3 - Very short price series")
    result = detect_price_pattern(pd.Series([100, 95]))
    print(f"  Detected: {result['detected']} (expect False)")
    assert result['detected'] == False, "Too short should not detect"
    
    print("\n✓ Edge case tests passed")

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("SWING RANKING SYSTEM - VALIDATION TESTS")
    print("=" * 60)
    
    try:
        test_pattern_detection()
        test_technical_scoring()
        test_composite_ranking()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe swing ranking system is working correctly!")
        print("You can now run:")
        print("  python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())

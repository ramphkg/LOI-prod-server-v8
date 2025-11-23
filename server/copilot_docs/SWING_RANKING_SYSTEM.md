# Swing Trading Ranking System - Documentation

## Overview
Brand new three-tier ranking system for identifying optimal swing trading opportunities based on price patterns and technical analysis.

## System Architecture

### Ranking 1: Price Pattern Detection
**Purpose**: Identify stocks showing bullish reversal pattern (recent uptrend after downtrend)

**Algorithm**:
1. Detects recent consecutive up days (d1)
2. Detects prior consecutive down days (d2) before d1
3. Qualifies pattern if: d1 ≥ min_recent_up AND d2 ≥ min_prior_down AND d2 > d1
4. Scores pattern quality (0-100) based on:
   - How close d1 is to ideal value (n1=3 days) - 40% weight
   - How large d2 is (higher = better) - 60% weight

**New Fields Added**:
- `PricePattern_RecentUpDays` (int): d1 - recent up days
- `PricePattern_PriorDownDays` (int): d2 - prior down days  
- `PricePattern_Detected` (bool): True if pattern qualifies
- `PricePattern_Score` (float 0-100): Pattern quality score
- `PricePattern_Rank` (int): Rank among detected patterns

**Default Parameters**:
```python
n1_ideal = 3              # ideal recent uptrend length
min_recent_up = 2         # minimum d1 to qualify
min_prior_down = 5        # minimum d2 to qualify
d2_max = 20               # cap for d2 scoring
lookback_days = 60        # history window
allow_flat_days = 1       # tolerance for flat days
```

---

### Ranking 2: Technical Analysis Scoring
**Purpose**: Score stocks based on technical indicators optimal for swing entries

**Components** (each 0-25 points, total 0-100):

1. **SMA Distance Score** (lower = better)
   - Measures distance between EMA20 and EMA50
   - ≤2%: 25 pts | ≤5%: 20 pts | ≤10%: 15 pts | ≤15%: 10 pts | else: 5 pts

2. **RSI Oversold Score** (deeper oversold = better)
   - ≤30: 25 pts | ≤35: 20 pts | ≤40: 15 pts | ≤45: 10 pts | ≤50: 5 pts

3. **52-Week High Distance Score** (higher = better value)
   - ≥40%: 25 pts | ≥30%: 20 pts | ≥20%: 15 pts | ≥10%: 10 pts | ≥5%: 5 pts

4. **GEM Rank Score** (lower rank = better quality)
   - ≤500: 25 pts | ≤1000: 20 pts | ≤1500: 15 pts | ≤2000: 10 pts | ≤3000: 5 pts

**New Fields Added**:
- `TechScore_SMA_Dist` (float 0-25): SMA component score
- `TechScore_RSI` (float 0-25): RSI component score
- `TechScore_Pct2H52` (float 0-25): 52W distance score
- `TechScore_GEM` (float 0-25): GEM quality score
- `TechScore_Total` (float 0-100): Sum of all components
- `TechScore_Rank` (int): Rank by technical score

---

### Ranking 3: Composite Ranking
**Purpose**: Combine pattern and technical scores into single actionable metric

**Algorithm**:
```python
CompositeScore = (PricePattern_Score × 0.45) + (TechScore_Total × 0.55)
```

**Signal Generation**:
- `Strong_Buy`: CompositeScore ≥ 80 AND pattern detected
- `Buy`: CompositeScore ≥ 65 AND pattern detected
- `Weak_Buy`: CompositeScore ≥ 50 AND pattern detected
- `Neutral`: Otherwise

**Ranking**: Only stocks with `PricePattern_Detected = True` are ranked

**New Fields Added**:
- `CompositeScore` (float 0-100): Weighted combination
- `CompositeRank` (int): Final rank (1 = best opportunity)
- `SwingTrade_Signal` (string): Buy signal strength

---

## Usage

### Basic Command
```bash
python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB
```

### Advanced Configuration
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --n1_ideal 3 \
    --min_recent_up 2 \
    --min_prior_down 5 \
    --lookback_days 60 \
    --pattern_weight 0.45 \
    --technical_weight 0.55 \
    --top_n 50 \
    --out_dir _out_swing
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--watchlist` | string | REQUIRED | Watchlist name (e.g., US_CORE) |
| `--source` | string | REQUIRED | Data source (e.g., FINNHUB) |
| `--n1_ideal` | int | 3 | Ideal recent uptrend length |
| `--min_recent_up` | int | 2 | Minimum d1 to qualify |
| `--min_prior_down` | int | 5 | Minimum d2 to qualify |
| `--lookback_days` | int | 60 | History window for pattern detection |
| `--pattern_weight` | float | 0.45 | Pattern weight in composite score |
| `--technical_weight` | float | 0.55 | Technical weight in composite score |
| `--top_n` | int | 50 | Number of top stocks to export |
| `--out_dir` | string | _out_swing | Output directory |

---

## Output Files

### 1. Full Rankings
**File**: `swing_rankings_full_<master>_<country>_<YYYYMMDD>.csv`

Contains all stocks with all ranking fields:
- All original tas_listing columns
- All Ranking 1 fields (pattern)
- All Ranking 2 fields (technical)
- All Ranking 3 fields (composite)

### 2. Top Opportunities
**File**: `swing_rankings_top_<master>_<country>_<YYYYMMDD>.csv`

Filtered view containing:
- Only stocks where `PricePattern_Detected = True`
- Sorted by `CompositeRank` (ascending)
- Top N stocks (default 50)

### 3. Strong Buy Signals
**File**: `swing_strong_buy_<master>_<country>_<YYYYMMDD>.csv`

Contains only stocks with:
- `SwingTrade_Signal = "Strong_Buy"`
- `CompositeScore ≥ 80`
- Pattern detected

---

## Output Summary

The script prints a comprehensive summary:

```
======================================================================
RANKING SUMMARY
======================================================================
Total stocks analyzed: 3846
Patterns detected: 127
Strong Buy signals: 8
Buy signals: 23
Weak Buy signals: 34
Average CompositeScore (detected): 62.45
Average TechScore (detected): 58.32
Average PatternScore (detected): 67.18

TOP 10 SWING TRADE OPPORTUNITIES:
----------------------------------------------------------------------
  #1: AAPL - Score: 88.3, Signal: Strong_Buy, Pattern: d1=3/d2=12
  #2: MSFT - Score: 86.7, Signal: Strong_Buy, Pattern: d1=4/d2=15
  #3: NVDA - Score: 84.2, Signal: Strong_Buy, Pattern: d1=3/d2=10
  ...
======================================================================
```

---

## Complete Field Reference

### Existing Fields (from tas_listings)
All original columns are preserved including:
- Price data: Date, Open, High, Low, Close, Volume
- Identifiers: Symbol, CountryName, IndustrySector
- Technical indicators: ADX, RSI, CCI, EMA20, EMA50, SMA200
- 52-week data: High52, Low52, Pct2H52, PctfL52
- Fundamentals: GEM_Rank, marketCap
- Signals: SignalClassifier_Rules, SignalClassifier_ML, TrendReversal_Rules, TrendReversal_ML

### New Fields (added by ranking system)

#### Ranking 1 Fields
- `PricePattern_RecentUpDays`: Recent consecutive up days (d1)
- `PricePattern_PriorDownDays`: Prior consecutive down days (d2)
- `PricePattern_Detected`: Boolean - pattern qualification
- `PricePattern_Score`: Float 0-100 - pattern quality
- `PricePattern_Rank`: Integer - rank by pattern score

#### Ranking 2 Fields
- `TechScore_SMA_Dist`: Float 0-25 - EMA20/EMA50 distance score
- `TechScore_RSI`: Float 0-25 - RSI oversold score
- `TechScore_Pct2H52`: Float 0-25 - 52W high distance score
- `TechScore_GEM`: Float 0-25 - GEM fundamental score
- `TechScore_Total`: Float 0-100 - sum of components
- `TechScore_Rank`: Integer - rank by technical score

#### Ranking 3 Fields
- `CompositeScore`: Float 0-100 - weighted combination
- `CompositeRank`: Integer - final rank (1 = best)
- `SwingTrade_Signal`: String - "Strong_Buy", "Buy", "Weak_Buy", "Neutral"

---

## Configuration Tuning Guide

### Pattern Detection
- **Increase `n1_ideal`**: If you prefer stocks with slightly longer recent uptrends
- **Decrease `min_prior_down`**: To detect patterns with shorter downtrends (more candidates)
- **Increase `min_prior_down`**: To require stronger reversal setups (fewer, higher quality)
- **Adjust `lookback_days`**: Longer window = detect patterns over longer timeframes

### Scoring Weights
- **Increase `pattern_weight`**: If price pattern is more important than technicals
- **Increase `technical_weight`**: If fundamentals/indicators matter more

### Signal Thresholds
Edit `DEFAULT_COMPOSITE_CONFIG` in code:
```python
'signal_thresholds': {
    'strong_buy': 80,  # Raise for stricter Strong_Buy criteria
    'buy': 65,         # Adjust mid-tier threshold
    'weak_buy': 50     # Adjust entry-level threshold
}
```

---

## Integration with Existing System

### Database Tables Used
- **Input**: `finnhub_tas_listings` (or configured master table)
- **History**: `finnhub_historical_prices` (for pattern detection)

### Dependencies
- `app_imports.getDbConnection()` - Database connection
- `ta_signals_mc_parallel.initialize_config()` - Configuration
- `ta_signals_mc_parallel.get_country_name()` - Country mapping

### Workflow Integration
```bash
# Step 1: Run technical analysis (existing system)
./ta_signals_mc_parallel.py --watchlist US_CORE --source FINNHUB

# Step 2: Run swing ranking system (new)
python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB

# Output: Rankings in _out_swing/ directory
```

---

## Examples

### Example 1: Conservative Settings (High Quality Only)
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --min_prior_down 10 \
    --n1_ideal 3 \
    --pattern_weight 0.6 \
    --technical_weight 0.4
```
Focuses on stronger reversals with emphasis on pattern quality.

### Example 2: Aggressive Settings (More Candidates)
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --min_prior_down 4 \
    --min_recent_up 1 \
    --pattern_weight 0.3 \
    --technical_weight 0.7
```
Lower pattern requirements, emphasizes technical indicators.

### Example 3: Custom Top List
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --top_n 100 \
    --out_dir my_rankings
```
Exports top 100 opportunities to custom directory.

---

## Troubleshooting

### No Patterns Detected
**Symptoms**: "No patterns detected" warning, empty top/strong_buy files

**Solutions**:
1. Lower `min_prior_down` (e.g., from 5 to 3)
2. Lower `min_recent_up` (e.g., from 2 to 1)
3. Check if historical price data exists in `finnhub_historical_prices`
4. Increase `lookback_days` to search further back

### Few Strong Buy Signals
**Symptoms**: Strong_Buy file empty or very small

**Solutions**:
1. Lower `strong_buy` threshold in code (from 80 to 75)
2. Adjust component score thresholds in `DEFAULT_TECHNICAL_CONFIG`
3. Verify RSI, Pct2H52, GEM_Rank data quality

### Slow Performance
**Symptoms**: Pattern detection taking too long

**Causes**: Fetches full price history for each symbol individually

**Solutions**:
1. Process smaller symbol lists
2. Optimize database queries (add indexes on Symbol, Date)
3. Consider batch loading price history

---

## Future Enhancements

### Planned Improvements
1. **Batch price loading**: Load all price history in one query
2. **Additional patterns**: Detect bearish patterns for short opportunities
3. **Volume confirmation**: Add volume trend to pattern scoring
4. **Backtest mode**: Historical performance of rankings
5. **Real-time updates**: Incremental pattern re-evaluation
6. **Custom weights UI**: Configuration file for easier tuning

### Extensibility
The system is modular and can be extended:
- Add new technical components to Ranking 2
- Create alternative pattern detectors for Ranking 1
- Implement custom composite scoring algorithms
- Export to different formats (JSON, database)

---

## Version History

### v1.0 (November 22, 2025)
- Initial implementation
- Three-tier ranking system
- Price pattern detection (d1/d2)
- Technical scoring (SMA, RSI, 52W, GEM)
- Composite ranking with signals
- CSV export functionality

---

## Support & Contact

For questions or issues:
1. Check existing tas_listings data completeness
2. Verify historical price data availability
3. Review log output for specific errors
4. Adjust parameters based on market conditions

---

**End of Documentation**

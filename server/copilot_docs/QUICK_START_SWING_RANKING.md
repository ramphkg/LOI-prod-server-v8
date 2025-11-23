# Quick Start Guide - Swing Ranking System

## ✅ System Ready

The brand new three-tier swing ranking system has been successfully implemented and tested!

## 🎯 What It Does

Identifies the best swing trading opportunities using:
1. **Price Pattern Detection** - Stocks showing recent uptrend after downtrend
2. **Technical Analysis** - Scores based on RSI, SMA distance, 52W position, GEM rank
3. **Composite Ranking** - Combines both into a single actionable score

## 🚀 Quick Start

### Run the Ranking System
```bash
cd /home/ram/dev/LOI/LOI-prod-server-v8/server
python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB
```

### Check the Output
Results will be in `_out_swing/` directory:
- `swing_rankings_full_*.csv` - All stocks with scores
- `swing_rankings_top_*.csv` - Top 50 opportunities
- `swing_strong_buy_*.csv` - Strong buy signals only

## 📊 Understanding the Output

### Key Columns to Watch

**Pattern Detection:**
- `PricePattern_Detected` - Boolean, is pattern present?
- `PricePattern_RecentUpDays` (d1) - How many days going up?
- `PricePattern_PriorDownDays` (d2) - How many days down before?
- `PricePattern_Score` - Pattern quality (0-100)
- `PricePattern_Rank` - Rank among detected patterns

**Technical Scores:**
- `TechScore_SMA_Dist` - EMA20/EMA50 proximity (0-25)
- `TechScore_RSI` - RSI oversold positioning (0-25)
- `TechScore_Pct2H52` - Distance from 52W high (0-25)
- `TechScore_GEM` - Fundamental quality (0-25)
- `TechScore_Total` - Sum of above (0-100)

**Final Ranking:**
- `CompositeScore` - Overall score (0-100)
- `CompositeRank` - Final rank (1 = best!)
- `SwingTrade_Signal` - Strong_Buy, Buy, Weak_Buy, or Neutral

## 🎮 Advanced Usage

### Conservative (High Quality Only)
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --min_prior_down 10 \
    --pattern_weight 0.6
```

### Aggressive (More Candidates)
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --min_prior_down 4 \
    --min_recent_up 1 \
    --technical_weight 0.7
```

### Custom Top List
```bash
python tas_swing_scoring.py \
    --watchlist US_CORE \
    --source FINNHUB \
    --top_n 100 \
    --out_dir my_custom_rankings
```

## 🔧 Key Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `--n1_ideal` | 3 | Ideal recent uptrend length (days) |
| `--min_recent_up` | 2 | Minimum days to qualify as uptrend |
| `--min_prior_down` | 5 | Minimum prior downtrend days |
| `--pattern_weight` | 0.45 | Pattern importance in composite |
| `--technical_weight` | 0.55 | Technical importance in composite |
| `--top_n` | 50 | How many top stocks to export |

## 📈 Example Output Summary

```
======================================================================
RANKING SUMMARY
======================================================================
Total stocks analyzed: 3846
Patterns detected: 127
Strong Buy signals: 8
Buy signals: 23
Weak Buy signals: 34

TOP 10 SWING TRADE OPPORTUNITIES:
----------------------------------------------------------------------
  #1: AAPL - Score: 88.3, Signal: Strong_Buy, Pattern: d1=3/d2=12
  #2: MSFT - Score: 86.7, Signal: Strong_Buy, Pattern: d1=4/d2=15
  ...
======================================================================
```

## ✅ Testing

Validate the system works:
```bash
python test_swing_scoring.py
```

Expected output: "✓ ALL TESTS PASSED SUCCESSFULLY"

## 📚 Full Documentation

See `SWING_RANKING_SYSTEM.md` for:
- Complete algorithm details
- All configuration options
- Field reference
- Tuning guide
- Troubleshooting

## 🔄 Integration with Existing Workflow

```bash
# Step 1: Run TA signals (existing system)
python ta_signals_mc_parallel.py --watchlist US_CORE --source FINNHUB

# Step 2: Run swing rankings (new system)
python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB

# Step 3: Review results
ls -lh _out_swing/
```

## 🎯 What Makes a Good Swing Trade Candidate?

According to this system, the best candidates have:

1. **Recent Reversal** (Pattern)
   - Was going down for 5+ days
   - Now going up for 2-4 days
   - The longer the downtrend, the better

2. **Good Technical Setup**
   - RSI in oversold zone (≤40)
   - EMA20 close to EMA50 (convergence)
   - Trading 20-40% below 52-week high
   - Strong GEM rank (fundamentals)

3. **High Composite Score**
   - CompositeScore ≥ 80 → Strong_Buy
   - CompositeScore ≥ 65 → Buy
   - CompositeScore ≥ 50 → Weak_Buy

## ⚙️ Tuning for Your Strategy

### If you want MORE candidates:
- Lower `--min_prior_down` to 4
- Lower `--min_recent_up` to 1

### If you want HIGHER QUALITY only:
- Raise `--min_prior_down` to 8-10
- Raise `--pattern_weight` to 0.6

### If you prefer FUNDAMENTALS over patterns:
- Lower `--pattern_weight` to 0.3
- Raise `--technical_weight` to 0.7

## 🐛 Troubleshooting

### "No patterns detected"
- Lower the `--min_prior_down` threshold
- Check if historical price data exists

### "Few Strong_Buy signals"
- This is normal - Strong_Buy is for exceptional setups only
- Look at "Buy" and "Weak_Buy" signals too
- Or lower thresholds in code

### Slow performance
- Pattern detection fetches history for each stock
- Consider running for smaller watchlists
- Or run during off-peak hours

## 💡 Tips

1. **Run Daily** - Market conditions change, rankings update
2. **Compare Dates** - Track which stocks consistently rank high
3. **Combine with Other Analysis** - This is a screening tool, not a complete strategy
4. **Check Volume** - Verify stocks have adequate liquidity
5. **Review Fundamentals** - Check the actual company behind high-ranked stocks

## 📞 Need Help?

- Check logs for specific errors
- Review existing data completeness
- Verify database connectivity
- Run test script to validate system

---

**Ready to find your next swing trade? Run the system and review the top rankings!**

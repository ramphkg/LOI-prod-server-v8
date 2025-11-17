import pandas as pd
from tas_swing_scoring import swing_score

# Construct sample rows emulating typical combinations
rows = [
    {"SignalClassifier_ML": 3, "SignalClassifier_Rules": 0, "Trend": "Bull[TrendingUp]", "TMA21_50_X": 1, "RSIUpTrend": 1, "ADX": 28,
     "TrendReversal_Rules": "BullishReversalModerate", "TrendReversal_ML": "BullishReversal-MLModerate", "LastTrendDays": 5,
     "Pct2H52": 10, "PctfL52": 30, "RSIUturnTypeOld": "BullishReversal[Strong]"},
    {"SignalClassifier_ML": -2, "SignalClassifier_Rules": -3, "Trend": "Bear[TrendingDown]", "TMA21_50_X": -1, "RSIUpTrend": 0, "ADX": 30,
     "TrendReversal_Rules": "BearishReversalModerate", "TrendReversal_ML": "BearishReversal-MLModerate", "LastTrendDays": -9,
     "Pct2H52": 1.5, "PctfL52": 50, "RSIUturnTypeOld": "BearishReversal[Weak]"},
    {"SignalClassifier_ML": 0, "SignalClassifier_Rules": 4, "Trend": "Bull[PullbackInBull]", "TMA21_50_X": 0, "RSIUpTrend": 1, "ADX": 22,
     "TrendReversal_Rules": "NoReversal", "TrendReversal_ML": "NoReversal", "LastTrendDays": 3,
     "Pct2H52": 3, "PctfL52": 4, "RSIUturnTypeOld": "ERR_NO_TREND"},
    {"SignalClassifier_ML": 0, "SignalClassifier_Rules": 0, "Trend": "Neutral[ShortTrend]", "TMA21_50_X": 0, "RSIUpTrend": 0, "ADX": 15,
     "TrendReversal_Rules": "NoReversal", "TrendReversal_ML": "NoReversal", "LastTrendDays": 0,
     "Pct2H52": 20, "PctfL52": 20, "RSIUturnTypeOld": "BearishReversal[Strong]"},
]

for i, r in enumerate(rows, 1):
    score = swing_score(pd.Series(r))
    print(f"Row {i} score: {score}  (RSIUturnTypeOld={r['RSIUturnTypeOld']})")

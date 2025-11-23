"""
Complete the following function specification properly "the file is supposed to have just a function that takes in a df of historic price, use indicators.py to build necessary features, then use the features to detect a reversals based on smart algorithm also confirmed just stared reversing from RSI overbought level (for signs of bearish reversal) or just started reversing from overbought levels (for signs of bullish reversals) and accordingly decide yesReversal or noReversal. Then for YesReversal ones, then use relevant and required features and machine learning to predict whether the detected reversal it is confirmed and highly likely to continue or or not. Accordingly generate string labels like BullishRerversal-MLStrong /Weak or /Moderate. Similarly on Bearish side. The function should return the labels. add more criteria into the reversal detection part to ensure that the trend that prevailed prior to the current signs of reversal was a reasonably prolonged one and not part of a ranging trend. Basically need to filter out any false positives"


Overview
Below is a single, drop‑in function that:
Accepts a pandas DataFrame of historical OHLCV data.
Builds features using indicators.py (with safe in-function fallbacks if indicators.py isn’t available).
Detects fresh reversals:
Bearish: RSI just crossed down from overbought.
Bullish: RSI just crossed up from oversold.
Filters out false positives by requiring a reasonably prolonged trend (not a range) prior to reversal (e.g., ADX, EMA slope, majority of bars aligned with trend, band extremes).
Trains a simple, in-sample, walk-forward style logistic model on past reversal signals to predict if similar reversals historically continued, then applies it to each detected signal to produce:
BullishReversal-MLStrong / BullishReversal-MLModerate / BullishReversal-MLWeak
BearishReversal-MLStrong / BearishReversal-MLModerate / BearishReversal-MLWeak
Returns a pandas Series of labels (indexed to your input df), defaulting to "NoReversal" and assigning the above labels at the bars where a reversal is detected.
Notes:
Assumes df has columns: open, high, low, close, volume, with a DateTimeIndex.
Uses indicators.py if available (rsi, adx, ema, sma, macd, atr, bollinger_bands). If not, internal fallbacks will compute these within the function.
For ML, if there are too few historical signals to train (or scikit-learn is unavailable), it falls back to a rule-based confidence score.

"""
import numpy as np
import pandas as pd

def detect_and_label_reversals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    adx_period: int = 14,
    ema_fast_len: int = 20,
    ema_slow_len: int = 50,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    # cross_lookback: int = 5,  # Removed: no longer used
    # min_rsi_hold_bars: int = 1,  # Removed: no longer used
    trend_lookback: int = 30,
    min_trend_bars_ratio: float = 0.5,  # Relaxed from 0.6
    min_trend_run: int = 3,
    adx_trend_threshold: float = 15.0,  # Relaxed from 18.0
    ema_slope_min: float = 0.005,  # Relaxed from 0.01
    divergence_lookback: int = 0,  # 0 = disable hard divergence filter; used as a scoring hint
    require_band_extreme: bool = False,  # False = do not hard-require BB touch; used as a scoring hint
    lookahead_bars: int = 10,
    target_rr_atr: float = 1.5,
    stop_rr_atr: float = 1.0,
    min_signals_to_train: int = 25,
    random_state: int = 42,
    fix_adx_mapping: bool = True,  # auto-fix the ADX/DI mis-mapping from indicators.py
):
    """
    Detects fresh RSI-based reversals with robust yet not over-restrictive trend filters and
    classifies them via a simple ML model (fallback to rules if insufficient data).
    Returns a Series of labels aligned to the input df index.
    Notes:
    - Uses indicators.compute_indicators to build all features (RSI, MACD, ADX/DI, ATR, EMAs, BB, etc.)
    - Auto-corrects the ADX/DI mapping issue from indicators.py by default
    - Emits one of: "NoReversal", "BullishReversal-MLStrong/Moderate/Weak", "BearishReversal-MLStrong/Moderate/Weak"
    - Updated: RSI now detects turn while extreme (not strict cross) for more signals.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=object)
    # Preserve original index
    orig_index = df.index
    work = df.copy()
    work["_orig_index"] = orig_index
    # Build indicators via indicators.py
    try:
        from indicators import compute_indicators
    except Exception as e:
        raise ImportError("indicators.py with compute_indicators(df, params) is required") from e
    params = {
        "rsi_period": rsi_period,
        "adx_period": adx_period,
        "atr_period": rsi_period,
        "bb_len": bb_period,
        "bb_std": bb_std,
        "ema_20": ema_fast_len,
        "ema_50": ema_slow_len,
    }
    ind = compute_indicators(work, params=params)
    # Column resolution
    close_col = "Close" if "Close" in ind.columns else ("close" if "close" in ind.columns else None)
    high_col = "High" if "High" in ind.columns else ("high" if "high" in ind.columns else None)
    low_col = "Low" if "Low" in ind.columns else ("low" if "low" in ind.columns else None)
    open_col = "Open" if "Open" in ind.columns else ("open" if "open" in ind.columns else None)
    vol_col = "Volume" if "Volume" in ind.columns else ("volume" if "volume" in ind.columns else None)
    if close_col is None:
        raise ValueError("Input df must contain a Close/close column")
    close = ind[close_col].astype(float)
    high = ind[high_col].astype(float) if high_col else close
    low = ind[low_col].astype(float) if low_col else close
    open_ = ind[open_col].astype(float) if open_col else close
    volume = ind[vol_col].astype(float) if vol_col else pd.Series(0.0, index=ind.index)
    # Indicators provided by indicators.py
    rsi = ind.get("rsi")
    macd_line = ind.get("macd")
    macd_signal = ind.get("macd_signal")
    macd_hist = ind.get("macd_histogram")
    adx_col = ind.get("adx")
    plus_di_col = ind.get("plus_di")
    minus_di_col = ind.get("minus_di")
    atr = ind.get("atr")
    ema_fast = ind.get("ema_20")
    ema_slow = ind.get("ema_50")
    bb_upper = ind.get("bb_upper")
    bb_mid = ind.get("bb_middle")
    bb_lower = ind.get("bb_lower")
    mfi = ind.get("mfi")
    # Fix ADX/DI mis-mapping from indicators.py: it assigns [DMP, DMN, ADX] to ['adx','plus_di','minus_di'] respectively.
    # Correct mapping should be: adx = ADX, plus_di = DMP, minus_di = DMN.
    if fix_adx_mapping and all(c in ind.columns for c in ["adx", "plus_di", "minus_di"]):
        # Re-map safely into new variables used below
        adx = minus_di_col  # this is actually ADX in indicators.py
        plus_di = adx_col   # this is actually +DI (DMP) in indicators.py
        minus_di = plus_di_col  # this is actually -DI (DMN) in indicators.py
    else:
        adx = adx_col
        plus_di = plus_di_col
        minus_di = minus_di_col
    # Safety fills
    def _nz(x, fill=0.0):
        return x.replace([np.inf, -np.inf], np.nan).fillna(fill) if isinstance(x, pd.Series) else (fill if x is None else x)
    rsi = _nz(rsi, 50.0)
    macd_line = _nz(macd_line)
    macd_signal = _nz(macd_signal)
    macd_hist = _nz(macd_hist)
    adx = _nz(adx)
    plus_di = _nz(plus_di)
    minus_di = _nz(minus_di)
    atr = _nz(atr, float(close.mean() * 0.01) if close.notna().any() else 1.0)
    ema_fast = _nz(ema_fast, close)
    ema_slow = _nz(ema_slow, close)
    bb_upper = _nz(bb_upper, close)
    bb_mid = _nz(bb_mid, close)
    bb_lower = _nz(bb_lower, close)
    mfi = _nz(mfi, 50.0)
    # Helpers
    def _safe_div(a, b, default=np.nan):
        b2 = b.copy() if isinstance(b, pd.Series) else b
        if isinstance(b2, pd.Series):
            b2 = b2.replace(0, np.nan)
        elif b2 == 0:
            b2 = np.nan
        out = a / b2
        return out.fillna(default) if isinstance(out, pd.Series) else (default if np.isnan(out) else out)
    # Scale-invariant slopes
    lookback_slow = max(5, min(trend_lookback, len(ind) // 4))
    ema_slow_slope = _safe_div(ema_slow - ema_slow.shift(lookback_slow), atr * lookback_slow, 0.0)
    ema_fast_slope = _safe_div(ema_fast - ema_fast.shift(5), atr * 5, 0.0)
    bb_width = _safe_div((bb_upper - bb_lower), bb_mid.abs().replace(0, np.nan), np.nan).fillna(0.0)
    price_above_ema_slow = (close > ema_slow).astype(int)
    price_below_ema_slow = (close < ema_slow).astype(int)
    vol_ma = volume.rolling(20, min_periods=5).mean()
    vol_spike = _safe_div(volume, vol_ma, 1.0).fillna(1.0)
    # Candles
    body = (close - open_).abs()
    upper_wick = (pd.concat([high, open_, close], axis=1).max(axis=1) - close.where(close >= open_, open_))
    lower_wick = (close.where(close <= open_, open_) - pd.concat([low, open_, close], axis=1).min(axis=1)).abs()
    bullish_engulf = ((close > open_) & (close.shift(1) < open_.shift(1)) & (close >= open_.shift(1)) & (open_ <= close.shift(1))).fillna(False)
    bearish_engulf = ((close < open_) & (close.shift(1) > open_.shift(1)) & (close <= open_.shift(1)) & (open_ >= close.shift(1))).fillna(False)
    hammer = ((lower_wick >= 2 * body) & (upper_wick <= 0.5 * body)).fillna(False)
    shooting_star = ((upper_wick >= 2 * body) & (lower_wick <= 0.5 * body)).fillna(False)
    # Updated RSI “just started reversing” logic: detect turn while extreme (no cross required)
    rsi_bearish_cross = (rsi.shift(1) > rsi_overbought) & (rsi < rsi.shift(1))
    rsi_bullish_cross = (rsi.shift(1) < rsi_oversold) & (rsi > rsi.shift(1))
    # Optional divergence hints (not required)
    def _bearish_div_hint(idx, win=30):
        if win <= 0 or idx - 2 * win < 0:
            return False
        prev = slice(idx - 2 * win, idx - win)
        curr = slice(idx - win, idx)
        try:
            ph_prev = close.iloc[prev].idxmax()
            ph_curr = close.iloc[curr].idxmax()
            return (close.loc[ph_curr] > close.loc[ph_prev]) and (rsi.loc[ph_curr] < rsi.loc[ph_prev])
        except Exception:
            return False
    def _bullish_div_hint(idx, win=30):
        if win <= 0 or idx - 2 * win < 0:
            return False
        prev = slice(idx - 2 * win, idx - win)
        curr = slice(idx - win, idx)
        try:
            pl_prev = close.iloc[prev].idxmin()
            pl_curr = close.iloc[curr].idxmin()
            return (close.loc[pl_curr] < close.loc[pl_prev]) and (rsi.loc[pl_curr] > rsi.loc[pl_prev])
        except Exception:
            return False
    def _run_length_bool(series_bool, i):
        cnt = 0
        j = i - 1
        while j >= 0 and bool(series_bool.iloc[j]):
            cnt += 1
            j -= 1
        return cnt
    # Prolonged-trend filters (relaxed)
    def _prolonged_uptrend(i):
        start = max(0, i - trend_lookback)
        sl = slice(start, i)
        if i - start < max(10, int(0.4 * trend_lookback)):
            return False
        cond_ratio = price_above_ema_slow.iloc[sl].mean() >= min_trend_bars_ratio
        adx_ok = adx.iloc[sl].median() >= adx_trend_threshold
        di_ok = plus_di.iloc[sl].median() > minus_di.iloc[sl].median()
        slope_ok = ema_slow_slope.iloc[i] >= ema_slope_min
        run_ok = _run_length_bool(close > ema_slow, i) >= min_trend_run
        if require_band_extreme:
            extreme_ok = (close.iloc[i - 1] >= bb_upper.iloc[i - 1]) if i - 1 >= 0 else False
            return bool(cond_ratio and adx_ok and di_ok and slope_ok and run_ok and extreme_ok)
        return bool(cond_ratio and adx_ok and di_ok and slope_ok and run_ok)
    def _prolonged_downtrend(i):
        start = max(0, i - trend_lookback)
        sl = slice(start, i)
        if i - start < max(10, int(0.4 * trend_lookback)):
            return False
        cond_ratio = price_below_ema_slow.iloc[sl].mean() >= min_trend_bars_ratio
        adx_ok = adx.iloc[sl].median() >= adx_trend_threshold
        di_ok = minus_di.iloc[sl].median() > plus_di.iloc[sl].median()
        slope_ok = ema_slow_slope.iloc[i] <= -ema_slope_min
        run_ok = _run_length_bool(close < ema_slow, i) >= min_trend_run
        if require_band_extreme:
            extreme_ok = (close.iloc[i - 1] <= bb_lower.iloc[i - 1]) if i - 1 >= 0 else False
            return bool(cond_ratio and adx_ok and di_ok and slope_ok and run_ok and extreme_ok)
        return bool(cond_ratio and adx_ok and di_ok and slope_ok and run_ok)
    # Signals
    labels = pd.Series(index=ind.index, dtype=object)
    labels[:] = "NoReversal"
    signal = pd.Series(0, index=ind.index, dtype=int)
    warmup = int(max(rsi_period, adx_period, bb_period, ema_slow_len) + 2)
    for i in range(len(ind)):
        if i < warmup:
            continue
        bear_candidate = bool(rsi_bearish_cross.iloc[i])
        bull_candidate = bool(rsi_bullish_cross.iloc[i])
        if bear_candidate and _prolonged_uptrend(i):
            # No hard divergence requirement
            signal.iloc[i] = -1
        if bull_candidate and _prolonged_downtrend(i):
            signal.iloc[i] = 1
    # Feature matrix
    feats = pd.DataFrame({
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "ema_fast_slope": ema_fast_slope,
        "ema_slow_slope": ema_slow_slope,
        "bb_z": _safe_div((close - bb_mid), ((bb_upper - bb_lower) / 2.0).replace(0, np.nan), 0.0),
        "bb_width": bb_width,
        "vol_spike": vol_spike,
        "mfi": mfi,
        "candle_bull_engulf": bullish_engulf.astype(int),
        "candle_bear_engulf": bearish_engulf.astype(int),
        "candle_hammer": hammer.astype(int),
        "candle_shooting": shooting_star.astype(int),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Label historical signals for ML using ATR-based target/stop
    def _label_outcome(i, side):
        if i >= len(ind) - 1:
            return np.nan
        future_end = min(len(ind) - 1, i + lookahead_bars)
        if future_end <= i + 1:
            return np.nan
        entry = close.iloc[i]
        atr_i = atr.iloc[i] if atr.iloc[i] > 0 else max(1e-8, float(close.iloc[i] * 0.01))
        tgt = target_rr_atr * atr_i
        stp = stop_rr_atr * atr_i
        highs = high.iloc[i + 1: future_end + 1]
        lows = low.iloc[i + 1: future_end + 1]
        if side == 1:
            hit_tgt = (highs >= entry + tgt).values
            hit_stp = (lows <= entry - stp).values
        else:
            hit_tgt = (lows <= entry - tgt).values
            hit_stp = (highs >= entry + stp).values
        t_indices = np.where(hit_tgt)[0]
        s_indices = np.where(hit_stp)[0]
        t_first = t_indices[0] if t_indices.size else None
        s_first = s_indices[0] if s_indices.size else None
        if t_first is None and s_first is None:
            return 0
        if t_first is not None and s_first is None:
            return 1
        if s_first is not None and t_first is None:
            return 0
        return 1 if t_first < s_first else 0
    sig_idx = np.where(signal.values != 0)[0]
    X_list, y_list = [], []
    for i in sig_idx:
        if i >= len(ind) - lookahead_bars - 1:
            continue
        side = signal.iloc[i]
        y_i = _label_outcome(i, side)
        if np.isnan(y_i):
            continue
        row = feats.iloc[i].copy()
        row["side"] = side
        X_list.append(row.values)
        y_list.append(int(y_i))
    X = np.array(X_list) if X_list else np.empty((0, feats.shape[1] + 1))
    y = np.array(y_list, dtype=int) if y_list else np.empty((0,), dtype=int)
    # ML training (balanced logistic regression). Fallback to rule score
    use_ml = False
    if X.shape[0] >= min_signals_to_train:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=random_state)
            clf.fit(Xs, y)
            use_ml = True
        except Exception:
            use_ml = False
    # Rule-based confidence score
    def _rule_score(i, side):
        adx_c = float(np.tanh((adx.iloc[i] - adx_trend_threshold) / 10.0) * 0.25 + 0.25)
        ema_c = float(np.clip(ema_fast_slope.iloc[i], -1.5, 1.5) / 3.0 + 0.5)
        macd_c = float(np.clip(macd_hist.iloc[i], -2.0, 2.0) / 4.0 + 0.5)
        band_hint = ((side == -1 and close.iloc[i - 1] >= bb_upper.iloc[i - 1]) or
                     (side == 1 and close.iloc[i - 1] <= bb_lower.iloc[i - 1]))
        band_c = 0.65 if band_hint else 0.45
        vol_c = float(np.clip(vol_spike.iloc[i] / 3.0, 0.0, 1.0))
        candle_c = 0.65 if ((side == 1 and (bullish_engulf.iloc[i] or hammer.iloc[i])) or
                            (side == -1 and (bearish_engulf.iloc[i] or shooting_star.iloc[i]))) else 0.45
        align = 0.65 if ((side == 1 and ema_fast_slope.iloc[i] > 0 and macd_hist.iloc[i] > 0) or
                         (side == -1 and ema_fast_slope.iloc[i] < 0 and macd_hist.iloc[i] < 0)) else 0.45
        # Divergence hint if enabled
        div_c = 0.05 if (divergence_lookback > 0 and
                         (_bullish_div_hint(i, divergence_lookback) if side == 1 else _bearish_div_hint(i, divergence_lookback))) else 0.0
        # MFI hint (leaving extremes)
        mfi_boost = 0.05 if ((side == -1 and mfi.iloc[i] > 70) or (side == 1 and mfi.iloc[i] < 30)) else 0.0
        score = 0.2 * adx_c + 0.2 * ema_c + 0.2 * macd_c + 0.15 * band_c + 0.15 * vol_c + 0.1 * candle_c + 0.1 * align + div_c + mfi_boost
        return float(np.clip(score, 0.0, 1.0))
    def _prob_to_strength(p):
        if p >= 0.70:
            return "MLStrong"
        elif p >= 0.55:
            return "MLModerate"
        else:
            return "MLWeak"
    # Assign labels
    out_labels = pd.Series("NoReversal", index=ind.index, dtype=object)
    for i in sig_idx:
        side = signal.iloc[i]
        if side == 0:
            continue
        if use_ml:
            row = feats.iloc[i].copy()
            row["side"] = side
            x = row.values.reshape(1, -1)
            p = float(clf.predict_proba(scaler.transform(x))[0, 1])
        else:
            p = _rule_score(i, side)
        strength = _prob_to_strength(p)
        out_labels.iloc[i] = f"{'Bullish' if side == 1 else 'Bearish'}Reversal-{strength}"
    # Map labels back to the original index using preserved column
    if "_orig_index" in ind.columns:
        mapped = pd.Series("NoReversal", index=orig_index, dtype=object)
        idx_to_orig = ind["_orig_index"]
        non_default = out_labels[out_labels != "NoReversal"]
        mapped.loc[idx_to_orig.loc[non_default.index]] = non_default.values
        return mapped
    else:
        return out_labels
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Optional ML pieces
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    _sklearn_available = True
except Exception:
    _sklearn_available = False


def detect_reversal_pro(
    df: pd.DataFrame,
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    open_col: str = "Open",
    vol_col: str = "Volume",
    min_recent_down: int = 5,
    min_total_move: float = 0.03,        # 3% prior move minimum
    rsi_period: int = 14,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    macd_fast: int = 12, macd_slow: int = 26, macd_sig: int = 9,
    ema_short: int = 8,
    adx_period: int = 14,
    atr_period: int = 14,
    mfi_period: int = 14,
    max_atr_pct: float = 0.12,
    vol_confirm_ratio: float = 0.75,
    slope_window_max: int = 10,
    strong_thresh: float = 0.70,
    moderate_thresh: float = 0.45,
    weak_thresh: float = 0.20,
    market_index: Optional[pd.Series] = None,   # optional market index series (Close) for regime check
    min_avg_volume: Optional[float] = None,     # liquidity filter e.g. 100k
    risk_per_trade_pct: float = 0.01,           # position sizing default (1% of equity)
    equity: float = 100000.0,
    train_meta: bool = False,  # optionally train a small logistic meta-model on historical features
    meta_lookahead_days: int = 5,
    verbose: bool = False
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Smart reversal detection function.

    Returns:
      - label (str): one of "BullishReversalStrong"/"Moderate"/"Weak", "BearishReversal...", or "NoReversal"
      - details (dict) if verbose=True: breakdown of scores, indicators, stop/target, sizing, meta info

    Notes:
      - df: DataFrame containing at least close_col (best if contains OHLCV).
      - Function is robust, modular and will fall back gracefully if some series are missing.
      - Set train_meta=True to let a simple time-series logistic calibrate weights on historical signals (sklearn required).
    """

    # ------------------------------
    # 1) Basic validation & preprocessing
    # ------------------------------
    if isinstance(close_col, int):
        close = df.iloc[:, close_col].astype(float).dropna().reset_index(drop=True)
    else:
        if close_col not in df:
            raise ValueError(f"Column '{close_col}' not found in dataframe")
        close = df[close_col].astype(float).reset_index(drop=True)

    n = len(close)
    if n < 30:
        # not enough history for reliable multi-indicator signals
        if verbose:
            return "NoReversal", {"reason": "insufficient_history", "n": n}
        return "NoReversal", None

    # preserve index
    # df.index.is_all_dates is deprecated; use inferred_type to detect datetime-like index
    try:
        inferred = getattr(df.index, "inferred_type", "")
        if isinstance(inferred, str) and inferred.startswith("datetime"):
            idx = df.index
        else:
            idx = pd.RangeIndex(n)
    except Exception:
        # fallback to checking DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        else:
            idx = pd.RangeIndex(n)

    # optional series
    has_highlow = (high_col in df) and (low_col in df)
    has_volume = (vol_col in df)
    high = df[high_col].astype(float).reset_index(drop=True) if has_highlow else close
    low  = df[low_col].astype(float).reset_index(drop=True) if has_highlow else close
    vol  = df[vol_col].astype(float).reset_index(drop=True) if has_volume else None
    open_s = df[open_col].astype(float).reset_index(drop=True) if open_col in df else close

    # core helpers
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=period, min_periods=period).mean()
        ma_down = down.rolling(window=period, min_periods=period).mean()
        rs = ma_up / (ma_down + 1e-9)
        r = 100 - (100 / (1 + rs))
        return r.fillna(50)

    def macd(series: pd.Series, fast=12, slow=26, sig=9):
        ef = ema(series, fast)
        es = ema(series, slow)
        ml = ef - es
        sigl = ml.ewm(span=sig, adjust=False).mean()
        hist = ml - sigl
        return ml, sigl, hist

    def atr(high_s, low_s, close_s, period=14):
        tr1 = high_s - low_s
        tr2 = (high_s - close_s.shift(1)).abs()
        tr3 = (low_s - close_s.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr_series.bfill()

    def adx(high_s, low_s, close_s, period=14):
        up_move = high_s.diff()
        down_move = -low_s.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        tr1 = high_s - low_s
        tr2 = (high_s - close_s.shift(1)).abs()
        tr3 = (low_s - close_s.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-9))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-9))
        dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) ) * 100
        adx_s = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx_s.bfill(), plus_di.fillna(0), minus_di.fillna(0)

    def mfi(high_s, low_s, close_s, vol_s, period=14):
        if vol_s is None:
            return pd.Series(np.nan, index=close_s.index)
        typical = (high_s + low_s + close_s) / 3.0
        mp = typical * vol_s
        pos = mp.where(typical > typical.shift(1), 0.0).rolling(window=period).sum()
        neg = mp.where(typical < typical.shift(1), 0.0).rolling(window=period).sum().abs()
        mfr = pos / (neg + 1e-9)
        mfi_s = 100 - (100 / (1 + mfr))
        return mfi_s.fillna(50)

    def cmf(high_s, low_s, close_s, vol_s, period=20):
        if vol_s is None:
            return pd.Series(np.nan, index=close_s.index)
        mfv = ((close_s - low_s) - (high_s - close_s)) / (high_s - low_s + 1e-9) * vol_s
        return mfv.rolling(window=period, min_periods=1).sum() / (vol_s.rolling(window=period, min_periods=1).sum() + 1e-9)

    def obv(close_s, vol_s):
        if vol_s is None:
            return pd.Series(np.nan, index=close_s.index)
        direction = np.sign(close_s.diff().fillna(0))
        obv_s = (direction * vol_s).fillna(0).cumsum()
        return obv_s

    # Calculate indicators
    rsi_s = rsi(close, rsi_period)
    macd_line, macd_signal, macd_hist = macd(close, macd_fast, macd_slow, macd_sig)
    ema_short_s = ema(close, ema_short)
    atr_s = atr(high, low, close) if has_highlow else pd.Series(np.nan, index=close.index)
    adx_s, plus_di_s, minus_di_s = adx(high, low, close) if has_highlow else (pd.Series(np.nan, index=close.index), None, None)
    mfi_s = mfi(high, low, close, vol, mfi_period) if has_volume else pd.Series(np.nan, index=close.index)
    cmf_s = cmf(high, low, close, vol) if has_volume else pd.Series(np.nan, index=close.index)
    obv_s = obv(close, vol) if has_volume else pd.Series(np.nan, index=close.index)

    # ------------------------------
    # 2) Pattern detection (consecutive runs), with tolerance for small counter days
    # ------------------------------
    diffs = close.diff()
    up_flags = diffs > 0
    down_flags = diffs < 0
    last_idx = n - 1

    # helper to measure consecutive run allowing occasional neutral days
    def count_run(flags: pd.Series, allow_neutral: int = 1, max_backward: int = 200):
        """Count consecutive True from end allowing up to allow_neutral False/NaN in between."""
        i = last_idx
        count = 0
        neutrals = 0
        steps = 0
        while i > 0 and steps < max_backward:
            if flags.iloc[i]:
                count += 1
            else:
                # treat small up/downs as neutral if within allow_neutral
                if neutrals < allow_neutral and not pd.isna(flags.iloc[i]):
                    neutrals += 1
                else:
                    break
            i -= 1
            steps += 1
        return count, i  # i is index before run started (approx)

    latest_up_days, idx_before_up = count_run(up_flags, allow_neutral=1)
    # count prior down run immediately before latest up
    # We find the end of prior down as idx_before_up
    def count_run_before(flags: pd.Series, end_idx: int, allow_neutral: int = 1, max_back: int = 200):
        i = end_idx
        count = 0
        neutrals = 0
        steps = 0
        while i > 0 and steps < max_back:
            if flags.iloc[i]:
                count += 1
            else:
                if neutrals < allow_neutral and not pd.isna(flags.iloc[i]):
                    neutrals += 1
                else:
                    break
            i -= 1
            steps += 1
        return count, i

    # For bullish: prior down run then latest up run at tail
    recent_down_days, idx_before_down = count_run_before(down_flags, idx_before_up, allow_neutral=1)
    # For bearish: prior up run then latest down run at tail
    latest_down_days, idx_before_down2 = count_run(down_flags, allow_neutral=1)
    recent_up_days, idx_before_up2 = count_run_before(up_flags, idx_before_down2, allow_neutral=1)

    # convenience window indices
    up_start_idx = idx_before_up + 1
    down_start_idx = idx_before_down + 1
    up_start_idx_bear = idx_before_up2 + 1
    down_start_idx_bear = idx_before_down2 + 1

    # ------------------------------
    # 3) Pattern scoring (time, magnitude, accel)
    # ------------------------------
    def linear_slope(series_slice: np.ndarray):
        if len(series_slice) < 2:
            return 0.0
        x = np.arange(len(series_slice))
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, series_slice, rcond=None)[0]
        return float(m)

    def pattern_score(prior_len: int, latest_len: int, prior_start: int, prior_end: int,
                      latest_start: int, latest_end: int, direction: str = "bull"):
        # minimal checks
        if prior_len < min_recent_down or latest_len < 1 or prior_len <= latest_len:
            return 0.0, {}
        # price references (use price just before the prior run if available)
        pre_idx = max(0, prior_start - 1)
        pre_price = close.iloc[pre_idx]
        prior_end_price = close.iloc[prior_end]
        prior_move = (prior_end_price - pre_price) / (pre_price + 1e-9)
        latest_pre_idx = max(0, latest_start - 1)
        latest_pre_price = close.iloc[latest_pre_idx]
        latest_end_price = close.iloc[latest_end]
        latest_move = (latest_end_price - latest_pre_price) / (latest_pre_price + 1e-9)

        if direction == "bull":
            total_drop = -prior_move
            if total_drop < min_total_move:
                return 0.0, {"reason": "drop_too_small", "total_drop": total_drop}
            up_gain = latest_move
            time_factor = float(max(0, prior_len - latest_len)) / (prior_len + 1e-9)
            rel_gain_after_drop = up_gain / (up_gain + total_drop + 1e-9)
            magnitude_factor = 1.0 - np.clip(rel_gain_after_drop, 0.0, 1.0)
            # slopes
            w_prior = min(slope_window_max, prior_len)
            prior_slice = close.iloc[max(0, prior_end - w_prior + 1): prior_end + 1].to_numpy()
            latest_slice = close.iloc[latest_start: latest_end + 1].to_numpy()
            prior_slope = linear_slope(prior_slice)
            latest_slope = linear_slope(latest_slice)
            slope_change = latest_slope - prior_slope
            accel_factor = np.tanh(slope_change / (abs(prior_slope) + 1e-9))
            accel_factor = np.clip((accel_factor + 1.0) / 2.0, 0.0, 1.0)
            score = 0.45 * time_factor + 0.35 * magnitude_factor + 0.20 * accel_factor
            meta = {"time_factor": time_factor, "magnitude_factor": magnitude_factor, "accel_factor": accel_factor,
                    "total_drop": total_drop, "up_gain": up_gain}
            return float(np.clip(score, 0.0, 1.0)), meta
        else:
            total_gain = prior_move
            if total_gain < min_total_move:
                return 0.0, {"reason": "gain_too_small", "total_gain": total_gain}
            down_loss = -latest_move
            time_factor = float(max(0, prior_len - latest_len)) / (prior_len + 1e-9)
            rel_loss_after_gain = down_loss / (down_loss + total_gain + 1e-9)
            magnitude_factor = 1.0 - np.clip(rel_loss_after_gain, 0.0, 1.0)
            w_prior = min(slope_window_max, prior_len)
            prior_slice = close.iloc[max(0, prior_end - w_prior + 1): prior_end + 1].to_numpy()
            latest_slice = close.iloc[latest_start: latest_end + 1].to_numpy()
            prior_slope = linear_slope(prior_slice)
            latest_slope = linear_slope(latest_slice)
            slope_change = prior_slope - latest_slope
            accel_factor = np.tanh(slope_change / (abs(prior_slope) + 1e-9))
            accel_factor = np.clip((accel_factor + 1.0) / 2.0, 0.0, 1.0)
            score = 0.45 * time_factor + 0.35 * magnitude_factor + 0.20 * accel_factor
            meta = {"time_factor": time_factor, "magnitude_factor": magnitude_factor, "accel_factor": accel_factor,
                    "total_gain": total_gain, "down_loss": down_loss}
            return float(np.clip(score, 0.0, 1.0)), meta

    # ------------------------------
    # 4) Confirmation scoring (rich ensemble)
    # ------------------------------
    def confirmation_score(direction: str, prior_start: int, prior_end: int, latest_start: int, latest_end: int):
        comps = {}
        score_sum = 0.0

        # RSI
        last_rsi = float(rsi_s.iloc[last_idx])
        if direction == "bull":
            rsi_ok = 1.0 if last_rsi <= rsi_oversold else max(0.0, (rsi_oversold * 1.2 - last_rsi) / (rsi_oversold * 1.2 + 1e-9))
        else:
            rsi_ok = 1.0 if last_rsi >= rsi_overbought else max(0.0, (last_rsi - rsi_overbought * 0.8) / (100 - rsi_overbought * 0.8 + 1e-9))
        comps['rsi'] = {"value": last_rsi, "score": rsi_ok}
        score_sum += 0.20 * rsi_ok

        # MACD
        macd_recent = macd_line.iloc[latest_start:last_idx + 1]
        macdsig_recent = macd_signal.iloc[latest_start:last_idx + 1]
        macd_ok = 0.0
        if (macd_recent > macdsig_recent).any():
            macd_ok = 1.0
        else:
            hist = macd_hist.iloc[latest_start:last_idx + 1].diff().fillna(0)
            macd_ok = float(np.clip((hist.iloc[-1] - hist.min()) / (abs(hist.max()) + 1e-9), 0.0, 1.0))
        comps['macd'] = {"macd_last": float(macd_line.iloc[last_idx]), "signal_last": float(macd_signal.iloc[last_idx]), "score": macd_ok}
        score_sum += 0.20 * macd_ok

        # Volume / OBV confirmation
        vol_ok = 0.5
        if has_volume and vol is not None:
            avg_vol_prior = float(vol.iloc[prior_start:prior_end + 1].mean()) if prior_end >= prior_start else float(vol.iloc[prior_start])
            avg_vol_latest = float(vol.iloc[latest_start:latest_end + 1].mean())
            ratio = (avg_vol_latest + 1e-9) / (avg_vol_prior + 1e-9)
            vol_ratio_score = float(np.clip((ratio - vol_confirm_ratio) / (3.0 - vol_confirm_ratio), 0.0, 1.0))
            obv_ok = 0.0
            if not obv_s.isna().all():
                obv_start = obv_s.iloc[latest_start]
                obv_last = obv_s.iloc[last_idx]
                obv_ok = 1.0 if obv_last > obv_start else max(0.0, (obv_last - obv_s.iloc[prior_start]) / (abs(obv_s.iloc[prior_start]) + 1e-9))
            vol_ok = 0.6 * vol_ratio_score + 0.4 * obv_ok
            comps['volume'] = {"avg_prior": avg_vol_prior, "avg_latest": avg_vol_latest, "ratio": ratio, "score": vol_ok}
        else:
            comps['volume'] = {"note": "no volume data", "score": vol_ok}
        score_sum += 0.18 * vol_ok

        # EMA short-term confirmation
        ema_val = float(ema_short_s.iloc[last_idx])
        ema_slope = float((ema_short_s.iloc[last_idx] - ema_short_s.iloc[max(0, last_idx - 3)]) / (3 if last_idx >= 3 else 1))
        ema_ok = 1.0 if close.iloc[last_idx] > ema_val and ema_slope > 0 else max(0.0, (close.iloc[last_idx] - ema_val) / (ema_val + 1e-9))
        comps['ema'] = {"ema_short": ema_val, "ema_slope_3": ema_slope, "score": ema_ok}
        score_sum += 0.12 * ema_ok

        # ADX / trend exhaustion
        adx_ok = 0.5
        if not adx_s.isna().all():
            adx_last = float(adx_s.iloc[last_idx])
            adx_prior = float(adx_s.iloc[max(0, prior_start)])
            if adx_last < adx_prior:
                adx_ok = float(np.clip((adx_prior - adx_last) / (adx_prior + 1e-9), 0.0, 1.0))
            else:
                adx_ok = 1.0 if adx_last < 25 else max(0.0, (40 - adx_last) / 15.0)
            comps['adx'] = {"adx_last": adx_last, "adx_prior": adx_prior, "score": adx_ok}
        else:
            comps['adx'] = {"note": "no high/low data", "score": adx_ok}
        score_sum += 0.08 * adx_ok

        # MFI / CMF money-flow confirmation
        mfi_ok = 0.5
        if not mfi_s.isna().all():
            mfi_last = float(mfi_s.iloc[last_idx])
            if direction == "bull":
                mfi_ok = 1.0 if mfi_last <= 40 else max(0.0, (45 - mfi_last) / 15.0)
            else:
                mfi_ok = 1.0 if mfi_last >= 60 else max(0.0, (mfi_last - 55) / 15.0)
            comps['mfi'] = {"mfi_last": mfi_last, "score": mfi_ok}
        else:
            comps['mfi'] = {"note": "no volume data", "score": mfi_ok}
        score_sum += 0.07 * mfi_ok

        # ATR filter
        atr_ok = 1.0
        if not atr_s.isna().all():
            atr_last = float(atr_s.iloc[last_idx])
            atr_pct = atr_last / (close.iloc[last_idx] + 1e-9)
            atr_ok = 1.0 if atr_pct <= max_atr_pct else max(0.0, (max_atr_pct * 1.5 - atr_pct) / (max_atr_pct * 1.5))
            comps['atr'] = {"atr_last": atr_last, "atr_pct": atr_pct, "score": atr_ok}
        else:
            comps['atr'] = {"note": "no high/low data", "score": atr_ok}
        score_sum += 0.05 * atr_ok

        # Market regime: check optional market_index alignment
        market_ok = 1.0
        if market_index is not None and len(market_index) == n:
            # use market RSI as simple filter: avoid buying reversals when market is deeply bearish
            market_rsi = rsi(pd.Series(market_index).reset_index(drop=True), rsi_period).iloc[last_idx]
            if direction == "bull":
                market_ok = 1.0 if market_rsi > 40 else max(0.0, (market_rsi - 30) / 10.0)
            else:
                market_ok = 1.0 if market_rsi < 60 else max(0.0, (70 - market_rsi) / 10.0)
            comps['market'] = {"market_rsi": float(market_rsi), "score": market_ok}
        else:
            comps['market'] = {"note": "no market index", "score": market_ok}
        score_sum += 0.05 * market_ok

        # Liquidity filter
        liq_ok = 1.0
        if min_avg_volume is not None and has_volume:
            avg_vol = float(vol.iloc[-20:].mean())
            liq_ok = 1.0 if avg_vol >= min_avg_volume else max(0.0, avg_vol / (min_avg_volume + 1e-9))
            comps['liquidity'] = {"avg_20": avg_vol, "min_required": min_avg_volume, "score": liq_ok}
        else:
            comps['liquidity'] = {"note": "no min volume requirement or no volume data", "score": liq_ok}
        score_sum += 0.03 * liq_ok

        # Normalize (weights chosen above sum approximately to 1)
        conf_score = float(np.clip(score_sum, 0.0, 1.0))
        comps['conf_score'] = conf_score
        return conf_score, comps

    # ------------------------------
    # 5) Evaluate bull & bear candidates combining pattern + confirmation
    # ------------------------------
    bull_score = 0.0; bull_meta = {}
    if recent_down_days >= min_recent_down and latest_up_days >= 1 and recent_down_days > latest_up_days:
        p_score, p_meta = pattern_score(prior_len=recent_down_days, latest_len=latest_up_days,
                                        prior_start=down_start_idx, prior_end=idx_before_up,
                                        latest_start=up_start_idx, latest_end=last_idx, direction="bull")
        if p_score > 0:
            conf_score, conf_meta = confirmation_score("bull", down_start_idx, idx_before_up, up_start_idx, last_idx)
            combined = p_score * conf_score
            bull_score = float(np.clip(combined, 0.0, 1.0))
            bull_meta = {"pattern": p_meta, "confirm": conf_meta, "pattern_raw": p_score, "conf_raw": conf_score}

    bear_score = 0.0; bear_meta = {}
    if recent_up_days >= min_recent_down and latest_down_days >= 1 and recent_up_days > latest_down_days:
        p_score, p_meta = pattern_score(prior_len=recent_up_days, latest_len=latest_down_days,
                                        prior_start=up_start_idx_bear, prior_end=idx_before_down2,
                                        latest_start=down_start_idx_bear, latest_end=last_idx, direction="bear")
        if p_score > 0:
            conf_score, conf_meta = confirmation_score("bear", up_start_idx_bear, idx_before_down2, down_start_idx_bear, last_idx)
            combined = p_score * conf_score
            bear_score = float(np.clip(combined, 0.0, 1.0))
            bear_meta = {"pattern": p_meta, "confirm": conf_meta, "pattern_raw": p_score, "conf_raw": conf_score}

    # ------------------------------
    # 6) Optional simple meta-model to re-weight features (train_meta)
    # ------------------------------
    meta_info = {}
    meta_multiplier = 1.0
    if train_meta and _sklearn_available:
        # Build features across history (sliding) and label: future return > 0 across lookahead window
        # This is a small, conservative model trained on the same symbol in-sample (user can persist offline)
        try:
            look = meta_lookahead_days
            features = []
            labels = []
            # We'll engineering a handful of features for each index t
            for t in range(30, n - look):
                # compute run lengths up to t using simple rule (no tolerance)
                window_close = close.iloc[:t + 1]
                # simple features: RSI, MACD hist, ATR pct, recent drop/gain length
                rsi_t = float(rsi_s.iloc[t])
                macdh_t = float(macd_hist.iloc[t])
                atr_pct_t = float(atr_s.iloc[t] / (close.iloc[t] + 1e-9)) if not atr_s.isna().all() else 0.0
                ema_diff = float(close.iloc[t] - ema_short_s.iloc[t])
                # run length features: count consecutive down days
                # simple consecutive counts
                k = t
                consec_down = 0
                while k > 0 and close.iloc[k] < close.iloc[k - 1] and consec_down < 50:
                    consec_down += 1; k -= 1
                k2 = t
                consec_up = 0
                while k2 > 0 and close.iloc[k2] > close.iloc[k2 - 1] and consec_up < 50:
                    consec_up += 1; k2 -= 1
                # volume ratio
                vol_ratio = 1.0
                if has_volume:
                    avg_prior = float(vol.iloc[max(0, t - 10):t + 1].mean())
                    avg_recent = float(vol.iloc[t - max(0, min(3, t)):t + 1].mean())
                    vol_ratio = (avg_recent + 1e-9) / (avg_prior + 1e-9)
                features.append([rsi_t, macdh_t, atr_pct_t, ema_diff, consec_down, consec_up, vol_ratio])
                # label: future return over look days
                future_ret = (close.iloc[t + look] - close.iloc[t]) / (close.iloc[t] + 1e-9)
                labels.append(1 if future_ret > 0.005 else 0)  # small threshold 0.5%
            if len(features) > 200:
                X = np.array(features)
                y = np.array(labels)
                tscv = TimeSeriesSplit(n_splits=3)
                base = LogisticRegression(max_iter=200)
                clf = CalibratedClassifierCV(base, cv=tscv)  # probability calibrated
                clf.fit(X, y)
                # Predict on last point (current state) features[-1]
                last_feat = X[-1:].copy()
                prob = float(clf.predict_proba(last_feat)[0, 1])
                meta_multiplier = 0.8 + 0.4 * prob  # scale from 0.8 .. 1.2 roughly
                meta_info['meta_prob'] = prob
            else:
                meta_info['meta_note'] = "insufficient history to train meta model"
        except Exception as e:
            meta_info['meta_error'] = str(e)
    elif train_meta and not _sklearn_available:
        meta_info['meta_note'] = "sklearn not available; skip meta model"

    # Combine bull/bear with meta multiplier
    if bull_score <= 0 and bear_score <= 0:
        if verbose:
            return "NoReversal", {"reason": "no_candidate", "bull_meta": bull_meta, "bear_meta": bear_meta}
        return "NoReversal", None

    if bull_score >= bear_score:
        final_raw = bull_score
        direction = "Bullish"
        chosen_meta = bull_meta
    else:
        final_raw = bear_score
        direction = "Bearish"
        chosen_meta = bear_meta

    final_score = float(np.clip(final_raw * meta_multiplier, 0.0, 1.0))
    meta_info['meta_multiplier'] = meta_multiplier

    # Map to strength label
    if final_score >= strong_thresh:
        strength = "Strong"
    elif final_score >= moderate_thresh:
        strength = "Moderate"
    elif final_score >= weak_thresh:
        strength = "Weak"
    else:
        if verbose:
            return "NoReversal", {"reason": "final_below_weak", "final_score": final_score, "meta": chosen_meta}
        return "NoReversal", None

    label = f"{direction}Reversal{strength}"

    # ------------------------------
    # 7) Stop, target (ATR-based) and position sizing suggestion
    # ------------------------------
    entry_price = float(close.iloc[last_idx])
    atr_last = float(atr_s.iloc[last_idx]) if not atr_s.isna().all() else (0.0)
    # stop multiplier: tighter for stronger signals
    stop_atr_mult = 1.0 if strength == "Strong" else (1.5 if strength == "Moderate" else 2.0)
    target_rr = 2.0 if strength == "Strong" else (1.8 if strength == "Moderate" else 1.5)

    if direction == "Bullish":
        stop_price = entry_price - stop_atr_mult * atr_last if atr_last > 0 else entry_price * 0.97
        stop_distance = entry_price - stop_price
        target_price = entry_price + target_rr * stop_distance
    else:
        stop_price = entry_price + stop_atr_mult * atr_last if atr_last > 0 else entry_price * 1.03
        stop_distance = stop_price - entry_price
        target_price = entry_price - target_rr * stop_distance

    # position size in shares
    risk_amount = equity * risk_per_trade_pct
    if stop_distance <= 0:
        position_size = 0.0
    else:
        position_size = max(0, int(risk_amount / (stop_distance + 1e-9)))

    # sanity check liquidity
    if min_avg_volume is not None and has_volume:
        avg20 = float(vol.iloc[-20:].mean())
        # scale down size if 20-day avg volume insufficient for trade (e.g., >0.5% daily volume)
        if avg20 > 0:
            # allow max participation pct = 1% of daily volume
            max_participation = 0.01
            max_shares = int(avg20 * max_participation)
            if position_size > max_shares:
                position_size = max_shares
                meta_info['sizing_note'] = "reduced_by_liquidity"

    # ------------------------------
    # 8) Verbose details and return
    # ------------------------------
    details = {
        "label": label,
        "direction": direction,
        "strength": strength,
        "final_score": final_score,
        "final_raw_score": final_raw,
        "pattern_meta": chosen_meta.get("pattern", {}) if chosen_meta else {},
        "confirm_meta": chosen_meta.get("confirm", {}) if chosen_meta else {},
        "meta_info": meta_info,
        "entry_price": entry_price,
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "stop_distance": float(stop_distance),
        "position_size_shares": int(position_size),
        "risk_amount": float(min(equity * risk_per_trade_pct, position_size * stop_distance)),
        "indicators": {
            "rsi": float(rsi_s.iloc[last_idx]),
            "macd_hist": float(macd_hist.iloc[last_idx]),
            "ema_short": float(ema_short_s.iloc[last_idx]),
            "atr": float(atr_last),
            "adx": float(adx_s.iloc[last_idx]) if not adx_s.isna().all() else None,
            "mfi": float(mfi_s.iloc[last_idx]) if not mfi_s.isna().all() else None,
            "cmf": float(cmf_s.iloc[last_idx]) if not cmf_s.isna().all() else None,
            "obv": float(obv_s.iloc[last_idx]) if not obv_s.isna().all() else None
        },
        "run_counts": {
            "recent_down_days": int(recent_down_days),
            "latest_up_days": int(latest_up_days),
            "recent_up_days": int(recent_up_days),
            "latest_down_days": int(latest_down_days)
        }
    }

    if verbose:
        return label, details
    return label, None

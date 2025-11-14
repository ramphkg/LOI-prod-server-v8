"""
TrendReversalDetector.py

Patched and hardened version of the original TrendReversalDetector.
Improvements:
 - Defensive validation and configurable minimum history
 - Robust _last_run: ignores zero-diff (flat) days when counting runs
 - Safe detect_runs with bounds checks
 - Corrected indicator computations:
     * ATR, RSI, ADX use Wilder-style exponential smoothing (ewm) for consistency
     * ADX uses correct Directional Movement logic (plus/minus DM selection)
     * Vectorized OBV implementation
 - Robust signal() and get_trend(): use positional iloc indexing, validate required data,
   defensive handling of insufficient/NaN indicator values
 - Ordered checks with explicit weights (no reliance on dict insertion ordering)
 - Clear, single-file drop-in replacement
"""

from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np


class TrendReversalDetector:
    ERR_MISSING = "ERR_MISSING_COLS"
    ERR_INSUFF = "ERR_INSUFF_DATA"
    ERR_NOTREND = "ERR_NO_TREND"

    def __init__(self,
                 df: pd.DataFrame,
                 min_prior: int = 4,
                 min_curr: int = 1,
                 atr_mult: float = 1.5,
                 alpha: float = 0.7,
                 adx_strong: float = 25,
                 adx_average: float = 20,
                 min_history: int = 260):
        """
        Args:
            df: input DataFrame that must contain at least ['Date','High','Low','Close','Volume'].
            min_prior/min_curr: minimum contiguous prior/current run lengths to be considered.
            atr_mult: multiplier to compare move size vs ATR.
            alpha: blending weight between indicator test score and length heuristic.
            adx_strong / adx_average: thresholds used in get_trend().
            min_history: minimum number of rows required to run detector (configurable).
        """
        self.df_raw = df.copy()
        self.min_prior = int(min_prior)
        self.min_curr = int(min_curr)
        self.atr_mult = float(atr_mult)
        self.alpha = float(alpha)
        self.adx_strong = float(adx_strong)
        self.adx_average = float(adx_average)
        self.min_history = int(min_history)

        self.df: Optional[pd.DataFrame] = None
        self.ind: Optional[pd.DataFrame] = None

    def validate(self) -> Optional[str]:
        """Validate raw dataframe and prepare self.df as a clean, sorted copy.

        Returns:
            None on success or an error code string on failure.
        """
        required = {'Date', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(set(self.df_raw.columns)):
            return self.ERR_MISSING

        try:
            df = (
                self.df_raw
                .assign(Date=lambda d: pd.to_datetime(d['Date']))
                .sort_values('Date')
                .reset_index(drop=True)
            )
        except (KeyError, ValueError, TypeError):
            return self.ERR_MISSING

        if len(df) < self.min_history:
            return self.ERR_INSUFF

        # Ensure numeric columns are numeric (coerce, keep NaNs for later checks)
        for col in ['High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        self.df = df
        return None

    @staticmethod
    def _last_run(prices: pd.Series) -> int:
        """
        Return signed length of last contiguous run of increasing (positive) or decreasing
        (negative) closes. Zero-change days are ignored when determining the last run.
        If unable to determine (all flat), returns 0.
        Example:
            last_run([1,2,3,2,2,3]) -> +1 (last contiguous non-zero move was +1 day up)
        """
        if prices is None or len(prices) == 0:
            return 0

        diffs = prices.diff()
        # sign: -1, 0, +1 ; keep alignment with prices index
        sgn = np.sign(diffs.fillna(0))

        # find last non-zero sign
        non_zero = sgn[sgn != 0]
        if non_zero.empty:
            return 0
        last_sign = int(non_zero.iat[-1])  # +1 or -1

        # count contiguous occurrences of last_sign from the end, ignoring zeros
        run = 0
        for x in sgn.iloc[::-1]:
            if x == 0:
                # ignore flat days
                continue
            if int(x) == last_sign:
                run += 1
            else:
                break
        return int(last_sign * run)

    def detect_runs(self) -> Tuple[Optional[str], dict]:
        """
        Detects a recent run (current) and a prior run right before it.
        Returns:
            (None, info_dict) on success where info_dict contains:
              {'curr_days', 'curr_type', 'prior_days', 'prior_type'}
            or (ERR_CODE, {}) on failure.
        """
        if self.df is None:
            return self.ERR_MISSING, {}

        n = len(self.df)
        if n == 0:
            return self.ERR_INSUFF, {}

        run1 = self._last_run(self.df['Close'])
        if run1 == 0:
            return self.ERR_NOTREND, {}

        cd = abs(run1)
        ct = 'Up' if run1 > 0 else 'Down'
        if cd < self.min_curr:
            return self.ERR_NOTREND, {}

        # ensure there is enough history before the current run to compute prior run
        if cd >= n:
            return self.ERR_NOTREND, {}

        prior_df = self.df.iloc[:-cd]
        if prior_df.empty:
            return self.ERR_NOTREND, {}

        run2 = self._last_run(prior_df['Close'])
        if run2 == 0:
            return self.ERR_NOTREND, {}

        pd_ = abs(run2)
        pt = 'Up' if run2 > 0 else 'Down'
        if pd_ < self.min_prior or pt == ct:
            return self.ERR_NOTREND, {}

        return None, {
            'curr_days': cd,
            'curr_type': ct,
            'prior_days': pd_,
            'prior_type': pt
        }

    def compute_indicators(self) -> None:
        """
        Compute and store indicators in self.ind (DataFrame indexed like self.df).
        Uses Wilder-style ewm smoothing for ATR/RSI/ADX (common, consistent behavior).
        """
        if self.df is None:
            raise ValueError("Data not validated. Call validate() first.")

        df = self.df.copy().reset_index(drop=True)
        ind = pd.DataFrame(index=df.index)

        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']

        # True Range (TR)
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # ATR with Wilder smoothing (alpha = 1/period)
        atr_period = 14
        ind['atr'] = tr.ewm(alpha=1/atr_period, adjust=False).mean()

        # Vectorized OBV
        sign = np.sign(close.diff()).fillna(0)
        obv = (vol * sign).cumsum().fillna(0)
        ind['obv'] = obv

        # RSI (Wilder's smoothing)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        ind['rsi'] = 100 - (100 / (1 + rs))

        # MACD (12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        ind['macd'] = macd - macd_signal

        # SMA 50 / 200
        ind['sma50'] = close.rolling(50).mean()
        ind['sma200'] = close.rolling(200).mean()

        # EMA 20 / 50
        ind['ema20'] = close.ewm(span=20, adjust=False).mean()
        ind['ema50'] = close.ewm(span=50, adjust=False).mean()

        # ADX: Proper directional movement + Wilder smoothing
        up = high.diff()
        down = -low.diff()  # positive when price moved down
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        # Smoothed ATR (reuse ind['atr'] but ensure alignment)
        atr_smoothed = ind['atr']

        # Smooth the DM series with Wilder's ewm (alpha=1/period)
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/atr_period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/atr_period, adjust=False).mean()

        plus_di = 100 * (plus_dm_smooth / (atr_smoothed + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (atr_smoothed + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        ind['adx'] = dx.ewm(alpha=1/atr_period, adjust=False).mean()

        # Save indicators aligned with df's index
        self.ind = ind

    def _has_enough_indicator_data(self, ind: pd.DataFrame, required_cols: List[str], lookback: int = 2) -> bool:
        """Helper: check that required indicator columns have at least `lookback` non-NaN values at the end."""
        if ind is None or ind.empty:
            return False
        n = len(ind)
        if n < lookback:
            return False
        # check last `lookback` rows for NaNs in required columns
        tail = ind.iloc[-lookback:][required_cols]
        return not tail.isnull().any().any()

    def signal(self) -> str:
        """
        Main reversal signal function. Returns:
          - ERR_* codes for validation / insufficient data / not a trend
          - "BullishReversal[Strong|Moderate|Weak]" or "BearishReversal[...]" on detection
        """
        if (e := self.validate()):
            return e

        err, info = self.detect_runs()
        if err:
            return err

        cd = int(info['curr_days'])
        ct = info['curr_type']
        pd_ = int(info['prior_days'])
        pt = info['prior_type']

        # Compute indicators
        try:
            self.compute_indicators()
        except Exception:
            return self.ERR_INSUFF

        ind = self.ind
        close_series = self.df['Close'].reset_index(drop=True)

        # Prior slice bounds: start..end (end exclusive)
        end_idx = len(close_series) - cd
        start_idx = end_idx - pd_
        if start_idx < 0 or end_idx <= start_idx:
            return self.ERR_INSUFF

        prior_slice = close_series.iloc[start_idx:end_idx]
        if prior_slice.empty:
            return self.ERR_INSUFF

        ph = prior_slice.max()
        pl = prior_slice.min()

        # Required indicators list
        required_for_tests = ['atr', 'obv', 'rsi', 'macd', 'sma50', 'sma200']
        if not self._has_enough_indicator_data(ind, required_for_tests, lookback=2):
            return self.ERR_INSUFF

        # Gather terminal indicator values with iloc (positional)
        try:
            atr_val = float(ind['atr'].iloc[-1])
        except Exception:
            atr_val = float('nan')

        obv_series = ind['obv']
        rsi_series = ind['rsi']
        macd_series = ind['macd']
        sma50 = ind['sma50']
        sma200 = ind['sma200']

        # Build ordered checks with explicit weights
        checks: List[Tuple[str, bool, int]] = []

        # 1) ATR move: price moved from prior extreme by at least atr_mult * ATR
        if np.isnan(atr_val) or atr_val <= 0:
            ok_atr = False
        else:
            if ct == 'Up':
                ok_atr = (close_series.iloc[-1] - pl) >= (self.atr_mult * atr_val)
            else:
                ok_atr = (ph - close_series.iloc[-1]) >= (self.atr_mult * atr_val)
        checks.append(("atr_move", bool(ok_atr), 2))

        # 2) OBV trend: OBV moving in direction of reversal (comparing last two)
        try:
            obv_ok = (obv_series.iloc[-1] > obv_series.iloc[-2]) if ct == 'Up' else (obv_series.iloc[-1] < obv_series.iloc[-2])
        except Exception:
            obv_ok = False
        checks.append(("volume", bool(obv_ok), 1))

        # 3) RSI reversal: oversold/overbought and turning
        try:
            if ct == 'Up':
                rsi_ok = (rsi_series.iloc[-2] < 30) and (rsi_series.iloc[-1] > rsi_series.iloc[-2])
            else:
                rsi_ok = (rsi_series.iloc[-2] > 70) and (rsi_series.iloc[-1] < rsi_series.iloc[-2])
        except Exception:
            rsi_ok = False
        checks.append(("rsi", bool(rsi_ok), 2))

        # 4) MACD zero-cross: momentum crossing zero in direction of reversal
        try:
            if ct == 'Up':
                macd_ok = (macd_series.iloc[-2] < 0) and (macd_series.iloc[-1] > 0)
            else:
                macd_ok = (macd_series.iloc[-2] > 0) and (macd_series.iloc[-1] < 0)
        except Exception:
            macd_ok = False
        checks.append(("macd", bool(macd_ok), 3))

        # 5) SMA crossover (50/200) indicating longer-term confirmation
        try:
            if ct == 'Up':
                sma_ok = (sma50.iloc[-2] < sma200.iloc[-2]) and (sma50.iloc[-1] > sma200.iloc[-1])
            else:
                sma_ok = (sma50.iloc[-2] > sma200.iloc[-2]) and (sma50.iloc[-1] < sma200.iloc[-1])
        except Exception:
            sma_ok = False
        checks.append(("sma_cross", bool(sma_ok), 4))

        # Weighted scoring
        total_w = sum(w for (_, _, w) in checks)
        passed_w = sum(w for (name, ok, w) in checks if ok)
        test_score = float(passed_w) / float(total_w) if total_w > 0 else 0.0

        # Length heuristic: relative size of prior trend vs current trend
        length_score = float(pd_) / float(pd_ + cd) if (pd_ + cd) > 0 else 0.0

        final_score = self.alpha * test_score + (1.0 - self.alpha) * length_score

        # Confidence buckets
        if final_score >= 0.5:
            conf = "Strong"
        elif final_score >= 0.4:
            conf = "Moderate"
        else:
            conf = "Weak"

        direction = "BullishReversal" if ct == 'Up' else "BearishReversal"

        return f"{direction}[{conf}]"

    def get_trend(self) -> str:
        """
        Returns the current trend label (e.g., 'StrongBullish', 'ModerateBearish', 'NoTrend').
        Uses EMA20/EMA50 and ADX strength to classify trend and strength.
        """
        if (e := self.validate()):
            return e

        try:
            self.compute_indicators()
        except Exception:
            return self.ERR_INSUFF

        ind = self.ind
        if ind is None or ind.empty:
            return "NoTrend"

        n = len(ind)
        if n < 3:
            return "NoTrend"

        # Use iloc (positional) rather than index arithmetic for robustness
        try:
            ema20 = ind['ema20'].iloc[-1]
            ema50 = ind['ema50'].iloc[-1]
            ema20_prev = ind['ema20'].iloc[-2]
            ema50_prev = ind['ema50'].iloc[-2]
            close = self.df['Close'].iloc[-1]
            close_prev = self.df['Close'].iloc[-2]
            close_prev2 = self.df['Close'].iloc[-3]
            adx = ind['adx'].iloc[-1]
        except Exception:
            return "NoTrend"

        # If any of these are NaN, cannot determine
        if np.isnan(ema20) or np.isnan(ema50) or np.isnan(adx):
            return "NoTrend"

        # MA Trend logic (similar to original, but robust)
        if (ema20 > ema50 and ema20 > ema20_prev and ema50 > ema50_prev and
                close > close_prev and close_prev > close_prev2):
            ma_trend = 'Bull+'
        elif (ema50 > ema20 and ema20 <= ema20_prev and ema50 <= ema50_prev and
              close < close_prev and close_prev < close_prev2):
            ma_trend = 'Bear+'
        elif (ema20 > ema50 and ema20 > ema20_prev and ema50 > ema50_prev):
            ma_trend = 'Bull'
        elif (ema50 > ema20 and ema20 <= ema20_prev and ema50 <= ema50_prev):
            ma_trend = 'Bear'
        else:
            ma_trend = 'Nil'

        # ADX Strength logic
        if adx >= self.adx_strong:
            adx_strength = 'Strong'
        elif adx >= self.adx_average:
            adx_strength = 'Average'
        else:
            adx_strength = 'Weak'

        # Combine into human-readable label
        if ma_trend == 'Bull+':
            return 'StrongBullish' if adx_strength == 'Strong' else 'WeakBullish'
        elif ma_trend == 'Bull':
            return 'ModerateBullish' if adx_strength != 'Weak' else 'WeakBullish'
        elif ma_trend == 'Bear+':
            return 'StrongBearish' if adx_strength == 'Strong' else 'WeakBearish'
        elif ma_trend == 'Bear':
            return 'ModerateBearish' if adx_strength != 'Weak' else 'WeakBearish'
        else:
            return 'NoTrend'
# indicators.py
# Rewritten to use pandas_ta for all technical indicators
# Returns DataFrame with sorted data and all indicators added as columns
#
# Usage:
#   from indicators import compute_indicators
#   df = compute_indicators(df, params={'rsi_period': 14, 'macd_fast': 12, ...})
#

import pandas as pd
import numpy as np
from typing import Dict, Optional
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compute technical indicators using pandas_ta and add them as columns to the DataFrame.
    
    The function:
    1. Sorts the input DataFrame by date (chronological order)
    2. Computes all technical indicators using pandas_ta
    3. Adds indicator columns to the DataFrame
    4. Returns the complete DataFrame with indicators
    
    Args:
        df (pd.DataFrame): Input DataFrame with at least 'Close' column.
                          Should have: 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'
        params (Optional[Dict]): Dictionary of indicator parameters with defaults:
            - rsi_period: 14
            - macd_fast: 12
            - macd_slow: 26
            - macd_sig: 9
            - ema_fast: 12
            - ema_slow: 26
            - ema_short: 8
            - ema_20: 20
            - ema_50: 50
            - atr_period: 14
            - adx_period: 14
            - mfi_period: 14
            - bb_len: 20
            - bb_std: 2.0
            - sma50: 50
            - sma200: 200
            - cci_length: 20
    
    Returns:
        pd.DataFrame: DataFrame sorted by date with all indicator columns added.
                     Indicator columns include:
                     - rsi, macd, macd_signal, macd_histogram
                     - ema_fast, ema_slow, ema_short, ema_20, ema_50
                     - sma50, sma200
                     - atr, adx, plus_di, minus_di
                     - mfi, obv
                     - bb_lower, bb_middle, bb_upper
                     - cci
    
    Notes:
        - Input DataFrame is NOT mutated; a copy is returned
        - DataFrame is sorted by 'Date' or 'date' column chronologically
        - All prices are converted to float64
        - Missing OHLCV data is handled gracefully
        - Indicator values may contain NaN for initial rows (warm-up period)
    """
    if df is None or df.empty:
        return df.copy()
    
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    p = {
        "rsi_period": params.get("rsi_period", 14),
        "macd_fast": params.get("macd_fast", 12),
        "macd_slow": params.get("macd_slow", 26),
        "macd_sig": params.get("macd_sig", 9),
        "ema_fast": params.get("ema_fast", 12),
        "ema_slow": params.get("ema_slow", 26),
        "ema_short": params.get("ema_short", 8),
        "ema_20": params.get("ema_20", 20),
        "ema_50": params.get("ema_50", 50),
        "atr_period": params.get("atr_period", 14),
        "adx_period": params.get("adx_period", 14),
        "mfi_period": params.get("mfi_period", 14),
        "bb_len": params.get("bb_len", 20),
        "bb_std": params.get("bb_std", 2.0),
        "sma50": params.get("sma50", 50),
        "sma200": params.get("sma200", 200),
        "cci_length": params.get("cci_length", 20)
    }
    
    # Create a working copy
    df_work = df.copy()
    
    # Find date column and sort
    date_col = None
    if "Date" in df_work.columns:
        date_col = "Date"
    elif "date" in df_work.columns:
        date_col = "date"
    elif "DateTime" in df_work.columns:
        date_col = "DateTime"
    elif "datetime" in df_work.columns:
        date_col = "datetime"
    
    # Sort by date if found
    if date_col is not None:
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_work = df_work.sort_values(by=date_col).reset_index(drop=True)
    else:
        # Try to sort by index if it's a datetime index
        if isinstance(df_work.index, pd.DatetimeIndex):
            df_work = df_work.sort_index()
            df_work = df_work.reset_index(drop=True)
    
    # Ensure OHLCV columns exist and are numeric
    close_col = None
    if "Close" in df_work.columns:
        close_col = "Close"
    elif "close" in df_work.columns:
        close_col = "close"
    
    if close_col is None:
        raise ValueError("DataFrame must contain 'Close' or 'close' column")
    
    # Get OHLCV columns
    high_col = "High" if "High" in df_work.columns else ("high" if "high" in df_work.columns else None)
    low_col = "Low" if "Low" in df_work.columns else ("low" if "low" in df_work.columns else None)
    open_col = "Open" if "Open" in df_work.columns else ("open" if "open" in df_work.columns else None)
    vol_col = "Volume" if "Volume" in df_work.columns else ("volume" if "volume" in df_work.columns else None)
    
    # Convert to numeric and fill missing OHLCV
    df_work[close_col] = pd.to_numeric(df_work[close_col], errors='coerce')
    
    if high_col:
        df_work[high_col] = pd.to_numeric(df_work[high_col], errors='coerce')
    else:
        df_work['High'] = df_work[close_col].copy()
        high_col = 'High'
    
    if low_col:
        df_work[low_col] = pd.to_numeric(df_work[low_col], errors='coerce')
    else:
        df_work['Low'] = df_work[close_col].copy()
        low_col = 'Low'
    
    if open_col:
        df_work[open_col] = pd.to_numeric(df_work[open_col], errors='coerce')
    else:
        df_work['Open'] = df_work[close_col].copy()
        open_col = 'Open'
    
    if vol_col:
        df_work[vol_col] = pd.to_numeric(df_work[vol_col], errors='coerce')
    else:
        df_work['Volume'] = 0.0
        vol_col = 'Volume'
    
    # ========== RSI ==========
    try:
        rsi_result = ta.rsi(close=df_work[close_col], length=p["rsi_period"])
        df_work['rsi'] = rsi_result if isinstance(rsi_result, pd.Series) else rsi_result.iloc[:, 0]
    except Exception as e:
        print(f"Warning: RSI calculation failed: {e}")
        df_work['rsi'] = np.nan
    
    # ========== MACD ==========
    try:
        macd_result = ta.macd(close=df_work[close_col], fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_sig"])
        if isinstance(macd_result, pd.DataFrame):
            macd_cols = macd_result.columns.tolist()
            df_work['macd'] = macd_result.iloc[:, 0] if len(macd_cols) > 0 else np.nan
            df_work['macd_signal'] = macd_result.iloc[:, 1] if len(macd_cols) > 1 else np.nan
            df_work['macd_histogram'] = macd_result.iloc[:, 2] if len(macd_cols) > 2 else np.nan
        else:
            df_work['macd'] = np.nan
            df_work['macd_signal'] = np.nan
            df_work['macd_histogram'] = np.nan
    except Exception as e:
        print(f"Warning: MACD calculation failed: {e}")
        df_work['macd'] = np.nan
        df_work['macd_signal'] = np.nan
        df_work['macd_histogram'] = np.nan
    
    # ========== EMAs ==========
    try:
        df_work['ema_fast'] = ta.ema(close=df_work[close_col], length=p["ema_fast"])
        df_work['ema_slow'] = ta.ema(close=df_work[close_col], length=p["ema_slow"])
        df_work['ema_short'] = ta.ema(close=df_work[close_col], length=p["ema_short"])
        df_work['ema_20'] = ta.ema(close=df_work[close_col], length=p["ema_20"])
        df_work['ema_50'] = ta.ema(close=df_work[close_col], length=p["ema_50"])
    except Exception as e:
        print(f"Warning: EMA calculation failed: {e}")
        df_work['ema_fast'] = np.nan
        df_work['ema_slow'] = np.nan
        df_work['ema_short'] = np.nan
        df_work['ema_20'] = np.nan
        df_work['ema_50'] = np.nan
    
    # ========== SMAs ==========
    try:
        df_work['sma50'] = ta.sma(close=df_work[close_col], length=p["sma50"])
        df_work['sma200'] = ta.sma(close=df_work[close_col], length=p["sma200"])
    except Exception as e:
        print(f"Warning: SMA calculation failed: {e}")
        df_work['sma50'] = np.nan
        df_work['sma200'] = np.nan
    
    # ========== ATR, ADX, DI ==========
    try:
        atr_result = ta.atr(high=df_work[high_col], low=df_work[low_col], close=df_work[close_col], length=p["atr_period"])
        df_work['atr'] = atr_result if isinstance(atr_result, pd.Series) else atr_result.iloc[:, 0]
    except Exception as e:
        print(f"Warning: ATR calculation failed: {e}")
        df_work['atr'] = np.nan
    
    try:
        adx_result = ta.adx(high=df_work[high_col], low=df_work[low_col], close=df_work[close_col], length=p["adx_period"])
        if isinstance(adx_result, pd.DataFrame):
            adx_cols = adx_result.columns.tolist()
            df_work['adx'] = adx_result.iloc[:, 0] if len(adx_cols) > 0 else np.nan
            df_work['plus_di'] = adx_result.iloc[:, 1] if len(adx_cols) > 1 else np.nan
            df_work['minus_di'] = adx_result.iloc[:, 2] if len(adx_cols) > 2 else np.nan
        else:
            df_work['adx'] = np.nan
            df_work['plus_di'] = np.nan
            df_work['minus_di'] = np.nan
    except Exception as e:
        print(f"Warning: ADX calculation failed: {e}")
        df_work['adx'] = np.nan
        df_work['plus_di'] = np.nan
        df_work['minus_di'] = np.nan
    
    # ========== MFI ==========
    try:
        mfi_result = ta.mfi(high=df_work[high_col], low=df_work[low_col], close=df_work[close_col], volume=df_work[vol_col], length=p["mfi_period"])
        df_work['mfi'] = mfi_result if isinstance(mfi_result, pd.Series) else mfi_result.iloc[:, 0]
    except Exception as e:
        print(f"Warning: MFI calculation failed: {e}")
        df_work['mfi'] = np.nan
    
    # ========== OBV ==========
    try:
        obv_result = ta.obv(close=df_work[close_col], volume=df_work[vol_col])
        df_work['obv'] = obv_result if isinstance(obv_result, pd.Series) else obv_result.iloc[:, 0]
    except Exception as e:
        print(f"Warning: OBV calculation failed: {e}")
        df_work['obv'] = np.nan
    
    # ========== Bollinger Bands ==========
    try:
        bb_result = ta.bbands(close=df_work[close_col], length=p["bb_len"], std=p["bb_std"])
        if isinstance(bb_result, pd.DataFrame):
            bb_cols = bb_result.columns.tolist()
            df_work['bb_lower'] = bb_result.iloc[:, 0] if len(bb_cols) > 0 else np.nan
            df_work['bb_middle'] = bb_result.iloc[:, 1] if len(bb_cols) > 1 else np.nan
            df_work['bb_upper'] = bb_result.iloc[:, 2] if len(bb_cols) > 2 else np.nan
        else:
            df_work['bb_lower'] = np.nan
            df_work['bb_middle'] = np.nan
            df_work['bb_upper'] = np.nan
    except Exception as e:
        print(f"Warning: Bollinger Bands calculation failed: {e}")
        df_work['bb_lower'] = np.nan
        df_work['bb_middle'] = np.nan
        df_work['bb_upper'] = np.nan
    
    # ========== CCI ==========
    try:
        cci_result = ta.cci(high=df_work[high_col], low=df_work[low_col], close=df_work[close_col], length=p["cci_length"])
        df_work['cci'] = cci_result if isinstance(cci_result, pd.Series) else cci_result.iloc[:, 0]
    except Exception as e:
        print(f"Warning: CCI calculation failed: {e}")
        df_work['cci'] = np.nan
    
    return df_work

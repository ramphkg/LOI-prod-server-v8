#!/usr/bin/env python3
# ta_signals_mc_parallel.py
# Simplified version with TrendReversalDetectorML removed:
#  - Removed imports and usage of TrendReversalDetectorML
#  - Removed related DB table fields from canonical schema
#  - Removed TRDML discovery and loading logic
#  - Kept SignalClassifier ML/rule processing intact

import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Iterable, Dict

import numpy as np
import pandas as pd
import argparse
import os
import logging

import pandas_ta as pta  # optional usage in some functions
from scipy import stats
from sqlalchemy import text

# Local application imports - expected to be present in your environment
from app_imports import getDbConnection, parallelLoggingSetter, printnlog, strUtcNow
from eod_api_prices import eod_lastPriceDetails, get_historic_prices_from_eod
from finnhub_api_prices import finnhub_lastPriceDetails, get_historic_prices_from_finnhub

from SignalClassifier import SignalClassifier
from TrendReversalDetector import TrendReversalDetector
from TrendReversalDetectorFunction import detect_reversal_pro
from ML_Predict_Price import predict_best_horizon

# Shared indicators utility
from indicators import compute_indicators

# Module globals set by main / worker_init
USE_ML: bool = True

# -------------------------
# Utility functions
# -------------------------
def canonical_table_schema() -> dict:
    """
    Central canonical schema mapping with updated field names for clarity.
    TrendReversalDetectorML-related fields removed.
    """
    return {
        "Date": "DATE",
        "Open": "DOUBLE PRECISION",
        "High": "DOUBLE PRECISION",
        "Low": "DOUBLE PRECISION",
        "Close": "DOUBLE PRECISION",
        "Volume": "BIGINT",
        "Symbol": "VARCHAR(191)",
        "ADX": "DOUBLE PRECISION",
        "DITrend": "VARCHAR(16)",
        "SMA200": "DOUBLE PRECISION",
        "EMA50": "DOUBLE PRECISION",
        "EMA20": "DOUBLE PRECISION",
        "CCI": "DOUBLE PRECISION",
        "RSI": "DOUBLE PRECISION",
        "MA_Trend": "VARCHAR(16)",
        "MADI_Trend": "VARCHAR(16)",
        "TMA21_50_X": "SMALLINT",
        "TodayPrice": "DOUBLE PRECISION",
        "marketCap": "DOUBLE PRECISION",
        "GEM_Rank": "VARCHAR(32)",
        "CountryName": "VARCHAR(191)",
        "IndustrySector": "VARCHAR(64)",
        "High52": "DOUBLE PRECISION",
        "Low52": "DOUBLE PRECISION",
        "Pct2H52": "DOUBLE PRECISION",
        "PctfL52": "DOUBLE PRECISION",
        "RSIUturnTypeOld": "VARCHAR(64)",
        "TrendReversal_Rules": "VARCHAR(64)",
        "RSIUpTrend": "BOOLEAN",
        "LastTrendDays": "INTEGER",
        "LastTrendType": "VARCHAR(32)",
        "Trend": "VARCHAR(64)",
        "ScanDate": "TIMESTAMP",
        "SignalClassifier_Rules": "INTEGER",
        "SignalClassifier_ML": "INTEGER",
        "ML_Target_Price": "DOUBLE PRECISION",
        "ML_Target_Price_Days": "INTEGER",
        "ML_Confidence_Score": "DOUBLE PRECISION",
        "ML_Target_Return_Pct": "DOUBLE PRECISION"
    }

# -------------------------
# DB / schema helpers
# -------------------------
def ensure_output_table(table_name: str, my_logger=None) -> None:
    """
    Ensure the table exists with canonical schema (MySQL-centric).
    """
    schema = canonical_table_schema()
    mysql_type_map = {
        "DOUBLE PRECISION": "DOUBLE",
        "BOOLEAN": "TINYINT(1)",
        "INTEGER": "INT",
        "SMALLINT": "SMALLINT",
        "BIGINT": "BIGINT",
    }

    def _sql_type_for_mysql(ctype: str) -> str:
        out = ctype
        for k, v in mysql_type_map.items():
            out = out.replace(k, v)
        return out

    if my_logger is None:
        try:
            my_logger = parallelLoggingSetter("ensure_output_table")
        except Exception:
            my_logger = logging.getLogger("ensure_output_table")
            if not my_logger.handlers:
                logging.basicConfig(level=logging.INFO)

    try:
        with getDbConnection() as con:
            q_table = f'`{table_name}`'
            col_defs = [f'`{col}` {_sql_type_for_mysql(ctype)}' for col, ctype in schema.items()]
            create_sql = f'CREATE TABLE IF NOT EXISTS {q_table} ({", ".join(col_defs)}) ENGINE=InnoDB;'
            try:
                con.execute(text(create_sql))
                my_logger.info(f"[ensure_output_table] CREATE TABLE IF NOT EXISTS executed for {table_name}")
            except Exception as e:
                my_logger.warning(f"[ensure_output_table] CREATE TABLE issued with warning: {e}")

            # Ensure each column exists
            for col, ctype in schema.items():
                q_check = text("""
                    SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name AND COLUMN_NAME = :col
                """)
                data_type = None
                char_max = None
                try:
                    res = con.execute(q_check, {"table_name": table_name, "col": col})
                    row = res.fetchone()
                    if row:
                        data_type = row[0]
                        char_max = row[1]
                except Exception:
                    # fallback inline
                    try:
                        inline = f"""
                            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{col}'
                        """
                        res = con.execute(text(inline))
                        row = res.fetchone()
                        if row:
                            data_type = row[0]
                            char_max = row[1]
                    except Exception as ie:
                        my_logger.error(f"[ensure_output_table] INFORMATION_SCHEMA query failed for {table_name}.{col}: {ie}")
                        raise

                if data_type is None:
                    add_sql = f'ALTER TABLE {q_table} ADD COLUMN `{col}` {_sql_type_for_mysql(ctype)};'
                    try:
                        con.execute(text(add_sql))
                        my_logger.info(f"[ensure_output_table] Added missing column `{col}` to {table_name}")
                    except Exception as e:
                        my_logger.error(f"[ensure_output_table] Failed to add column `{col}` to {table_name}: {e}\nSQL: {add_sql}")
                        raise
                else:
                    my_logger.debug(f"[ensure_output_table] Column {col} already exists in {table_name} (type={data_type}, char_max={char_max})")

            # Ensure indexes: Symbol, CountryName, Date
            idx_defs = {
                f'idx_{table_name}_symbol': ('Symbol',),
                f'idx_{table_name}_country': ('CountryName',),
                f'idx_{table_name}_date': ('Date',)
            }
            for idx_name, cols in idx_defs.items():
                q_idx_check = text("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name AND INDEX_NAME = :idx_name
                """)
                try:
                    res = con.execute(q_idx_check, {"table_name": table_name, "idx_name": idx_name})
                    r = res.fetchone()
                    idx_cnt = int(r[0]) if r is not None else 0
                except Exception:
                    idx_cnt = 0

                if idx_cnt == 0:
                    col = cols[0]
                    try:
                        res = con.execute(text("""
                            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name AND COLUMN_NAME = :col
                        """), {"table_name": table_name, "col": col})
                        meta = res.fetchone()
                        data_type = meta[0] if meta is not None else None
                        char_max = meta[1] if meta is not None else None
                    except Exception:
                        data_type = None
                        char_max = None

                    if data_type is not None and data_type.lower() in ('text', 'mediumtext', 'longtext', 'tinytext'):
                        prefix = min(191, int(char_max) if char_max else 191)
                        create_idx_sql = f'CREATE INDEX `{idx_name}` ON {q_table} (`{col}`({prefix}));'
                    elif data_type is not None and data_type.lower() in ('varchar', 'char'):
                        if char_max and int(char_max) > 191:
                            prefix = 191
                            create_idx_sql = f'CREATE INDEX `{idx_name}` ON {q_table} (`{col}`({prefix}));'
                        else:
                            create_idx_sql = f'CREATE INDEX `{idx_name}` ON {q_table} (`{col}`);'
                    else:
                        create_idx_sql = f'CREATE INDEX `{idx_name}` ON {q_table} (`{col}`);'

                    try:
                        con.execute(create_idx_sql)
                        my_logger.info(f"[ensure_output_table] Created index {idx_name} on {table_name}({', '.join(cols)})")
                    except Exception as e:
                        my_logger.warning(f"[ensure_output_table] Failed to create index {idx_name} on {table_name}: {e} -- SQL: {create_idx_sql} (continuing)")
                else:
                    my_logger.debug(f"[ensure_output_table] Index {idx_name} already present on {table_name}")

    except Exception as final_e:
        my_logger.error(f"[ensure_output_table] Fatal error ensuring table {table_name}: {final_e}\n{traceback.format_exc()}")
        raise

# -------------------------
# Price retrieval wrappers used by initialize_config
# -------------------------
def get_historic_prices_from_finnhub_local(symbol: str) -> pd.DataFrame:
    with getDbConnection() as con:
        q = text("""
        SELECT symbol, date AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM finnhub_stock_prices
        WHERE symbol = :symbol
        ORDER BY date ASC
        """)
        df = pd.read_sql(q, con=con, params={"symbol": symbol})
    return df

def get_historic_prices_from_eod_local(symbol: str) -> pd.DataFrame:
    with getDbConnection() as con:
        q = text("""
        SELECT symbol, date AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM eod_stock_prices
        WHERE symbol = :symbol
        ORDER BY date ASC
        """)
        df = pd.read_sql(q, con=con, params={"symbol": symbol})
    return df

# -------------------------
# Configuration and helpers
# -------------------------
def initialize_config(price_source: str) -> dict:
    config = {}
    config['PRICE_SOURCE'] = price_source.upper()
    config['DIWidth_Strong_threshold'] = 15
    config['DIWidth_Average_threshold'] = 10
    config['ADX_Strong_threshold'] = 25
    config['ADX_Average_threshold'] = 20
    config['EMA_CROSS_LOOKBACK'] = 3
    config['SMA_CROSS_LOOKBACK'] = 10
    config['HISTORY_BARS'] = 600

    ps = config['PRICE_SOURCE']
    if ps == 'FINNHUB':
        config['tal_master_tablename'] = 'finnhub_tas_listings'
        config['FUNDAMENTALS_TABLENAME'] = "finnhub_gem_listings"
        config['CLOSING_PRICES_FUNCTION'] = get_historic_prices_from_finnhub
        config['LAST_PRICES_DETLS_FUNCTION'] = finnhub_lastPriceDetails
        config['WATCHLIST_TABLENAME'] = "finnhub_watchlist"
    elif ps == 'FINNHUB_LOCAL':
        config['tal_master_tablename'] = 'finnhub_tas_listings'
        config['FUNDAMENTALS_TABLENAME'] = "finnhub_gem_listings"
        config['CLOSING_PRICES_FUNCTION'] = get_historic_prices_from_finnhub_local
        config['LAST_PRICES_DETLS_FUNCTION'] = finnhub_lastPriceDetails
        config['WATCHLIST_TABLENAME'] = "finnhub_watchlist"
    elif ps == 'EOD':
        config['tal_master_tablename'] = 'eod_tas_listings'
        config['FUNDAMENTALS_TABLENAME'] = "eod_gem_listings"
        config['CLOSING_PRICES_FUNCTION'] = get_historic_prices_from_eod
        config['LAST_PRICES_DETLS_FUNCTION'] = eod_lastPriceDetails
        config['WATCHLIST_TABLENAME'] = "eod_watchlist"
    elif ps == 'EOD_LOCAL':
        config['tal_master_tablename'] = 'eod_tas_listings'
        config['FUNDAMENTALS_TABLENAME'] = "eod_gem_listings"
        config['CLOSING_PRICES_FUNCTION'] = get_historic_prices_from_eod_local
        config['LAST_PRICES_DETLS_FUNCTION'] = eod_lastPriceDetails
        config['WATCHLIST_TABLENAME'] = "eod_watchlist"
    else:
        raise ValueError(f"Invalid PRICE_SOURCE: {ps}")

    config['tal_temp_tablename'] = config['tal_master_tablename'] + '_temp'
    return config

def get_country_name(watchlist: str) -> str:
    wl_country_names = {'US': 'USA', 'IN': 'India', 'BS': 'India-BSE', 'HK': 'Hong Kong'}
    if len(watchlist) >= 2:
        country_code = watchlist[:2].upper()
        return wl_country_names.get(country_code, "Unknown")
    else:
        return "Invalid"

# -------------------------
# Small utilities used by detectors
# -------------------------
def get_lasttrend_days(df: pd.DataFrame) -> int:
    if df.empty or 'Close' not in df.columns:
        return 0
    if df['Close'].max() == df['Close'].min():
        return 0
    days = 0
    prevtrend = None
    for idx in range(len(df) - 1, 0, -1):
        price = float(df.at[idx, 'Close'])
        prevprice = float(df.at[idx - 1, 'Close'])
        trend = -1 if price < prevprice else 1
        if prevtrend is None:
            prevtrend = trend
        if trend != prevtrend:
            break
        days += 1
        prevtrend = trend
    return days * prevtrend

def detect_rsi_uptrend(df: pd.DataFrame,
                       downturn_lookback: int = 12,
                       upturn_lookback: int = 5,
                       oversold_threshold: float = 35.0,
                       exit_oversold: float = 30.0,
                       min_downturn_bars: int = 4,
                       min_slope_threshold: float = 0.02,
                       rsi_smoothing: int = 2,
                       p_value_threshold: float = 0.2,
                       strength_threshold: float = 50.0,
                       debug: bool = False) -> Tuple[bool, float]:
    if df.empty or 'RSI' not in df.columns or len(df) < (downturn_lookback + upturn_lookback + 1):
        if debug: print("ERR: Insufficient data or no RSI")
        return False, 0.0

    rsi_ser = df['RSI'].dropna()
    if len(rsi_ser) < (downturn_lookback + upturn_lookback):
        if debug: print("ERR: Too few RSI values")
        return False, 0.0

    if rsi_smoothing > 1:
        rsi_ser = rsi_ser.ewm(span=rsi_smoothing).mean()

    recent_rsi = rsi_ser.iloc[-upturn_lookback:]
    prior_rsi = rsi_ser.iloc[-(downturn_lookback + upturn_lookback):-upturn_lookback]

    if len(prior_rsi) < min_downturn_bars:
        if debug: print(f"ERR: Prior window too short ({len(prior_rsi)} < {min_downturn_bars})")
        return False, 0.0

    x_prior = np.arange(len(prior_rsi))
    slope_prior, _, _, p_prior, _ = stats.linregress(x_prior, prior_rsi.to_numpy())
    downturn_confirmed = (
        slope_prior < -min_slope_threshold and
        p_prior < p_value_threshold and
        prior_rsi.iloc[-1] < oversold_threshold
    )
    if debug: print(f"Downturn check: slope={slope_prior:.3f}, p={p_prior:.3f}, end={prior_rsi.iloc[-1]:.1f} -> {downturn_confirmed}")
    if not downturn_confirmed:
        return False, 0.0

    x_recent = np.arange(len(recent_rsi))
    slope_recent, _, _, p_recent, _ = stats.linregress(x_recent, recent_rsi.to_numpy())
    net_increase = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
    small_dips_ok = (recent_rsi.diff().dropna() >= -recent_rsi.iloc[-1] * 0.08).all()
    crosses_exit = (recent_rsi.iloc[-1] > exit_oversold) and (recent_rsi.min() <= exit_oversold)
    uptrend_confirmed = (
        slope_recent > min_slope_threshold and
        p_recent < p_value_threshold and
        net_increase > 0 and
        small_dips_ok and
        crosses_exit
    )
    if debug: print(f"Upturn check: slope={slope_recent:.3f}, p={p_recent:.3f}, net={net_increase:.1f}, dips_ok={small_dips_ok}, cross={crosses_exit} -> {uptrend_confirmed}")
    if not uptrend_confirmed:
        return False, 0.0

    strength = 100.0
    is_uptrend = strength >= strength_threshold
    return is_uptrend, round(strength, 1)

def detect_trend_reversal(df, min_prior_trend_days=4, min_current_trend_days=1,
                          period_cci=20, period_rsi=14, atr_multiplier=1.5):
    required_cols = ['Date', 'Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return "ERR_MISSING_COLS"
    try:
        df_sorted = df.copy()
        df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
        df_sorted = df_sorted.sort_values('Date').reset_index(drop=True)
    except (ValueError, TypeError):
        return "ERR_INVALID_DATE"

    required_length = max(period_cci, period_rsi, 252) + min_prior_trend_days + min_current_trend_days + 3
    if len(df_sorted) < required_length:
        return "ERR_INSUFF_DATA"

    current_trend_result = get_lasttrend_days(df_sorted)
    if current_trend_result == 0:
        return "ERR_NO_TREND"

    current_trend_days = abs(current_trend_result)
    current_trend_type = "Up" if current_trend_result > 0 else "Down"
    if current_trend_days < min_current_trend_days:
        return "ERR_CURR_DAYS"

    prior_df = df_sorted.iloc[:-current_trend_days]
    prior_trend_result = get_lasttrend_days(prior_df)
    if prior_trend_result == 0:
        return "ERR_NO_PRIOR_TREND"
    prior_trend_days = abs(prior_trend_result)
    prior_trend_type = "Up" if prior_trend_result > 0 else "Down"

    if prior_trend_days < min_prior_trend_days:
        return "ERR_PRIOR_DAYS"
    if current_trend_days >= prior_trend_days:
        return "ERR_CURR_GT_PRIOR"
    if current_trend_type == prior_trend_type:
        return "ERR_SAME_TRENDS"

    base_code = f"{current_trend_type[0]}{current_trend_days}-{prior_trend_type[0]}{prior_trend_days}"
    current_price = df_sorted['Close'].iloc[-1]

    try:
        cci = pta.cci(high=df_sorted['High'], low=df_sorted['Low'], close=df_sorted['Close'], length=period_cci)
        rsi = pta.rsi(close=df_sorted['Close'], length=period_rsi)
        macd = pta.macd(close=df_sorted['Close'])['MACDh_12_26_9']
        adx = pta.adx(high=df_sorted['High'], low=df_sorted['Low'], close=df_sorted['Close'])['ADX_14']
        obv = pta.obv(close=df_sorted['Close'], volume=df_sorted['Volume'])
        atr = pta.atr(high=df_sorted['High'], low=df_sorted['Low'], close=df_sorted['Close'], length=14)
        bb = pta.bbands(close=df_sorted['Close'], length=20, std=2.0)
        stoch = pta.stoch(high=df_sorted['High'], low=df_sorted['Low'], close=df_sorted['Close'], k=14, d=3, smooth_k=3)
    except Exception as e:
        return f"ERR_INDICATOR: {str(e)}"

    price_data = {
        'current': current_price,
        'year_high': df_sorted['Close'].iloc[-252:].max(),
        'year_low': df_sorted['Close'].iloc[-252:].min(),
        'prior_high': df_sorted['Close'].iloc[-(current_trend_days + prior_trend_days):-current_trend_days].max(),
        'prior_low': df_sorted['Close'].iloc[-(current_trend_days + prior_trend_days):-current_trend_days].min()
    }

    indicator_values = {
        'cci': {'current': cci.iloc[-1], 'prev': cci.iloc[-2]},
        'rsi': {'current': rsi.iloc[-1], 'prev': rsi.iloc[-2]},
        'macd': {'current': macd.iloc[-1], 'prev': macd.iloc[-2]},
        'adx': {'current': adx.iloc[-1], 'prev': adx.iloc[-2]},
        'obv': {'current': obv.iloc[-1], 'prev': obv.iloc[-2]},
        'atr': atr.iloc[-1],
        'stoch': {
            'k_current': stoch['STOCHk_14_3_3'].iloc[-1],
            'd_current': stoch['STOCHd_14_3_3'].iloc[-1],
            'k_prev': stoch['STOCHk_14_3_3'].iloc[-2],
            'd_prev': stoch['STOCHd_14_3_3'].iloc[-2]
        },
        'bb': {
            'upper': bb['BBU_20_2.0'].iloc[-1],
            'lower': bb['BBL_20_2.0'].iloc[-1]
        },
        'sma': {
            '50': df_sorted['Close'].rolling(50).mean(),
            '200': df_sorted['Close'].rolling(200).mean()
        }
    }

    direction = "BULL" if current_trend_type == "Up" and prior_trend_type == "Down" else "BEAR"
    tests = {
        'price_bound': (current_price >= price_data['year_low']) if direction == "BULL"
        else (current_price <= price_data['year_high']),
        'volatility_move': (current_price - price_data['prior_low']) >= atr_multiplier * indicator_values['atr'] if direction == "BULL"
        else (price_data['prior_high'] - current_price) >= atr_multiplier * indicator_values['atr'],
        'volume_conf': (indicator_values['obv']['current'] > indicator_values['obv']['prev']) if direction == "BULL"
        else (indicator_values['obv']['current'] < indicator_values['obv']['prev']),
        'bollinger': (current_price > indicator_values['bb']['lower']) if direction == "BULL"
        else (current_price < indicator_values['bb']['upper']),
        'stochastic': (indicator_values['stoch']['k_current'] > indicator_values['stoch']['d_current'] and
                       indicator_values['stoch']['k_current'] < 20 and
                       indicator_values['stoch']['d_current'] < 20) if direction == "BULL"
        else (indicator_values['stoch']['k_current'] < indicator_values['stoch']['d_current'] and
              indicator_values['stoch']['k_current'] > 80 and
              indicator_values['stoch']['d_current'] > 80)
    }

    if direction == "BULL":
        tests.update({
            'cci': (indicator_values['cci']['prev'] < -100 and indicator_values['cci']['current'] > -100),
            'rsi': (indicator_values['rsi']['prev'] < 30 and indicator_values['rsi']['current'] > indicator_values['rsi']['prev']),
            'macd': (indicator_values['macd']['prev'] < 0 and indicator_values['macd']['current'] > 0),
            'sma_cross': any(indicator_values['sma']['50'].iloc[-i] > indicator_values['sma']['200'].iloc[-i] and
                             indicator_values['sma']['50'].iloc[-i - 1] < indicator_values['sma']['200'].iloc[-i - 1]
                             for i in range(1, 4)),
            'adx': (indicator_values['adx']['prev'] > indicator_values['adx']['current'])
        })
    else:
        tests.update({
            'cci': (indicator_values['cci']['prev'] > 100 and indicator_values['cci']['current'] < 100),
            'rsi': (indicator_values['rsi']['prev'] > 70 and indicator_values['rsi']['current'] < indicator_values['rsi']['prev']),
            'macd': (indicator_values['macd']['prev'] > 0 and indicator_values['macd']['current'] < 0),
            'sma_cross': any(indicator_values['sma']['50'].iloc[-i] < indicator_values['sma']['200'].iloc[-i] and
                             indicator_values['sma']['50'].iloc[-i - 1] > indicator_values['sma']['200'].iloc[-i - 1]
                             for i in range(1, 4)),
            'adx': (indicator_values['adx']['prev'] > indicator_values['adx']['current'])
        })

    total_tests = len(tests)
    confirmations = sum(1 for v in tests.values() if v)
    ratio = confirmations / total_tests if total_tests > 0 else 0
    if ratio >= 0.85:
        conf_level = "STRONG"
    elif ratio >= 0.65:
        conf_level = "MOD"
    else:
        conf_level = "WEAK"

    return f"{base_code}:{direction}-REVERSAL-{conf_level}_{confirmations}/{total_tests}"

def analyze_trend(df: pd.DataFrame):
    if len(df) < 2:
        return 0, "No Trend"
    df_temp = df.copy()
    df_ = df.iloc[:-1]
    trends = []
    trend = None
    duration = 0
    for i in range(1, len(df_)):
        if df_.at[i - 1, 'Close'] < df_.at[i, 'Close']:
            current_trend = 1
        elif df_.at[i - 1, 'Close'] > df_.at[i, 'Close']:
            current_trend = -1
        else:
            continue
        if current_trend != trend:
            if trend is not None:
                trends.append((trend, duration))
            trend = current_trend
            duration = 1
        else:
            duration += 1
    if trend is not None:
        trends.append((trend, duration))
    if not trends:
        return 0, "No Trend"
    latest_trend, latest_duration = trends[-1]
    latest_close = df_temp.iloc[-1]['Close']
    prev_close = df_.iloc[-1]['Close']
    if latest_close > prev_close and latest_trend == -1:
        reversal_direction = 'Reversal_Up'
    elif latest_close < prev_close and latest_trend == 1:
        reversal_direction = 'Reversal_Down'
    else:
        latest_duration += 1
        reversal_direction = "Continuation"
    return latest_trend * latest_duration, reversal_direction

# -------------------------
# Indicator mapping for legacy functions
# -------------------------
def get_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators using pandas_ta and add as uppercase columns.
    compute_indicators now returns the full DataFrame with indicator columns added.
    """
    if df is None or df.empty:
        return df
    
    # compute_indicators now returns df with all indicators as columns
    df = compute_indicators(df, params={"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9})
    
    # Map lowercase indicator columns to uppercase (expected by downstream code)
    if 'adx' in df.columns:
        df['ADX'] = df['adx']
    if 'plus_di' in df.columns:
        df['DIPLUS'] = df['plus_di']
    if 'minus_di' in df.columns:
        df['DIMINUS'] = df['minus_di']
    if 'sma200' in df.columns:
        df['SMA200'] = df['sma200']
    if 'sma50' in df.columns:
        df['SMA50'] = df['sma50']
    if 'ema_50' in df.columns:
        df['EMA50'] = df['ema_50']
    elif 'ema_fast' in df.columns:
        df['EMA50'] = df['ema_fast']
    if 'ema_20' in df.columns:
        df['EMA20'] = df['ema_20']
    elif 'ema_short' in df.columns:
        df['EMA20'] = df['ema_short']
    if 'cci' in df.columns:
        df['CCI'] = df['cci']
    if 'rsi' in df.columns:
        df['RSI'] = df['rsi']
    if 'obv' in df.columns:
        df['OBV'] = df['obv']
    if 'atr' in df.columns:
        df['ATR'] = df['atr']
    if 'bb_lower' in df.columns:
        df['BBL_20_2.0'] = df['bb_lower']
    if 'bb_middle' in df.columns:
        df['BBM_20_2.0'] = df['bb_middle']
    if 'bb_upper' in df.columns:
        df['BBU_20_2.0'] = df['bb_upper']
    
    # Create EMA200 if not present (for primary/secondary trend analysis)
    if 'EMA200' not in df.columns:
        if 'Close' in df.columns:
            df['EMA200'] = pta.ema(df['Close'], length=200)
        elif 'close' in df.columns:
            df['EMA200'] = pta.ema(df['close'], length=200)
    
    # Create BB_width (Bollinger Band width as percentage) if not present
    if 'BB_width' not in df.columns:
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'Close' in df.columns:
            df['BB_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
        elif 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns and 'Close' in df.columns:
            df['BB_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['Close']
        else:
            df['BB_width'] = 0.0
    
    # Ensure critical columns are filled with appropriate defaults to avoid None/NaN comparison errors
    df['BB_width'] = df['BB_width'].fillna(0.0)
    if 'ATR' in df.columns:
        df['ATR'] = df['ATR'].fillna(0.0)
    if 'ADX' in df.columns:
        df['ADX'] = df['ADX'].fillna(0.0)
    
    return df

# Config with thresholds (tune to your universe)
DEFAULT_INDICATORS_CONFIG = {
    'ADX_Strong_threshold': 25,
    'ADX_Average_threshold': 20,
    'ADX_Weak_threshold': 15,
    'BB_width_ranging_pct': 0.02,   # Bollinger band width relative to price -> small == ranging
    'ATR_volatility_multiplier': 0.015,  # ATR/price above this -> volatile
    'slope_window': 50,  # slope window for EMA200 or price trend
    'persistence_bars': 3,  # require condition for N bars
    'pullback_pct': 0.03,  # pullback definition relative to short EMA / recent high
    'weights': {'ema_cross': 1.0, 'di': 1.0, 'adx': 1.0, 'slope': 0.5}
}

def compute_linear_slope(series: pd.Series, window: int) -> pd.Series:
    """Return normalized slope (slope / mean price) using rolling linear regression."""
    def slope_calc(x):
        if np.isnan(x).any():
            return np.nan
        y = x
        x_idx = np.arange(len(y))
        m, _, _, _, _ = stats.linregress(x_idx, y)
        # normalize by mean price to compare across symbols
        meanp = np.mean(y) if np.mean(y) != 0 else 1.0
        return m / meanp
    return series.rolling(window).apply(slope_calc, raw=True)

def get_primary_secondary_trends(df: pd.DataFrame, config: dict = DEFAULT_INDICATORS_CONFIG) -> pd.DataFrame:
    df = df.copy()
    C = config

    # INPUT ASSUMPTIONS: indicators.py populated these columns:
    # 'EMA20', 'EMA50', 'EMA200', 'ADX', 'DITrend' (strings 'Bull'/'Bear'), 'BB_width' (bb_high - bb_low)/price, 'ATR', 'Close'
    # If not present, compute them in indicators.py before calling this function.
    
    # Fill NaN/None values to prevent comparison errors
    numeric_cols = ['EMA20', 'EMA50', 'EMA200', 'ADX', 'BB_width', 'ATR', 'Close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    # Ensure DITrend column exists with proper default
    if 'DITrend' in df.columns:
        df['DITrend'] = df['DITrend'].fillna('Neutral')
    else:
        df['DITrend'] = 'Neutral'

    # 1) ADX strength categories (re-using your helper idea)
    adx_conditions = [
        (df['ADX'] >= C['ADX_Strong_threshold']),
        (df['ADX'] >= C['ADX_Average_threshold']),
        (df['ADX'] < C['ADX_Average_threshold'])
    ]
    adx_choices = ['Strong', 'Average', 'Weak']
    df['ADX_Strength'] = np.select(adx_conditions, adx_choices, default='Weak')

    # 2) Primary trend: combine EMA50/EMA200, DITrend, ADX_Strength, and slope of EMA200
    # Score features (numeric) to reduce brittleness; then threshold
    df['feat_ema'] = np.where(df['EMA50'] > df['EMA200'], 1.0, -1.0)  # 1 = bullish bias, -1 = bearish
    df['feat_di'] = np.where(df['DITrend'] == 'Bull', 1.0, np.where(df['DITrend'] == 'Bear', -1.0, 0.0))
    df['feat_adx'] = np.where(df['ADX'] >= C['ADX_Strong_threshold'], 1.0, np.where(df['ADX'] >= C['ADX_Average_threshold'], 0.5, 0.0))
    # slope of EMA200 or price; positive slope => bullish
    slope = compute_linear_slope(df['EMA200'].fillna(df['Close']), C['slope_window']).fillna(0.0)
    df['feat_slope'] = np.sign(slope) * np.minimum(np.abs(slope) * 10, 1.0)  # compress to [-1,1]

    w = C['weights']
    df['primary_score'] = (w['ema_cross'] * df['feat_ema'] +
                           w['di'] * df['feat_di'] +
                           w['adx'] * df['feat_adx'] +
                           w['slope'] * df['feat_slope'])

    # Map numeric score to Primary label; tune thresholds to your asset class
    df['Primary'] = 'Neutral'
    df.loc[df['primary_score'] >= 1.0, 'Primary'] = 'Bull'
    df.loc[df['primary_score'] <= -1.0, 'Primary'] = 'Bear'

    # 3) Secondary trend: categorize shorter-term behavior
    # Ranging detection: low ADX and small BB width
    is_low_adx = df['ADX'] < C['ADX_Average_threshold']
    is_narrow_bb = df['BB_width'] < C['BB_width_ranging_pct']
    is_volatile = (df['ATR'] / df['Close']) > C['ATR_volatility_multiplier']

    # Short-term cross logic: EMA 20 vs EMA 50
    cross_up = (df['EMA20'] > df['EMA50'])
    cross_down = (df['EMA20'] < df['EMA50'])

    # Pullback detection: price retraces towards EMA20/EMA50 while primary remains Bull/Bear
    # Simple heuristic: if Primary==Bull and Close < EMA20 but Close > EMA50 => pullback in bull
    is_pullback_bull = (df['Primary'] == 'Bull') & (df['Close'] < df['EMA20']) & (df['Close'] > df['EMA50'])
    is_pullback_bear = (df['Primary'] == 'Bear') & (df['Close'] > df['EMA20']) & (df['Close'] < df['EMA50'])

    # Compose secondary
    df['Secondary'] = 'Unknown'
    df.loc[is_volatile, 'Secondary'] = 'Volatile'
    df.loc[is_low_adx & is_narrow_bb & ~is_volatile, 'Secondary'] = 'Ranging'
    df.loc[(~is_low_adx) & (df['ADX'] >= C['ADX_Strong_threshold']) & cross_up, 'Secondary'] = 'TrendingUp'
    df.loc[(~is_low_adx) & (df['ADX'] >= C['ADX_Strong_threshold']) & cross_down, 'Secondary'] = 'TrendingDown'
    df.loc[is_pullback_bull, 'Secondary'] = 'PullbackInBull'
    df.loc[is_pullback_bear, 'Secondary'] = 'PullbackInBear'

    # Fallback: if primary = Neutral and short-term crosses with ADX average -> classify as ShortTrend
    df.loc[(df['Primary'] == 'Neutral') & (df['ADX'] >= C['ADX_Average_threshold']) & (cross_up | cross_down), 'Secondary'] = 'ShortTrend'

    # 4) Persistence smoothing: require N-of-last M bars agreement (majority) before label fully applied
    persistence = C['persistence_bars']
    # helper to majority-smooth categorical series
    def majority_smooth(series, window):
        # For categorical data, use expanding window mode calculation with proper type handling
        result = pd.Series(index=series.index, dtype=object)
        for i in range(len(series)):
            start = max(0, i - window + 1)
            window_data = series.iloc[start:i+1]
            mode_vals = window_data.mode()
            result.iloc[i] = mode_vals.iloc[0] if len(mode_vals) > 0 else window_data.iloc[-1]
        return result

    df['Primary_smoothed'] = majority_smooth(df['Primary'].astype(str), persistence)
    df['Secondary_smoothed'] = majority_smooth(df['Secondary'].astype(str), persistence)

    # final composite string for reporting
    df['Trend'] = df['Primary_smoothed'].astype(str) + '[' + df['Secondary_smoothed'].astype(str) + ']' 

    # Convert to categorical to save memory and for faster comparisons
    df['Primary'] = df['Primary_smoothed'].astype('category')
    df['Secondary'] = df['Secondary_smoothed'].astype('category')
    df['ADX_Strength'] = df['ADX_Strength'].astype('category')

    # drop intermediate features if you want
    to_drop = ['feat_ema', 'feat_di', 'feat_adx', 'feat_slope', 'primary_score', 'Primary_smoothed', 'Secondary_smoothed']
    for c in to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df

def get_MA_Trend(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA20_prev'] = df['EMA20'].shift(1)
    df['EMA50_prev'] = df['EMA50'].shift(1)
    df['Close_prev'] = df['Close'].shift(1)
    df['Close_prev2'] = df['Close'].shift(2)
    conditions = [
        (df['EMA20'] > df['EMA50']) &
        (df['EMA20'] > df['EMA20_prev']) &
        (df['EMA50'] > df['EMA50_prev']) &
        (df['Close'] > df['Close_prev']) &
        (df['Close_prev'] > df['Close_prev2']),
        (df['EMA50'] > df['EMA20']) &
        (df['EMA20'] <= df['EMA20_prev']) &
        (df['EMA50'] <= df['EMA50_prev']) &
        (df['Close'] < df['Close_prev']) &
        (df['Close_prev'] < df['Close_prev2']),
        (df['EMA20'] > df['EMA50']) &
        (df['EMA20'] > df['EMA20_prev']) &
        (df['EMA50'] > df['EMA50_prev']),
        (df['EMA50'] > df['EMA20']) &
        (df['EMA20'] <= df['EMA20_prev']) &
        (df['EMA50'] <= df['EMA50_prev'])
    ]
    choices = ['Bull+', 'Bear+', 'Bull', 'Bear']
    df['MA_Trend'] = np.select(conditions, choices, default='Nil')
    df.drop(['EMA20_prev', 'EMA50_prev', 'Close_prev', 'Close_prev2'], axis=1, inplace=True)
    return df

def get_ADX_Strength(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    conditions = [
        (df['ADX'] >= config['ADX_Strong_threshold']),
        (df['ADX'] >= config['ADX_Average_threshold']),
        (df['ADX'] < config['ADX_Average_threshold'])
    ]
    choices = ['Strong', 'Average', 'Weak']
    df['ADX_Strength'] = np.select(conditions, choices, default='Unknown')
    return df

def get_DI_Strength(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    conditions = [
        (df['DIWidth'] >= config['DIWidth_Strong_threshold']),
        (df['DIWidth'] >= config['DIWidth_Average_threshold']),
        (df['DIWidth'] < config['DIWidth_Average_threshold'])
    ]
    choices = ['Strong', 'Average', 'Weak']
    df['DI_Strength'] = np.select(conditions, choices, default='Unknown')
    return df

def get_MADI_Trend(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df['EMA20'] > df['EMA50']) & (df['DITrend'] == 'Bull'),
        (df['EMA50'] > df['EMA20']) & (df['DITrend'] == 'Bear')
    ]
    choices = ['Bull', 'Bear']
    df['MADI_Trend'] = np.select(conditions, choices, default='Nil')
    return df

def add_tma_single_crossover_flag(df: pd.DataFrame, fast_col: str, slow_col: str, lookback: int, out_col: str) -> pd.DataFrame:
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty or 'Close' not in df.columns:
        if isinstance(df, pd.DataFrame) and out_col not in df.columns:
            df[out_col] = 0
        return df
    df = df.copy().reset_index(drop=True)
    n = len(df)
    last_idx = n - 1
    lb = max(1, int(lookback) if lookback is not None else 1)
    window_start = max(0, last_idx - lb + 1)
    def _parse_len(col_name, default):
        s = ''.join(ch for ch in str(col_name) if ch.isdigit())
        try:
            return int(s) if s else int(default)
        except Exception:
            return int(default)
    fast_len = _parse_len(fast_col, 21)
    slow_len = _parse_len(slow_col, 50)
    def compute_tma(series: pd.Series, length: int) -> pd.Series:
        k = max(1, (length + 1) // 2)
        sma1 = series.rolling(window=k, min_periods=k).mean()
        tma = sma1.rolling(window=k, min_periods=k).mean()
        return tma
    if fast_col not in df.columns:
        df[fast_col] = compute_tma(df['Close'], fast_len)
    if slow_col not in df.columns:
        df[slow_col] = compute_tma(df['Close'], slow_len)
    tma_fast = df[fast_col]
    tma_slow = df[slow_col]
    diff = tma_fast - tma_slow
    cross_up = (diff > 0) & (diff.shift(1) <= 0)
    cross_dn = (diff < 0) & (diff.shift(1) >= 0)
    slope_fast = tma_fast.diff()
    slope_slow = tma_slow.diff()
    price = df['Close']
    above_both = price >= pd.concat([tma_fast, tma_slow], axis=1).max(axis=1)
    below_both = price <= pd.concat([tma_fast, tma_slow], axis=1).min(axis=1)
    if 'ADX' in df.columns:
        adx_ok = (df['ADX'] >= 20)
    else:
        adx_ok = pd.Series(True, index=df.index)
    bull_conf = cross_up & (slope_fast > 0) & (slope_slow >= 0) & above_both & adx_ok
    bear_conf = cross_dn & (slope_fast < 0) & (slope_slow <= 0) & below_both & adx_ok
    df[out_col] = 0
    conf_idxs = np.where((bull_conf | bear_conf).to_numpy())[0]
    if conf_idxs.size:
        recent = conf_idxs[conf_idxs >= window_start]
        if recent.size:
            pos = int(recent.max())
            df.at[last_idx, out_col] = 1 if bool(bull_conf.iloc[pos]) else -1
    try:
        df[out_col] = df[out_col].astype(np.int8)
    except Exception:
        pass
    return df

def get_technicals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = get_technical_indicators(df)
    df['DIWidth'] = (df['DIPLUS'] - df['DIMINUS']).abs()
    df['DITrend'] = np.select(
        [df['DIPLUS'] > df['DIMINUS'], df['DIPLUS'] == df['DIMINUS'], df['DIPLUS'] < df['DIMINUS']],
        ['Bull', 'Nil', 'Bear'],
        default='Unknown'
    )
    df = get_ADX_Strength(df, config)
    df = get_DI_Strength(df, config)
    df = get_MA_Trend(df)
    df = get_MADI_Trend(df)
    df = get_primary_secondary_trends(df)
    df = add_tma_single_crossover_flag(df, 'TMA_fast', 'TMA_slow', lookback=10, out_col='TMA21_50_X')
    return df

def get_symbols_forwatchlist(watchlist: str, config: dict) -> pd.DataFrame:
    with getDbConnection() as conn:
        query = text(f"SELECT DISTINCT symbol FROM {config['WATCHLIST_TABLENAME']} WHERE watchlist = :watchlist;")
        df = pd.read_sql(query, con=conn, params={"watchlist": watchlist})
    return df

# -------------------------
# Worker logic (per-symbol processing)
# -------------------------
def get_tlib_tadata(underlying: str, price_source: str, my_logger, df: Optional[pd.DataFrame] = None, mainrun: bool = True) -> Optional[pd.DataFrame]:
    printnlog(f"[get_tlib_tadata : {underlying}, {price_source}, df : {len(df) if df is not None else 'None'}, mainrun: {mainrun}]",
              my_logger=my_logger)
    config = initialize_config(price_source)
    if df is None:
        try:
            df = config['CLOSING_PRICES_FUNCTION'](underlying)
        except Exception as e:
            my_logger.info(f'[get_tlib_tadata :: {config["CLOSING_PRICES_FUNCTION"].__name__} ERROR! - {e}\n {traceback.format_exc()}]')
            return None
    if df is None or df.empty:
        my_logger.info(f"[get_tlib_tadata : Symbol {underlying}, Received Empty df]")
        return None

    # Normalize columns
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'volume': 'Volume', 'date': 'Date', 'symbol': 'Symbol'
    }, inplace=True)

    if 'Symbol' not in df.columns or df['Symbol'].isnull().all():
        df['Symbol'] = underlying
    df['Symbol'] = df['Symbol'].astype(str)

    df = df.sort_values(['Date'], ascending=[True]).reset_index(drop=True)
    history_bars = int(config.get('HISTORY_BARS', 600))
    if history_bars > 0 and len(df) > history_bars:
        df = df.tail(history_bars).reset_index(drop=True)
    df = get_technicals(df, config)
    result = {}
    try:
        result = config['LAST_PRICES_DETLS_FUNCTION'](underlying)
    except Exception as e:
        my_logger.info(f'[get_tlib_tadata :: {config["LAST_PRICES_DETLS_FUNCTION"].__name__} Error in getting last price details - {e}\n {traceback.format_exc()}]')

    d = df.tail(1).copy()
    d.drop([c for c in ['Unnamed', 'index', 'level_0'] if c in d.columns], axis=1, inplace=True)
    # Force numeric types for result fields to prevent truncation errors
    d['TodayPrice'] = float(result.get('TodayPrice', 0.0) or 0.0)
    d['marketCap'] = float(result.get('marketCap', 0.0) or 0.0)
    d['GEM_Rank'] = str(result.get('GEM_Rank', '') or '')
    # Keep CountryName as None/pd.NA instead of converting to string '<NA>'
    country = result.get('CountryName')
    d['CountryName'] = country if country else pd.NA
    d['IndustrySector'] = str(result.get('Sector', '') or '')
    d['High52'] = float(result.get('High52', 0.0) or 0.0)
    d['Low52'] = float(result.get('Low52', 0.0) or 0.0)
    d['Pct2H52'] = float(result.get('Pct2H52', 0.0) or 0.0)
    d['PctfL52'] = float(result.get('PctfL52', 0.0) or 0.0)

    # RSIUturnTypeOld (legacy, keep as is)
    try:
        detector_old = TrendReversalDetector(df)
        d['RSIUturnTypeOld'] = detector_old.signal()
    except Exception as e:
        my_logger.info(f"get_tlib_tadata : Symbol {underlying}, TrendReversalDetector error - {e}")
        d['RSIUturnTypeOld'] = pd.NA

    # TrendReversal_Rules
    try:
        label, details = detect_reversal_pro(df)
        d['TrendReversal_Rules'] = label
    except Exception as e:
        my_logger.info(f"get_tlib_tadata : Symbol {underlying}, detect_reversal_pro error - {e}")
        d['TrendReversal_Rules'] = pd.NA

    # SignalClassifier_Rules level
    rules_signal = pd.NA
    sc = SignalClassifier()
    try:
        rules_signal = sc.get_rule_signal_int(df)
        d['SignalClassifier_Rules'] = rules_signal
    except Exception as e:
        my_logger.info(f"[SignalClassifier.rule error for {underlying}: {e}]")
        d['SignalClassifier_Rules'] = pd.NA

    # SignalClassifier_ML
    ml_signal = pd.NA
    if USE_ML:
        try:
            ml_signal = sc.get_ml_signal_int(df)
            d['SignalClassifier_ML'] = ml_signal
        except Exception as e:
            my_logger.info(f"[SignalClassifier.ml error for {underlying}: {e}]")
            d['SignalClassifier_ML'] = pd.NA
    else:
        d['SignalClassifier_ML'] = pd.NA

    # RSI uptrend + trend analysis
    is_uptrend, strength = detect_rsi_uptrend(df, oversold_threshold=35, min_slope_threshold=0.02, debug=False)
    d['RSIUpTrend'] = is_uptrend
    LastTrendDays, LastTrendType = analyze_trend(df)
    d['LastTrendDays'] = LastTrendDays
    d['LastTrendType'] = LastTrendType

    """ Trend determination 
    try:
        # d is a single-row DataFrame; use .iat[0] on the Series to extract scalar safely
        trend_val = str(d['ADX_Strength'].iat[0]) + "_" + str(d['MADI_Trend'].iat[0])
        d['Trend'] = trend_val
    except Exception as e:
        my_logger.info(f"get_tlib_tadata : Symbol {underlying}, ERROR! in Determining Trend - {e}\n {traceback.format_exc()}")
        return None
    """

    # ML Price Prediction (Multi-Horizon)
    if USE_ML:
        try:
            # Pass the full df with historical OHLCV data
            prediction_result = predict_best_horizon(df, min_confidence=0.40)
            if prediction_result is not None:
                d['ML_Target_Price'] = round(prediction_result['price'], 2)
                d['ML_Target_Price_Days'] = prediction_result['days']
                d['ML_Confidence_Score'] = round(prediction_result['confidence'], 2)
                d['ML_Target_Return_Pct'] = round(prediction_result['return_pct'], 2)
            else:
                d['ML_Target_Price'] = pd.NA
                d['ML_Target_Price_Days'] = pd.NA
                d['ML_Confidence_Score'] = pd.NA
                d['ML_Target_Return_Pct'] = pd.NA
        except Exception as e:
            my_logger.info(f"[ML_Predict_Price error for {underlying}: {e}]")
            d['ML_Target_Price'] = pd.NA
            d['ML_Target_Price_Days'] = pd.NA
            d['ML_Confidence_Score'] = pd.NA
            d['ML_Target_Return_Pct'] = pd.NA
    else:
        d['ML_Target_Price'] = pd.NA
        d['ML_Target_Price_Days'] = pd.NA
        d['ML_Confidence_Score'] = pd.NA
        d['ML_Target_Return_Pct'] = pd.NA

    d['ScanDate'] = strUtcNow()

    canonical_cols = list(canonical_table_schema().keys())
    for col in canonical_cols:
        if col not in d.columns:
            d[col] = None
    try:
        out = d[canonical_cols].reset_index(drop=True)
    except Exception:
        out = d.copy().reindex(columns=canonical_cols)
    return out

def process_symbol(symbol: str, price_source: str, my_logger_name: str) -> Tuple[str, Optional[pd.DataFrame], Optional[float], Optional[str]]:
    my_logger = parallelLoggingSetter(my_logger_name)
    start_time = time.time()
    try:
        df_row = get_tlib_tadata(underlying=symbol, price_source=price_source, my_logger=my_logger)
        time_taken = round(time.time() - start_time, 4)
        return (symbol, df_row, time_taken, None)
    except Exception as e:
        return (symbol, None, None, f"[get_tlib_tadata Error : {e}, \n {traceback.format_exc()}]")

# -------------------------
# Worker initializer
# -------------------------
def worker_init(use_ml: bool):
    """
    Worker initializer invoked by ProcessPoolExecutor.
    Caches global classifier payload per worker process.
    TrendReversalDetectorML removed from worker init.
    """
    global USE_ML
    USE_ML = use_ml

    try:
        worker_logger = parallelLoggingSetter("worker_init")
    except Exception:
        worker_logger = logging.getLogger("worker_init")
        if not worker_logger.handlers:
            logging.basicConfig(level=logging.INFO)

    worker_logger.info(f"[worker_init] INIT worker: USE_ML={USE_ML}")

# -------------------------
# Bulk DB insert
# -------------------------
def bulk_insert_dataframe(table_name: str, df: pd.DataFrame, chunksize: int = 500) -> None:
    if df is None or df.empty:
        return
    fallback_logger = logging.getLogger("ta_signals_mc_parallel")
    # Force numeric types to prevent truncation errors
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'ADX', 'SMA200', 'EMA50', 'EMA20', 'CCI', 'RSI', 
                    'TodayPrice', 'marketCap', 'High52', 'Low52', 'Pct2H52', 'PctfL52', 'LastTrendDays', 
                    'SignalClassifier_Rules', 'SignalClassifier_ML', 'ML_Target_Price', 
                    'ML_Target_Price_Days', 'ML_Confidence_Score', 'ML_Target_Return_Pct']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Boolean
    if 'RSIUpTrend' in df.columns:
        df['RSIUpTrend'] = df['RSIUpTrend'].astype(bool)
    try:
        with getDbConnection() as con:
            try:
                df.to_sql(table_name, con=con, index=False, if_exists='append', method='multi', chunksize=chunksize)
            except TypeError:
                df.to_sql(table_name, con=con, index=False, if_exists='append', chunksize=chunksize)
            try:
                if hasattr(con, 'commit') and callable(con.commit):
                    con.commit()
            except Exception:
                pass
    except Exception as e:
        printnlog(f"[bulk_insert_dataframe] Error inserting into {table_name}: {e}\n{traceback.format_exc()}", my_logger=fallback_logger)
        raise

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Technical Analysis Signals Processor')
    parser.add_argument('-w', '--watchlist', nargs='?', help='Watchlist name', required=True)
    parser.add_argument('-s', '--source', nargs='?', help='Price source (FINNHUB/EOD/FINNHUB_LOCAL/EOD_LOCAL)', required=True)
    parser.add_argument('--use_ml', choices=['yes', 'no'], default='yes', help='Whether to use ML scoring (default: yes).')
    parser.add_argument('--max_workers', type=int, default=4, help='Max parallel workers')
    args = parser.parse_args()

    global USE_ML

    USE_ML = str(args.use_ml).lower() == 'yes'
    if not USE_ML:
        printnlog("[INFO] ML scoring disabled via --use_ml=no.")

    config = initialize_config(args.source)
    WATCHLIST = args.watchlist.upper()
    my_logger_name = f'ta_signals_mc_parallel_{WATCHLIST}'
    my_logger = parallelLoggingSetter(my_logger_name)
    printnlog(config, my_logger=my_logger)

    countryName = get_country_name(WATCHLIST)
    if countryName == "Invalid":
        printnlog(f'\n[WATCHLIST = {WATCHLIST} not supported]', my_logger=my_logger)
        sys.exit()

    symbols_df = get_symbols_forwatchlist(WATCHLIST, config)
    symbols = symbols_df['symbol'].tolist()
    if not symbols:
        printnlog(f"[get_symbols_forwatchlist: 0 Symbols found for {WATCHLIST}]", my_logger=my_logger)
        sys.exit()
    printnlog(f"[Count of total Symbols : {len(symbols)}]", my_logger=my_logger)

    try:
        ensure_output_table(config['tal_temp_tablename'], my_logger=my_logger)
        ensure_output_table(config['tal_master_tablename'], my_logger=my_logger)
        my_logger.info(f"[Ensured tables {config['tal_temp_tablename']} and {config['tal_master_tablename']} exist with canonical schema]")
    except Exception as e:
        my_logger.error(f"[Failed to ensure output tables: {e}\n{traceback.format_exc()}]")
        sys.exit(1)

    # Delete old temp rows for this country
    try:
        printnlog('[Deleting outdated records from temporary table]', my_logger=my_logger)
        delete_sql = text(f'DELETE FROM `{config["tal_temp_tablename"]}` WHERE `CountryName` = :country')
        with getDbConnection() as con:
            con.execute(delete_sql, {"country": countryName})
            try:
                if hasattr(con, 'commit') and callable(con.commit):
                    con.commit()
            except Exception:
                pass
            my_logger.info("[Delete operation completed successfully]")
    except Exception as e:
        my_logger.error(f"[Error during delete from {config['tal_temp_tablename']}: {e}\n {traceback.format_exc()}]")

    # Parallel processing
    rows: List[pd.DataFrame] = []
    df_temp_all = pd.DataFrame()
    i = 1
    total_time = 0.0
    with ProcessPoolExecutor(max_workers=args.max_workers, initializer=worker_init, initargs=(USE_ML,)) as executor:
        futures = {executor.submit(process_symbol, symbol, config['PRICE_SOURCE'], my_logger_name): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, df_row, time_taken, error = future.result()
                if error:
                    my_logger.error(error)
                else:
                    if df_row is not None and isinstance(df_row, pd.DataFrame) and not df_row.empty:
                        rows.append(df_row)
                    total_time += time_taken or 0.0
                    avg_time = round(total_time / i, 4) if i > 0 else 0
                    remaining_time = round(avg_time * (len(symbols) - i), 4)
                    log_str = f"[End Processing {symbol} - time taken {time_taken} seconds. Avg {avg_time}, time remaining {remaining_time}]"
                    printnlog(log_str, my_logger=my_logger)
                i += 1
            except Exception as exc:
                my_logger.error(f"[Parallel processing error for {symbol}: {exc}]")

    # Bulk insert collected rows into temp
    if rows:
        try:
            df_temp_all = pd.concat(rows, ignore_index=True)
            canonical_cols = list(canonical_table_schema().keys())
            for c in canonical_cols:
                if c not in df_temp_all.columns:
                    df_temp_all[c] = None
            df_temp_all = df_temp_all[canonical_cols]

            # Fill missing CountryName
            try:
                mask = df_temp_all['CountryName'].isna() | (df_temp_all['CountryName'].astype(str).str.strip() == '')
                filled_count = int(mask.sum())
                if filled_count > 0:
                    df_temp_all.loc[mask, 'CountryName'] = countryName
                    my_logger.info(f"[CountryName fill] Filled {filled_count} rows in temp with CountryName='{countryName}'")
            except Exception as e:
                my_logger.warning(f"[CountryName fill] Error while filling CountryName: {e}")

            # Symbol protection
            try:
                if 'Symbol' not in df_temp_all.columns:
                    df_temp_all['Symbol'] = ''
                mask_sym = df_temp_all['Symbol'].isna() | (df_temp_all['Symbol'].astype(str).str.strip() == '')
                sym_filled = int(mask_sym.sum())
                if sym_filled > 0:
                    df_temp_all.loc[mask_sym, 'Symbol'] = ''
                    my_logger.info(f"[Symbol check] Found {sym_filled} blank Symbol rows in temp payload")
            except Exception as e:
                my_logger.warning(f"[Symbol check] Error while checking/filling Symbol: {e}")

            bulk_insert_dataframe(config['tal_temp_tablename'], df_temp_all, chunksize=500)
            my_logger.info(f"[Inserted {len(df_temp_all)} rows into {config['tal_temp_tablename']}]")
        except Exception as e:
            my_logger.error(f"[Error inserting into temp table: {e}\n{traceback.format_exc()}]")
    else:
        my_logger.info('[No rows collected from workers; skipping temp insert]')

    # Reuse in-memory df_temp_all to insert into master
    if not df_temp_all.empty:
        df_final = df_temp_all.copy()
    else:
        df_final = pd.DataFrame()

    if not df_final.empty:
        try:
            with getDbConnection() as con:
                sql_delete_master = text(f'DELETE FROM `{config["tal_master_tablename"]}` WHERE `CountryName` = :country')
                result = con.execute(sql_delete_master, {"country": countryName})
                try:
                    if hasattr(con, 'commit') and callable(con.commit):
                        con.commit()
                except Exception:
                    pass
                try:
                    my_logger.info(f"[Deleted {getattr(result, 'rowcount', 'N/A')} rows from {config['tal_master_tablename']}]")
                except Exception:
                    my_logger.info(f"[Cleared rows from {config['tal_master_tablename']}]")
        except Exception as e:
            my_logger.error(f"[Deleting rows from {config['tal_master_tablename']}: Error -- {e} --\n {traceback.format_exc()}]")

        try:
            canonical_cols = list(canonical_table_schema().keys())
            for c in canonical_cols:
                if c not in df_final.columns:
                    df_final[c] = None
            df_final = df_final[canonical_cols]

            try:
                mask = df_final['CountryName'].isna() | (df_final['CountryName'].astype(str).str.strip() == '')
                filled_count = int(mask.sum())
                if filled_count > 0:
                    df_final.loc[mask, 'CountryName'] = countryName
                    my_logger.info(f"[CountryName fill] Filled {filled_count} rows in master payload with CountryName='{countryName}'")
            except Exception as e:
                my_logger.warning(f"[CountryName fill] Error while filling CountryName in master payload: {e}")

            try:
                if 'Symbol' not in df_final.columns:
                    df_final['Symbol'] = ''
                mask_sym = df_final['Symbol'].isna() | (df_final['Symbol'].astype(str).str.strip() == '')
                sym_filled = int(mask_sym.sum())
                if sym_filled > 0:
                    df_final.loc[mask_sym, 'Symbol'] = ''
                    my_logger.info(f"[Symbol fill] Filled {sym_filled} Symbol rows with empty-string before master insert")
            except Exception as e:
                my_logger.warning(f"[Symbol fill] Error while final-filling Symbol: {e}")

            bulk_insert_dataframe(config['tal_master_tablename'], df_final, chunksize=500)
            my_logger.info(f"[{len(df_final)} rows inserted into {config['tal_master_tablename']}]")
        except Exception as e:
            my_logger.error(f"[Insert into {config['tal_master_tablename']}: Error -- {e} --\n {traceback.format_exc()}]")
    else:
        my_logger.info('[Empty temp dataset or an error occurred; nothing to insert into master]')

if __name__ == '__main__':
    main()


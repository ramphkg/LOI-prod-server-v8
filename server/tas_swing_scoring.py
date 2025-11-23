#!/usr/bin/env python3
# tas_swing_scoring.py
#
# Purpose:
#   Three-tier ranking system for swing trading stock selection:
#   - Ranking 1: Price Pattern Detection (recent uptrend after downtrend)
#   - Ranking 2: Technical Analysis Scoring (SMA distance, RSI, Pct2H52, GEM)
#   - Ranking 3: Composite Ranking (weighted combination of Ranking 1 & 2)
#
# Usage:
#   python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB --n1_ideal 3 --top_n 50
#
# Output:
#   - swing_rankings_full_<master>_<country>_<YYYYMMDD>.csv (all stocks with rankings)
#   - swing_rankings_top_<master>_<country>_<YYYYMMDD>.csv (top N opportunities)
#   - swing_strong_buy_<master>_<country>_<YYYYMMDD>.csv (strong buy signals only)

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

# Project imports
from app_imports import getDbConnection, parallelLoggingSetter
from ta_signals_mc_parallel import initialize_config, get_country_name

# --------------------------
# Logging Setup (parallel-safe file logger)
# --------------------------
LOG = parallelLoggingSetter(module="tas_swing_scoring")

# --------------------------
# Configuration Constants
# --------------------------
DEFAULT_PATTERN_CONFIG = {
    'n1_ideal': 3,              # ideal recent uptrend length
    'min_recent_up': 2,         # minimum d1 to qualify
    'min_prior_down': 5,        # minimum d2 to qualify
    'd2_max': 20,               # cap for d2 scoring
    'd1_weight': 0.4,           # weight for d1 component
    'd2_weight': 0.6,           # weight for d2 component
    'lookback_days': 60,        # history window to analyze
    'allow_flat_days': 1        # allow occasional flat days within trend
}

DEFAULT_TECHNICAL_CONFIG = {
    'sma_thresholds': [2, 5, 10, 15],           # distance % thresholds
    'rsi_thresholds': [30, 35, 40, 45, 50],     # RSI level thresholds
    'pct2h52_thresholds': [5, 10, 20, 30, 40],  # 52W distance thresholds
    'gem_thresholds': [500, 1000, 1500, 2000, 3000]  # GEM rank thresholds
}

DEFAULT_COMPOSITE_CONFIG = {
    'pattern_weight': 0.45,
    'technical_weight': 0.55,
    'signal_thresholds': {
        'strong_buy': 80,
        'buy': 65,
        'weak_buy': 50
    }
}

# Price filters by country (avoid penny stocks and extremely expensive stocks)
PRICE_FILTERS = {
    'USA': {'min': 5.0, 'max': 500.0},           # US stocks: $5-$500
    'India': {'min': 50.0, 'max': 5000.0},       # Indian stocks: ₹50-₹5000
    'India-BSE': {'min': 50.0, 'max': 5000.0},   # BSE stocks: ₹50-₹5000
    'Hong Kong': {'min': 5.0, 'max': 500.0},     # HK stocks: HK$5-HK$500
    'Unknown': {'min': 1.0, 'max': 10000.0}      # Fallback: minimal filtering
}

# --------------------------
# Utility Functions
# --------------------------
def utc_datestr() -> str:
    """Return current UTC date as YYYYMMDD string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def safe_float(value, default=0.0) -> float:
    """Safely convert value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0) -> int:
    """Safely convert value to int."""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

# --------------------------
# RANKING 1: Price Pattern Detection
# --------------------------
def detect_price_pattern(close_prices: pd.Series, config: Dict = None) -> Dict:
    """
    Detect recent uptrend (d1) after prior downtrend (d2) pattern.
    
    Args:
        close_prices: Series of closing prices (oldest to newest)
        config: Pattern configuration dict
        
    Returns:
        Dict with pattern detection results:
        {
            'detected': bool,
            'd1_recent_up': int,
            'd2_prior_down': int,
            'pattern_score': float (0-100)
        }
    """
    cfg = DEFAULT_PATTERN_CONFIG.copy()
    if config:
        cfg.update(config)
    
    result = {
        'detected': False,
        'd1_recent_up': 0,
        'd2_prior_down': 0,
        'pattern_score': 0.0
    }
    
    if close_prices is None or len(close_prices) < (cfg['min_recent_up'] + cfg['min_prior_down']):
        return result
    
    prices = close_prices.values
    n = len(prices)
    
    # Step 1: Count recent consecutive up days (d1) from the end
    d1 = 0
    flat_count = 0
    i = n - 1
    
    while i > 0 and (d1 + flat_count) < cfg['lookback_days']:
        diff = prices[i] - prices[i-1]
        if diff > 1e-9:  # up day (use small threshold for float comparison)
            d1 += 1
            i -= 1
        elif abs(diff) < 1e-9:  # flat day
            if flat_count < cfg['allow_flat_days']:
                flat_count += 1
                i -= 1
            else:
                break
        else:
            break
    
    # Must have minimum recent up days
    if d1 < cfg['min_recent_up']:
        return result
    
    # Step 2: Count prior consecutive down days (d2) before d1
    d2 = 0
    flat_count_d2 = 0  # Reset flat count for d2
    start_d2 = i
    
    while i > 0 and (d2 + flat_count_d2) < cfg['lookback_days']:
        diff = prices[i] - prices[i-1]
        if diff < -1e-9:  # down day (use small threshold for float comparison)
            d2 += 1
            i -= 1
        elif abs(diff) < 1e-9:  # flat day
            if flat_count_d2 < cfg['allow_flat_days']:
                flat_count_d2 += 1
                i -= 1
            else:
                break
        else:
            break
    
    # Check pattern criteria
    if d2 < cfg['min_prior_down']:
        return result
    
    if d2 <= d1:
        return result
    
    # Pattern detected!
    result['detected'] = True
    result['d1_recent_up'] = d1
    result['d2_prior_down'] = d2
    
    # Step 3: Calculate pattern quality score (0-100)
    # Component 1: d1 close to ideal (n1)
    d1_distance = abs(d1 - cfg['n1_ideal'])
    d1_component = max(0, 100 - (d1_distance * 10))
    
    # Component 2: d2 higher is better (up to cap)
    d2_component = min(100, (d2 / cfg['d2_max']) * 100)
    
    # Weighted combination
    pattern_score = (d1_component * cfg['d1_weight']) + (d2_component * cfg['d2_weight'])
    result['pattern_score'] = round(pattern_score, 2)
    
    return result

def compute_pattern_ranking_with_history(symbols: list, source: str, config: Dict = None) -> pd.DataFrame:
    """
    Compute pattern ranking by fetching full price history for each symbol.
    This is the production-ready version.
    
    Args:
        symbols: List of stock symbols
        source: Data source ('FINNHUB' or 'EOD')
        config: Pattern configuration
        
    Returns:
        DataFrame with pattern results per symbol
    """
    LOG.info(f"Computing Ranking 1: Pattern Detection for {len(symbols)} symbols...")
    
    cfg = DEFAULT_PATTERN_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Determine price table based on source
    price_table = 'finnhub_stock_prices' if source.upper().startswith('FINNHUB') else 'eod_stock_prices'
    
    results = []
    db_conn = getDbConnection()
    
    for idx, symbol in enumerate(symbols):
        if (idx + 1) % 100 == 0:
            LOG.info(f"  Progress: {idx + 1}/{len(symbols)} symbols processed")
        
        try:
            # Fetch recent price history
            query = text(f"""
                SELECT Date, Close 
                FROM {price_table}
                WHERE Symbol = :symbol 
                ORDER BY Date DESC 
                LIMIT {cfg['lookback_days'] + 10}
            """)
            
            price_df = pd.read_sql(query, db_conn, params={'symbol': symbol})
            
            if price_df.empty:
                LOG.debug(f"No price history for {symbol}")
                results.append({
                    'Symbol': symbol,
                    'PricePattern_RecentUpDays': 0,
                    'PricePattern_PriorDownDays': 0,
                    'PricePattern_Detected': False,
                    'PricePattern_Score': 0.0
                })
                continue
            
            # Sort oldest to newest for pattern detection
            price_df = price_df.sort_values('Date')
            close_series = price_df['Close']
            
            # Detect pattern
            pattern = detect_price_pattern(close_series, cfg)
            
            results.append({
                'Symbol': symbol,
                'PricePattern_RecentUpDays': pattern['d1_recent_up'],
                'PricePattern_PriorDownDays': pattern['d2_prior_down'],
                'PricePattern_Detected': pattern['detected'],
                'PricePattern_Score': pattern['pattern_score']
            })
            
        except Exception as e:
            LOG.warning(f"Error processing pattern for {symbol}: {e}")
            results.append({
                'Symbol': symbol,
                'PricePattern_RecentUpDays': 0,
                'PricePattern_PriorDownDays': 0,
                'PricePattern_Detected': False,
                'PricePattern_Score': 0.0
            })
    
    pattern_df = pd.DataFrame(results)
    
    # Rank only detected patterns
    detected = pattern_df[pattern_df['PricePattern_Detected'] == True].copy()
    if not detected.empty:
        detected = detected.sort_values('PricePattern_Score', ascending=False)
        detected['PricePattern_Rank'] = range(1, len(detected) + 1)
        
        # Merge ranks back
        pattern_df = pattern_df.merge(
            detected[['Symbol', 'PricePattern_Rank']], 
            on='Symbol', 
            how='left'
        )
    else:
        pattern_df['PricePattern_Rank'] = np.nan
    
    LOG.info(f"✓ Pattern detected in {len(detected)} stocks")
    
    return pattern_df

# --------------------------
# RANKING 2: Technical Analysis Scoring
# --------------------------
def score_sma_distance(ema20: float, ema50: float, config: Dict = None) -> float:
    """
    Score based on distance between EMA20 and EMA50.
    Lower distance = better (0-25 points).
    """
    cfg = DEFAULT_TECHNICAL_CONFIG.copy()
    if config:
        cfg.update(config)
    
    if pd.isna(ema20) or pd.isna(ema50) or ema50 == 0:
        return 0.0
    
    distance_pct = abs(ema20 - ema50) / ema50 * 100
    thresholds = cfg['sma_thresholds']
    
    if distance_pct <= thresholds[0]:    # <= 2%
        return 25.0
    elif distance_pct <= thresholds[1]:  # <= 5%
        return 20.0
    elif distance_pct <= thresholds[2]:  # <= 10%
        return 15.0
    elif distance_pct <= thresholds[3]:  # <= 15%
        return 10.0
    else:
        return 5.0

def score_rsi_oversold(rsi: float, config: Dict = None) -> float:
    """
    Score based on RSI oversold positioning.
    Deeper oversold = better (0-25 points).
    """
    cfg = DEFAULT_TECHNICAL_CONFIG.copy()
    if config:
        cfg.update(config)
    
    if pd.isna(rsi):
        return 0.0
    
    thresholds = cfg['rsi_thresholds']
    
    if rsi <= thresholds[0]:    # <= 30
        return 25.0
    elif rsi <= thresholds[1]:  # <= 35
        return 20.0
    elif rsi <= thresholds[2]:  # <= 40
        return 15.0
    elif rsi <= thresholds[3]:  # <= 45
        return 10.0
    elif rsi <= thresholds[4]:  # <= 50
        return 5.0
    else:
        return 0.0

def score_pct2h52(pct2h52: float, config: Dict = None) -> float:
    """
    Score based on distance from 52-week high.
    Higher distance = better value (0-25 points).
    """
    cfg = DEFAULT_TECHNICAL_CONFIG.copy()
    if config:
        cfg.update(config)
    
    if pd.isna(pct2h52):
        return 0.0
    
    thresholds = cfg['pct2h52_thresholds']
    
    if pct2h52 >= thresholds[4]:    # >= 40%
        return 25.0
    elif pct2h52 >= thresholds[3]:  # >= 30%
        return 20.0
    elif pct2h52 >= thresholds[2]:  # >= 20%
        return 15.0
    elif pct2h52 >= thresholds[1]:  # >= 10%
        return 10.0
    elif pct2h52 >= thresholds[0]:  # >= 5%
        return 5.0
    else:
        return 0.0

def score_gem_rank(gem_rank: float, config: Dict = None) -> float:
    """
    Score based on GEM fundamental rank.
    Lower rank = better quality (0-25 points).
    """
    cfg = DEFAULT_TECHNICAL_CONFIG.copy()
    if config:
        cfg.update(config)
    
    if pd.isna(gem_rank):
        return 0.0
    
    thresholds = cfg['gem_thresholds']
    
    if gem_rank <= thresholds[0]:    # <= 500
        return 25.0
    elif gem_rank <= thresholds[1]:  # <= 1000
        return 20.0
    elif gem_rank <= thresholds[2]:  # <= 1500
        return 15.0
    elif gem_rank <= thresholds[3]:  # <= 2000
        return 10.0
    elif gem_rank <= thresholds[4]:  # <= 3000
        return 5.0
    else:
        return 0.0

def compute_technical_score(row: pd.Series, config: Dict = None) -> Dict:
    """
    Compute technical analysis score components for a single stock.
    
    Returns:
        Dict with score components and total
    """
    ema20 = safe_float(row.get('EMA20'))
    ema50 = safe_float(row.get('EMA50'))
    rsi = safe_float(row.get('RSI'))
    pct2h52 = safe_float(row.get('Pct2H52'))
    gem_rank = safe_float(row.get('GEM_Rank'))
    
    sma_score = score_sma_distance(ema20, ema50, config)
    rsi_score = score_rsi_oversold(rsi, config)
    pct_score = score_pct2h52(pct2h52, config)
    gem_score = score_gem_rank(gem_rank, config)
    
    total = sma_score + rsi_score + pct_score + gem_score
    
    return {
        'TechScore_SMA_Dist': round(sma_score, 2),
        'TechScore_RSI': round(rsi_score, 2),
        'TechScore_Pct2H52': round(pct_score, 2),
        'TechScore_GEM': round(gem_score, 2),
        'TechScore_Total': round(total, 2)
    }

def compute_technical_ranking(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Compute technical analysis ranking for all stocks.
    
    Args:
        df: DataFrame with technical indicator columns
        config: Technical scoring configuration
        
    Returns:
        DataFrame with technical score columns added
    """
    LOG.info("Computing Ranking 2: Technical Analysis Scoring...")
    
    # Compute scores for each row
    tech_scores = df.apply(lambda row: compute_technical_score(row, config), axis=1)
    tech_df = pd.DataFrame(tech_scores.tolist())
    
    # Add scores to dataframe
    for col in tech_df.columns:
        df[col] = tech_df[col]
    
    # Rank by total technical score
    df_sorted = df.sort_values('TechScore_Total', ascending=False).copy()
    df_sorted['TechScore_Rank'] = range(1, len(df_sorted) + 1)
    
    # Merge ranks back to original order
    df = df.merge(df_sorted[['Symbol', 'TechScore_Rank']], on='Symbol', how='left', suffixes=('', '_new'))
    if 'TechScore_Rank_new' in df.columns:
        df['TechScore_Rank'] = df['TechScore_Rank_new']
        df = df.drop('TechScore_Rank_new', axis=1)
    
    LOG.info(f"✓ Technical scoring complete for {len(df)} stocks")
    
    return df

# --------------------------
# RANKING 3: Composite Ranking
# --------------------------
def compute_composite_score(pattern_score: float, tech_score: float, config: Dict = None) -> float:
    """
    Compute composite score from pattern and technical scores.
    
    Args:
        pattern_score: Score from Ranking 1 (0-100)
        tech_score: Score from Ranking 2 (0-100)
        config: Composite configuration
        
    Returns:
        Composite score (0-100)
    """
    cfg = DEFAULT_COMPOSITE_CONFIG.copy()
    if config:
        cfg.update(config)
    
    composite = (
        (pattern_score * cfg['pattern_weight']) + 
        (tech_score * cfg['technical_weight'])
    )
    
    return round(composite, 2)

def generate_signal(composite_score: float, pattern_detected: bool, config: Dict = None) -> str:
    """
    Generate swing trade signal based on composite score.
    
    Args:
        composite_score: Composite score (0-100)
        pattern_detected: Whether price pattern was detected
        config: Composite configuration
        
    Returns:
        Signal string: "Strong_Buy", "Buy", "Weak_Buy", or "Neutral"
    """
    cfg = DEFAULT_COMPOSITE_CONFIG.copy()
    if config:
        cfg.update(config)
    
    if not pattern_detected:
        return "Neutral"
    
    thresholds = cfg['signal_thresholds']
    
    if composite_score >= thresholds['strong_buy']:
        return "Strong_Buy"
    elif composite_score >= thresholds['buy']:
        return "Buy"
    elif composite_score >= thresholds['weak_buy']:
        return "Weak_Buy"
    else:
        return "Neutral"

def compute_composite_ranking(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Compute composite ranking combining pattern and technical scores.
    
    Args:
        df: DataFrame with both pattern and technical scores
        config: Composite configuration
        
    Returns:
        DataFrame with composite score, rank, and signal columns added
    """
    LOG.info("Computing Ranking 3: Composite Ranking...")
    
    # Compute composite score for each row
    df['CompositeScore'] = df.apply(
        lambda row: compute_composite_score(
            safe_float(row.get('PricePattern_Score')),
            safe_float(row.get('TechScore_Total')),
            config
        ),
        axis=1
    )
    
    # Generate signals
    df['SwingTrade_Signal'] = df.apply(
        lambda row: generate_signal(
            safe_float(row.get('CompositeScore')),
            row.get('PricePattern_Detected', False),
            config
        ),
        axis=1
    )
    
    # Rank only stocks with detected patterns
    detected = df[df['PricePattern_Detected'] == True].copy()
    
    if not detected.empty:
        detected = detected.sort_values('CompositeScore', ascending=False)
        detected['CompositeRank'] = range(1, len(detected) + 1)
        
        # Merge ranks back
        df = df.merge(
            detected[['Symbol', 'CompositeRank']], 
            on='Symbol', 
            how='left'
        )
    else:
        df['CompositeRank'] = np.nan
        LOG.warning("No patterns detected - no composite ranking generated")
    
    LOG.info(f"✓ Composite ranking complete. Ranked {len(detected)} stocks with patterns")
    
    return df

# --------------------------
# Data Loading & Export
# --------------------------
def load_tas_listings(watchlist: str, source: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Load latest tas_listings data from database.
    
    Args:
        watchlist: Watchlist name (e.g., 'US_CORE')
        source: Data source (e.g., 'FINNHUB')
        
    Returns:
        Tuple of (DataFrame, country_name, master_table)
    """
    LOG.info(f"Loading tas_listings for watchlist={watchlist}, source={source}")
    
    # Initialize config - takes only source parameter
    config = initialize_config(source)
    country_name = get_country_name(watchlist)
    
    master_table = config.get('tal_master_tablename', 'finnhub_tas_listings')
    
    # Load from database
    db_conn = getDbConnection()
    
    query = text(f"""
        SELECT * FROM {master_table}
        WHERE CountryName = :country
        ORDER BY Symbol
    """)
    
    df = pd.read_sql(query, db_conn, params={'country': country_name})
    
    LOG.info(f"✓ Loaded {len(df)} stocks from {master_table}")
    
    # Apply price filtering to remove penny stocks and extremely expensive stocks
    if country_name in PRICE_FILTERS:
        price_range = PRICE_FILTERS[country_name]
        price_col = 'TodayPrice' if 'TodayPrice' in df.columns else 'Close'
        
        if price_col in df.columns:
            original_count = len(df)
            # Filter stocks within price range (handle NaN/None values)
            df = df[
                (df[price_col].notna()) & 
                (df[price_col] >= price_range['min']) & 
                (df[price_col] <= price_range['max'])
            ].copy()
            filtered_count = original_count - len(df)
            LOG.info(f"✓ Price filter ({price_range['min']}-{price_range['max']}): "
                     f"removed {filtered_count} stocks, {len(df)} remaining")
        else:
            LOG.warning(f"Price column not found, skipping price filter")
    else:
        LOG.warning(f"No price filter configured for country: {country_name}")
    
    return df, country_name, master_table

def export_rankings_db(df: pd.DataFrame, country: str, top_n: int = 25) -> None:
    """
    Write only stocks with detected patterns to tas_swing_listings.
    Simplified: filter detected, create table if needed, purge country, insert, verify count.
    """
    db_conn = getDbConnection()
    db_table = 'tas_swing_listings'
    
    LOG.info("=" * 70)
    LOG.info(f"DB EXPORT: {db_table}")
    LOG.info("=" * 70)
    LOG.info(f"Total stocks analyzed: {len(df)}")
    
    # Ensure table exists
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {db_table} (
        Symbol VARCHAR(20),
        CountryName VARCHAR(50),
        GEM_Rank VARCHAR(32),
        TodayPrice FLOAT,
        IndustrySector VARCHAR(64),
        RSI FLOAT,
        EMA20 FLOAT,
        EMA50 FLOAT,
        ADX FLOAT,
        High52 FLOAT,
        Low52 FLOAT,
        Pct2H52 FLOAT,
        PctfL52 FLOAT,
        LastTrendDays INT,
        LastTrendType VARCHAR(32),
        Trend VARCHAR(64),
        MA_Trend VARCHAR(16),
        MADI_Trend VARCHAR(16),
        RSIUturnTypeOld VARCHAR(64),
        TrendReversal_Rules VARCHAR(64),
        TrendReversal_ML VARCHAR(64),
        RSIUpTrend TINYINT,
        PricePattern_RecentUpDays INT,
        PricePattern_PriorDownDays INT,
        PricePattern_Detected TINYINT,
        PricePattern_Score FLOAT,
        PricePattern_Rank INT,
        TechScore_SMA_Dist FLOAT,
        TechScore_RSI FLOAT,
        TechScore_Pct2H52 FLOAT,
        TechScore_GEM FLOAT,
        TechScore_Total FLOAT,
        TechScore_Rank INT,
        CompositeScore FLOAT,
        CompositeRank INT,
        SwingTrade_Signal VARCHAR(20),
        ScanDate DATETIME,
        INDEX idx_country (CountryName),
        INDEX idx_symbol (Symbol),
        INDEX idx_rank (CompositeRank)
    )"""
    try:
        db_conn.execute(text(create_sql))
        db_conn.commit()
        LOG.info(f"✓ Ensured {db_table} exists")
    except Exception as e:
        LOG.warning(f"Table creation warning (may already exist): {e}")
    
    # Filter to only rows where pattern was detected
    detected = df[df['PricePattern_Detected'] == True].copy()
    LOG.info(f"Patterns detected: {len(detected)}")
    
    if detected.empty:
        LOG.warning("No patterns detected - nothing to write to DB")
        print("[tas_swing_scoring] No patterns detected - no DB writes")
        return
    
    # Add timestamp
    detected['ScanDate'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert boolean to int
    detected['PricePattern_Detected'] = detected['PricePattern_Detected'].astype(int)
    
    # Select only the columns we need (including important tas_listings fields)
    export_cols = [
        'Symbol', 'CountryName', 'GEM_Rank', 'TodayPrice', 'IndustrySector',
        'RSI', 'EMA20', 'EMA50', 'ADX',
        'High52', 'Low52', 'Pct2H52', 'PctfL52',
        'LastTrendDays', 'LastTrendType', 'Trend', 'MA_Trend', 'MADI_Trend',
        'RSIUturnTypeOld', 'TrendReversal_Rules', 'TrendReversal_ML', 'RSIUpTrend',
        'PricePattern_RecentUpDays', 'PricePattern_PriorDownDays',
        'PricePattern_Detected', 'PricePattern_Score', 'PricePattern_Rank',
        'TechScore_SMA_Dist', 'TechScore_RSI', 'TechScore_Pct2H52', 'TechScore_GEM',
        'TechScore_Total', 'TechScore_Rank',
        'CompositeScore', 'CompositeRank', 'SwingTrade_Signal',
        'ScanDate'
    ]
    
    df_export = detected[[c for c in export_cols if c in detected.columns]].copy()
    LOG.info(f"Exporting {len(df_export)} rows with {len(df_export.columns)} columns")
    
    try:
        # Purge existing rows for this country
        LOG.info(f"Purging existing {country} rows from {db_table}...")
        purge_sql = text(f"DELETE FROM {db_table} WHERE CountryName = :country")
        result = db_conn.execute(purge_sql, {'country': country})
        db_conn.commit()
        LOG.info(f"Purged rows for {country}")
        
        # Insert new rows
        LOG.info(f"Inserting {len(df_export)} rows...")
        df_export.to_sql(db_table, db_conn, if_exists='append', index=False, chunksize=500)
        db_conn.commit()
        LOG.info(f"✓ Inserted {len(df_export)} rows into {db_table}")
        
        # Verify
        verify_sql = text(f"SELECT COUNT(*) FROM {db_table} WHERE CountryName = :country")
        result = db_conn.execute(verify_sql, {'country': country})
        count = result.fetchone()[0]
        LOG.info(f"✓ Verified: {count} rows in {db_table} for {country}")
        print(f"\n[SUCCESS] {count} rows written to {db_table} for {country}\n")
        
    except Exception as e:
        LOG.error(f"DB write failed: {e}")
        print(f"[ERROR] DB write failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return
    
    LOG.info("")
    LOG.info("RANKING SUMMARY")
    LOG.info("-" * 70)
    LOG.info(f"Strong Buy: {len(detected[detected['SwingTrade_Signal'] == 'Strong_Buy'])}")
    LOG.info(f"Buy: {len(detected[detected['SwingTrade_Signal'] == 'Buy'])}")
    LOG.info(f"Weak Buy: {len(detected[detected['SwingTrade_Signal'] == 'Weak_Buy'])}")
    LOG.info(f"Avg Composite Score: {detected['CompositeScore'].mean():.2f}")
    LOG.info(f"Avg Technical Score: {detected['TechScore_Total'].mean():.2f}")
    LOG.info(f"Avg Pattern Score: {detected['PricePattern_Score'].mean():.2f}")
    
    LOG.info("")
    LOG.info("TOP 10 OPPORTUNITIES:")
    LOG.info("-" * 70)
    top10 = detected.sort_values('CompositeRank').head(10)
    for _, row in top10.iterrows():
        LOG.info(f"  #{int(row['CompositeRank'])}: {row['Symbol']} - "
                 f"Score={row['CompositeScore']:.1f}, "
                 f"Signal={row['SwingTrade_Signal']}, "
                 f"d1={int(row['PricePattern_RecentUpDays'])}/d2={int(row['PricePattern_PriorDownDays'])}")
    LOG.info("=" * 70)

# --------------------------
# Main Execution
# --------------------------
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Three-tier ranking system for swing trading stock selection"
    )
    
    # Required arguments
    parser.add_argument('--watchlist', required=True, 
                        help='Watchlist name (e.g., US_CORE)')
    parser.add_argument('--source', required=True, 
                        help='Data source (e.g., FINNHUB)')
    
    # Pattern configuration
    parser.add_argument('--n1_ideal', type=int, default=3,
                        help='Ideal recent uptrend length (default: 3)')
    parser.add_argument('--min_recent_up', type=int, default=2,
                        help='Minimum recent up days (default: 2)')
    parser.add_argument('--min_prior_down', type=int, default=5,
                        help='Minimum prior down days (default: 5)')
    parser.add_argument('--lookback_days', type=int, default=60,
                        help='Lookback window for pattern detection (default: 60)')
    
    # Composite configuration
    parser.add_argument('--pattern_weight', type=float, default=0.45,
                        help='Weight for pattern score in composite (default: 0.45)')
    parser.add_argument('--technical_weight', type=float, default=0.55,
                        help='Weight for technical score in composite (default: 0.55)')
    
    # Output / persistence configuration
    parser.add_argument('--top_n', type=int, default=25,
                        help='Number of top stocks for summary logs (default: 25)')
    
    args = parser.parse_args()
    
    try:
        LOG.info("=" * 70)
        LOG.info("SWING TRADING RANKING SYSTEM - Three-Tier Analysis")
        LOG.info("=" * 70)
        
        # Build configuration dicts
        pattern_config = {
            'n1_ideal': args.n1_ideal,
            'min_recent_up': args.min_recent_up,
            'min_prior_down': args.min_prior_down,
            'lookback_days': args.lookback_days
        }
        
        composite_config = {
            'pattern_weight': args.pattern_weight,
            'technical_weight': args.technical_weight
        }
        
        # Step 1: Load tas_listings data
        df, country, master = load_tas_listings(args.watchlist, args.source)
        
        if df.empty:
            LOG.error("No data loaded. Exiting.")
            return 1
        
        # Step 2: Compute Ranking 1 (Price Pattern)
        symbols = df['Symbol'].unique().tolist()
        pattern_df = compute_pattern_ranking_with_history(symbols, args.source, pattern_config)
        df = df.merge(pattern_df, on='Symbol', how='left')
        
        # Step 3: Compute Ranking 2 (Technical Analysis)
        df = compute_technical_ranking(df, None)
        
        # Step 4: Compute Ranking 3 (Composite)
        df = compute_composite_ranking(df, composite_config)
        
        # Step 5: Persist results to DB (always to tas_swing_listings)
        export_rankings_db(df, country, args.top_n)
        
        LOG.info("")
        LOG.info("=" * 70)
        LOG.info("✓ SWING RANKING PIPELINE COMPLETED SUCCESSFULLY")
        LOG.info("=" * 70)
        return 0
        
    except Exception as e:
        LOG.error(f"Pipeline failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

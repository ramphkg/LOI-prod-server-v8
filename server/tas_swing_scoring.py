#!/usr/bin/env python3
# tas_swing_scoring.py
#
# Purpose:
#   - Load latest tas_listings snapshot (by CountryName) from DB.
#   - Compute SwingScore per row using classifier signals, trend, TMA cross, RSIUpTrend, ADX, reversals, and 52W proximity.
#   - Optionally write SwingScore back to tas_listings (UPDATE JOIN).
#   - Export full snapshot with SwingScore and top-N long/short shortlists as CSV files to ../data.
#
# Usage:
#   python tas_swing_scoring.py --watchlist US_CORE --source FINNHUB --top_n 25 --use_ml_preferred yes --update_db no --out_dir ../data
#
# Assumptions:
#   - DB connection helper app_imports.getDbConnection is available.
#   - Table and config naming are aligned with ta_signals_mc_parallel.initialize_config and get_country_name.
#   - tas_listings has the canonical schema produced by ta_signals_mc_parallel.py, including all fields referenced here.
#
# Notes:
#   - Classifier scale: integers -4..+4; 1–2 = emerging reversals/retracements, 3–4 = stronger trend-aligned; sign gives direction (+ bull, - bear).
#   - TrendReversal_Rules and TrendReversal_ML use the same label set; literal "0" (or 0) is treated as null/ignore.
#   - Trend format is "Primary[Secondary]".
#   - Outputs:
#       ../data/swingscores_<master>_<country>_<YYYYMMDD>.csv
#       ../data/shortlist_longs_<master>_<country>_<YYYYMMDD>.csv
#       ../data/shortlist_shorts_<master>_<country>_<YYYYMMDD>.csv

import argparse
import logging
import math
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

# Project imports (expected in your environment)
from app_imports import getDbConnection  # DB engine/connection
from ta_signals_mc_parallel import initialize_config, get_country_name  # config and CountryName mapping [[11]]

# --------------------------
# Configurable constants
# --------------------------
REVERSAL_MAP = {
    "NoReversal": 0,
    "BullishReversalWeak": +1,
    "BullishReversalModerate": +2,
    "BullishReversalStrong": +3,
    "BearishReversalWeak": -1,
    "BearishReversalModerate": -2,
    "BearishReversalStrong": -3,
    # TrendReversal_ML uses different naming convention
    "BullishReversal-MLWeak": +1,
    "BullishReversal-MLModerate": +2,
    "BullishReversal-MLStrong": +3,
    "BearishReversal-MLWeak": -1,
    "BearishReversal-MLModerate": -2,
    "BearishReversal-MLStrong": -3
}

DEFAULT_WEIGHTS = {
    "base_classifier_ml": 0.35,
    "base_classifier_rules": 0.25,
    "trend_primary": 0.15,
    "trend_secondary": 0.10,
    "tma_cross": 0.12,
    "rsi_up": 0.10,
    "adx": 0.05,
    "reversal_rules": 0.08,
    "reversal_ml": 0.08,
    "rsi_uturn_old": 0.06,  # legacy RSIUturnTypeOld directional confirmation
    "late_trend_penalty": -0.05,
    "near_52w_high_penalty": -0.05,
    "near_52w_low_bonus": 0.05
}

# --------------------------
# Helpers
# --------------------------
def utc_datestr() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rr_label_to_int(label: object) -> int:
    """Map reversal label to signed integer; treat '0'/'0.0'/0 as null."""
    if label is None:
        return 0
    s = str(label).strip()
    if s == "" or s.lower() in {"nan", "none"} or s == "0" or s == "0.0":
        return 0
    return REVERSAL_MAP.get(s, 0)

def parse_trend(trend: object) -> tuple[str, str]:
    """Return (Primary, Secondary) from 'Primary[Secondary]'; defaults to ('Neutral','Unknown')."""
    if trend is None:
        return ("Neutral", "Unknown")
    t = str(trend)
    if "[" in t and t.endswith("]"):
        primary = t.split("[", 1)[0]
        secondary = t.split("[", 1)[1][:-1]
        return (primary or "Neutral", secondary or "Unknown")
    return (t if t else "Neutral", "Unknown")

def parse_rsi_uturn_old(label: object) -> float:
    """Map legacy RSIUturnTypeOld string to a normalized directional value in [-1,+1].
    Accepted patterns observed:
      - ERR_NO_TREND, ERR_INSUFF_DATA -> 0
      - BullishReversal[Weak|Moderate|Strong]
      - BearishReversal[Weak|Moderate|Strong]
    Strength scaling: Weak=0.4, Moderate=0.7, Strong=1.0
    """
    if label is None:
        return 0.0
    s = str(label).strip()
    if not s or s.startswith("ERR"):
        return 0.0
    if "BullishReversal" in s:
        strength = 1.0 if "Strong" in s else 0.7 if "Moderate" in s else 0.4 if "Weak" in s else 0.0
        return strength
    if "BearishReversal" in s:
        strength = 1.0 if "Strong" in s else 0.7 if "Moderate" in s else 0.4 if "Weak" in s else 0.0
        return -strength
    # Unknown pattern
    return 0.0

def swing_score(row: pd.Series, weights: dict | None = None) -> float:
    """
    Compute SwingScore for a tas_listings row.
    Inputs (by column):
      - SignalClassifier_ML, SignalClassifier_Rules (±1..±4, 0=neutral)
      - Trend ("Primary[Secondary]")
      - TMA21_50_X (-1/0/+1)
      - RSIUpTrend (bool/int)
      - ADX (double)
      - TrendReversal_Rules, TrendReversal_ML (labels or 0 to ignore)
      - LastTrendDays (int)
      - Pct2H52, PctfL52 (double)
    """
    W = DEFAULT_WEIGHTS.copy()
    if weights:
        W.update(weights)

    score = 0.0

    # 1) Base classifier: prefer ML if present and non-zero, else Rules.
    ml_sig = row.get("SignalClassifier_ML")
    rules_sig = row.get("SignalClassifier_Rules")
    base_used = None
    if pd.notna(ml_sig) and float(ml_sig) != 0:
        base = max(min(float(ml_sig), 4.0), -4.0) / 4.0
        score += W["base_classifier_ml"] * base
        base_used = "ML"
    elif pd.notna(rules_sig) and float(rules_sig) != 0:
        base = max(min(float(rules_sig), 4.0), -4.0) / 4.0
        score += W["base_classifier_rules"] * base
        base_used = "Rules"

    # 2) Trend (Primary[Secondary])
    primary, secondary = parse_trend(row.get("Trend"))
    if primary == "Bull":
        score += W["trend_primary"] * 1.0
    elif primary == "Bear":
        score += W["trend_primary"] * -1.0

    sec_boost = 0.0
    if secondary == "PullbackInBull":
        sec_boost = 1.0
    elif secondary == "TrendingUp":
        sec_boost = 0.8
    elif secondary == "PullbackInBear":
        sec_boost = -1.0
    elif secondary == "TrendingDown":
        sec_boost = -0.8
    elif secondary == "Volatile":
        sec_boost = 0.0  # Neutral - no clear directional bias
    elif secondary == "ShortTrend":
        sec_boost = 0.3 if primary == "Bull" else -0.3 if primary == "Bear" else 0.0
    elif secondary == "Ranging":
        sec_boost = 0.0  # Neutral - sideways movement
    score += W["trend_secondary"] * sec_boost

    # 3) TMA cross
    try:
        tma = float(row.get("TMA21_50_X") or 0.0)
    except Exception:
        tma = 0.0
    if not math.isnan(tma):
        score += W["tma_cross"] * (1.0 if tma > 0 else -1.0 if tma < 0 else 0.0)

    # 4) RSIUpTrend (boolean-like)
    rsi_up = row.get("RSIUpTrend")
    try:
        rsi_up = bool(int(rsi_up)) if isinstance(rsi_up, (int, np.integer, np.int64, np.int32)) else bool(rsi_up)
    except Exception:
        rsi_up = False
    if rsi_up:
        score += W["rsi_up"]

    # 5) ADX confirmation (toward Primary): Strong ≥25, Average ≥20
    try:
        adx = float(row.get("ADX") or 0.0)
    except Exception:
        adx = 0.0
    dir_primary = 1.0 if primary == "Bull" else -1.0 if primary == "Bear" else 0.0
    if adx >= 25:
        score += W["adx"] * dir_primary
    elif adx >= 20:
        score += W["adx"] * 0.5 * dir_primary

    # 6) Reversals: treat "0" as null (ignore)
    rr_rules = rr_label_to_int(row.get("TrendReversal_Rules"))
    if rr_rules != 0:
        score += W["reversal_rules"] * (rr_rules / 3.0)
    rr_ml = rr_label_to_int(row.get("TrendReversal_ML"))
    if rr_ml != 0:
        score += W["reversal_ml"] * (rr_ml / 3.0)

    # 6b) Legacy RSIUturnTypeOld directional confirmation (optional)
    try:
        uturn_val = parse_rsi_uturn_old(row.get("RSIUturnTypeOld"))
    except Exception:
        uturn_val = 0.0
    if uturn_val != 0.0:
        score += W["rsi_uturn_old"] * uturn_val

    # 7) Late leg penalty
    try:
        ltd = float(row.get("LastTrendDays") or 0.0)
    except Exception:
        ltd = 0.0
    if abs(ltd) > 8:
        score += W["late_trend_penalty"] * min(1.0, (abs(ltd) - 8) / 10.0)

    # 8) 52W proximity adjustments
    try:
        p2h = float(row.get("Pct2H52") or 0.0)
    except Exception:
        p2h = 0.0
    try:
        pfl = float(row.get("PctfL52") or 0.0)
    except Exception:
        pfl = 0.0
    if p2h <= 2.0 and base_used is not None:
        score += W["near_52w_high_penalty"]
    if pfl <= 5.0 and (primary == "Bull" or rsi_up):
        score += W["near_52w_low_bonus"]

    # Scale to 0-100 range for readability (preserves ordering)
    return int(round(score * 100))

# --------------------------
# DB helpers (optional update)
# --------------------------
def ensure_score_column(table_name: str, logger) -> None:
    """Adds SwingScore column if missing."""
    with getDbConnection() as con:
        try:
            q = text("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
                 WHERE TABLE_SCHEMA = DATABASE()
                   AND TABLE_NAME = :t
                   AND COLUMN_NAME = 'SwingScore'
            """)
            r = con.execute(q, {"t": table_name}).fetchone()
            exists = bool(r and int(r[0]) > 0)
        except Exception as e:
            logger.warning(f"[ensure_score_column] INFO_SCHEMA check failed: {e}; attempting ALTER")
            exists = False

        if not exists:
            try:
                con.execute(text(f"ALTER TABLE `{table_name}` ADD COLUMN `SwingScore` DOUBLE"))
                logger.info(f"[ensure_score_column] Added SwingScore to {table_name}")
            except Exception as e:
                logger.error(f"[ensure_score_column] Failed to add SwingScore: {e}")
                raise

def update_master_scores(master_table: str, temp_table: str, country: str, logger) -> int:
    """Join-update SwingScore from temp to master for given CountryName."""
    with getDbConnection() as con:
        sql = text(f"""
            UPDATE `{master_table}` m
            JOIN `{temp_table}` t
              ON m.Symbol = t.Symbol
             AND m.CountryName = t.CountryName
             AND m.Date = t.Date
            SET m.SwingScore = t.SwingScore
            WHERE m.CountryName = :country
        """)
        res = con.execute(sql, {"country": country})
        try:
            if hasattr(con, 'commit') and callable(con.commit):
                con.commit()
        except Exception:
            pass
        rowcount = getattr(res, "rowcount", 0)
        logger.info(f"[update_master_scores] Updated {rowcount} rows in {master_table}")
        return int(rowcount)

def upsert_temp_scores_table(temp_table: str, df_scores: pd.DataFrame, logger) -> None:
    """Create/empty temp table and insert scores for update-join."""
    with getDbConnection() as con:
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{temp_table}` (
                `Symbol` VARCHAR(191),
                `CountryName` VARCHAR(191),
                `Date` DATE,
                `SwingScore` DOUBLE
            ) ENGINE=InnoDB
        """))
        con.execute(text(f"TRUNCATE TABLE `{temp_table}`"))
        try:
            df_scores[['Symbol', 'CountryName', 'Date', 'SwingScore']].to_sql(
                temp_table, con=con, index=False, if_exists='append', method='multi', chunksize=1000
            )
        except TypeError:
            df_scores[['Symbol', 'CountryName', 'Date', 'SwingScore']].to_sql(
                temp_table, con=con, index=False, if_exists='append', chunksize=1000
            )
        except Exception as e:
            logger.error(f"[upsert_temp_scores_table] to_sql failed: {e}")
            raise
        try:
            if hasattr(con, 'commit') and callable(con.commit):
                con.commit()
        except Exception:
            pass
        logger.info(f"[upsert_temp_scores_table] Inserted {len(df_scores)} rows into {temp_table}")

# --------------------------
# DB table management for tas_swing_listings
# --------------------------
def ensure_swing_table(table_name: str, logger) -> None:
    """Ensure tas_swing_listings table exists with necessary schema."""
    with getDbConnection() as con:
        # Check if table exists
        try:
            q = text("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t
            """)
            r = con.execute(q, {"t": table_name}).fetchone()
            exists = bool(r and int(r[0]) > 0)
        except Exception as e:
            logger.warning(f"[ensure_swing_table] INFO_SCHEMA check failed: {e}; attempting CREATE")
            exists = False

        if not exists:
            create_sql = text(f"""
                CREATE TABLE `{table_name}` (
                    `Date` DATE,
                    `Symbol` VARCHAR(191),
                    `CountryName` VARCHAR(191),
                    `SwingScore` DOUBLE,
                    `Direction` VARCHAR(16),
                    `Trend` VARCHAR(64),
                    `SignalClassifier_ML` INTEGER,
                    `SignalClassifier_Rules` INTEGER,
                    `TMA21_50_X` SMALLINT,
                    `RSIUpTrend` BOOLEAN,
                    `ADX` DOUBLE,
                    `Pct2H52` DOUBLE,
                    `PctfL52` DOUBLE,
                    `GEM_Rank` VARCHAR(32),
                    `marketCap` DOUBLE,
                    `IndustrySector` VARCHAR(64),
                    `ScanDate` TIMESTAMP,
                    INDEX idx_symbol (Symbol),
                    INDEX idx_country (CountryName),
                    INDEX idx_date (Date)
                ) ENGINE=InnoDB
            """)
            try:
                con.execute(create_sql)
                logger.info(f"[ensure_swing_table] Created table {table_name}")
            except Exception as e:
                logger.error(f"[ensure_swing_table] Failed to create {table_name}: {e}")
                raise

def purge_country_swing_records(table_name: str, country: str, logger) -> int:
    """Delete existing swing records for given country."""
    with getDbConnection() as con:
        sql = text(f"DELETE FROM `{table_name}` WHERE CountryName = :country")
        res = con.execute(sql, {"country": country})
        try:
            if hasattr(con, 'commit') and callable(con.commit):
                con.commit()
        except Exception:
            pass
        rowcount = getattr(res, "rowcount", 0)
        logger.info(f"[purge_country_swing_records] Deleted {rowcount} rows for {country}")
        return int(rowcount)

def insert_swing_scores(table_name: str, df_full: pd.DataFrame, country: str, logger) -> None:
    """Bulk insert swing scores with Direction labels into tas_swing_listings."""
    if df_full.empty:
        logger.warning(f"[insert_swing_scores] No data to insert for {country}")
        return

    # Prepare columns for insert
    cols = ['Date', 'Symbol', 'CountryName', 'SwingScore', 'Direction', 'Trend',
            'SignalClassifier_ML', 'SignalClassifier_Rules', 'TMA21_50_X', 'RSIUpTrend',
            'ADX', 'Pct2H52', 'PctfL52', 'GEM_Rank', 'marketCap', 'IndustrySector', 'ScanDate']

    df_insert = df_full.copy()
    # Add Direction column based on SwingScore
    df_insert['Direction'] = df_insert['SwingScore'].apply(
        lambda x: 'Long' if pd.notna(x) and x > 0 else 'Short' if pd.notna(x) and x < 0 else 'Neutral'
    )

    # Ensure all required columns exist
    for col in cols:
        if col not in df_insert.columns:
            df_insert[col] = None

    df_insert = df_insert[cols]

    # Convert Date to proper format
    if 'Date' in df_insert.columns:
        try:
            df_insert['Date'] = pd.to_datetime(df_insert['Date']).dt.date
        except Exception:
            pass

    with getDbConnection() as con:
        try:
            df_insert.to_sql(table_name, con=con, index=False, if_exists='append', method='multi', chunksize=500)
        except TypeError:
            df_insert.to_sql(table_name, con=con, index=False, if_exists='append', chunksize=500)
        except Exception as e:
            logger.error(f"[insert_swing_scores] Insert failed: {e}")
            raise
        try:
            if hasattr(con, 'commit') and callable(con.commit):
                con.commit()
        except Exception:
            pass
    logger.info(f"[insert_swing_scores] Inserted {len(df_insert)} rows into {table_name} for {country}")

# --------------------------
# Main job
# --------------------------
def run_job(watchlist: str, price_source: str, top_n: int, use_ml_preferred: bool,
            update_db: bool, out_dir: str):
    logger = logging.getLogger("tas_swing_scoring")
    logger.info(f"[run_job] watchlist={watchlist}, source={price_source}, top_n={top_n}, use_ml_preferred={use_ml_preferred}, update_db={update_db}, out_dir={out_dir}")

    # Resolve tables and CountryName via the TA pipeline config [[11]]
    cfg = initialize_config(price_source)  # [[11]]
    master_table = cfg["tal_master_tablename"]  # [[11]]
    country = get_country_name(watchlist)       # [[11]]
    swing_table = "tas_swing_listings"
    logger.info(f"[run_job] master={master_table}, country={country}, swing_table={swing_table}")

    # Load latest snapshot rows for CountryName
    with getDbConnection() as con:
        df = pd.read_sql(
            text(f"SELECT * FROM `{master_table}` WHERE CountryName = :country"),
            con=con, params={"country": country}
        )

    if df.empty:
        logger.warning(f"[run_job] No rows found in {master_table} for CountryName={country}")
        return

    # If user does not prefer ML, mask ML signals to force rules usage in scoring
    if not use_ml_preferred:
        if "SignalClassifier_ML" in df.columns:
            df["SignalClassifier_ML"] = np.nan

    # Compute SwingScore
    df["SwingScore"] = df.apply(swing_score, axis=1)

    # Optional DB update to master table
    if update_db:
        try:
            ensure_score_column(master_table, logger)
            tmp_table = f"{master_table}_swingscore_temp"
            upsert_temp_scores_table(tmp_table, df[["Symbol", "CountryName", "Date", "SwingScore"]].copy(), logger)
            update_master_scores(master_table, tmp_table, country, logger)
        except Exception as e:
            logger.error(f"[run_job] DB update failed: {e}")

    # Write all scored records to tas_swing_listings (purge country first)
    try:
        ensure_swing_table(swing_table, logger)
        purge_country_swing_records(swing_table, country, logger)
        insert_swing_scores(swing_table, df, country, logger)
    except Exception as e:
        logger.error(f"[run_job] Failed to write to {swing_table}: {e}")

    # Build top-N longs and shorts DataFrames for logging/display
    df_nonnull = df[df["SwingScore"].notna()].copy()
    df_longs = (df_nonnull[df_nonnull["SwingScore"] > 0]
                        .sort_values("SwingScore", ascending=False)
                        .head(top_n)
                        [["Date", "Symbol", "CountryName", "SwingScore", "SignalClassifier_ML",
                          "SignalClassifier_Rules", "Trend", "TMA21_50_X", "RSIUpTrend", "ADX",
                          "Pct2H52", "PctfL52", "GEM_Rank", "marketCap", "IndustrySector", "ScanDate"]]
                        .reset_index(drop=True))

    df_shorts = (df_nonnull[df_nonnull["SwingScore"] < 0]
                         .sort_values("SwingScore", ascending=True)
                         .head(top_n)
                         [["Date", "Symbol", "CountryName", "SwingScore", "SignalClassifier_ML",
                           "SignalClassifier_Rules", "Trend", "TMA21_50_X", "RSIUpTrend", "ADX",
                           "Pct2H52", "PctfL52", "GEM_Rank", "marketCap", "IndustrySector", "ScanDate"]]
                         .reset_index(drop=True))

    logger.info(f"[run_job] Computed {len(df_longs)} long candidates and {len(df_shorts)} short candidates (top {top_n})")
    logger.info(f"[run_job] All scored records written to {swing_table} for country={country}")

def main():
    ap = argparse.ArgumentParser(description="Compute SwingScore and export shortlist CSVs")
    ap.add_argument("-w", "--watchlist", required=True, help="Watchlist name (e.g., US_CORE)")
    ap.add_argument("-s", "--source", required=True,
                    choices=["FINNHUB", "EOD", "FINNHUB_LOCAL", "EOD_LOCAL"],
                    help="Price source (aligns with TA pipeline)")
    ap.add_argument("--top_n", type=int, default=25, help="Top-N longs and shorts to export (default: 25)")
    ap.add_argument("--use_ml_preferred", choices=["yes", "no"], default="yes",
                    help="Prefer ML classifier over rules when both exist (default: yes)")
    ap.add_argument("--update_db", choices=["yes", "no"], default="no",
                    help="Update SwingScore back into tas_listings (default: no)")
    ap.add_argument("--out_dir", default="../data", help="Output directory for CSVs (default: ../data)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        run_job(
            watchlist=args.watchlist.upper(),
            price_source=args.source.upper(),
            top_n=int(args.top_n),
            use_ml_preferred=(args.use_ml_preferred.lower() == "yes"),
            update_db=(args.update_db.lower() == "yes"),
            out_dir=args.out_dir
        )
    except Exception as e:
        logging.getLogger("tas_swing_scoring").error(f"[FATAL] {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

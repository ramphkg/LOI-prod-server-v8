# fundamentals_common.py
"""
Common constants and pure utilities shared by fetcher and ranker.
(Strictly shared pieces; FX-related helpers/fields removed)
"""
from __future__ import annotations
import time
import math
import threading
import logging
import pandas as pd
import datetime
from typing import Optional, List, Dict, Any

# small environment helper (original t.py imported strUtcNow from app_imports)
from app_imports import strUtcNow  # used for timestamps in scanDate / output

LOG = logging.getLogger("fundamentals_common")

# Defaults & constants (shared)
DEFAULT_API_SLEEP = 0.25
DEFAULT_FETCH_BATCH = 50
DEFAULT_DB_PREFIXES = {
    "EOD": ("eod_historical_fundamentals", "eod_gem_listings"),
    "FINNHUB": ("finnhub_historical_fundamentals", "finnhub_gem_listings"),
}
WATCHLIST_TABLE = "eod_watchlist"
DEFAULT_QUARTERS_TO_FETCH = 6
DEFAULT_WORKERS = 8

DEFAULT_RATE = 600
DEFAULT_RATE_PERIOD = 60.0

DEFAULT_MIN_MKTCAP_MLN_BY_COUNTRY: Dict[str, float] = { "USA": 2.0, "India": 160.0, "Hong Kong": 15.0 }

INPUT_COLUMNS = [
    "symbol", "report_date",
    "name", "exchange", "currency", "CountryName", "industry", "Sector",
    "MarketCapitalization", "MarketCapitalizationMln",
    "low52Week", "high52Week",
    "ev", "roicTTM",
    "ebit", "ebitda", "netIncome",
    "revenue", "RevenueTTM",
    "totalAssets", "totalCurrentAssets",
    "netOperatingCashFlow", "totalCashFromOperatingCashFlow",
    "totalCashFromOperatingActivities",
    "EnterpriseValueRevenue",
    "peTTM", "TrailingPE", "EPS",
    "last_close",
    "sharesOutstanding", "avgVolume", "totalDebt", "cashAndCashEquivalents",
    "scanDate"
]

# Note: FX-related fields removed here (no fx_rate_to_usd, MarketCapitalizationUSD, etc.)
OUTPUT_COLUMNS = [
    "symbol", "CountryName", "report_date", "name", "exchange", "industry", "Sector",
    "MarketCapitalization", "MarketCapitalizationMln",
    "ADTV_dollars",
    "low52Week", "high52Week",
    "ROIC", "Earnings_Yield", "Earnings_Quality",
    "GEM_Score", "GEM_Rank", "GEM_Percentile",
    "GEM_Rank_one", "GEM_Rank_two", "GEM_Rank_three",
    "GEM_Data_Tier", "GEM_Data_Tier_Rank",
    "CAP_Category", "CAP_GEM_Rank",
    "last_close", "peTTM", "TrailingPE", "EPS",
    "scanDate"
]

COMPLETENESS_WEIGHT_DEFAULT = 0.20

REQUIRED_QUARTER_FIELDS_BY_SOURCE: Dict[str, List[str]] = {
    "EOD": [
        "ebit",
        "totalAssets",
        "totalCurrentAssets",
        "totalCashFromOperatingActivities",
        "netIncome",
        "EnterpriseValueRevenue",
        "RevenueTTM",
        "MarketCapitalization",
    ],
    "FINNHUB": [
        "ebit",
        "ev",
        "roicTTM",
        "netOperatingCashFlow",
        "netIncome",
        "MarketCapitalization",
    ],
}

def get_required_quarter_fields(source: str) -> List[str]:
    return REQUIRED_QUARTER_FIELDS_BY_SOURCE.get(source.upper(), REQUIRED_QUARTER_FIELDS_BY_SOURCE["EOD"])

DEFAULT_MCAP_FLOOR_DOLLARS = 1_000_000.0
DEFAULT_EV_FLOOR_DOLLARS = 1_000_000.0
DEFAULT_ADTV_MIN_DOLLARS = 0.0

# -------------------------
# Utilities
# -------------------------
def normalize_symbol(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    s = s.replace('\u00A0', ' ').replace('\u200B', '').replace('\ufeff', '')
    s = ' '.join(s.split())
    return s.strip().upper()

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "" or (isinstance(v, str) and v.strip().upper() in ("N/A","NA","NONE")):
            return None
        return float(v)
    except Exception:
        return None

def _sql_commit(db):
    try:
        db.commit()
    except Exception:
        pass

def sanitize_report_date_for_db(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        if isinstance(val, pd.Timestamp):
            if pd.isna(val):
                return None
            return val.date().isoformat()
        if isinstance(val, datetime.date) and not isinstance(val, datetime.datetime):
            return val.isoformat()
        if isinstance(val, datetime.datetime):
            return val.date().isoformat()
    except Exception:
        pass
    s = str(val).strip()
    if s == "" or s in ("0000-00-00", "0000-00-00 00:00:00"):
        return None
    try:
        ts = pd.to_datetime(s, errors='coerce')
        if pd.isna(ts):
            return None
        if ts.year < 1900:
            return None
        return ts.date().isoformat()
    except Exception:
        return None

def _coalesce(*vals) -> Optional[float]:
    for v in vals:
        f = _safe_float(v)
        if f is not None and not (isinstance(f, float) and (math.isnan(f) or math.isinf(f))):
            return f
    return None

# -------------------------
# Rate limiter (token-bucket)
# -------------------------
class RateLimiter:
    def __init__(self, max_calls: int = DEFAULT_RATE, period_seconds: float = DEFAULT_RATE_PERIOD):
        self.max_calls = float(max_calls)
        self.period = float(period_seconds)
        self._allowance = float(max_calls)
        self._last_check = time.time()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._last_check
                self._last_check = now
                self._allowance += elapsed * (self.max_calls / self.period)
                if self._allowance > self.max_calls:
                    self._allowance = self.max_calls
                if self._allowance >= 1.0:
                    self._allowance -= 1.0
                    return
                missing = 1.0 - self._allowance
                if self.max_calls > 0:
                    wait = missing * (self.period / self.max_calls)
                else:
                    wait = 1.0
            time.sleep(max(wait, 0.01))

# -------------------------
# DB schema management (shared)
# -------------------------
from sqlalchemy import text

def ensure_input_table(db, table_name: str):
    cols_sql = """
        symbol VARCHAR(64) NOT NULL,
        report_date DATE NOT NULL,
        name VARCHAR(255),
        exchange VARCHAR(64),
        currency VARCHAR(16),
        CountryName VARCHAR(64),
        industry VARCHAR(255),
        Sector VARCHAR(128),
        MarketCapitalization DOUBLE,
        MarketCapitalizationMln DOUBLE,
        low52Week DOUBLE,
        high52Week DOUBLE,
        ev DOUBLE,
        roicTTM DOUBLE,
        ebit DOUBLE,
        ebitda DOUBLE,
        netIncome DOUBLE,
        revenue DOUBLE,
        RevenueTTM DOUBLE,
        totalAssets DOUBLE,
        totalCurrentAssets DOUBLE,
        netOperatingCashFlow DOUBLE,
        totalCashFromOperatingCashFlow DOUBLE,
        totalCashFromOperatingActivities DOUBLE,
        EnterpriseValueRevenue DOUBLE,
        peTTM DOUBLE,
        TrailingPE DOUBLE,
        EPS DOUBLE,
        last_close DOUBLE,
        sharesOutstanding DOUBLE,
        avgVolume DOUBLE,
        totalDebt DOUBLE,
        cashAndCashEquivalents DOUBLE,
        scanDate DATETIME,
        PRIMARY KEY (symbol, report_date)
    """
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({cols_sql}) ENGINE=InnoDB;"
    db.execute(text(create_sql))
    _sql_commit(db)
    LOG.info("Ensured input table: %s", table_name)

def ensure_output_table(db, table_name: str):
    cols_sql = """
        symbol VARCHAR(64) NOT NULL,
        CountryName VARCHAR(64) NOT NULL,
        report_date DATE NOT NULL,
        name VARCHAR(255),
        exchange VARCHAR(64),
        industry VARCHAR(255),
        Sector VARCHAR(128),
        MarketCapitalization DOUBLE,
        MarketCapitalizationMln DOUBLE,
        ADTV_dollars DOUBLE,
        low52Week DOUBLE,
        high52Week DOUBLE,
        ROIC DOUBLE,
        Earnings_Yield DOUBLE,
        Earnings_Quality DOUBLE,
        GEM_Score DOUBLE,
        GEM_Rank INT,
        GEM_Percentile DOUBLE,
        GEM_Rank_one INT,
        GEM_Rank_two INT,
        GEM_Rank_three INT,
        GEM_Data_Tier INT,
        GEM_Data_Tier_Rank INT,
        CAP_Category VARCHAR(32),
        CAP_GEM_Rank INT,
        last_close DOUBLE,
        peTTM DOUBLE,
        TrailingPE DOUBLE,
        EPS DOUBLE,
        scanDate DATETIME,
        PRIMARY KEY (symbol, CountryName)
    """
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({cols_sql}) ENGINE=InnoDB;"
    db.execute(text(create_sql))
    _sql_commit(db)
    LOG.info("Ensured output table: %s", table_name)

# fundamentals_ranker.py
from __future__ import annotations
import math
import logging
import re
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sqlalchemy import text

from app_imports import getDbConnection, parallelLoggingSetter, strUtcNow

from fundamentals_common import (
    OUTPUT_COLUMNS, DEFAULT_DB_PREFIXES, COMPLETENESS_WEIGHT_DEFAULT,
    normalize_symbol, _safe_float, sanitize_report_date_for_db, _coalesce,
    get_required_quarter_fields, ensure_output_table,
    DEFAULT_MCAP_FLOOR_DOLLARS, DEFAULT_EV_FLOOR_DOLLARS, DEFAULT_ADTV_MIN_DOLLARS,
    WATCHLIST_TABLE, _sql_commit,
    DEFAULT_MIN_MKTCAP_MLN_BY_COUNTRY
)

LOG = parallelLoggingSetter("fundamentals_ranker")
logging.getLogger("requests").setLevel(logging.WARNING)

# Map country -> native currency. Keys are lower-cased for easy matching.
_COUNTRY_NATIVE_CURRENCY = {
    "usa": "USD",
    "india": "INR",
    "hong kong": "HKD"
}

# -------------------------
# GEM scoring & completeness helpers (FX removed)
# -------------------------
def calculate_gem_rank_latest(
    df: pd.DataFrame,
    mcap_floor: float = DEFAULT_MCAP_FLOOR_DOLLARS,
    ev_floor: float = DEFAULT_EV_FLOOR_DOLLARS,
    adtv_min: float = DEFAULT_ADTV_MIN_DOLLARS
) -> pd.DataFrame:
    out = df.copy()

    for c in [
        "roicTTM","ebit","ebitda","totalAssets","totalCurrentAssets","EnterpriseValueRevenue","RevenueTTM",
        "netOperatingCashFlow","totalCashFromOperatingActivities","netIncome","MarketCapitalization",
        "ev","last_close","peTTM","TrailingPE","EPS","sharesOutstanding","avgVolume","totalDebt","cashAndCashEquivalents","currency"
    ]:
        if c not in out.columns:
            out[c] = np.nan
        else:
            if c == "currency":
                out[c] = out[c].astype(object)
            else:
                out[c] = pd.to_numeric(out[c], errors='coerce')

    has_valid_mcap = out["MarketCapitalization"].notna() & (out["MarketCapitalization"] > 0)
    can_recompute_mcap = out["sharesOutstanding"].notna() & out["last_close"].notna()
    recomputed_mcap = np.where(
        (~has_valid_mcap) & can_recompute_mcap,
        out["sharesOutstanding"] * out["last_close"],
        out["MarketCapitalization"]
    )
    out["MarketCapitalization"] = pd.to_numeric(recomputed_mcap, errors='coerce')

    if "MarketCapitalizationMln" not in out.columns:
        out["MarketCapitalizationMln"] = out["MarketCapitalization"] / 1e6
    else:
        out["MarketCapitalizationMln"] = pd.to_numeric(out.get("MarketCapitalizationMln", out["MarketCapitalization"]/1e6), errors='coerce')
        mcapm_from_mcap = out["MarketCapitalization"] / 1e6
        out["MarketCapitalizationMln"] = np.where(out["MarketCapitalizationMln"].isna(), mcapm_from_mcap, out["MarketCapitalizationMln"])

    out["ADTV_dollars"] = out["avgVolume"] * out["last_close"]

    out["NetDebt"] = np.where(
        out["totalDebt"].notna() & out["cashAndCashEquivalents"].notna(),
        out["totalDebt"].fillna(0.0) - out["cashAndCashEquivalents"].fillna(0.0),
        np.nan
    )

    ev_fallback = np.where(
        out["EnterpriseValueRevenue"].notna() & out["RevenueTTM"].notna(),
        out["EnterpriseValueRevenue"] * out["RevenueTTM"],
        np.nan
    )
    ev_mc_net = np.where(
        out["MarketCapitalization"].notna() & out["NetDebt"].notna(),
        out["MarketCapitalization"].fillna(np.nan) + out["NetDebt"].fillna(np.nan),
        np.nan
    )
    out["EnterpriseValue"] = out["ev"].where(out["ev"].notna(), ev_mc_net)
    out["EnterpriseValue"] = out["EnterpriseValue"].where(out["EnterpriseValue"].notna(), ev_fallback)
    out["EnterpriseValue"] = pd.to_numeric(out["EnterpriseValue"], errors='coerce')

    denom = out["totalAssets"] - out["totalCurrentAssets"]
    roic_fallback = np.where((denom == 0) | denom.isna(), np.nan, out["ebit"] / denom)
    out["ROIC"] = out["roicTTM"].where(out["roicTTM"].notna(), roic_fallback)

    cfo = out["netOperatingCashFlow"].where(out["netOperatingCashFlow"].notna(), out["totalCashFromOperatingActivities"])

    investable_by_mcap = out["MarketCapitalization"].notna() & (out["MarketCapitalization"].abs() >= float(mcap_floor))
    if adtv_min and adtv_min > 0:
        investable_by_adtv = out["ADTV_dollars"].notna() & (out["ADTV_dollars"].abs() >= float(adtv_min))
        investable = investable_by_mcap & investable_by_adtv
    else:
        investable = investable_by_mcap

    out["Earnings_Quality"] = np.where(
        investable & (cfo.notna() | out["netIncome"].notna()) & out["MarketCapitalization"].notna(),
        (cfo.fillna(np.nan) - out["netIncome"].fillna(np.nan)) / out["MarketCapitalization"],
        np.nan
    )

    base_profit = out["ebitda"].where(out["ebitda"].notna(), out["ebit"])

    ev_series = out["EnterpriseValue"]
    ev_ok = ev_series.notna() & (ev_series.abs() >= float(ev_floor))

    out["Earnings_Yield"] = np.nan
    base_profit_usd = base_profit  # native units (no FX)
    ey_series = base_profit_usd.where(ev_ok) / ev_series.where(ev_ok)
    # assign the Series directly to preserve index alignment (avoid deprecated .values)
    out["Earnings_Yield"] = ey_series

    out["Earnings_Yield"] = pd.to_numeric(out["Earnings_Yield"], errors='coerce').clip(lower=-100.0, upper=100.0)
    out["Earnings_Quality"] = pd.to_numeric(out["Earnings_Quality"], errors='coerce').clip(lower=-100.0, upper=100.0)
    out["ROIC"] = pd.to_numeric(out["ROIC"], errors='coerce').clip(lower=-10.0, upper=10.0)

    out['ROIC_Percentile'] = out['ROIC'].rank(pct=True, na_option='bottom')
    out['Earnings_Yield_Percentile'] = out['Earnings_Yield'].rank(pct=True, na_option='bottom')
    out['Earnings_Quality_Percentile'] = out['Earnings_Quality'].rank(pct=True, na_option='bottom')

    out['GEM_Score'] = (
        out['ROIC_Percentile'].fillna(0.0) * 0.4 +
        out['Earnings_Yield_Percentile'].fillna(0.0) * 0.5 +
        out['Earnings_Quality_Percentile'].fillna(0.0) * 0.1
    )

    out['GEM_Score_Percentile'] = out['GEM_Score'].rank(pct=True, na_option='bottom')
    out['GEM_Rank'] = out['GEM_Score_Percentile'].rank(ascending=False, method='first', na_option='bottom').round().astype('Int64')

    out['ADTV_dollars'] = out['ADTV_dollars']
    out['NetDebt'] = out['NetDebt']
    out['EnterpriseValue_final'] = out['EnterpriseValue']

    return out

def compute_completeness_ratio(db, input_table: str, symbol: str, required_fields: List[str], quarters: int = 4) -> float:
    norm = normalize_symbol(symbol)
    if not required_fields:
        return 0.0
    fields_sql = ", ".join(required_fields)
    sql = f"""
    SELECT {fields_sql}
    FROM {input_table}
    WHERE TRIM(symbol) = :symbol
    ORDER BY report_date DESC
    LIMIT :limit
    """
    try:
        dfq = pd.read_sql(text(sql), con=db, params={"symbol": norm, "limit": int(quarters)})
    except Exception:
        return 0.0
    if dfq.empty:
        return 0.0
    total = len(required_fields)
    ratios = []
    for _, r in dfq.iterrows():
        present = 0
        for f in required_fields:
            if f in r and not pd.isna(r[f]):
                present += 1
        ratios.append(present / total if total > 0 else 0.0)
    return float(sum(ratios) / len(ratios))

def apply_completeness_weighting(df_scores: pd.DataFrame, completeness_map: Dict[str, float], w_comp: float = COMPLETENESS_WEIGHT_DEFAULT) -> pd.DataFrame:
    out = df_scores.copy()
    if 'GEM_Score' not in out.columns:
        out['GEM_Score'] = 0.0
    gnum = pd.to_numeric(out['GEM_Score'], errors='coerce').fillna(0.0)
    mn = gnum.min(skipna=True)
    mx = gnum.max(skipna=True)
    norm = gnum if mx == mn else (gnum - mn) / (mx - mn)
    out['__completeness'] = out['symbol'].map(completeness_map).fillna(0.0).astype(float)
    out['GEM_Score_Composite'] = (1.0 - float(w_comp)) * norm + float(w_comp) * out['__completeness']
    out['GEM_Score_Percentile'] = out['GEM_Score_Composite'].rank(pct=True, method='min', na_option='bottom')
    out['GEM_Rank'] = out['GEM_Score_Composite'].rank(ascending=False, method='first', na_option='bottom').astype('Int64')
    out['GEM_Score'] = out['GEM_Score_Composite']
    out = out.drop(columns=['__completeness','GEM_Score_Composite'], errors='ignore')
    return out

def compute_prior_period_gem_ranks(db, input_table: str, country: str, use_periods: List[int] = [1,2,3]) -> Dict[int, Dict[str,int]]:
    cols = [
        "symbol", "report_date", "ebit", "ev", "roicTTM",
        "netOperatingCashFlow", "totalCashFromOperatingActivities",
        "netIncome", "MarketCapitalization", "EnterpriseValueRevenue", "RevenueTTM",
        "totalAssets","totalCurrentAssets"
    ]
    sql = f"""
    SELECT {', '.join(cols)}
    FROM {input_table}
    WHERE CountryName = :country
    """
    df_all = pd.read_sql(text(sql), con=db, params={"country": country})
    if df_all.empty:
        return {p:{} for p in use_periods}

    df_all['report_date'] = pd.to_datetime(df_all['report_date'], errors='coerce')
    df_all = df_all.dropna(subset=['symbol','report_date'])
    df_all['symbol'] = df_all['symbol'].astype(str).apply(normalize_symbol)

    for c in ["ebit","ev","roicTTM","netOperatingCashFlow","totalCashFromOperatingActivities",
              "netIncome","MarketCapitalization","EnterpriseValueRevenue","RevenueTTM",
              "totalAssets","totalCurrentAssets"]:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

    df_all["CFO"] = df_all["netOperatingCashFlow"].where(df_all["netOperatingCashFlow"].notna(), df_all["totalCashFromOperatingActivities"])
    df_all["EV_final"] = df_all["ev"].where(
        df_all["ev"].notna(),
        np.where(df_all["EnterpriseValueRevenue"].notna() & df_all["RevenueTTM"].notna(),
                 df_all["EnterpriseValueRevenue"] * df_all["RevenueTTM"], np.nan)
    )
    denom = df_all["totalAssets"] - df_all["totalCurrentAssets"]
    roic_fallback = np.where((denom == 0) | denom.isna(), np.nan, df_all["ebit"] / denom)
    df_all["ROIC_final"] = df_all["roicTTM"].where(df_all["roicTTM"].notna(), roic_fallback)

    df_all = df_all.sort_values(["symbol","report_date"], ascending=[True, False]).copy()
    df_all["period_index"] = df_all.groupby("symbol").cumcount()

    prior_rank_maps: Dict[int, Dict[str,int]] = {}

    for p in use_periods:
        dfp = df_all[df_all["period_index"] == p].copy()
        if dfp.empty:
            prior_rank_maps[p] = {}
            continue
        dfp["Earnings_Yield"] = np.where(dfp["EV_final"].notna() & (dfp["EV_final"] != 0), dfp["ebit"] / dfp["EV_final"], np.nan)
        dfp["Earnings_Quality"] = np.where(
            dfp["MarketCapitalization"].notna(), (dfp["CFO"] - dfp["netIncome"]) / dfp["MarketCapitalization"], np.nan
        )
        dfp['ROIC_Percentile'] = dfp['ROIC_final'].rank(pct=True, na_option='keep')
        dfp['EY_Percentile'] = dfp['Earnings_Yield'].rank(pct=True, na_option='keep')
        dfp['EQ_Percentile'] = dfp['Earnings_Quality'].rank(pct=True, na_option='keep')
        dfp['GEM_Score'] = dfp['ROIC_Percentile'] * 0.4 + dfp['EY_Percentile'] * 0.5 + dfp['EQ_Percentile'] * 0.1
        dfp['GEM_Score_Percentile'] = dfp['GEM_Score'].rank(pct=True, na_option='keep')
        dfp['GEM_Rank'] = dfp['GEM_Score_Percentile'].rank(ascending=False, method='first').round().astype('Int64')
        prior_rank_maps[p] = {row['symbol']: int(row['GEM_Rank']) if not pd.isna(row['GEM_Rank']) else None for _, row in dfp.iterrows()}

    return prior_rank_maps

# -------------------------
# Ranking helpers & upsert/watchlist (FX removed)
# -------------------------
def load_latest_per_symbol(db, input_table: str, country: str, freshness_days: int) -> pd.DataFrame:
    sql = f"SELECT * FROM {input_table} WHERE CountryName = :country"
    df = pd.read_sql(text(sql), con=db, params={"country": country})
    if df.empty:
        return df
    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    latest_idx = df.groupby('symbol')['report_date'].idxmax()
    df = df.loc[latest_idx].reset_index(drop=True)
    if freshness_days and freshness_days > 0:
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=freshness_days)
        df = df[df['report_date'] >= cutoff].copy()
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).apply(normalize_symbol)
    return df

def filter_microcaps(df: pd.DataFrame, min_mktcap_mln: float) -> pd.DataFrame:
    # Use MarketCapitalization (full dollars) as the authoritative field.
    if 'MarketCapitalization' not in df.columns or min_mktcap_mln is None:
        return df.copy()
    mcap = pd.to_numeric(df['MarketCapitalization'], errors='coerce')
    # min_mktcap_mln is provided in millions; compare using full-dollar units.
    threshold = float(min_mktcap_mln) * 1e6
    return df[mcap >= threshold].copy()

def upsert_gem_output(db, output_table: str, rows: List[Dict[str,Any]]):
    if not rows:
        LOG.info("No GEM rows to upsert into %s", output_table)
        return
    for r in rows:
        if 'symbol' in r:
            r['symbol'] = normalize_symbol(r.get('symbol'))
        if 'report_date' in r:
            r['report_date'] = sanitize_report_date_for_db(r.get('report_date'))

    insert_sql = f"""
    INSERT INTO `{output_table}` (
        symbol, CountryName, report_date, name, exchange, industry, Sector,
        MarketCapitalization, MarketCapitalizationMln, ADTV_dollars,
        low52Week, high52Week,
        ROIC, Earnings_Yield, Earnings_Quality,
        GEM_Score, GEM_Rank, GEM_Percentile,
        GEM_Rank_one, GEM_Rank_two, GEM_Rank_three,
        GEM_Data_Tier, GEM_Data_Tier_Rank,
        CAP_Category, CAP_GEM_Rank,
        last_close, peTTM, TrailingPE, EPS,
        scanDate
    ) VALUES (
        :symbol, :CountryName, :report_date, :name, :exchange, :industry, :Sector,
        :MarketCapitalization, :MarketCapitalizationMln, :ADTV_dollars,
        :low52Week, :high52Week,
        :ROIC, :Earnings_Yield, :Earnings_Quality,
        :GEM_Score, :GEM_Rank, :GEM_Percentile,
        :GEM_Rank_one, :GEM_Rank_two, :GEM_Rank_three,
        :GEM_Data_Tier, :GEM_Data_Tier_Rank,
        :CAP_Category, :CAP_GEM_Rank,
        :last_close, :peTTM, :TrailingPE, :EPS,
        :scanDate
    )
    ON DUPLICATE KEY UPDATE
        report_date=VALUES(report_date), name=VALUES(name), exchange=VALUES(exchange),
        industry=VALUES(industry), Sector=VALUES(Sector),
        MarketCapitalization=VALUES(MarketCapitalization), MarketCapitalizationMln=VALUES(MarketCapitalizationMln),
        ADTV_dollars=VALUES(ADTV_dollars),
        low52Week=VALUES(low52Week), high52Week=VALUES(high52Week),
        ROIC=VALUES(ROIC), Earnings_Yield=VALUES(Earnings_Yield), Earnings_Quality=VALUES(Earnings_Quality),
        GEM_Score=VALUES(GEM_Score), GEM_Rank=VALUES(GEM_Rank), GEM_Percentile=VALUES(GEM_Percentile),
        GEM_Rank_one=VALUES(GEM_Rank_one), GEM_Rank_two=VALUES(GEM_Rank_two), GEM_Rank_three=VALUES(GEM_Rank_three),
        GEM_Data_Tier=VALUES(GEM_Data_Tier), GEM_Data_Tier_Rank=VALUES(GEM_Data_Tier_Rank),
        CAP_Category=VALUES(CAP_Category), CAP_GEM_Rank=VALUES(CAP_GEM_Rank),
        last_close=VALUES(last_close), peTTM=VALUES(peTTM), TrailingPE=VALUES(TrailingPE), EPS=VALUES(EPS),
        scanDate=VALUES(scanDate)
    """
    params = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            # Convert pandas / numpy missing sentinels (pd.NA, np.nan) to None first,
            # so DB parameter binding uses NULL rather than the literal string "<NA>".
            try:
                if pd.isna(v):
                    rr[k] = None
                    continue
            except Exception:
                # If pd.isna fails for some type, fall through to normal handling.
                pass

            if isinstance(v, pd.Timestamp):
                rr[k] = v.date().isoformat()
            elif isinstance(v, (np.integer,)):
                rr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                vv = float(v)
                rr[k] = None if (math.isnan(vv) or math.isinf(vv)) else vv
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rr[k] = None
            else:
                rr[k] = v
        params.append(rr)

    try:
        placeholders = set(re.findall(r":([A-Za-z0-9_]+)", insert_sql))
        if params:
            union_keys = set().union(*(set(p.keys()) for p in params))
        else:
            union_keys = set()
        missing = placeholders - union_keys
        if missing:
            LOG.debug("upsert_gem_output: filling missing placeholders with None: %s", missing)
            for p in params:
                for m in missing:
                    p.setdefault(m, None)
    except Exception:
        pass

    try:
        db.execute(text(insert_sql), params)
        _sql_commit(db)
        LOG.info("Upserted %d GEM rows into %s", len(params), output_table)
    except Exception as e:
        LOG.exception("Batch upsert into %s failed: %s", output_table, e)
        for p in params:
            try:
                db.execute(text(insert_sql), p)
            except Exception as e2:
                LOG.error("Per-row upsert failed for %s: %s", p.get("symbol"), e2)
        _sql_commit(db)

def refresh_watchlists_from_output(db, output_table: str):
    watchlists = {
        'GEMS_TOP_TIER1': {'sql': f"SELECT symbol AS Symbol FROM {output_table} WHERE GEM_Data_Tier = 1 AND GEM_Rank IS NOT NULL ORDER BY GEM_Rank ASC LIMIT 100"},
        'GEMS_ALL': {'sql': f"SELECT symbol AS Symbol FROM {output_table} ORDER BY GEM_Data_Tier ASC, GEM_Rank ASC"}
    }
    for wl_name, meta in watchlists.items():
        try:
            df = pd.read_sql(text(meta['sql']), con=db)
            if df.empty:
                LOG.info("No symbols for watchlist %s", wl_name)
                continue
            df.insert(1, "Watchlist", value=wl_name)
            try:
                db.execute(text(f"DELETE FROM {WATCHLIST_TABLE} WHERE Watchlist = :wl"), {"wl": wl_name})
            except Exception:
                pass
            df.to_sql(WATCHLIST_TABLE, con=db, if_exists='append', index=False)
            LOG.info("Updated watchlist %s with %d symbols", wl_name, len(df))
        except Exception as e:
            LOG.exception("Failed to refresh watchlist %s: %s", wl_name, e)

# -------------------------
# Ranking orchestration (native-currency only)
# -------------------------
def rank_source_country(
    source: str,
    country: str,
    min_mktcap_mln: float = 0.0,
    freshness_days: int = 9999,
    purge_first: bool = True,
    completeness_weight: float = COMPLETENESS_WEIGHT_DEFAULT,
    adtv_min_dollars: float = DEFAULT_ADTV_MIN_DOLLARS,
    mcap_floor_dollars: float = DEFAULT_MCAP_FLOOR_DOLLARS,
    ev_floor_dollars: float = DEFAULT_EV_FLOOR_DOLLARS
):
    src_upper = source.upper()
    if src_upper not in DEFAULT_DB_PREFIXES:
        LOG.error("Unknown source: %s", source)
        raise SystemExit(1)
    input_table, output_table = DEFAULT_DB_PREFIXES[src_upper]
    db = getDbConnection()
    ensure_output_table(db, output_table)

    LOG.info("Loading latest-per-symbol from %s for country %s", input_table, country)
    df_latest = load_latest_per_symbol(db, input_table, country, freshness_days)
    LOG.info("Loaded %d symbols (after freshness filter)", len(df_latest))
    if df_latest.empty:
        LOG.warning("No rows to rank for %s/%s", source, country)
        return
    
    # determine effective min_mktcap_mln (in native currency millions)
    if not min_mktcap_mln or float(min_mktcap_mln) <= 0.0:
        # try per-country default (case-insensitive keys)
        country_key = country.strip()
        # try exact match, then title-case, then upper
        eff = DEFAULT_MIN_MKTCAP_MLN_BY_COUNTRY.get(country_key)
        if eff is None:
            eff = DEFAULT_MIN_MKTCAP_MLN_BY_COUNTRY.get(country_key.title())
        if eff is None:
            eff = DEFAULT_MIN_MKTCAP_MLN_BY_COUNTRY.get(country_key.upper())
        if eff is None:
            eff = float(0.0)
        min_mktcap_mln = float(eff)
    LOG.info("Using min_mktcap_mln (native-currency millions) = %s for country=%s", min_mktcap_mln, country)

    # Enforce native-currency-only universe (non-destructive to input table)
    expected = _COUNTRY_NATIVE_CURRENCY.get(country.strip().lower())
    if expected:
        before = len(df_latest)
        if 'currency' in df_latest.columns:
            df_latest['currency_norm'] = df_latest['currency'].astype(object).fillna('').astype(str).str.strip().str.upper()
            df_latest = df_latest[df_latest['currency_norm'] == expected].drop(columns=['currency_norm'])
        else:
            df_latest = df_latest.iloc[0:0]
        LOG.info("Filtered by native currency=%s: before=%d after=%d", expected, before, len(df_latest))

    if df_latest.empty:
        LOG.warning("No rows to rank after currency filter for %s/%s", source, country)
        return

    symbols = df_latest['symbol'].tolist()
    LOG.info("Assigning GEM_Data_Tier based on last 4 quarters completeness for %d symbols", len(symbols))
    required_fields = get_required_quarter_fields(src_upper)

    tier_map: Dict[str,int] = {}
    for i, sym in enumerate(symbols, start=1):
        sql = f"""
        SELECT report_date, {', '.join(required_fields)}
        FROM {input_table}
        WHERE TRIM(symbol) = :symbol
        AND CountryName = :country
        ORDER BY report_date DESC
        LIMIT 4
        """
        try:
            dfq = pd.read_sql(text(sql), con=db, params={"symbol": normalize_symbol(sym), "country": country})
        except Exception:
            dfq = pd.DataFrame()
        count_full = 0
        if not dfq.empty:
            for _, r in dfq.iterrows():
                all_present = True
                for f in required_fields:
                    if f not in r or pd.isna(r[f]):
                        all_present = False
                        break
                if all_present:
                    count_full += 1
        if count_full >= 4:
            tier = 1
        elif count_full >= 2:
            tier = 2
        elif count_full == 1:
            tier = 3
        else:
            tier = 4
        tier_map[sym] = tier
        if i % 200 == 0 or i == len(symbols):
            LOG.info("Tier assignment progress: %d/%d processed", i, len(symbols))

    df_latest['GEM_Data_Tier'] = df_latest['symbol'].map(tier_map).fillna(4).astype(int)

    LOG.info("Scoring latest-per-symbol GEM metrics (native-currency universe)")
    df_scored_latest = calculate_gem_rank_latest(
        df_latest,
        mcap_floor=float(mcap_floor_dollars),
        ev_floor=float(ev_floor_dollars),
        adtv_min=float(adtv_min_dollars)
    )

    completeness_map: Dict[str, float] = {}
    db_conn = getDbConnection()
    syms = df_scored_latest['symbol'].unique().tolist()
    LOG.info("Computing completeness ratio for %d symbols", len(syms))
    for i, s in enumerate(syms, start=1):
        completeness_map[s] = compute_completeness_ratio(db_conn, input_table, s, required_fields=required_fields, quarters=4)
        if i % 200 == 0 or i == len(syms):
            LOG.info("Completeness computation progress: %d/%d", i, len(syms))

    df_scored_latest = apply_completeness_weighting(df_scored_latest, completeness_map, w_comp=float(completeness_weight))

    if 'GEM_Score_Percentile' in df_scored_latest.columns and 'GEM_Percentile' not in df_scored_latest.columns:
        df_scored_latest['GEM_Percentile'] = df_scored_latest['GEM_Score_Percentile']

    LOG.info("Computing prior-period GEM ranks (one/two/three) from historical table")
    prior_rank_maps = compute_prior_period_gem_ranks(db, input_table, country, use_periods=[1,2,3])
    df_scored_latest["GEM_Rank_one"] = df_scored_latest["symbol"].map(prior_rank_maps.get(1, {}))
    df_scored_latest["GEM_Rank_two"] = df_scored_latest["symbol"].map(prior_rank_maps.get(2, {}))
    df_scored_latest["GEM_Rank_three"] = df_scored_latest["symbol"].map(prior_rank_maps.get(3, {}))

    # --- New behaviour: do NOT drop below-threshold rows here.
    # Determine investable threshold (native-currency full dollars)
    threshold_dollars = float(min_mktcap_mln) * 1e6
    mcap_col = 'MarketCapitalization'
    cap_cat_col = 'CAP_Category'
    cap_rank_col = 'CAP_GEM_Rank'

    # Ensure numeric MarketCapitalization
    df_scored_latest[mcap_col] = pd.to_numeric(df_scored_latest.get(mcap_col), errors='coerce')

    investable_mask = df_scored_latest[mcap_col].notna() & (df_scored_latest[mcap_col] >= threshold_dollars)
    df_investable = df_scored_latest.loc[investable_mask].copy()
    df_below = df_scored_latest.loc[~investable_mask].copy()

    LOG.info("Investable count=%d; BelowMin count=%d (threshold=%s mln native)", len(df_investable), len(df_below), min_mktcap_mln)

    # Compute cap thresholds from investable universe only (so buckets are meaningful)
    mcap_vals = pd.to_numeric(df_investable[mcap_col].dropna(), errors='coerce')
    thresholds = None
    if len(mcap_vals) >= 10:
        try:
            p40, p75, p95 = np.nanpercentile(mcap_vals, [40, 75, 95])
            thresholds = (float(p40), float(p75), float(p95))
            LOG.info("Computed cap thresholds (native units, investable) for %s: p40=%.2f p75=%.2f p95=%.2f", country, p40, p75, p95)
        except Exception as e:
            LOG.warning("Failed to compute cap percentiles for %s: %s", country, e)
    else:
        if len(mcap_vals) > 0:
            try:
                p33, p66 = np.nanpercentile(mcap_vals, [33, 66])
                thresholds = (float(p33), float(p66), float(np.nanmax(mcap_vals)))
                LOG.info("Small-investable-universe cap thresholds fallback for %s: p33=%.2f p66=%.2f max=%.2f", country, p33, p66, float(np.nanmax(mcap_vals)))
            except Exception:
                thresholds = None
        else:
            thresholds = None

    def _cap_bucket_native(m_val):
        if pd.isna(m_val):
            return "Unknown"
        try:
            mv = float(m_val)
        except Exception:
            return "Unknown"
        # Anything below the configured threshold is BelowMin regardless of percentiles
        if mv < threshold_dollars:
            return "BelowMin"
        if thresholds is None:
            # No computed thresholds (few investable names) - mark as Unknown for now
            return "Unknown"
        p40, p75, p95 = thresholds
        if mv < p40:
            return "Micro"
        if mv < p75:
            return "Small"
        if mv < p95:
            return "Mid"
        return "Large"

    df_scored_latest[cap_cat_col] = df_scored_latest[mcap_col].apply(_cap_bucket_native)

    # CAP_GEM_Rank: rank within each cap category by GEM_Score (descending). Include BelowMin group as its own category.
    if 'GEM_Score' not in df_scored_latest.columns:
        df_scored_latest['GEM_Score'] = 0.0
    df_scored_latest[cap_rank_col] = df_scored_latest.groupby(cap_cat_col)['GEM_Score'] \
        .rank(ascending=False, method='first', na_option='bottom').astype('Int64')

    # GEM_Rank: assign only to investable symbols (sequential ranking by GEM_Score across investable set).
    # Build mapping symbol -> rank for investable
    gem_rank_map: Dict[str, int] = {}
    if not df_investable.empty:
        tmp = df_investable.copy()
        tmp = tmp.sort_values(['GEM_Score', 'symbol'], ascending=[False, True]).reset_index(drop=True)
        # 1-based rank
        tmp['__GEM_Rank_calc'] = (tmp.index + 1).astype(int)
        gem_rank_map = dict(zip(tmp['symbol'], tmp['__GEM_Rank_calc']))

    # Apply GEM_Rank mapping to full dataframe (non-investable will be NA)
    df_scored_latest['GEM_Rank'] = df_scored_latest['symbol'].map(gem_rank_map).astype('Int64')

    # Now compute GEM_Data_Tier_Rank (unchanged logic - per tier)
    tiered_rows = []
    for tier_val in sorted(df_scored_latest['GEM_Data_Tier'].unique()):
        sub = df_scored_latest[df_scored_latest['GEM_Data_Tier'] == tier_val].copy()
        sub = sub.sort_values(['GEM_Score','symbol'], ascending=[False, True]).reset_index(drop=True)
        sub['GEM_Data_Tier_Rank'] = (sub.index + 1).astype(int)
        tiered_rows.append(sub)
    df_out_all = pd.concat(tiered_rows, axis=0, ignore_index=True, sort=False)

    df_out_all['scanDate'] = strUtcNow()
    for c in OUTPUT_COLUMNS:
        if c not in df_out_all.columns:
            df_out_all[c] = None

    if purge_first and country:
        try:
            db.execute(text(f"DELETE FROM {output_table} WHERE CountryName = :country"), {"country": country})
            _sql_commit(db)
            LOG.info("Purged existing listings for CountryName=%s in %s", country, output_table)
        except Exception as e:
            LOG.warning("Failed to purge existing listings: %s", e)

    rows = df_out_all[OUTPUT_COLUMNS].to_dict(orient='records')
    upsert_gem_output(db, output_table, rows)
    LOG.info("Ranking finished: upserted %d rows into %s", len(rows), output_table)

    try:
        LOG.info("Refreshing watchlists from %s", output_table)
        refresh_watchlists_from_output(db, output_table)
        LOG.info("Watchlist refresh complete.")
    except Exception as e:
        LOG.exception("Watchlist refresh failed: %s", e)

# -------------------------
# Minimal ranker CLI/main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Ranker: compute GEM ranks and upsert to output table.")
    parser.add_argument("--source", required=True, choices=["EOD", "FINNHUB"])
    parser.add_argument("--country", required=True)
    parser.add_argument("--min-mktcap-mln", type=float, default=0.0)
    parser.add_argument("--min-adtv", type=float, default=DEFAULT_ADTV_MIN_DOLLARS)
    parser.add_argument("--mcap-floor", type=float, default=DEFAULT_MCAP_FLOOR_DOLLARS)
    parser.add_argument("--ev-floor", type=float, default=DEFAULT_EV_FLOOR_DOLLARS)
    parser.add_argument("--freshness-days", type=int, default=9999)
    parser.add_argument("--purge-first", action="store_true", default=True)
    parser.add_argument("--no-purge", action="store_false", dest="purge_first")
    parser.add_argument("--completeness-weight", type=float, default=COMPLETENESS_WEIGHT_DEFAULT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    try:
        rank_source_country(
            source=args.source,
            country=args.country,
            min_mktcap_mln=float(args.min_mktcap_mln),
            freshness_days=int(args.freshness_days),
            purge_first=bool(args.purge_first),
            completeness_weight=float(args.completeness_weight),
            adtv_min_dollars=float(args.min_adtv),
            mcap_floor_dollars=float(args.mcap_floor),
            ev_floor_dollars=float(args.ev_floor)
        )
    except Exception as e:
        LOG.exception("Ranker main failed: %s", e)
        raise SystemExit(1)

if __name__ == "__main__":
    main()

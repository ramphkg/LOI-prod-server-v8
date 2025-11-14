# fundamentals_fetcher.py
from __future__ import annotations
import time
import logging
import threading
from typing import List, Dict, Any, Type, Tuple, Optional
import requests
import pandas as pd
import argparse
import math
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed

from app_imports import getDbConnection, EOD_API_KEY, FINNHUB_API_KEY, parallelLoggingSetter, strUtcNow

from fundamentals_common import (
    DEFAULT_API_SLEEP, DEFAULT_FETCH_BATCH, DEFAULT_QUARTERS_TO_FETCH, DEFAULT_WORKERS,
    DEFAULT_RATE, DEFAULT_RATE_PERIOD, DEFAULT_DB_PREFIXES,
    INPUT_COLUMNS, normalize_symbol, _safe_float, sanitize_report_date_for_db, _coalesce,
    RateLimiter, ensure_input_table, _sql_commit
)

LOG = parallelLoggingSetter("fundamentals_fetcher")
logging.getLogger("requests").setLevel(logging.WARNING)

# BaseFetcher and provider implementations (copied/adapted from t.py)
class BaseFetcher:
    def __init__(
        self,
        api_key: Optional[str],
        api_sleep: float = DEFAULT_API_SLEEP,
        rate_limiter: Optional[RateLimiter] = None,
        stop_event: Optional[threading.Event] = None,
        symbol_exchange_map: Optional[Dict[str,str]] = None
    ):
        self.api_key = api_key
        self.api_sleep = api_sleep
        self.session = requests.Session()
        self.rate_limiter = rate_limiter
        self.stop_event = stop_event
        self.symbol_exchange_map = symbol_exchange_map or {}

    def request_get(self, url: str, params: Dict[str,Any] = None, timeout: int = 30, max_retries: int = 3) -> requests.Response:
        if self.stop_event is not None and self.stop_event.is_set():
            raise RuntimeError("Fetch aborted due to stop_event set (likely Payment Required earlier)")
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except Exception as e:
                if attempt <= max_retries:
                    sleep_for = 2 ** attempt
                    LOG.debug("Request exception (attempt %d/%d) for %s: %s â€” sleeping %ds", attempt, max_retries, url, e, sleep_for)
                    time.sleep(sleep_for)
                    continue
                raise
            if resp.status_code == 402:
                LOG.error("Payment Required (402) from API for URL: %s", resp.url)
                if self.stop_event is not None:
                    self.stop_event.set()
                resp.raise_for_status()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt <= max_retries:
                sleep_for = 2 ** attempt
                LOG.warning("Transient HTTP %d for %s (attempt %d/%d). Backing off %ds", resp.status_code, resp.url, attempt, max_retries, sleep_for)
                time.sleep(sleep_for)
                if self.stop_event is not None and self.stop_event.is_set():
                    raise RuntimeError("Fetch aborted during retry backoff")
                continue
            resp.raise_for_status()
            return resp

    def fetch_symbols(self, country: str) -> List[str]:
        raise NotImplementedError()

    def fetch_quarters(self, symbol: str, country: str, quarters: int = DEFAULT_QUARTERS_TO_FETCH) -> List[Dict[str,Any]]:
        raise NotImplementedError()

class EODFetcher(BaseFetcher):
    country_exchanges = {
        'USA': ['NYSE','NASDAQ'],
        'India': ['NSE'],
        'India-BSE': ['BSE'],
        'Hong Kong': ['HK'],
    }

    def fetch_symbols(self, country: str) -> List[str]:
        exchanges = self.country_exchanges.get(country)
        if not exchanges:
            LOG.error("EODFetcher: unsupported country: %s", country)
            return []
        symbols: List[str] = []
        for exch in exchanges:
            url = f"https://eodhistoricaldata.com/api/exchange-symbol-list/{exch}"
            params = {"api_token": self.api_key, "fmt": "json"}
            try:
                r = self.request_get(url, params=params, timeout=30)
                data = r.json()
                df = pd.DataFrame(data)
                if 'Code' not in df.columns:
                    LOG.warning("EODFetcher: symbol list schema unexpected for exchange %s", exch)
                    continue
                if 'Type' in df.columns:
                    df = df[df['Type'] == 'Common Stock']
                if 'Isin' in df.columns:
                    df = df[~df['Isin'].isna()]
                df = df.rename(columns={'Code':'symbol'})
                if exch not in ('NYSE','NASDAQ'):
                    df['symbol'] = df['symbol'].astype(str) + '.' + exch
                syms = df['symbol'].dropna().astype(str).unique().tolist()
                syms = [normalize_symbol(s) for s in syms]
                LOG.info("EODFetcher: %s -> %d symbols", exch, len(syms))
                symbols.extend(syms)
            except Exception as e:
                LOG.warning("EODFetcher: failed to fetch symbol list for %s: %s", exch, e)
        out = []
        seen = set()
        for s in symbols:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        LOG.info("EODFetcher: total symbols for %s = %d", country, len(out))
        return out

    def fetch_quarters(self, symbol: str, country: str, quarters: int = DEFAULT_QUARTERS_TO_FETCH) -> List[Dict[str,Any]]:
        # Full implementation adapted from your t.py (kept behavior)
        raw_symbol = symbol
        url = f"https://eodhistoricaldata.com/api/fundamentals/{raw_symbol}"
        params = {"api_token": self.api_key, "fmt": "json"}
        try:
            r = self.request_get(url, params=params, timeout=30)
            j = r.json()
        except Exception as e:
            LOG.debug("EODFetcher: fundamentals fetch failed for %s: %s", raw_symbol, e)
            return []

        general = j.get("General", {}) or {}
        highlights = j.get("Highlights", {}) or {}
        valuation = j.get("Valuation", {}) or {}
        technicals = j.get("Technicals", {}) or {}
        financials = j.get("Financials", {}) or {}

        country_name = (general.get("CountryName") or country) or None
        exch = general.get("Exchange")
        if exch == 'BSE' and country_name:
            country_name = f"{country_name}-{exch}"

        bs_q = {}
        ic_q = {}
        cf_q = {}
        try:
            bs_q = ((financials.get("Balance_Sheet") or {}).get("quarterly") or {}) if isinstance(financials.get("Balance_Sheet"), dict) else {}
            ic_q = ((financials.get("Income_Statement") or {}).get("quarterly") or {}) if isinstance(financials.get("Income_Statement"), dict) else {}
            cf_q = ((financials.get("Cash_Flow") or {}).get("quarterly") or {}) if isinstance(financials.get("Cash_Flow"), dict) else {}
        except Exception:
            bs_q, ic_q, cf_q = {}, {}, {}

        all_dates = set()
        for m in (bs_q, ic_q, cf_q):
            if isinstance(m, dict):
                all_dates.update([d for d in m.keys() if d])

        dates_sorted: List[pd.Timestamp] = []
        try:
            dates_sorted = sorted(
                [pd.to_datetime(d, errors='coerce') for d in all_dates if pd.to_datetime(d, errors='coerce') is not pd.NaT],
                reverse=True
            )
        except Exception:
            dates_sorted = []

        last_px: Optional[float] = None
        try:
            price_url = f"https://eodhistoricaldata.com/api/real-time/{raw_symbol}"
            r2 = self.request_get(price_url, params={"api_token": self.api_key, "fmt":"json"}, timeout=10)
            if r2.ok:
                pj = r2.json()
                last_px = _safe_float(pj.get("close") or pj.get("Close") or pj.get("last"))
        except Exception:
            pass

        def _try_highlight(key_candidates):
            for k in key_candidates:
                if k in highlights and highlights.get(k) is not None:
                    return _safe_float(highlights.get(k))
            return None

        rows: List[Dict[str,Any]] = []
        if dates_sorted:
            for idx, dt in enumerate(dates_sorted[:quarters]):
                key = dt.date().isoformat()
                bs = bs_q.get(key) or {}
                ic = ic_q.get(key) or {}
                cf = cf_q.get(key) or {}

                ebit_val = _coalesce(ic.get("ebit"), ic.get("EBIT"))
                ebitda_val = _coalesce(ic.get("ebitda"), ic.get("EBITDA"))
                net_income = _safe_float(ic.get("netIncome"))
                total_assets = _safe_float(bs.get("totalAssets"))
                total_current_assets = _safe_float(bs.get("totalCurrentAssets"))
                cfo = _safe_float(cf.get("totalCashFromOperatingCashFlow"))
                rev = _coalesce(ic.get("revenue"), ic.get("totalRevenue"))

                total_debt = _coalesce(bs.get("totalDebt"), bs.get("shortTermDebt"), bs.get("longTermDebt"), highlights.get("TotalDebt"))
                cash_eq = _coalesce(bs.get("cashAndCashEquivalents"), bs.get("cash"), highlights.get("CashAndEquivalent"))
                shares_out = _coalesce(highlights.get("SharesOutstanding"), general.get("SharesOutstanding"))
                avg_vol = _coalesce(technicals.get("AverageVolume"), technicals.get("avgVolume"), technicals.get("avgVol"))

                row = {
                    "symbol": normalize_symbol(raw_symbol),
                    "report_date": key,
                    "name": general.get("Name"),
                    "exchange": exch,
                    "currency": general.get("CurrencyCode"),
                    "CountryName": country_name,
                    "industry": general.get("Industry"),
                    "Sector": general.get("Sector"),
                    "MarketCapitalization": _safe_float(highlights.get("MarketCapitalization")),
                    "MarketCapitalizationMln": _safe_float(highlights.get("MarketCapitalizationMln")),
                    "low52Week": _safe_float(technicals.get("52WeekLow")),
                    "high52Week": _safe_float(technicals.get("52WeekHigh")),
                    "ev": None,
                    "roicTTM": None,
                    "ebit": _safe_float(ebit_val),
                    "ebitda": _safe_float(ebitda_val),
                    "netIncome": _safe_float(net_income),
                    "revenue": _safe_float(rev),
                    "RevenueTTM": _safe_float(highlights.get("RevenueTTM")),
                    "totalAssets": _safe_float(total_assets),
                    "totalCurrentAssets": _safe_float(total_current_assets),
                    "netOperatingCashFlow": _safe_float(cfo),
                    "totalCashFromOperatingCashFlow": None,
                    "totalCashFromOperatingActivities": _safe_float(cfo),
                    "EnterpriseValueRevenue": _safe_float(valuation.get("EnterpriseValueRevenue")) if 'valuation' in locals() else None,
                    "peTTM": _safe_float(valuation.get("TrailingPE")) if 'valuation' in locals() else None,
                    "TrailingPE": _safe_float(valuation.get("TrailingPE")) if 'valuation' in locals() else None,
                    "EPS": _safe_float(highlights.get("EarningsShare")),
                    "last_close": last_px if idx == 0 else None,
                    "sharesOutstanding": _safe_float(shares_out),
                    "avgVolume": _safe_float(avg_vol),
                    "totalDebt": _safe_float(total_debt),
                    "cashAndCashEquivalents": _safe_float(cash_eq),
                    "scanDate": strUtcNow()
                }
                rows.append(row)

            LOG.info("EOD: %s -> assembled %d quarterly rows", raw_symbol, len(rows))
            if self.api_sleep:
                time.sleep(self.api_sleep)
            return rows

        # snapshot fallback
        row = {
            "symbol": normalize_symbol(raw_symbol),
            "report_date": sanitize_report_date_for_db(highlights.get("MostRecentQuarter")) or pd.Timestamp.today().date().isoformat(),
            "name": general.get("Name"),
            "exchange": exch,
            "currency": general.get("CurrencyCode"),
            "CountryName": country_name,
            "industry": general.get("Industry"),
            "Sector": general.get("Sector"),
            "MarketCapitalization": _safe_float(highlights.get("MarketCapitalization")),
            "MarketCapitalizationMln": _safe_float(highlights.get("MarketCapitalizationMln")),
            "low52Week": _safe_float(technicals.get("52WeekLow")),
            "high52Week": _safe_float(technicals.get("52WeekHigh")),
            "ev": None,
            "roicTTM": None,
            "ebit": None,
            "ebitda": None,
            "netIncome": _safe_float(highlights.get("NetIncome")),
            "revenue": None,
            "RevenueTTM": _safe_float(highlights.get("RevenueTTM")),
            "totalAssets": None,
            "totalCurrentAssets": None,
            "netOperatingCashFlow": _safe_float(highlights.get("TotalCashFromOperatingActivities")),
            "totalCashFromOperatingCashFlow": None,
            "totalCashFromOperatingActivities": _safe_float(highlights.get("TotalCashFromOperatingActivities")),
            "EnterpriseValueRevenue": _safe_float(valuation.get("EnterpriseValueRevenue")) if 'valuation' in locals() else None,
            "peTTM": _safe_float(valuation.get("TrailingPE")) if 'valuation' in locals() else None,
            "TrailingPE": _safe_float(highlights.get("TrailingPE")),
            "EPS": _safe_float(highlights.get("EarningsShare")),
            "last_close": last_px,
            "sharesOutstanding": _safe_float(highlights.get("SharesOutstanding") or general.get("SharesOutstanding")),
            "avgVolume": _safe_float(technicals.get("AverageVolume") or technicals.get("avgVolume")),
            "totalDebt": _safe_float(highlights.get("TotalDebt")),
            "cashAndCashEquivalents": _safe_float(highlights.get("CashAndEquivalent")),
            "scanDate": strUtcNow()
        }
        LOG.info("EOD: %s -> no quarterly statements; returning 1-row snapshot", raw_symbol)
        if self.api_sleep:
            time.sleep(self.api_sleep)
        return [row]

class FinnhubFetcher(BaseFetcher):
    _MICs = ['XNYS','XNAS']
    _TYPES = ['Common Stock','ADR']

    def fetch_symbols(self, country: str) -> List[str]:
        if country != "USA":
            LOG.error("FinnhubFetcher: only USA supported for symbol enumeration")
            return []
        url = "https://finnhub.io/api/v1/stock/symbol"
        params = {"exchange": "US", "token": self.api_key}
        try:
            r = self.request_get(url, params=params, timeout=30)
            data = r.json()
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                LOG.warning("FinnhubFetcher: symbol list returned unexpected schema")
                return []
            if 'mic' in df.columns:
                df = df[df['mic'].isin(self._MICs)]
            if 'type' in df.columns:
                df = df[df['type'].isin(self._TYPES)]
            syms = df['symbol'].dropna().astype(str).unique().tolist()
            return [normalize_symbol(s) for s in syms]
        except Exception as e:
            LOG.warning("FinnhubFetcher: failed to fetch symbol list: %s", e)
            return []

    def get_symbol_exchange_map(self) -> Dict[str,str]:
        url = "https://finnhub.io/api/v1/stock/symbol"
        params = {"exchange": "US", "token": self.api_key}
        try:
            r = self.request_get(url, params=params, timeout=40)
            data = r.json()
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                return {}
            if 'mic' in df.columns:
                df = df[df['mic'].isin(self._MICs)]
                mic_map = {'XNAS': 'NASDAQ', 'XNYS': 'NYSE'}
                df['exchange_std'] = df['mic'].map(mic_map).fillna(df.get('mic'))
                df['symbol_norm'] = df['symbol'].astype(str).apply(normalize_symbol)
                mapping = dict(zip(df['symbol_norm'], df['exchange_std']))
                LOG.info("Built symbol->exchange map with %d entries (Finnhub)", len(mapping))
                return mapping
            return {}
        except Exception as e:
            LOG.warning("Failed to build Finnhub symbol->exchange map from Finnhub: %s", e)
            return {}

    def _merge_bucket(self, assembled_by_date: Dict[str, Dict[str, Any]], report_date_raw: str, metric_map: Dict[str, Any]):
        if not report_date_raw:
            return
        try:
            rd = pd.to_datetime(report_date_raw).normalize()
        except Exception:
            return
        key = rd.date().isoformat()
        bucket = assembled_by_date.setdefault(key, {})
        for k, v in metric_map.items():
            if v is None:
                continue
            if bucket.get(k) is None:
                bucket[k] = _safe_float(v)

    def _fetch_financials_statement(self, symbol: str, statement: str, assembled_by_date: Dict[str, Dict[str, Any]]):
        url = "https://finnhub.io/api/v1/stock/financials"
        params = {"symbol": symbol, "statement": statement, "freq": "quarterly", "token": self.api_key}
        try:
            r = self.request_get(url, params=params, timeout=20)
            j = r.json() if r.content else {}
            finlist = j.get("financials") or j.get("data") or []
            if not isinstance(finlist, list):
                return
            for it in finlist:
                rd = it.get("period") or it.get("reportDate") or it.get("endDate") or it.get("date")
                mapped: Dict[str, Any] = {}
                if statement == "ic":
                    mapped["netIncome"] = it.get("netIncome")
                    mapped["ebit"] = it.get("ebit") or it.get("operatingIncome")
                    mapped["ebitda"] = it.get("ebitda")
                    mapped["revenue"] = it.get("revenue")
                elif statement == "bs":
                    mapped["totalAssets"] = it.get("totalAssets")
                    ca = it.get("currentAssets") or it.get("totalCurrentAssets")
                    if ca is not None:
                        mapped["totalCurrentAssets"] = ca
                    mapped["totalDebt"] = it.get("totalDebt") or it.get("shortTermDebt") or it.get("longTermDebt")
                    mapped["cashAndCashEquivalents"] = it.get("cashAndCashEquivalents") or it.get("cash")
                    mapped["sharesOutstanding"] = it.get("sharesOutstanding")
                elif statement == "cf":
                    ocf = it.get("netOperatingCashFlow") or it.get("cashFromOperatingActivities") or it.get("operatingCashFlow")
                    if ocf is not None:
                        mapped["netOperatingCashFlow"] = ocf
                        mapped["totalCashFromOperatingActivities"] = ocf
                if mapped:
                    self._merge_bucket(assembled_by_date, rd, mapped)
        except Exception as e:
            LOG.debug("Finnhub /stock/financials fetch failed for statement=%s symbol=%s: %s", statement, symbol, e)

    def _fetch_profile(self, symbol: str) -> Dict[str, Any]:
        prof = {}
        try:
            url = "https://finnhub.io/api/v1/stock/profile2"
            r = self.request_get(url, params={"symbol": symbol, "token": self.api_key}, timeout=15)
            if r.ok:
                pj = r.json() or {}
                prof["name"] = pj.get("name")
                prof["exchange_raw"] = pj.get("exchange")
                prof["currency"] = pj.get("currency")
                prof["industry"] = pj.get("finnhubIndustry")
                prof["gsector"] = pj.get("gsector")
                prof["sharesOutstanding"] = pj.get("shareOutstanding") or pj.get("sharesOutstanding")
                prof["avgVolume"] = pj.get("fiftyTwoWeekAverageVolume") or pj.get("avgVolume")
        except Exception:
            pass
        return prof

    def _fetch_metric_and_series(self, symbol: str):
        metric_latest: Dict[str, Any] = {}
        series_map: Dict[str, Dict[str, float]] = {"ev":{}, "roicTTM":{}, "peTTM":{}, "eps":{}}
        try:
            url = "https://finnhub.io/api/v1/stock/metric"
            r = self.request_get(url, params={"symbol": symbol, "metric": "all", "token": self.api_key}, timeout=20)
            if not r.ok:
                return metric_latest, series_map
            j = r.json() or {}
            metric = j.get("metric") or {}
            series_q = ((j.get("series") or {}).get("quarterly")) or {}

            # Finnhub reports marketCapitalization in millions per-runner note.
            mc_mln = _safe_float(metric.get("marketCapitalization"))
            metric_latest["MarketCapitalizationMln"] = mc_mln
            metric_latest["MarketCapitalization"] = (mc_mln * 1e6) if mc_mln is not None else None

            metric_latest["low52Week"] = _safe_float(metric.get("52WeekLow"))
            metric_latest["high52Week"] = _safe_float(metric.get("52WeekHigh"))
            metric_latest["RevenueTTM"] = _safe_float(metric.get("revenueTTM"))
            metric_latest["TrailingPE"] = _safe_float(metric.get("peTTM") or metric.get("peNormalized"))
            metric_latest["peTTM"] = _safe_float(metric.get("peTTM"))
            metric_latest["sharesOutstanding"] = _safe_float(metric.get("sharesOutstanding") or metric.get("shareOutstanding"))
            metric_latest["avgVolume"] = _safe_float(metric.get("avgVolume") or metric.get("50DayAverageVolume"))

            def map_series(key: str, target_key: str):
                arr = series_q.get(key) or []
                for item in arr:
                    period_raw = item.get("period") or item.get("t")
                    val = _safe_float(item.get("v"))
                    if not period_raw:
                        continue
                    try:
                        rd = pd.to_datetime(period_raw).date().isoformat()
                    except Exception:
                        rd = str(period_raw)
                    if val is not None:
                        # Convert EV series to full-dollar units to match MarketCapitalization conversion.
                        # Finnhub metric.marketCapitalization is converted to dollars above (mc_mln * 1e6),
                        # and the EV series appears to follow the same "millions" convention.
                        if key == "ev":
                            series_map[target_key][rd] = val * 1e6
                        else:
                            series_map[target_key][rd] = val

            map_series("ev", "ev")
            map_series("roicTTM", "roicTTM")
            map_series("peTTM", "peTTM")
            map_series("eps", "eps")
        except Exception:
            pass
        return metric_latest, series_map

    def _fetch_quote(self, symbol: str) -> Optional[float]:
        try:
            url = "https://finnhub.io/api/v1/quote"
            r = self.request_get(url, params={"symbol": symbol, "token": self.api_key}, timeout=10)
            if r.ok:
                q = r.json() or {}
                return _safe_float(q.get("c") or q.get("pc"))
        except Exception:
            pass
        return None

    def fetch_quarters(self, symbol: str, country: str, quarters: int = DEFAULT_QUARTERS_TO_FETCH) -> List[Dict[str,Any]]:
        # Full implementation copied from t.py
        symbol_raw = symbol
        assembled_by_date: Dict[str, Dict[str,Any]] = {}

        for stmt in ("ic","bs","cf"):
            self._fetch_financials_statement(symbol_raw, stmt, assembled_by_date)

        metric_latest, series_map = self._fetch_metric_and_series(symbol_raw)

        need_profile = not metric_latest.get("MarketCapitalization") or not metric_latest.get("RevenueTTM")
        prof = {}
        last_px = None
        if need_profile:
            prof = self._fetch_profile(symbol_raw)
            last_px = self._fetch_quote(symbol_raw)
        else:
            prof = {"name": None, "exchange_raw": None, "currency": None, "industry": None, "gsector": None, "sharesOutstanding": metric_latest.get("sharesOutstanding"), "avgVolume": metric_latest.get("avgVolume")}

        symbol_norm = normalize_symbol(symbol_raw)
        exch_std = None
        if self.symbol_exchange_map and symbol_norm in self.symbol_exchange_map:
            exch_std = self.symbol_exchange_map[symbol_norm]
        else:
            exch_std = prof.get("exchange_raw") or None

        sector_std = None
        gsec = prof.get("gsector")
        if gsec:
            # keep mapping inline (small helper)
            mapping = {
                'Materials': "Basic Materials",
                'Consumer Staples': "Consumer Defensive",
                'Consumer Discretionary': "Consumer Cyclical",
                'Financials': "Financial Services",
                'Health Care': "Healthcare",
                'Information Technology': "Technology",
            }
            sector_std = mapping.get(gsec, gsec)
        else:
            sector_std = prof.get("industry")

        if assembled_by_date:
            rows: List[Dict[str,Any]] = []
            for rd in sorted(assembled_by_date.keys(), reverse=True)[:quarters]:
                b = assembled_by_date[rd]
                # prefer explicit mln if available; metric_latest stores MarketCapitalization as full dollars and MarketCapitalizationMln as mln
                mc_full = metric_latest.get("MarketCapitalization")
                mc_mln = metric_latest.get("MarketCapitalizationMln") if metric_latest.get("MarketCapitalizationMln") is not None else (mc_full / 1e6 if mc_full is not None else None)
                row = {
                    "symbol": normalize_symbol(symbol_raw),
                    "report_date": rd,
                    "name": prof.get("name"),
                    "exchange": exch_std,
                    "currency": prof.get("currency"),
                    "CountryName": country,
                    "industry": prof.get("industry"),
                    "Sector": sector_std,
                    "MarketCapitalization": mc_full,
                    "MarketCapitalizationMln": mc_mln,
                    "low52Week": metric_latest.get("low52Week"),
                    "high52Week": metric_latest.get("high52Week"),
                    "ev": _safe_float(series_map["ev"].get(rd)),
                    "roicTTM": _safe_float(series_map["roicTTM"].get(rd)),
                    "ebit": _safe_float(b.get("ebit")),
                    "ebitda": _safe_float(b.get("ebitda")),
                    "netIncome": _safe_float(b.get("netIncome")),
                    "revenue": _safe_float(b.get("revenue")),
                    "RevenueTTM": metric_latest.get("RevenueTTM"),
                    "totalAssets": _safe_float(b.get("totalAssets")),
                    "totalCurrentAssets": _safe_float(b.get("totalCurrentAssets")),
                    "netOperatingCashFlow": _safe_float(b.get("netOperatingCashFlow")),
                    "totalCashFromOperatingCashFlow": None,
                    "totalCashFromOperatingActivities": _safe_float(b.get("netOperatingCashFlow")),
                    "EnterpriseValueRevenue": None,
                    "peTTM": _safe_float(series_map["peTTM"].get(rd) or metric_latest.get("peTTM")),
                    "TrailingPE": metric_latest.get("TrailingPE"),
                    "EPS": _safe_float(series_map["eps"].get(rd)),
                    "last_close": last_px,
                    "sharesOutstanding": _safe_float(b.get("sharesOutstanding") or metric_latest.get("sharesOutstanding") or prof.get("sharesOutstanding")),
                    "avgVolume": _safe_float(metric_latest.get("avgVolume") or prof.get("avgVolume")),
                    "totalDebt": _safe_float(b.get("totalDebt")),
                    "cashAndCashEquivalents": _safe_float(b.get("cashAndCashEquivalents")),
                    "scanDate": strUtcNow()
                }
                rows.append(row)
            LOG.info("Finnhub: %s -> assembled %d quarterly rows (with series mapping)", symbol_raw, len(rows))
            if self.api_sleep:
                time.sleep(self.api_sleep)
            return rows

        # fallback snapshot
        LOG.info("Finnhub: %s -> no quarterly financials found; returning single snapshot fallback", symbol_raw)
        row: Dict[str,Any] = {k: None for k in INPUT_COLUMNS}
        row["symbol"] = normalize_symbol(symbol_raw)
        row["CountryName"] = country
        row["scanDate"] = strUtcNow()
        row["exchange"] = exch_std
        if prof:
            row["name"] = prof.get("name")
            row["currency"] = prof.get("currency")
            row["industry"] = prof.get("industry")
            row["sharesOutstanding"] = _safe_float(prof.get("sharesOutstanding"))
            row["avgVolume"] = _safe_float(prof.get("avgVolume"))
        if metric_latest:
            mc_full = metric_latest.get("MarketCapitalization")
            mc_mln = metric_latest.get("MarketCapitalizationMln") if metric_latest.get("MarketCapitalizationMln") is not None else (mc_full / 1e6 if mc_full is not None else None)
            row["MarketCapitalization"] = mc_full
            row["MarketCapitalizationMln"] = mc_mln
            row["low52Week"] = metric_latest.get("low52Week")
            row["high52Week"] = metric_latest.get("high52Week")
            row["RevenueTTM"] = metric_latest.get("RevenueTTM")
            row["TrailingPE"] = metric_latest.get("TrailingPE")
            row["peTTM"] = metric_latest.get("peTTM")
            if metric_latest.get("sharesOutstanding") is not None and row.get("sharesOutstanding") is None:
                row["sharesOutstanding"] = metric_latest.get("sharesOutstanding")
            if metric_latest.get("avgVolume") is not None and row.get("avgVolume") is None:
                row["avgVolume"] = metric_latest.get("avgVolume")
        row["EPS"] = row.get("EPS")
        row["Sector"] = sector_std
        row["last_close"] = last_px
        row["report_date"] = pd.Timestamp.today().date().isoformat()
        if self.api_sleep:
            time.sleep(self.api_sleep)
        return [row]

FETCHER_MAP: Dict[str, Type[BaseFetcher]] = {
    "EOD": EODFetcher,
    "FINNHUB": FinnhubFetcher,
}

# Upsert helper for input table
def batch_upsert_input(db, table_name: str, rows: List[Dict[str,Any]]):
    if not rows:
        return
    for r in rows:
        if 'symbol' in r:
            r['symbol'] = normalize_symbol(r.get('symbol'))
        if 'report_date' in r:
            r['report_date'] = sanitize_report_date_for_db(r.get('report_date'))
    for r in rows:
        for c in INPUT_COLUMNS:
            r.setdefault(c, None)

    insert_sql = f"""
    INSERT INTO `{table_name}` (
        symbol, report_date,
        name, exchange, currency, CountryName, industry, Sector,
        MarketCapitalization, MarketCapitalizationMln,
        low52Week, high52Week,
        ev, roicTTM,
        ebit, ebitda, netIncome,
        revenue, RevenueTTM,
        totalAssets, totalCurrentAssets,
        netOperatingCashFlow, totalCashFromOperatingCashFlow, totalCashFromOperatingActivities,
        EnterpriseValueRevenue,
        peTTM, TrailingPE, EPS,
        last_close, sharesOutstanding, avgVolume, totalDebt, cashAndCashEquivalents, scanDate
    ) VALUES (
        :symbol, :report_date,
        :name, :exchange, :currency, :CountryName, :industry, :Sector,
        :MarketCapitalization, :MarketCapitalizationMln,
        :low52Week, :high52Week,
        :ev, :roicTTM,
        :ebit, :ebitda, :netIncome,
        :revenue, :RevenueTTM,
        :totalAssets, :totalCurrentAssets,
        :netOperatingCashFlow, :totalCashFromOperatingCashFlow, :totalCashFromOperatingActivities,
        :EnterpriseValueRevenue,
        :peTTM, :TrailingPE, :EPS,
        :last_close, :sharesOutstanding, :avgVolume, :totalDebt, :cashAndCashEquivalents, :scanDate
    )
    ON DUPLICATE KEY UPDATE
        name=VALUES(name), exchange=VALUES(exchange), currency=VALUES(currency),
        CountryName=VALUES(CountryName), industry=VALUES(industry), Sector=VALUES(Sector),
        MarketCapitalization=VALUES(MarketCapitalization), MarketCapitalizationMln=VALUES(MarketCapitalizationMln),
        low52Week=VALUES(low52Week), high52Week=VALUES(high52Week),
        ev=VALUES(ev), roicTTM=VALUES(roicTTM),
        ebit=VALUES(ebit), ebitda=VALUES(ebitda), netIncome=VALUES(netIncome),
        revenue=VALUES(revenue), RevenueTTM=VALUES(RevenueTTM),
        totalAssets=VALUES(totalAssets), totalCurrentAssets=VALUES(totalCurrentAssets),
        netOperatingCashFlow=VALUES(netOperatingCashFlow),
        totalCashFromOperatingCashFlow=VALUES(totalCashFromOperatingCashFlow),
        totalCashFromOperatingActivities=VALUES(totalCashFromOperatingActivities),
        EnterpriseValueRevenue=VALUES(EnterpriseValueRevenue),
        peTTM=VALUES(peTTM), TrailingPE=VALUES(TrailingPE), EPS=VALUES(EPS),
        last_close=VALUES(last_close), sharesOutstanding=VALUES(sharesOutstanding),
        avgVolume=VALUES(avgVolume), totalDebt=VALUES(totalDebt), cashAndCashEquivalents=VALUES(cashAndCashEquivalents),
        scanDate=VALUES(scanDate)
    """
    params = []
    for r in rows:
        p = {}
        for k in INPUT_COLUMNS:
            v = r.get(k)
            if k == "report_date":
                v = sanitize_report_date_for_db(v)
            if isinstance(v, (pd.Timestamp,)):
                v = v.date().isoformat()
            if isinstance(v, str) and v.strip() in ("0000-00-00","0000-00-00 00:00:00"):
                v = None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            p[k] = v
        params.append(p)
    try:
        db.execute(text(insert_sql), params)
        _sql_commit(db)
        LOG.info("Upserted %d rows into %s", len(params), table_name)
    except Exception as e:
        LOG.exception("Batch upsert failed: %s", e)
        for p in params:
            try:
                db.execute(text(insert_sql), p)
            except Exception as e2:
                LOG.error("Per-row upsert failed for %s: %s", p.get('symbol'), e2)
        _sql_commit(db)

# Fetch orchestration
def fetch_source_country(
    source: str,
    country: str,
    api_sleep: float = DEFAULT_API_SLEEP,
    quarters: int = DEFAULT_QUARTERS_TO_FETCH,
    workers: int = DEFAULT_WORKERS,
    rate: int = DEFAULT_RATE,
    rate_period: float = DEFAULT_RATE_PERIOD,
    force_fetch: bool = False,
    stale_hours: int = 24
):
    src_upper = source.upper()
    if src_upper not in DEFAULT_DB_PREFIXES:
        LOG.error("Unknown source: %s", source)
        raise SystemExit(1)
    input_table, _ = DEFAULT_DB_PREFIXES[src_upper]

    db_main = getDbConnection()
    ensure_input_table(db_main, input_table)

    FetcherCls = FETCHER_MAP.get(src_upper)
    if not FetcherCls:
        LOG.error("No fetcher implemented for source %s", src_upper)
        raise SystemExit(1)
    api_key = EOD_API_KEY if src_upper == "EOD" else FINNHUB_API_KEY

    if src_upper == "FINNHUB" and country != "USA":
        LOG.error("FINNHUB source currently supports only USA. Received country=%s. Exiting.", country)
        raise SystemExit(1)

    rate_limiter = RateLimiter(max_calls=int(rate), period_seconds=float(rate_period))
    stop_event = threading.Event()

    fetcher_for_list = FetcherCls(api_key=api_key, api_sleep=api_sleep, rate_limiter=rate_limiter, stop_event=stop_event)
    symbols = fetcher_for_list.fetch_symbols(country)
    if not symbols:
        LOG.warning("No symbols to fetch for source=%s country=%s", source, country)
        return

    symbol_exchange_map: Dict[str,str] = {}
    if src_upper == "FINNHUB":
        try:
            symbol_exchange_map = fetcher_for_list.get_symbol_exchange_map()
        except Exception:
            symbol_exchange_map = {}

    symbols_to_fetch = symbols
    if not force_fetch:
        try:
            sql = f"""
            SELECT TRIM(symbol) AS symbol, MAX(scanDate) AS last_scan
            FROM {input_table}
            WHERE CountryName = :country
            GROUP BY TRIM(symbol)
            """
            df_scan = pd.read_sql(text(sql), con=db_main, params={"country": country})
            if not df_scan.empty:
                df_scan['symbol'] = df_scan['symbol'].astype(str).apply(normalize_symbol)
                df_scan['last_scan_dt'] = pd.to_datetime(df_scan['last_scan'], errors='coerce')
                cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(hours=int(stale_hours))
                fresh_set = set(df_scan.loc[df_scan['last_scan_dt'] >= cutoff, 'symbol'].astype(str).tolist())
                symbols_to_fetch = [s for s in symbols if s not in fresh_set]
                LOG.info("Stale filtering: total symbols=%d, fresh=%d, to_fetch=%d (stale_hours=%d)", len(symbols), len(fresh_set), len(symbols_to_fetch), stale_hours)
            else:
                LOG.info("No existing scanDate entries found; will fetch all symbols")
                symbols_to_fetch = symbols
        except Exception as e:
            LOG.warning("Failed to compute scanDate-based filtering (fetching all symbols): %s", e)
            symbols_to_fetch = symbols
    else:
        LOG.info("force_fetch enabled: will fetch all %d symbols", len(symbols))

    if not symbols_to_fetch:
        LOG.info("No symbols to fetch after stale-check; exiting fetch stage")
        return

    total_symbols = len(symbols_to_fetch)
    LOG.info("Beginning parallel, rate-limited fetch for %d symbols (source=%s, country=%s) with %d workers; rate=%d/%ss; quarters=%d",
             total_symbols, source, country, workers, int(rate), rate_period, quarters)

    workers = max(1, min(int(workers), total_symbols))

    def worker_fetch(sym: str) -> Tuple[str, List[Dict[str,Any]], Optional[Exception]]:
        if stop_event.is_set():
            return (sym, [], RuntimeError("Aborted: stop_event set before starting fetch"))
        try:
            local_fetcher = FetcherCls(api_key=api_key, api_sleep=api_sleep, rate_limiter=rate_limiter, stop_event=stop_event, symbol_exchange_map=symbol_exchange_map)
            LOG.debug("Thread %s starting fetch for %s", threading.current_thread().name, sym)
            rows = local_fetcher.fetch_quarters(sym, country, quarters=quarters)
            LOG.debug("Thread %s finished fetch for %s (rows=%d)", threading.current_thread().name, sym, len(rows))
            return (sym, rows, None)
        except Exception as e:
            return (sym, [], e)

    batch: List[Dict[str,Any]] = []
    total_upserted = 0
    submitted = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=workers) as exe:
        future_to_sym = {exe.submit(worker_fetch, sym): sym for sym in symbols_to_fetch}
        submitted = len(future_to_sym)
        LOG.info("Dispatched %d fetch tasks across %d workers", submitted, workers)

        for fut in as_completed(future_to_sym):
            sym = future_to_sym.get(fut)
            try:
                sym_ret, rows, exc = fut.result()
                if exc:
                    LOG.warning("Fetch failed for %s: %s", sym_ret, exc)
                if rows:
                    for r in rows:
                        if 'symbol' in r:
                            r['symbol'] = normalize_symbol(r.get('symbol'))
                        if 'report_date' in r:
                            r['report_date'] = sanitize_report_date_for_db(r.get('report_date'))
                    batch.extend(rows)
                    LOG.info("Worker returned %d rows for %s (thread=%s)", len(rows), sym_ret, threading.current_thread().name)
                else:
                    LOG.debug("No rows returned for %s (thread=%s)", sym_ret, threading.current_thread().name)

                processed += 1

                if stop_event.is_set():
                    LOG.error("Global stop_event set (likely Payment Required). Aborting remaining processing.")

                if len(batch) >= DEFAULT_FETCH_BATCH:
                    try:
                        batch_upsert_input(db_main, input_table, batch)
                        total_upserted += len(batch)
                        LOG.info("Progress: processed %d/%d fetch tasks, upserted total %d rows so far", processed, submitted, total_upserted)
                    except Exception as e:
                        LOG.exception("Batch upsert during parallel fetch failed: %s", e)
                    batch = []
            except Exception as e:
                LOG.exception("Future.result() failed for %s: %s", sym, e)

    if batch:
        try:
            batch_upsert_input(db_main, input_table, batch)
            total_upserted += len(batch)
            LOG.info("Final flush: upserted %d rows", len(batch))
        except Exception as e:
            LOG.exception("Final batch upsert failed: %s", e)

    LOG.info("Parallel fetch completed: total upserted rows = %d into %s", total_upserted, input_table)

# Minimal fetcher CLI/main
def main():
    parser = argparse.ArgumentParser(description="Fetcher: fetch fundamentals into input table.")
    parser.add_argument("--source", required=True, choices=["EOD", "FINNHUB"])
    parser.add_argument("--country", required=True, help="Country (e.g. India, Hong Kong, USA). FINNHUB supports only USA.")
    parser.add_argument("--quarters", type=int, default=DEFAULT_QUARTERS_TO_FETCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE)
    parser.add_argument("--rate-period", type=float, default=DEFAULT_RATE_PERIOD)
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--stale-hours", type=int, default=24)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    try:
        fetch_source_country(
            source=args.source,
            country=args.country,
            api_sleep=DEFAULT_API_SLEEP,
            quarters=int(args.quarters),
            workers=int(args.workers),
            rate=int(args.rate),
            rate_period=float(args.rate_period),
            force_fetch=bool(args.force_fetch),
            stale_hours=int(args.stale_hours)
        )
    except Exception as e:
        LOG.exception("Fetcher main failed: %s", e)
        raise SystemExit(1)

if __name__ == "__main__":
    main()

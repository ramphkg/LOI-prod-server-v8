#!/usr/bin/env python3
"""
ML_daily_incremental_update.py

Daily incremental update job for the continuous-learning swing trading pipeline (Option B).

What it does each run:
- Identifies snapshots whose label horizon has just completed (ScanDate = today - horizon).
- For each symbol in the watchlist:
    - Loads historical price series (via initialize_config CLOSING_PRICES_FUNCTION).
    - Reconstructs the point-in-time snapshot at ScanDate (features computed only using data <= ScanDate).
    - Computes the future outcome label (target-before-stop) using subsequent horizon days.
- Collects new labeled rows into a Parquet batch and saves it into artifact_store.
- Loads the latest online model (SGDClassifier) + transformer bundle (scaler + hasher). If missing, initializes defaults.
- Transforms the new batch into features and calls online_model.partial_fit(X, y, classes=[0,1]).
- Persists the updated online model artifact and writes an update metrics JSON.
- Records artifacts consistently under ../data/<program>/<watchlist>/ using artifact_store.

Notes:
- This is intended to be run daily (e.g., cron) after market data for the new day is available.
- Requires: ta_signals_mc_parallel.py, artifact_store.py, sklearn, pandas, numpy.
- The script is defensive: if there is no prior online model it will create one and partial_fit it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# machine learning
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import csr_matrix, hstack as sparse_hstack

# local project imports
from ta_signals_mc_parallel import initialize_config, get_symbols_forwatchlist, get_technicals, detect_reversal_pro
from app_imports import parallelLoggingSetter, printnlog
import ML_artifact_store as artifact_store

LOG: Optional[logging.Logger] = None


def setup_logger():
    global LOG
    if LOG:
        return LOG
    try:
        LOG = parallelLoggingSetter("ML_incremental_update")
    except Exception:
        LOG = logging.getLogger("ML_incremental_update")
        if not LOG.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return LOG


# -------------------------
# Helpers: labeling & snapshots (copied/adapted from bootstrap)
# -------------------------
def _utcnow_date() -> pd.Timestamp:
    return pd.to_datetime(datetime.utcnow().date())


def compute_future_metrics(prices: pd.DataFrame, idx: int, horizon: int,
                           target_pct: float, stop_pct: float, use_intraday: bool = True) -> Dict[str, Any]:
    """
    Compute future metrics for a snapshot at prices.index[idx].
    Returns dict with label information.
    """
    if idx < 0 or idx >= len(prices):
        return {"max_future_return": np.nan, "min_future_return": np.nan,
                "days_to_target": None, "stop_hit": False, "label_success": 0, "realized_return": np.nan}
    P0 = float(prices['Close'].iat[idx])
    n = len(prices)
    end = min(n - 1, idx + horizon)
    if end <= idx:
        return {"max_future_return": np.nan, "min_future_return": np.nan,
                "days_to_target": None, "stop_hit": False, "label_success": 0, "realized_return": np.nan}

    target_price = P0 * (1.0 + target_pct)
    stop_price = P0 * (1.0 - stop_pct)
    max_ret = -np.inf
    min_ret = np.inf
    first_target_day = None
    first_stop_day = None

    for u, j in enumerate(range(idx + 1, end + 1), start=1):
        if use_intraday and 'High' in prices.columns:
            high = float(prices['High'].iat[j])
            low = float(prices['Low'].iat[j]) if 'Low' in prices.columns else float(prices['Close'].iat[j])
            if first_target_day is None and high >= target_price:
                first_target_day = u
            if first_stop_day is None and low <= stop_price:
                first_stop_day = u
            ret = float(prices['Close'].iat[j]) / P0 - 1.0
        else:
            close = float(prices['Close'].iat[j])
            if first_target_day is None and close >= target_price:
                first_target_day = u
            if first_stop_day is None and close <= stop_price:
                first_stop_day = u
            ret = close / P0 - 1.0
        max_ret = max(max_ret, ret)
        min_ret = min(min_ret, ret)

    max_future_return = float(max_ret) if max_ret != -np.inf else np.nan
    min_future_return = float(min_ret) if min_ret != np.inf else np.nan

    label_success = 0
    realized_return = np.nan
    if first_target_day is not None and (first_stop_day is None or first_target_day <= first_stop_day):
        label_success = 1
        j = idx + first_target_day
        realized_return = float(prices['Close'].iat[j]) / P0 - 1.0
    elif first_stop_day is not None and (first_target_day is None or first_stop_day < first_target_day):
        label_success = 0
        j = idx + first_stop_day
        realized_return = float(prices['Close'].iat[j]) / P0 - 1.0
    else:
        j = end
        realized_return = float(prices['Close'].iat[j]) / P0 - 1.0
        label_success = 1 if (not math.isnan(max_future_return) and max_future_return >= target_pct) else 0

    return {"max_future_return": max_future_return, "min_future_return": min_future_return,
            "days_to_target": int(first_target_day) if first_target_day is not None else None,
            "stop_hit": bool(first_stop_day is not None), "label_success": int(label_success),
            "realized_return": float(realized_return)}


def build_snapshot_from_slice(df_slice: pd.DataFrame, cfg: dict, symbol: str, country_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Compute a point-in-time snapshot dict from df_slice (inclusive).
    Uses get_technicals from ta_signals_mc_parallel. Returns None on failure.
    """
    if df_slice is None or df_slice.empty:
        return None
    try:
        ind_df = get_technicals(df_slice.copy(), cfg)
    except Exception as e:
        setup_logger().warning(f"[snapshot] get_technicals failed for {symbol}: {e}")
        return None
    last = ind_df.tail(1).iloc[0]
    rec = {}
    rec['Symbol'] = symbol
    rec['Date'] = pd.to_datetime(last.get('Date', df_slice['Date'].iat[-1] if 'Date' in df_slice.columns else None))
    rec['TodayPrice'] = float(last.get('Close', np.nan) or np.nan)
    for fld in ['Open', 'High', 'Low', 'Close', 'Volume', 'ADX', 'CCI', 'RSI',
                'EMA20', 'EMA50', 'SMA200', 'TMA21_50_X', 'High52', 'Low52', 'Pct2H52', 'PctfL52']:
        rec[fld] = float(last.get(fld, np.nan)) if last.get(fld, None) is not None else np.nan
    try:
        tr_label = detect_reversal_pro(ind_df)
    except Exception:
        tr_label = None
    rec['TrendReversal_Rules'] = tr_label
    rec['TrendReversal_ML'] = None  # Placeholder for future ML-based reversal detection
    rec['SignalClassifier_Rules'] = int(last.get('SignalClassifier_Rules', 0) or 0)
    rec['SignalClassifier_ML'] = int(last.get('SignalClassifier_ML', 0) or 0)
    
    # Derived features
    ema20_val = rec.get('EMA20')
    if ema20_val and not math.isnan(ema20_val) and ema20_val > 0:
        rec['price_over_ema20'] = rec['TodayPrice'] / ema20_val
    else:
        rec['price_over_ema20'] = np.nan
    
    rec['LastTrendDays'] = int(last.get('LastTrendDays', 0) or 0)
    rec['Primary'] = str(last.get('Primary', '') or '')
    rec['Secondary'] = str(last.get('Secondary', '') or '')
    rec['DITrend'] = str(last.get('DITrend', '') or '')
    rec['MA_Trend'] = str(last.get('MA_Trend', '') or '')
    rec['MADI_Trend'] = str(last.get('MADI_Trend', '') or '')
    rec['CountryName'] = country_name or str(last.get('CountryName', '') or '')
    rec['IndustrySector'] = str(last.get('IndustrySector', '') or '')
    rec['marketCap'] = float(last.get('marketCap', 0.0) or 0.0)
    rec['GEM_Rank'] = str(last.get('GEM_Rank', '') or '')
    return rec


# -------------------------
# Feature transformation helpers
# -------------------------
def transform_features(df_snaps: pd.DataFrame, feature_cfg: Dict[str, Any],
                       hasher: FeatureHasher, scaler: StandardScaler) -> Tuple[Any, Any]:
    """
    Build sparse feature matrix X and label y from snapshot DataFrame using feature configuration,
    hasher and scaler. Returns (X_sparse, y_array).
    """
    numeric_cols = feature_cfg.get("numeric_columns", [])
    categorical_cols = feature_cfg.get("categorical_columns", [])
    # ensure price_over_ema20 exists
    if 'price_over_ema20' not in df_snaps.columns:
        df_snaps['price_over_ema20'] = df_snaps['TodayPrice'] / pd.to_numeric(df_snaps.get('EMA20', pd.Series([np.nan]*len(df_snaps))), errors='coerce')
    # numeric matrix
    num_mat = np.zeros((len(df_snaps), len(numeric_cols)), dtype=float)
    for i, c in enumerate(numeric_cols):
        if c in df_snaps.columns:
            num_mat[:, i] = pd.to_numeric(df_snaps[c], errors='coerce').fillna(0.0).to_numpy()
        else:
            num_mat[:, i] = 0.0
    # partial-fit scaler if needed (fit outside for bootstrap; here we assume scaler provided)
    num_scaled = scaler.transform(num_mat)
    # categorical hashing
    cat_inputs = []
    for _, row in df_snaps.iterrows():
        toks = []
        for col in categorical_cols:
            if col in row and row[col] is not None:
                val = row[col]
                if isinstance(val, (list, tuple)):
                    for v in val:
                        toks.append(f"{col}={v}")
                else:
                    toks.append(f"{col}={val}")
        cat_inputs.append(toks)
    X_cat = hasher.transform(cat_inputs)  # sparse
    # combine numeric + cat
    X_num_sparse = csr_matrix(num_scaled)
    X = sparse_hstack([X_num_sparse, X_cat], format='csr')
    y = df_snaps['label_success'].astype(int).to_numpy() if 'label_success' in df_snaps.columns else None
    return X, y


# -------------------------
# Daily incremental update orchestration
# -------------------------
def daily_incremental_update(source: str, watchlist: str,
                             horizon: int, target: float, stop: float,
                             program_name: str = "swing_buy_recommender",
                             max_symbols: Optional[int] = None):
    log = setup_logger()
    cfg = initialize_config(source)
    closing_prices_fn = cfg.get("CLOSING_PRICES_FUNCTION")
    if closing_prices_fn is None:
        log.error("[daily_update] No closing prices function configured for source %s", source)
        return

    # find target scan date (snapshots whose ScanDate = today - horizon have labels available now)
    today = pd.to_datetime(datetime.utcnow().date())
    scan_date_to_label = today - pd.Timedelta(days=horizon)  # approximate using calendar days; price data uses trading days
    # We will match exact calendar date in prices Date column; if markets skip weekends, many scan dates not present.
    log.info(f"[daily_update] Labeling snapshots with ScanDate = {scan_date_to_label.date()} (horizon={horizon})")

    # list symbols
    try:
        syms_df = get_symbols_forwatchlist(watchlist, cfg)
        symbols = syms_df['symbol'].tolist()
    except Exception as e:
        log.error(f"[daily_update] get_symbols_forwatchlist failed: {e}")
        return
    if max_symbols:
        symbols = symbols[:max_symbols]
    total = len(symbols)
    if total == 0:
        log.info("[daily_update] No symbols found for watchlist %s", watchlist)
        return

    labeled_rows = []
    processed = 0
    for i, sym in enumerate(symbols, start=1):
        try:
            prices = closing_prices_fn(sym)
            if prices is None or prices.empty:
                continue
            # normalize dates and columns
            prices = prices.rename(columns={c: c.title() for c in prices.columns})
            if 'Date' not in prices.columns:
                if hasattr(prices, 'index'):
                    prices = prices.reset_index().rename(columns={prices.index.name or 0: 'Date'})
            prices['Date'] = pd.to_datetime(prices['Date']).dt.normalize()
            prices = prices.sort_values('Date').reset_index(drop=True)
            # find index where Date == scan_date_to_label
            matches = prices.index[prices['Date'] == scan_date_to_label.normalize()]
            if len(matches) == 0:
                # no exact match (market holiday or not traded) -> skip
                continue
            idx = int(matches[0])
            # ensure enough history before idx
            if idx < 10:
                continue
            # slice up to idx inclusive
            df_slice = prices.iloc[: idx + 1].copy()
            snap = build_snapshot_from_slice(df_slice, cfg, sym, country_name=None)
            if snap is None:
                continue
            # compute label using full horizon from prices (we already have prices)
            metrics = compute_future_metrics(prices, idx, horizon=horizon, target_pct=target, stop_pct=stop, use_intraday=True)
            # merge
            snap.update({
                "ScanDate": prices['Date'].iat[idx].to_pydatetime(),
                "horizon": horizon,
                "target_pct": target,
                "stop_pct": stop,
                "max_future_return": metrics["max_future_return"],
                "min_future_return": metrics["min_future_return"],
                "days_to_target": metrics["days_to_target"],
                "stop_hit": metrics["stop_hit"],
                "label_success": int(metrics["label_success"]),
                "realized_return": float(metrics["realized_return"])
            })
            labeled_rows.append(snap)
            processed += 1
        except Exception as e:
            log.warning(f"[daily_update] symbol {sym} processing failed: {e}")
            continue

    if not labeled_rows:
        log.info("[daily_update] No labeled snapshots created for this run (no matching ScanDate or insufficient data).")
        return

    df_new = pd.DataFrame(labeled_rows)
    run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    labeled_name = f"labeled_batch_{run_timestamp}"
    labeled_path = artifact_store.save_dataframe(df_new, labeled_name, program_name, watchlist,
                                                 ext=".parquet", metadata={
                                                     "rows": int(len(df_new)),
                                                     "scan_date_labeled": str(scan_date_to_label.date()),
                                                     "label_config": {"horizon": horizon, "target_pct": target, "stop_pct": stop},
                                                     "created_at": datetime.utcnow().isoformat()
                                                 })
    log.info(f"[daily_update] Saved labeled batch with {len(df_new)} rows to {labeled_path}")

    # Load transformer bundle (scaler + hasher) and feature_config
    feature_cfg_path = artifact_store.latest_artifact_path(program_name, watchlist, name_contains="feature_config", ext=".json")
    transformer_path = artifact_store.latest_artifact_path(program_name, watchlist, name_contains="transformer_bundle", ext=".pkl")
    if feature_cfg_path is None:
        log.error("[daily_update] feature_config not found in artifact store; aborting incremental update.")
        return
    feature_cfg = artifact_store.load_json(feature_cfg_path)

    # prepare hasher and scaler: if transformer bundle exists, load scaler from it
    if transformer_path is not None:
        try:
            bundle = artifact_store.load_model(transformer_path)
            scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
        except Exception as e:
            log.warning(f"[daily_update] failed to load transformer_bundle: {e}")
            scaler = None
    else:
        scaler = None

    # If scaler missing, create new and fit on this small batch (not ideal but workable)
    hasher = FeatureHasher(n_features=feature_cfg.get("feature_hasher_n_features", 4096), input_type='string')
    if scaler is None:
        scaler = StandardScaler()
        # fit scaler on numeric columns of this batch
        numeric_cols = feature_cfg.get("numeric_columns", [])
        num_mat = np.zeros((len(df_new), len(numeric_cols)), dtype=float)
        for i, c in enumerate(numeric_cols):
            if c in df_new.columns:
                num_mat[:, i] = pd.to_numeric(df_new[c], errors='coerce').fillna(0.0).to_numpy()
            else:
                num_mat[:, i] = 0.0
        scaler.partial_fit(num_mat)

    # transform new batch
    try:
        X_batch, y_batch = transform_features(df_new, feature_cfg, hasher, scaler)
    except Exception as e:
        log.error(f"[daily_update] feature transform failed: {e}\n{traceback.format_exc()}")
        return

    # Load latest online model
    online_path = artifact_store.latest_artifact_path(program_name, watchlist, name_contains="model_online", ext=".pkl")
    if online_path is None:
        log.info("[daily_update] No existing online model found; creating a new SGDClassifier.")
        online_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
        # initialize with a single partial_fit call (classes specified)
        try:
            online_model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
        except Exception as e:
            log.error(f"[daily_update] partial_fit failed on new model: {e}")
            return
        updated = True
        prev_model_name = None
    else:
        try:
            online_model = artifact_store.load_model(online_path)
            # partial_fit on batch
            online_model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
            updated = True
            prev_model_name = online_path.name
        except Exception as e:
            log.error(f"[daily_update] Failed to load/partial_fit online model: {e}\n{traceback.format_exc()}")
            return

    # Evaluate on the new batch (post-update) for quick metrics
    try:
        probs = online_model.predict_proba(X_batch)[:, 1] if hasattr(online_model, "predict_proba") else online_model.decision_function(X_batch)
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "n_rows": int(len(y_batch)),
            "precision": float(precision_score(y_batch, preds, zero_division=0)),
            "recall": float(recall_score(y_batch, preds, zero_division=0)),
            "f1": float(f1_score(y_batch, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_batch, probs)) if len(np.unique(y_batch)) > 1 else None
        }
    except Exception as e:
        log.warning(f"[daily_update] Evaluation on batch failed: {e}")
        metrics = {"n_rows": int(len(y_batch)), "error": str(e)}

    # Persist updated online model
    new_model_name = f"model_online_update_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    try:
        saved_path = artifact_store.save_model(online_model, new_model_name, program_name, watchlist,
                                              metadata={"updated_at": datetime.utcnow().isoformat(), "prev_model": prev_model_name, "rows": int(len(y_batch)), "metrics": metrics})
        log.info(f"[daily_update] Saved updated online model to {saved_path}")
    except Exception as e:
        log.error(f"[daily_update] Failed to save updated model: {e}\n{traceback.format_exc()}")
        return

    # Save update metrics to artifact store
    metrics_name = f"online_update_metrics_{datetime.utcnow().strftime('%Y%m%d')}"
    try:
        artifact_store.save_json(metrics, metrics_name, program_name, watchlist, ext=".json", metadata={"labeled_batch": labeled_path.name, "model_saved": saved_path.name})
    except Exception as e:
        log.warning(f"[daily_update] Failed to save metrics JSON: {e}")

    log.info(f"[daily_update] Processed {len(df_new)} labeled snapshots and updated online model. Metrics: {metrics}")


def parse_args():
    p = argparse.ArgumentParser(description="Daily incremental updater (partial_fit) for swing recommender")
    p.add_argument("--source", required=True, help="Price source (FINNHUB | FINNHUB_LOCAL | EOD | EOD_LOCAL)")
    p.add_argument("--watchlist", required=True, help="Watchlist code (e.g., US01)")
    p.add_argument("--horizon", type=int, default=10, help="Label horizon in trading days (N)")
    p.add_argument("--target", type=float, default=0.08, help="Target return threshold (R)")
    p.add_argument("--stop", type=float, default=0.06, help="Stop loss threshold (S)")
    p.add_argument("--max-symbols", type=int, default=None, help="Limit symbols processed (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()
    try:
        daily_incremental_update(
            source=args.source,
            watchlist=args.watchlist,
            horizon=args.horizon,
            target=args.target,
            stop=args.stop,
            program_name="swing_buy_recommender",
            max_symbols=args.max_symbols
        )
    except Exception as e:
        LOG = setup_logger()
        LOG.error(f"[daily_incremental_update] Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

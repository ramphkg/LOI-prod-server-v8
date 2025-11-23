#!/usr/bin/env python3
"""
ML_bootstrap.py

Bootstrap training script (Option A) for the continuous-learning swing trading pipeline.

What it does
- Builds a labeled dataset from historical price series:
    - For each symbol in the chosen watchlist, iterates historical dates,
    - Computes point-in-time features using ta_signals_mc_parallel.get_technicals() and helpers,
    - Computes future outcome labels over a horizon N using intraday High/Low when available (target-before-stop logic).
- Trains two models:
    - online_model: SGDClassifier (logistic, supports partial_fit) — used for fast incremental updates,
    - batch_model: LightGBM (if available) or LogisticRegression as fallback — used for weekly/full retrain.
- Persists artifacts and metadata into a consistent artifact folder via artifact_store:
    ../data/<program>/<watchlist>/
    (models, feature_config.json, labeled_data.parquet, train_metrics.json, production_pointer.json)
- This bootstrap run is intended to be run once to create initial artifacts before enabling daily incremental updates.

Usage example:
    python bootstrap.py --source FINNHUB_LOCAL --watchlist US01 --horizon 10 --target 0.08 --stop 0.06 \
        --train-years 2 --max-symbols 200

Notes:
- Requires the project modules: ta_signals_mc_parallel (for price access and indicator computations),
  artifact_store (the common storage utility), app_imports.getDbConnection etc.
- Training can be slow for large universes; use --max-symbols and --train-years to limit scope for the bootstrap.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import csr_matrix
from typing import Any

# local project imports
from ta_signals_mc_parallel import initialize_config, get_symbols_forwatchlist, get_technicals, detect_reversal_pro
from app_imports import parallelLoggingSetter, printnlog
import artifact_store

# sklearn & lightgbm (optional)
try:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
except Exception as e:
    raise RuntimeError("scikit-learn is required for bootstrap.py") from e

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# -------------------------
# Logging
# -------------------------
LOG = None


def setup_logger():
    global LOG
    if LOG:
        return LOG
    try:
        LOG = parallelLoggingSetter("ML_bootstrap")
    except Exception:
        LOG = logging.getLogger("bootstrap")
        if not LOG.handlers:
            logging.basicConfig(level=logging.INFO)
    return LOG


# -------------------------
# Labeling utilities
# -------------------------
def compute_future_metrics(prices: pd.DataFrame, idx: int, horizon: int = 10,
                           target_pct: float = 0.08, stop_pct: float = 0.06,
                           use_intraday: bool = True) -> Dict[str, object]:
    """
    Compute future metrics for a snapshot at prices.index[idx].
    prices: DataFrame with ascending dates, columns: Close and optionally High, Low.
    Returns dict with keys: max_future_return, min_future_return, days_to_target, stop_hit, label_success, realized_return
    """
    if idx < 0 or idx >= len(prices):
        return {
            "max_future_return": np.nan,
            "min_future_return": np.nan,
            "days_to_target": None,
            "stop_hit": False,
            "label_success": 0,
            "realized_return": np.nan
        }

    P0 = float(prices['Close'].iat[idx])
    n = len(prices)
    end = min(n - 1, idx + horizon)
    if end <= idx:
        return {
            "max_future_return": np.nan,
            "min_future_return": np.nan,
            "days_to_target": None,
            "stop_hit": False,
            "label_success": 0,
            "realized_return": np.nan
        }

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

        if ret > max_ret:
            max_ret = ret
        if ret < min_ret:
            min_ret = ret

    if max_ret == -np.inf:
        max_ret = np.nan
    if min_ret == np.inf:
        min_ret = np.nan

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
        label_success = 1 if (not math.isnan(max_ret) and max_ret >= target_pct) else 0

    return {
        "max_future_return": float(max_ret) if not math.isnan(max_ret) else np.nan,
        "min_future_return": float(min_ret) if not math.isnan(min_ret) else np.nan,
        "days_to_target": int(first_target_day) if first_target_day is not None else None,
        "stop_hit": bool(first_stop_day is not None),
        "label_success": int(label_success),
        "realized_return": float(realized_return)
    }


# -------------------------
# Snapshot builder (point-in-time features)
# -------------------------
def build_snapshot_from_slice(df_slice: pd.DataFrame, cfg: dict, symbol: str, country_name: Optional[str] = None) -> Optional[pd.Series]:
    """
    Compute point-in-time snapshot row (as Series) from df_slice (prices up to and including snapshot date).
    Re-uses get_technicals() and detect_reversal_pro() from ta_signals_mc_parallel where available.
    Returns Series with canonical columns (subset) or None on failure.
    """
    if df_slice is None or df_slice.empty:
        return None
    try:
        # compute technicals (this function is expected to exist and be robust)
        ind_df = get_technicals(df_slice.copy(), cfg)
    except Exception as e:
        setup_logger().warning(f"[build_snapshot] get_technicals failed for {symbol}: {e}")
        return None

    last = ind_df.tail(1).iloc[0]

    row = {}
    # fill a subset of production features (the model can use a defined subset)
    row['Symbol'] = symbol
    row['Date'] = pd.to_datetime(last.get('Date', df_slice.index[-1] if df_slice.index is not None else None))
    row['TodayPrice'] = float(last.get('Close', np.nan) or np.nan)
    for fld in ['Open', 'High', 'Low', 'Close', 'Volume', 'ADX', 'CCI', 'RSI',
                'EMA20', 'EMA50', 'SMA200', 'TMA21_50_X', 'High52', 'Low52', 'Pct2H52', 'PctfL52']:
        row[fld] = float(last.get(fld, np.nan)) if last.get(fld, None) is not None else np.nan

    # classifier & reversal labels if functions present
    try:
        tr_label = detect_reversal_pro(ind_df)
    except Exception:
        tr_label = None
    row['TrendReversal_Rules'] = tr_label
    row['TrendReversal_ML'] = None  # Placeholder for future ML-based reversal detection

    # SignalClassifier fields may be computed by get_technicals or separate; attempt safe retrieval
    row['SignalClassifier_Rules'] = int(last.get('SignalClassifier_Rules', 0) or 0)
    row['SignalClassifier_ML'] = int(last.get('SignalClassifier_ML', 0) or 0)
    
    # Derived features
    ema20_val = row.get('EMA20')
    if ema20_val and not math.isnan(ema20_val) and ema20_val > 0:
        row['price_over_ema20'] = row['TodayPrice'] / ema20_val
    else:
        row['price_over_ema20'] = np.nan

    # LastTrendDays
    row['LastTrendDays'] = int(last.get('LastTrendDays', 0) or 0)
    row['Primary'] = str(last.get('Primary', '') or '')
    row['Secondary'] = str(last.get('Secondary', '') or '')
    row['DITrend'] = str(last.get('DITrend', '') or '')
    row['MA_Trend'] = str(last.get('MA_Trend', '') or '')

    # country/industry metadata
    row['CountryName'] = country_name or str(last.get('CountryName', '') or '')
    row['IndustrySector'] = str(last.get('IndustrySector', '') or '')
    row['marketCap'] = float(last.get('marketCap', 0.0) or 0.0)
    row['GEM_Rank'] = str(last.get('GEM_Rank', '') or '')

    return pd.Series(row)


# -------------------------
# Training helpers
# -------------------------
def make_feature_matrices(df_snapshots: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                          hasher: FeatureHasher, scaler: Optional[StandardScaler] = None) -> Tuple[csr_matrix, np.ndarray, List[List[str]]]:
    """
    Build numeric and hashed categorical matrices for a DataFrame of snapshots.
    Returns (X_sparse, y_array, cat_inputs)
      - X_sparse: scipy csr sparse (numeric scaled + hashed categorical)
      - y_array: labels array (if 'label_success' in df_snapshots), else None
      - cat_inputs: list of list-of-strings passed to FeatureHasher (useful for introspection)
    """
    # Prepare numeric matrix
    num_mat = np.zeros((len(df_snapshots), len(numeric_cols)), dtype=float)
    for i, c in enumerate(numeric_cols):
        if c in df_snapshots.columns:
            num_mat[:, i] = pd.to_numeric(df_snapshots[c], errors="coerce").fillna(0.0).to_numpy()
        else:
            num_mat[:, i] = 0.0

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(num_mat)
    num_scaled = scaler.transform(num_mat)
    num_sparse = csr_matrix(num_scaled)

    # Prepare categorical hashed features as list of iterable-of-strings per sample
    cat_inputs: List[List[str]] = []
    for _, row in df_snapshots.iterrows():
        toks: List[str] = []
        for col in categorical_cols:
            if col in row and row[col] is not None:
                val = row[col]
                # flatten lists/tuples if needed
                if isinstance(val, (list, tuple)):
                    for v in val:
                        toks.append(f"{col}={str(v)}")
                else:
                    toks.append(f"{col}={str(val)}")
        cat_inputs.append(toks)

    # Transform categorical via FeatureHasher (input_type='string' expects list of strings per row)
    X_cat = hasher.transform(cat_inputs)  # returns sparse matrix

    # Combine numeric + categorical
    X = sparse_hstack([num_sparse, X_cat], format="csr")

    y = None
    if 'label_success' in df_snapshots.columns:
        y = df_snapshots['label_success'].astype(int).to_numpy()

    return X, y, cat_inputs, scaler


def train_online_model(X_csr, y, classes=(0, 1), random_state: int = 42) -> Tuple[SGDClassifier, Dict]:
    """
    Train (or fit) an SGDClassifier with partial_fit (one-shot here).
    Returns model and training metrics dict.
    """
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-4, random_state=random_state)
    # partial_fit requires classes known
    model.partial_fit(X_csr, y, classes=classes)
    # metrics on training set (quick)
    probs = model.predict_proba(X_csr)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "train_samples": int(len(y)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None
    }
    return model, metrics


def train_batch_model(X_csr, y, use_lightgbm: bool = True, random_state: int = 42) -> Tuple[Optional[Any], Dict]:
    """
    Train a stronger batch model (LightGBM if available, otherwise LogisticRegression).
    Returns model and metrics.
    """
    if use_lightgbm and lgb is not None:
        dtrain = lgb.Dataset(X_csr, label=y, free_raw_data=False)
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": random_state,
            "num_leaves": 31,
            "learning_rate": 0.05,
        }
        # small training rounds for bootstrap; later full retrain will increase rounds
        num_round = 200
        bst = lgb.train(params, dtrain, num_boost_round=num_round)
        probs = bst.predict(X_csr)
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "train_samples": int(len(y)),
            "auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None,
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0))
        }
        return bst, metrics
    else:
        # fallback to LogisticRegression
        clf = LogisticRegression(solver='liblinear', random_state=random_state, max_iter=200)
        clf.fit(X_csr, y)
        probs = clf.predict_proba(X_csr)[:, 1]
        preds = clf.predict(X_csr)
        metrics = {
            "train_samples": int(len(y)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None
        }
        return clf, metrics


# -------------------------
# Orchestration: build dataset & train
# -------------------------
def bootstrap_train(source: str, watchlist: str,
                    horizon: int = 10, target: float = 0.08, stop: float = 0.06,
                    train_years: int = 2, max_symbols: Optional[int] = 200,
                    program_name: str = "swing_buy_recommender"):
    """
    Build labeled dataset and train initial models. Save artifacts via artifact_store.
    """
    log = setup_logger()
    cfg = initialize_config(source)
    closing_prices_fn = cfg.get("CLOSING_PRICES_FUNCTION")
    if closing_prices_fn is None:
        raise RuntimeError("No CLOSING_PRICES_FUNCTION in config for source " + source)

    # get symbols from watchlist
    try:
        symbols_df = get_symbols_forwatchlist(watchlist, cfg)
        symbols = symbols_df['symbol'].tolist()
    except Exception as e:
        log.error(f"[bootstrap] get_symbols_forwatchlist failed: {e}")
        raise

    if max_symbols:
        symbols = symbols[:max_symbols]

    # date window
    today = pd.to_datetime("today").normalize()
    start_date = (today - pd.DateOffset(years=train_years)).date()

    rows = []
    total = len(symbols)
    log.info(f"[bootstrap] Building labeled dataset for {len(symbols)} symbols from {start_date} to {today.date()}")

    for i, sym in enumerate(symbols, start=1):
        try:
            prices = closing_prices_fn(sym)
            if prices is None or prices.empty:
                log.debug(f"[bootstrap] no prices for {sym}")
                continue
            # normalize columns & sort ascending
            prices = prices.rename(columns={c: c.title() for c in prices.columns})
            if 'Date' not in prices.columns:
                # if index is date-like
                if hasattr(prices, 'index'):
                    prices = prices.reset_index().rename(columns={prices.index.name or 0: 'Date'})
            prices['Date'] = pd.to_datetime(prices['Date'])
            prices = prices.sort_values('Date').reset_index(drop=True)
            # filter start_date..today
            prices = prices[prices['Date'].dt.date >= start_date]
            if prices.shape[0] <= horizon:
                continue

            # for each possible snapshot index (ensure horizon ahead exists)
            # to limit work sample every k days (e.g., every 1 day)
            for idx in range(0, len(prices) - horizon):
                try:
                    df_slice = prices.iloc[: idx + 1].copy()
                    if len(df_slice) < 20:  # skip too short windows
                        continue
                    snap_series = build_snapshot_from_slice(df_slice, cfg, sym, country_name=None)
                    if snap_series is None:
                        continue
                    # compute labels using full future window from prices
                    metrics = compute_future_metrics(prices, idx, horizon=horizon,
                                                     target_pct=target, stop_pct=stop, use_intraday=True)
                    # merge features + label
                    record = snap_series.to_dict()
                    record.update({
                        "ScanDate": pd.to_datetime(record.get("Date")).to_pydatetime(),
                        "horizon_days": horizon,
                        "target_pct": target,
                        "stop_pct": stop,
                        "max_future_return": metrics["max_future_return"],
                        "min_future_return": metrics["min_future_return"],
                        "days_to_target": metrics["days_to_target"],
                        "stop_hit": metrics["stop_hit"],
                        "label_success": int(metrics["label_success"]),
                        "realized_return": metrics["realized_return"]
                    })
                    rows.append(record)
                except Exception as e:
                    log.debug(f"[bootstrap:{sym}] snapshot idx {idx} failed: {e}")
                    continue

            log.info(f"[bootstrap] [{i}/{total}] symbol {sym} -> collected snapshots so far: {len(rows)}")
        except Exception as e:
            log.warning(f"[bootstrap] failed symbol {sym}: {e}\n{traceback.format_exc()}")
            continue

    if not rows:
        log.error("[bootstrap] No labeled rows produced; aborting")
        return

    df_labeled = pd.DataFrame(rows)
    # save labeled dataset
    labeled_name = f"labeled_data_{train_years}y_{datetime.utcnow().strftime('%Y%m%d')}"
    labeled_path = artifact_store.save_dataframe(df_labeled, labeled_name, program_name, watchlist,
                                                 ext=".parquet", metadata={
                                                     "rows": int(len(df_labeled)),
                                                     "label_config": {"horizon": horizon, "target_pct": target, "stop_pct": stop},
                                                     "generated_at": datetime.utcnow().isoformat()
                                                 })
    log.info(f"[bootstrap] Saved labeled dataset to {labeled_path} ({len(df_labeled)} rows)")

    # feature configuration
    numeric_cols = ['ADX', 'RSI', 'EMA20', 'EMA50', 'SMA200', 'price_over_ema20', 'Volume', 'LastTrendDays',
                    'Pct2H52', 'PctfL52', 'marketCap']
    categorical_cols = ['TrendReversal_Rules', 'TrendReversal_ML', 'DITrend', 'MA_Trend', 'MADI_Trend',
                        'Primary', 'Secondary', 'IndustrySector', 'GEM_Rank']
    feature_config = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_hasher_n_features": 4096,
        "label_column": "label_success",
        "timestamp": datetime.utcnow().isoformat()
    }
    artifact_store.save_json(feature_config, "feature_config_v1", program_name, watchlist,
                             ext=".json", metadata={"note": "bootstrap feature config"})

    # Build features (numeric + hashed categorical)
    hasher = FeatureHasher(n_features=feature_config["feature_hasher_n_features"], input_type='string')
    # ensure price_over_ema20 exists
    if 'price_over_ema20' not in df_labeled.columns:
        df_labeled['price_over_ema20'] = df_labeled['TodayPrice'] / pd.to_numeric(df_labeled.get('EMA20', pd.Series([np.nan]*len(df_labeled))), errors='coerce')

    # Build matrices
    X, y, cat_inputs, fitted_scaler = None, None, None, None
    try:
        X, y, cat_inputs, fitted_scaler = make_feature_matrices(df_labeled, numeric_cols, categorical_cols, hasher, scaler=None)
    except Exception as e:
        log.error(f"[bootstrap] feature matrix construction failed: {e}\n{traceback.format_exc()}")
        return

    # Train/validation split for proper evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log.info(f"[bootstrap] Train size: {len(y_train)}, Val size: {len(y_val)}")
    
    # Train online model
    online_model, online_train_metrics = train_online_model(X_train, y_train, classes=(0, 1))
    # Validate on holdout set
    val_probs_online = online_model.predict_proba(X_val)[:, 1]
    val_preds_online = (val_probs_online >= 0.5).astype(int)
    online_val_metrics = {
        "val_precision": float(precision_score(y_val, val_preds_online, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_preds_online, zero_division=0)),
        "val_f1": float(f1_score(y_val, val_preds_online, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs_online)) if len(np.unique(y_val)) > 1 else None
    }
    online_metrics = {**online_train_metrics, **online_val_metrics}
    online_meta = {
        "kind": "online",
        "created_at": datetime.utcnow().isoformat(),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "label_config": {"horizon": horizon, "target_pct": target, "stop_pct": stop}
    }
    online_path = artifact_store.save_model(online_model, "model_online_v1", program_name, watchlist, metadata=online_meta)
    log.info(f"[bootstrap] Saved online model to {online_path} with val metrics: {online_val_metrics}")
    artifact_store.save_json({"metrics": online_metrics}, "online_train_metrics_v1", program_name, watchlist)

    # Train batch model
    batch_model, batch_train_metrics = train_batch_model(X_train, y_train, use_lightgbm=(lgb is not None))
    # Validate batch model
    if lgb is not None and hasattr(batch_model, 'predict'):
        val_probs_batch = batch_model.predict(X_val)
    else:
        val_probs_batch = batch_model.predict_proba(X_val)[:, 1] if hasattr(batch_model, 'predict_proba') else batch_model.predict(X_val)
    val_preds_batch = (val_probs_batch >= 0.5).astype(int)
    batch_val_metrics = {
        "val_precision": float(precision_score(y_val, val_preds_batch, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_preds_batch, zero_division=0)),
        "val_f1": float(f1_score(y_val, val_preds_batch, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs_batch)) if len(np.unique(y_val)) > 1 else None
    }
    batch_metrics = {**batch_train_metrics, **batch_val_metrics}
    batch_meta = {
        "kind": "batch",
        "created_at": datetime.utcnow().isoformat(),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "label_config": {"horizon": horizon, "target_pct": target, "stop_pct": stop},
        "framework": "lightgbm" if lgb is not None else "sklearn_logistic"
    }
    batch_path = artifact_store.save_model(batch_model, "model_batch_v1", program_name, watchlist, metadata=batch_meta)
    log.info(f"[bootstrap] Saved batch model to {batch_path} with val metrics: {batch_val_metrics}")
    artifact_store.save_json({"metrics": batch_metrics}, "batch_train_metrics_v1", program_name, watchlist)

    # Save scaler & hasher & feature_config as a single object for inference pipeline
    transformer_bundle = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "hasher_n_features": feature_config["feature_hasher_n_features"],
        "scaler": fitted_scaler,  # scaler object (StandardScaler)
        "hasher": hasher  # Save actual hasher object for consistent feature hashing
    }
    artifact_store.save_model(transformer_bundle, "transformer_bundle_v1", program_name, watchlist,
                              metadata={"note": "scaler + hasher for inference"})

    # Save production pointer (explicitly reference artifacts to use for production)
    production_pointer = {
        "production_online": os.path.basename(str(online_path)),
        "production_batch": os.path.basename(str(batch_path)),
        "feature_config": "feature_config_v1.json",
        "saved_at": datetime.utcnow().isoformat()
    }
    artifact_store.save_json(production_pointer, "production_pointer_v1", program_name, watchlist,
                             ext=".json", metadata={"note": "explicit production artifact pointer"})

    log.info("[bootstrap] Bootstrap complete. Artifacts saved under artifact store.")
    return {
        "labeled_path": str(labeled_path),
        "online_model_path": str(online_path),
        "batch_model_path": str(batch_path),
        "feature_config": feature_config,
        "online_metrics": online_metrics,
        "batch_metrics": batch_metrics
    }


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap initial labeled dataset and models for swing recommender")
    p.add_argument("--source", required=True, help="Price source: FINNHUB | FINNHUB_LOCAL | EOD | EOD_LOCAL")
    p.add_argument("--watchlist", required=True, help="Watchlist code (e.g., US01)")
    p.add_argument("--horizon", type=int, default=10, help="Label horizon in trading days")
    p.add_argument("--target", type=float, default=0.08, help="Target return threshold (e.g., 0.08 = 8%)")
    p.add_argument("--stop", type=float, default=0.06, help="Stop loss threshold (e.g., 0.06 = 6%)")
    p.add_argument("--train-years", type=int, default=2, help="How many historical years to build labeled dataset from")
    p.add_argument("--max-symbols", type=int, default=200, help="Limit number of symbols to process (for bootstrap speed)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()
    try:
        out = bootstrap_train(
            source=args.source,
            watchlist=args.watchlist,
            horizon=args.horizon,
            target=args.target,
            stop=args.stop,
            train_years=args.train_years,
            max_symbols=args.max_symbols,
            program_name="swing_buy_recommender"
        )
        print("Bootstrap finished. Summary:")
        print(json.dumps(out, indent=2, default=str))
    except Exception as e:
        LOG = setup_logger()
        LOG.error(f"Bootstrap failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

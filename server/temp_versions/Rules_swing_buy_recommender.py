#!/usr/bin/env python3
"""
Rules_swing_buy_recommender.py

Model-backed daily swing-buy recommender.

- Loads artifacts from artifact_store (feature_config, transformer_bundle, model_online, model_batch).
- Reads today's snapshot table determined by initialize_config(source) and optional country filter.
- Builds point-in-time features (safe defaults if columns missing).
- Transforms features using transformer bundle (scaler + hasher) if present, otherwise falls back to an on-the-fly transform.
- Predicts with online and batch models, ensembles probabilities and returns top-K recommendations.
- Saves daily predictions artifact into ../data/swing_buy_recommender/<watchlist>/ via artifact_store.

Usage:
  python swing_buy_recommender_ML.py --source FINNHUB_LOCAL --watchlist US_ALL --country USA --top-k 20

Dependencies:
- artifact_store.py (placed in repo)
- ta_signals_mc_parallel.initialize_config
- app_imports.getDbConnection, loggingSetter, strUtcNow
- pandas, numpy, sklearn (FeatureHasher, StandardScaler) recommended
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from scipy.sparse import csr_matrix, hstack as sparse_hstack

# Local project utilities (must be on PYTHONPATH)
import ML_artifact_store as artifact_store
from ta_signals_mc_parallel import initialize_config
from app_imports import getDbConnection, loggingSetter, strUtcNow

# Optional sklearn imports (FeatureHasher + StandardScaler)
try:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.preprocessing import StandardScaler
except Exception:
    FeatureHasher = None
    StandardScaler = None

# -------------------------
# Logger
# -------------------------
logger: Optional[logging.Logger] = None


def setup_logger(name: str = "ML_swing_buy_recommender") -> logging.Logger:
    global logger
    if logger:
        return logger
    try:
        logger = loggingSetter(name)
    except Exception:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
    return logger


# -------------------------
# DB helpers (SQLAlchemy 2.x safe)
# -------------------------
def get_table_columns(table_name: str) -> List[str]:
    log = setup_logger()
    try:
        with getDbConnection() as con:
            stmt = text("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :tname
                ORDER BY ORDINAL_POSITION
            """)
            res = con.execute(stmt, {"tname": table_name})
            rows = res.fetchall()
            cols = [r[0] for r in rows] if rows else []
            return cols
    except Exception:
        log.debug("[get_table_columns] INFORMATION_SCHEMA failed; falling back to SELECT LIMIT 0")
        try:
            with getDbConnection() as con:
                df0 = pd.read_sql_query(text(f"SELECT * FROM `{table_name}` LIMIT 0"), con=con)
                return list(df0.columns)
        except Exception:
            log.exception("[get_table_columns] fallback failed")
            return []


def read_snapshot_data(table_name: str,
                       country_filter: Optional[str] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
    """
    Read snapshot rows from table_name. Only apply CountryName filter if column exists.
    Uses sqlalchemy.text and passes params as a mapping to avoid SQLAlchemy param issues.
    """
    log = setup_logger()
    cols = get_table_columns(table_name)
    if not cols:
        raise RuntimeError(f"Cannot determine columns for table {table_name}")

    where_clauses: List[str] = []
    params: Dict[str, Any] = {}

    if country_filter and "CountryName" in cols:
        where_clauses.append("`CountryName` LIKE :country")
        params["country"] = f"%{country_filter}%"
    elif country_filter:
        log.debug(f"[read_snapshot_data] Table {table_name} has no CountryName; ignoring country_filter")

    q = f"SELECT * FROM `{table_name}`"
    if where_clauses:
        q += " WHERE " + " AND ".join(where_clauses)
    if "ScanDate" in cols:
        q += " ORDER BY `ScanDate` DESC, `Symbol` ASC"
    else:
        q += " ORDER BY `Symbol` ASC"
    if limit and isinstance(limit, int):
        q += f" LIMIT {int(limit)}"

    sql_text = text(q)
    with getDbConnection() as con:
        try:
            df = pd.read_sql_query(sql_text, con=con, params=params)
        except Exception:
            # fallback manual execute
            try:
                res = con.execute(sql_text, params) if params else con.execute(sql_text)
                try:
                    rows = res.mappings().all()
                    df = pd.DataFrame(rows)
                except Exception:
                    rows = res.fetchall()
                    df = pd.DataFrame(rows, columns=res.keys() if hasattr(res, "keys") else None)
            except Exception as e:
                log.error(f"[read_snapshot_data] Manual execute failed: {e}\nSQL: {q}\nparams: {params}")
                raise
    df.columns = [str(c) for c in df.columns]
    log.info(f"[read_snapshot_data] Read {len(df)} rows from {table_name}")
    return df


# -------------------------
# Feature engineering
# -------------------------
DEFAULT_NUMERIC = [
    "ADX", "RSI", "EMA20", "EMA50", "SMA200", "TodayPrice", "Volume",
    "High52", "Low52", "Pct2H52", "PctfL52", "LastTrendDays", "marketCap"
]
DEFAULT_CATEGORICAL = [
    "TrendReversal_Rules", "TrendReversal_ML", "DITrend", "MA_Trend",
    "MADI_Trend", "Primary", "Secondary", "IndustrySector", "GEM_Rank"
]


def synthesize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive feature synthesis used by model transforms and fallback rules.
    """
    X = df.copy()
    # numeric coercion
    for c in DEFAULT_NUMERIC:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = np.nan

    # price_over_ema20
    if "TodayPrice" in X.columns and "EMA20" in X.columns:
        X["price_over_ema20"] = X["TodayPrice"] / X["EMA20"].replace({0: np.nan})
    else:
        X["price_over_ema20"] = np.nan

    # ema_struct
    X["ema_struct"] = np.nan
    if {"EMA20", "EMA50", "SMA200"}.issubset(X.columns):
        cond_best = (X["EMA20"] > X["EMA50"]) & (X["EMA50"] > X["SMA200"])
        cond_mid = (X["EMA20"] > X["EMA50"])
        cond_worse = (X["EMA50"] > X["EMA20"])
        X.loc[cond_best.fillna(False), "ema_struct"] = 2
        X.loc[(cond_mid & ~cond_best).fillna(False), "ema_struct"] = 1
        X.loc[cond_worse.fillna(False), "ema_struct"] = -1

    # categorical safe strings
    for c in DEFAULT_CATEGORICAL:
        X[c] = X[c].fillna("").astype(str) if c in X.columns else ""

    # LastTrendDays safe int
    if "LastTrendDays" in X.columns:
        X["LastTrendDays"] = pd.to_numeric(X["LastTrendDays"], errors="coerce").fillna(0).astype(int)
    else:
        X["LastTrendDays"] = 0

    # Volume safe
    if "Volume" in X.columns:
        X["Volume"] = pd.to_numeric(X["Volume"], errors="coerce").fillna(0)
    else:
        X["Volume"] = 0

    return X


# -------------------------
# Artifacts loader & transforms
# -------------------------
def load_artifacts(program: str, watchlist: str) -> Dict[str, Any]:
    log = setup_logger()
    out: Dict[str, Any] = {"feature_config": None, "transformer": None, "online_model": None, "batch_model": None, "paths": {}}

    # feature_config (exclude .meta.json files)
    all_fcfg = artifact_store.list_artifacts(program, watchlist, pattern="feature_config", ext=".json")
    fcfg_candidates = [p for p in all_fcfg if not p.name.endswith(".meta.json")]
    fcfg_path = fcfg_candidates[0] if fcfg_candidates else None
    if fcfg_path:
        try:
            out["feature_config"] = artifact_store.load_json(fcfg_path)
            out["paths"]["feature_config"] = str(fcfg_path)
            log.info(f"[load_artifacts] loaded feature_config: {fcfg_path.name}")
        except Exception:
            log.exception("[load_artifacts] failed to load feature_config")

    tb_path = artifact_store.latest_artifact_path(program, watchlist, name_contains="transformer_bundle", ext=".pkl")
    if tb_path:
        try:
            out["transformer"] = artifact_store.load_model(tb_path)
            out["paths"]["transformer"] = str(tb_path)
            log.info(f"[load_artifacts] loaded transformer_bundle: {tb_path.name}")
        except Exception:
            log.exception("[load_artifacts] failed to load transformer_bundle")

    online_p = artifact_store.latest_artifact_path(program, watchlist, name_contains="model_online", ext=".pkl")
    if online_p:
        try:
            out["online_model"] = artifact_store.load_model(online_p)
            out["paths"]["online_model"] = str(online_p)
            log.info(f"[load_artifacts] loaded online model: {online_p.name}")
        except Exception:
            log.exception("[load_artifacts] failed to load online_model")

    batch_p = artifact_store.latest_artifact_path(program, watchlist, name_contains="model_batch", ext=".pkl")
    if batch_p:
        try:
            out["batch_model"] = artifact_store.load_model(batch_p)
            out["paths"]["batch_model"] = str(batch_p)
            log.info(f"[load_artifacts] loaded batch model: {batch_p.name}")
        except Exception:
            log.exception("[load_artifacts] failed to load batch_model")

    return out


def transform_features(df: pd.DataFrame, feature_config: Optional[Dict[str, Any]], transformer: Optional[Dict[str, Any]]):
    """
    Transform features to model input matrix X (sparse). Returns (X, used_scaler, used_hasher).
    If transformer bundle contains scaler/hasher objects, use them. Otherwise create and fit on this batch (fallback).
    """
    log = setup_logger()
    numeric_cols = feature_config.get("numeric_columns", DEFAULT_NUMERIC) if feature_config else DEFAULT_NUMERIC
    categorical_cols = feature_config.get("categorical_columns", DEFAULT_CATEGORICAL) if feature_config else DEFAULT_CATEGORICAL
    n_features = feature_config.get("feature_hasher_n_features", 4096) if feature_config else 4096

    # ensure price_over_ema20 exists
    if "price_over_ema20" not in df.columns:
        df["price_over_ema20"] = df["TodayPrice"] / pd.to_numeric(df.get("EMA20", pd.Series([np.nan]*len(df))), errors="coerce")

    # numeric matrix
    num_mat = np.zeros((len(df), len(numeric_cols)), dtype=float)
    for i, c in enumerate(numeric_cols):
        if c in df.columns:
            num_mat[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
        else:
            num_mat[:, i] = 0.0

    # categorical tokens
    cat_inputs: List[List[str]] = []
    for _, row in df.iterrows():
        toks: List[str] = []
        for c in categorical_cols:
            val = row.get(c, "")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            if isinstance(val, (list, tuple)):
                for v in val:
                    toks.append(f"{c}={v}")
            else:
                toks.append(f"{c}={val}")
        cat_inputs.append(toks)

    # use transformer bundle if present
    scaler = None
    hasher = None
    if transformer and isinstance(transformer, dict):
        scaler = transformer.get("scaler", None)
        hasher = transformer.get("hasher", None)

    # create fallback hasher/scaler if missing
    if hasher is None:
        if FeatureHasher is None:
            raise RuntimeError("FeatureHasher not available; install scikit-learn")
        hasher = FeatureHasher(n_features=n_features, input_type="string")
    if scaler is None:
        if StandardScaler is None:
            raise RuntimeError("StandardScaler not available; install scikit-learn")
        scaler = StandardScaler().fit(num_mat) if num_mat.shape[1] > 0 else StandardScaler().fit(np.zeros((len(df), 1)))

    # apply transforms
    num_scaled = scaler.transform(num_mat) if num_mat.shape[1] > 0 else np.zeros((len(df), 0))
    X_cat = hasher.transform(cat_inputs)

    # combine numeric + cat into sparse matrix
    X_num = csr_matrix(num_scaled) if num_scaled.size else csr_matrix((len(df), 0))
    X = sparse_hstack([X_num, X_cat], format="csr")
    return X, scaler, hasher


# -------------------------
# Prediction helpers
# -------------------------
def predict_probs(model: Any, X):
    """
    Return probability of positive class for various model types.
    """
    if model is None:
        return np.zeros(X.shape[0], dtype=float)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            return np.asarray(probs, dtype=float)
        # LightGBM Booster: predict returns probabilities
        if hasattr(model, "predict"):
            preds = model.predict(X)
            return np.asarray(preds, dtype=float)
        # fallback: decision_function -> sigmoid
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))
    except Exception:
        setup_logger().exception("[predict_probs] Model prediction failed")
    return np.zeros(X.shape[0], dtype=float)


def ensemble_scores(online_prob: np.ndarray, batch_prob: np.ndarray, alpha_online: float) -> np.ndarray:
    if online_prob is None and batch_prob is None:
        return np.zeros(0, dtype=float)
    if online_prob is None or len(online_prob) == 0:
        return batch_prob
    if batch_prob is None or len(batch_prob) == 0:
        return online_prob
    alpha = float(alpha_online)
    return alpha * online_prob + (1.0 - alpha) * batch_prob


# -------------------------
# Main flow
# -------------------------
def run_recommender(source: str, watchlist: str, country: Optional[str], top_k: int = 20, alpha_online: float = 0.7, limit: Optional[int] = None):
    log = setup_logger()
    # config -> table name
    cfg = initialize_config(source)
    table_name = cfg.get("tal_master_tablename")
    if not table_name:
        raise RuntimeError(f"No tal_master_tablename for source={source}")

    # 1. read snapshots
    df_snap = read_snapshot_data(table_name, country_filter=country, limit=limit)
    if df_snap is None or df_snap.empty:
        log.warning("[run_recommender] No snapshot rows found")
        return pd.DataFrame()

    # 2. synthesize basic features
    df_feats = synthesize_features(df_snap)

    # 3. load artifacts
    artifacts = load_artifacts("swing_buy_recommender", watchlist)

    # 4. transform features -> X
    feature_config = artifacts.get("feature_config") or {
        "numeric_columns": DEFAULT_NUMERIC,
        "categorical_columns": DEFAULT_CATEGORICAL,
        "feature_hasher_n_features": 4096
    }
    try:
        X, used_scaler, used_hasher = transform_features(df_feats, feature_config, artifacts.get("transformer"))
    except Exception:
        # fallback: try local transformer creation
        log.exception("[run_recommender] transform_features failed; attempting fallback")
        # create simple transformer bundle on the fly
        fallback_transformer = {}
        try:
            X, used_scaler, used_hasher = transform_features(df_feats, feature_config, fallback_transformer)
        except Exception:
            log.exception("[run_recommender] fallback transform also failed; aborting")
            raise

    # 5. predict
    online_prob = predict_probs(artifacts.get("online_model"), X)
    batch_prob = predict_probs(artifacts.get("batch_model"), X)

    # 6. ensemble
    final_score = ensemble_scores(online_prob, batch_prob, alpha_online)

    # 7. assemble output
    out = df_feats.reset_index(drop=True).copy()
    out["online_prob"] = online_prob
    out["batch_prob"] = batch_prob
    out["final_score"] = final_score
    out["model_online"] = artifacts.get("paths", {}).get("online_model")
    out["model_batch"] = artifacts.get("paths", {}).get("batch_model")
    out["scored_at"] = pd.Timestamp.utcnow().isoformat()

    # sort and pick top-K
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    top = out.head(top_k)

    # 8. save predictions artifact
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pred_name = f"predictions_{ts}"
    try:
        artifact_store.save_dataframe(out, pred_name, "swing_buy_recommender", watchlist,
                                      ext=".parquet", metadata={
                                          "source_table": table_name,
                                          "country_filter": country,
                                          "scored_at": pd.Timestamp.utcnow().isoformat(),
                                          "rows": int(len(out)),
                                          "model_online": artifacts.get("paths", {}).get("online_model"),
                                          "model_batch": artifacts.get("paths", {}).get("batch_model")
                                      })
        log.info(f"[run_recommender] Wrote predictions artifact {pred_name}")
    except Exception:
        log.exception("[run_recommender] Failed to save predictions artifact")

    return top


def parse_args():
    p = argparse.ArgumentParser(description="Model-backed Swing Buy Recommender")
    p.add_argument("--source", required=True, help="Price source: FINNHUB | FINNHUB_LOCAL | EOD | EOD_LOCAL")
    p.add_argument("--watchlist", required=True, help="Watchlist folder/name under artifact_store")
    p.add_argument("--country", required=False, default=None, help="CountryName substring filter (optional)")
    p.add_argument("--top-k", type=int, default=20, help="Top K recommendations")
    p.add_argument("--alpha-online", type=float, default=0.7, help="Weight for online model in ensemble (0..1)")
    p.add_argument("--limit", type=int, default=None, help="Limit rows read from snapshot (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logger()
    try:
        top = run_recommender(source=args.source, watchlist=args.watchlist, country=args.country,
                              top_k=args.top_k, alpha_online=args.alpha_online, limit=args.limit)
        if top is None or top.empty:
            print("No recommendations (no data or models).")
            return
        display_cols = [c for c in ["Symbol", "TodayPrice", "final_score", "online_prob", "batch_prob", "TrendReversal_Rules", "LastTrendDays", "ScanDate"] if c in top.columns]
        print("\n" + "=" * 100)
        print(f"Top {len(top)} recommendations:")
        print(top[display_cols].to_string(index=False))
        print("=" * 100 + "\n")
    except Exception as e:
        log.exception(f"[main] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

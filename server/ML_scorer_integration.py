#!/usr/bin/env python3
"""
ML_scorer_integration.py

Production scorer for the swing trading system (Component C).

What this script does:
- Loads the latest artifacts (online model, batch model, transformer bundle, feature_config)
  from the artifact_store for a given program/watchlist.
- Reads today's snapshot rows from the appropriate tal_master table (selected by initialize_config(source))
  using a SQLAlchemy-safe query (named parameters).
- Synthesizes a consistent set of features (numeric + categorical) required by the transformer.
- Transforms features using the transformer bundle (StandardScaler + FeatureHasher) if present,
  otherwise falls back to on-the-fly transforms (with a warning).
- Produces predictions from the online model and the batch model (if both available), ensembles them,
  and writes a predictions artifact (Parquet) using artifact_store.
- Prints the top-K recommended symbols by final_score and returns the DataFrame.

Intended usage:
  python scorer_integration.py --source FINNHUB_LOCAL --watchlist US01 --country USA --top-k 20

Notes:
- The script expects artifact_store.py in the PYTHONPATH (the common functions for artifact IO).
- It assumes models were saved with artifact_store.save_model (joblib/pickle compatible).
- It uses safe SQL through sqlalchemy.text and passes params as a dict to avoid SQLAlchemy 2.0 pitfalls.
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

# local modules
import ML_artifact_store as artifact_store
from ta_signals_mc_parallel import initialize_config, canonical_table_schema
from app_imports import getDbConnection, loggingSetter

# sklearn helpers
try:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import csr_matrix, hstack as sparse_hstack
except Exception:
    FeatureHasher = None
    StandardScaler = None
    csr_matrix = None
    sparse_hstack = None

LOG: Optional[logging.Logger] = None


def setup_logger(name: str = "ML_scorer") -> logging.Logger:
    global LOG
    if LOG:
        return LOG
    try:
        LOG = loggingSetter(name)
    except Exception:
        LOG = logging.getLogger(name)
        if not LOG.handlers:
            logging.basicConfig(level=logging.INFO)
    return LOG


# -------------------------
# DB read (safe)
# -------------------------
def get_table_columns(table_name: str) -> List[str]:
    """
    Read INFORMATION_SCHEMA.COLUMNS to discover columns for table_name.
    Falls back to SELECT * LIMIT 0 if INFORMATION_SCHEMA is unavailable.
    """
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
    except Exception as e:
        log.warning(f"[get_table_columns] INFORMATION_SCHEMA failed: {e}; falling back to SELECT LIMIT 0")
        try:
            with getDbConnection() as con:
                df0 = pd.read_sql_query(text(f"SELECT * FROM `{table_name}` LIMIT 0"), con=con)
                return list(df0.columns)
        except Exception as ie:
            log.error(f"[get_table_columns] fallback failed: {ie}")
            return []


def read_snapshot_table(table_name: str,
                        country_filter: Optional[str] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
    """
    Safely read rows from tal_master table applying country substring filter if the column exists.
    Uses sqlalchemy.text and passes params as a dict.
    """
    log = setup_logger()
    cols = get_table_columns(table_name)
    if not cols:
        raise RuntimeError(f"Unable to determine columns for table {table_name}")

    where = []
    params: Dict[str, Any] = {}
    if country_filter and "CountryName" in cols:
        where.append("`CountryName` LIKE :country")
        params["country"] = f"%{country_filter}%"
    elif country_filter:
        log.debug(f"[read_snapshot_table] table {table_name} has no CountryName; ignoring country_filter")

    q = f"SELECT * FROM `{table_name}`"
    if where:
        q += " WHERE " + " AND ".join(where)
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
        except Exception as e:
            # fallback: manual execute with mapping
            log.warning(f"[read_snapshot_table] pd.read_sql_query failed: {e}; falling back to manual execute")
            try:
                res = con.execute(sql_text, params) if params else con.execute(sql_text)
                try:
                    rows = res.mappings().all()
                    df = pd.DataFrame(rows)
                except Exception:
                    rows = res.fetchall()
                    df = pd.DataFrame(rows, columns=res.keys() if hasattr(res, "keys") else None)
            except Exception as e2:
                log.error(f"[read_snapshot_table] manual execute failed: {e2}\nSQL: {q}\nparams: {params}")
                raise
    df.columns = [str(c) for c in df.columns]
    log.info(f"[read_snapshot_table] Read {len(df)} rows from {table_name}")
    return df


# -------------------------
# Feature synthesis (lightweight and aligned with transformer expectation)
# -------------------------
NUMERIC_COLUMNS = [
    "ADX", "RSI", "EMA20", "EMA50", "SMA200", "TodayPrice", "Volume",
    "High52", "Low52", "Pct2H52", "PctfL52", "LastTrendDays", "marketCap"
]


def synthesize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features used by the transformer/model (safe if columns missing)."""
    X = df.copy()

    # numeric coercion
    for c in NUMERIC_COLUMNS:
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

    # classifier consensus
    if "SignalClassifier_Rules" in X.columns:
        X["SignalClassifier_Rules"] = pd.to_numeric(X["SignalClassifier_Rules"], errors="coerce").fillna(0)
    else:
        X["SignalClassifier_Rules"] = 0
    if "SignalClassifier_ML" in X.columns:
        X["SignalClassifier_ML"] = pd.to_numeric(X["SignalClassifier_ML"], errors="coerce").fillna(0)
    else:
        X["SignalClassifier_ML"] = 0
    X["classifier_consensus"] = X["SignalClassifier_Rules"].clip(0, 1) + X["SignalClassifier_ML"].clip(0, 1)

    # text/categorical uniform strings
    for c in ["DITrend", "MA_Trend", "MADI_Trend", "Primary", "Secondary", "TrendReversal_Rules", "TrendReversal_ML"]:
        if c in X.columns:
            X[c] = X[c].fillna("").astype(str)
        else:
            X[c] = ""

    if "LastTrendDays" in X.columns:
        X["LastTrendDays"] = pd.to_numeric(X["LastTrendDays"], errors="coerce").fillna(0).astype(int)
    else:
        X["LastTrendDays"] = 0

    if "Volume" in X.columns:
        X["Volume"] = pd.to_numeric(X["Volume"], errors="coerce").fillna(0)
    else:
        X["Volume"] = 0

    return X


# -------------------------
# Transformation & prediction
# -------------------------
def load_artifacts(program: str, watchlist: str) -> Dict[str, Optional[Any]]:
    """
    Load latest artifacts:
      - feature_config (json)
      - transformer_bundle (pickle) -> {scaler, hasher}
      - online model (pickle)
      - batch model (pickle)
    Returns dict with keys: feature_config, transformer, online_model, batch_model and their artifact filenames.
    """
    log = setup_logger()
    out = {
        "feature_config": None,
        "transformer": None,
        "online_model": None,
        "batch_model": None,
        "feature_config_path": None,
        "transformer_path": None,
        "online_path": None,
        "batch_path": None
    }

    # feature_config (exclude .meta.json files)
    all_fcfg = artifact_store.list_artifacts(program, watchlist, pattern="feature_config", ext=".json")
    fcfg_candidates = [p for p in all_fcfg if not p.name.endswith(".meta.json")]
    fcfg_path = fcfg_candidates[0] if fcfg_candidates else None
    if fcfg_path:
        try:
            out["feature_config"] = artifact_store.load_json(fcfg_path)
            out["feature_config_path"] = fcfg_path
            log.info(f"[load_artifacts] Loaded feature_config: {fcfg_path.name}")
        except Exception as e:
            log.warning(f"[load_artifacts] Failed to load feature_config: {e}")

    # transformer bundle
    t_path = artifact_store.latest_artifact_path(program, watchlist, name_contains="transformer_bundle", ext=".pkl")
    if t_path:
        try:
            out["transformer"] = artifact_store.load_model(t_path)
            out["transformer_path"] = t_path
            log.info(f"[load_artifacts] Loaded transformer bundle: {t_path.name}")
        except Exception as e:
            log.warning(f"[load_artifacts] Failed to load transformer bundle: {e}")

    # online model
    online_path = artifact_store.latest_artifact_path(program, watchlist, name_contains="model_online", ext=".pkl")
    if online_path:
        try:
            out["online_model"] = artifact_store.load_model(online_path)
            out["online_path"] = online_path
            log.info(f"[load_artifacts] Loaded online model: {online_path.name}")
        except Exception as e:
            log.warning(f"[load_artifacts] Failed to load online model: {e}")

    # batch model
    batch_path = artifact_store.latest_artifact_path(program, watchlist, name_contains="model_batch", ext=".pkl")
    if batch_path:
        try:
            out["batch_model"] = artifact_store.load_model(batch_path)
            out["batch_path"] = batch_path
            log.info(f"[load_artifacts] Loaded batch model: {batch_path.name}")
        except Exception as e:
            log.warning(f"[load_artifacts] Failed to load batch model: {e}")

    return out


def transform_for_models(df: pd.DataFrame, feature_config: Dict[str, Any], transformer: Optional[Dict[str, Any]]) -> Tuple[Any, np.ndarray]:
    """
    Produce X (sparse or dense) and array y (if present) ready for model.predict_proba.
    If transformer bundle includes 'scaler' and 'hasher' it will be used; otherwise a best-effort transformation is performed.
    """
    log = setup_logger()
    numeric_cols = feature_config.get("numeric_columns", [])
    categorical_cols = feature_config.get("categorical_columns", [])
    n_hasher = feature_config.get("feature_hasher_n_features", 4096)

    # ensure price_over_ema20 present
    if 'price_over_ema20' not in df.columns:
        df['price_over_ema20'] = df['TodayPrice'] / pd.to_numeric(df.get('EMA20', pd.Series([np.nan]*len(df))), errors='coerce')

    # numeric matrix
    num_mat = np.zeros((len(df), len(numeric_cols)), dtype=float)
    for i, c in enumerate(numeric_cols):
        if c in df.columns:
            num_mat[:, i] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).to_numpy()
        else:
            num_mat[:, i] = 0.0

    # categorical tokenization -> list[str]
    cat_inputs = []
    for _, row in df.iterrows():
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

    # Use provided transformer when possible
    if transformer and isinstance(transformer, dict):
        scaler = transformer.get("scaler", None)
        hasher_obj = transformer.get("hasher", None)
        # hasher may be stored as config instead of object; fallback to creating one
        if hasher_obj is None:
            if FeatureHasher is None:
                raise RuntimeError("FeatureHasher not available for categorical transforms")
            hasher = FeatureHasher(n_features=n_hasher, input_type='string')
        else:
            hasher = hasher_obj
        if scaler is None:
            # create and fit on current numeric (not ideal) but necessary fallback
            if StandardScaler is None:
                raise RuntimeError("StandardScaler not available for numeric transforms")
            scaler = StandardScaler().fit(num_mat)
        num_scaled = scaler.transform(num_mat)
    else:
        # fallback: instantiate new hasher and scaler (fit scaler on current batch)
        if FeatureHasher is None or StandardScaler is None:
            raise RuntimeError("Sklearn FeatureHasher/StandardScaler required for transformation")
        hasher = FeatureHasher(n_features=n_hasher, input_type='string')
        scaler = StandardScaler().fit(num_mat)
        num_scaled = scaler.transform(num_mat)

    # transform categorical
    X_cat = hasher.transform(cat_inputs)  # sparse matrix
    # combine numeric + cat
    if csr_matrix is None or sparse_hstack is None:
        raise RuntimeError("scipy.sparse not available for matrix operations")
    X_num_sparse = csr_matrix(num_scaled)
    X = sparse_hstack([X_num_sparse, X_cat], format='csr')
    return X


def score_snapshots(df_snap: pd.DataFrame, artifacts: Dict[str, Any],
                    alpha_online: float = 0.7, alpha_batch: float = 0.3) -> pd.DataFrame:
    """
    Compute online_prob, batch_prob (if available) and final_score (ensemble) for each snapshot row.
    Returns DataFrame with predictions appended.
    """
    log = setup_logger()
    feature_config = artifacts.get("feature_config")
    transformer = artifacts.get("transformer")
    online_model = artifacts.get("online_model")
    batch_model = artifacts.get("batch_model")

    if feature_config is None:
        # fallback feature_config default
        feature_config = {
            "numeric_columns": ["ADX", "RSI", "EMA20", "EMA50", "SMA200", "price_over_ema20", "Volume", "LastTrendDays", "Pct2H52", "PctfL52", "marketCap"],
            "categorical_columns": ["TrendReversal_Rules", "TrendReversal_ML", "DITrend", "MA_Trend", "MADI_Trend", "Primary", "Secondary"],
            "feature_hasher_n_features": 4096
        }
        log.warning("[score_snapshots] feature_config missing, using default set")

    # ensure synthesized features present
    df_feats = synthesize_features(df_snap)

    # compute X
    try:
        X = transform_for_models(df_feats, feature_config, transformer)
    except Exception as e:
        log.error(f"[score_snapshots] feature transform failed: {e}")
        raise

    # online probabilities
    online_prob = np.zeros(len(df_feats), dtype=float)
    batch_prob = np.zeros(len(df_feats), dtype=float)
    online_version = artifacts.get("online_path").name if artifacts.get("online_path") else None
    batch_version = artifacts.get("batch_path").name if artifacts.get("batch_path") else None

    if online_model is not None:
        try:
            if hasattr(online_model, "predict_proba"):
                online_prob = online_model.predict_proba(X)[:, 1]
            else:
                # some linear models provide decision_function
                scores = online_model.decision_function(X)
                online_prob = 1 / (1 + np.exp(-scores))
        except Exception as e:
            log.warning(f"[score_snapshots] online model predict failed: {e}")
            online_prob = np.zeros(len(df_feats), dtype=float)

    if batch_model is not None:
        try:
            if hasattr(batch_model, "predict_proba"):
                batch_prob = batch_model.predict_proba(X)[:, 1]
            else:
                # LightGBM Booster has predict returning raw probability
                batch_prob = batch_model.predict(X)
        except Exception as e:
            log.warning(f"[score_snapshots] batch model predict failed: {e}")
            batch_prob = np.zeros(len(df_feats), dtype=float)

    # ensemble logic: if only one model present, use it; if both, weighted ensemble
    if (online_model is not None) and (batch_model is None):
        final_score = online_prob
    elif (batch_model is not None) and (online_model is None):
        final_score = batch_prob
    else:
        # prefer online for recency (alpha_online) and batch for stability
        alpha_o = float(alpha_online)
        alpha_b = 1.0 - alpha_o
        final_score = alpha_o * online_prob + alpha_b * batch_prob

    out = df_feats.copy().reset_index(drop=True)
    out["online_prob"] = online_prob
    out["batch_prob"] = batch_prob
    out["final_score"] = final_score
    out["model_online"] = online_version
    out["model_batch"] = batch_version
    out["scored_at"] = pd.Timestamp.utcnow().isoformat()
    return out


# -------------------------
# Main CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Scorer integration: load models and score today's snapshots")
    p.add_argument("--source", required=True, help="Price source (FINNHUB | FINNHUB_LOCAL | EOD | EOD_LOCAL)")
    p.add_argument("--watchlist", required=True, help="Watchlist code (e.g., US01)")
    p.add_argument("--country", required=False, help="CountryName substring filter applied when reading tal_master (optional)")
    p.add_argument("--top-k", type=int, default=20, help="Number of top recommendations to print")
    p.add_argument("--out", default=None, help="Optional CSV/Parquet path to write top-k results")
    p.add_argument("--alpha-online", type=float, default=0.7, help="Ensemble weight for online model (0..1)")
    p.add_argument("--limit", type=int, default=None, help="Limit rows read from tal_master (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logger()
    try:
        cfg = initialize_config(args.source)
    except Exception as e:
        log.error(f"[main] initialize_config failed: {e}")
        sys.exit(2)

    table_name = cfg.get("tal_master_tablename", "")
    if not table_name:
        log.error("[main] No tal_master_tablename configured for source")
        sys.exit(2)

    # Read snapshots (safe)
    try:
        df_snap = read_snapshot_table(table_name, country_filter=args.country, limit=args.limit)
    except Exception as e:
        log.error(f"[main] Failed to read snapshot table: {e}\n{traceback.format_exc()}")
        sys.exit(3)

    if df_snap is None or df_snap.empty:
        log.info("[main] No snapshot rows to score")
        return

    # Load artifacts
    artifacts = load_artifacts("swing_buy_recommender", args.watchlist)

    # If transformer bundle missing, attempt to load transformer from artifact (scaler/hasher), else warn
    transformer = artifacts.get("transformer")
    if transformer is None:
        log.warning("[main] transformer bundle not found; transform will fit scaler on current snapshot (not ideal)")

    # Score
    try:
        scored = score_snapshots(df_snap, artifacts, alpha_online=args.alpha_online)
    except Exception as e:
        log.error(f"[main] Scoring failed: {e}\n{traceback.format_exc()}")
        sys.exit(4)

    # Persist predictions artifact
    run_ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    predictions_name = f"predictions_{run_ts}"
    try:
        preds_path = artifact_store.save_dataframe(scored, predictions_name, "swing_buy_recommender", args.watchlist,
                                                  ext=".parquet", metadata={
                                                      "source_table": table_name,
                                                      "country_filter": args.country,
                                                      "model_online": artifacts.get("online_path").name if artifacts.get("online_path") else None,
                                                      "model_batch": artifacts.get("batch_path").name if artifacts.get("batch_path") else None,
                                                      "scored_at": pd.Timestamp.utcnow().isoformat()
                                                  })
        log.info(f"[main] Saved predictions to artifact: {preds_path}")
    except Exception as e:
        log.error(f"[main] Failed to save predictions artifact: {e}\n{traceback.format_exc()}")

    # Output top-K
    topk = scored.dropna(subset=["final_score"]).sort_values("final_score", ascending=False).head(args.top_k)
    display_cols = ["Symbol", "TodayPrice", "final_score", "online_prob", "batch_prob", "TrendReversal_Rules", "LastTrendDays", "ScanDate"]
    present_cols = [c for c in display_cols if c in topk.columns]
    print("\n" + "=" * 80)
    print(f"Top {args.top_k} recommendations (scored at {pd.Timestamp.utcnow().isoformat()}):")
    print(topk[present_cols].to_string(index=False))
    print("=" * 80 + "\n")

    # Optional write out top-K to user-specified path
    if args.out:
        try:
            ext = args.out.lower().split(".")[-1]
            if ext in ("parquet", "pq"):
                topk.to_parquet(args.out, index=False)
            else:
                topk.to_csv(args.out, index=False)
            log.info(f"[main] Wrote top-K to {args.out}")
        except Exception as e:
            log.error(f"[main] Failed to write out file {args.out}: {e}")

    return scored


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ML_monitor.py

Daily monitoring job for the swing trading pipeline (Component D).

Responsibilities:
- Locate the latest predictions artifact(s) for a given program/watchlist (produced by scorer_integration).
- Locate labeled outcomes (produced by daily incremental updater) and join them to predictions by (Symbol, ScanDate).
- Compute evaluation metrics (precision@K, recall@K, AUC when possible, average realized return for picks).
- Compute simple distribution-shift / drift checks (PSI) on key numeric features and on predicted probability distribution.
- Produce a monitor report artifact (JSON) and a short tabular summary (Parquet) saved to the artifact store.
- Trigger simple alerts (logged warnings) when metrics cross thresholds.

Notes:
- Relies on artifact_store.py for artifact discovery + IO.
- Assumes predictions artifacts contain at least: Symbol, ScanDate, final_score, online_prob, batch_prob, scored_at.
- Assumes labeled data artifacts contain at least: Symbol, ScanDate, label_success, realized_return.
- Designed to be conservative and robust to missing data: if labels are not yet available, the monitor will note it.

Usage:
  python monitor.py --watchlist US01 --program swing_buy_recommender --predictions-days-back 1 --top-k 10

Exit codes:
 - 0 success (report produced)
 - 1 fatal error
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import ML_artifact_store as artifact_store
from app_imports import loggingSetter

LOG: Optional[logging.Logger] = None


def setup_logger(name: str = "ML_monitor") -> logging.Logger:
    global LOG
    if LOG:
        return LOG
    try:
        LOG = loggingSetter(name)
    except Exception:
        LOG = logging.getLogger(name)
        if not LOG.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return LOG


# -------------------------
# Helper functions
# -------------------------
def find_latest_predictions(program: str, watchlist: str, days_back: int = 1) -> Optional[str]:
    """
    Try to find a predictions artifact within the last `days_back` days.
    Returns artifact path or None.
    """
    log = setup_logger()
    manifest = artifact_store.read_manifest(program, watchlist)
    if not manifest:
        log.info("[find_latest_predictions] No manifest entries found.")
        return None

    # Look for predictions_ in filename; manifest entries include filename key
    candidates = []
    for ent in manifest:
        fn = ent.get("filename") or ent.get("filename", "")
        if "predictions_" in fn and (fn.endswith(".parquet") or fn.endswith(".csv")):
            # parse timestamp embedded if present
            candidates.append((fn, ent))
    if not candidates:
        log.info("[find_latest_predictions] No predictions artifacts found in manifest.")
        return None

    # Sort by created_at in manifest entries (most recent first)
    sorted_c = sorted(candidates, key=lambda x: x[1].get("created_at", ""), reverse=True)
    # Optionally restrict to days_back window
    now = pd.Timestamp.utcnow()
    for fn, ent in sorted_c:
        created_at = ent.get("created_at")
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at)
            except Exception:
                dt = None
            if dt is None or (now - dt) <= timedelta(days=days_back):
                # find artifact path
                p = artifact_store.find_artifact_by_filename(program, watchlist, fn)
                if p:
                    return str(p)
        else:
            p = artifact_store.find_artifact_by_filename(program, watchlist, fn)
            if p:
                return str(p)
    # fallback to latest artifact path heuristics
    p = artifact_store.latest_artifact_path(program, watchlist, name_contains="predictions", ext=".parquet")
    if p:
        return str(p)
    p = artifact_store.latest_artifact_path(program, watchlist, name_contains="predictions", ext=".csv")
    if p:
        return str(p)
    return None


def gather_labeled_data(program: str, watchlist: str) -> pd.DataFrame:
    """
    Load all labeled_data artifacts and concat into a single DataFrame.
    Returns empty DataFrame if none found.
    """
    log = setup_logger()
    manifest = artifact_store.read_manifest(program, watchlist)
    rows = []
    for ent in manifest:
        fn = ent.get("filename", "")
        if fn.startswith("labeled_data") and (fn.endswith(".parquet") or fn.endswith(".csv")):
            p = artifact_store.find_artifact_by_filename(program, watchlist, fn)
            if p:
                try:
                    df = artifact_store.load_dataframe(p)
                    rows.append(df)
                except Exception as e:
                    log.warning(f"[gather_labeled_data] failed loading {p}: {e}")
    if not rows:
        log.info("[gather_labeled_data] no labeled data artifacts found")
        return pd.DataFrame()
    df_all = pd.concat(rows, ignore_index=True)
    # normalize ScanDate column
    if 'ScanDate' in df_all.columns:
        df_all['ScanDate'] = pd.to_datetime(df_all['ScanDate']).dt.normalize()
    return df_all


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between two 1D numeric arrays.
    Implementation: create quantile buckets on expected, compute distribution differences.
    """
    # remove NaNs
    exp = expected[~np.isnan(expected)]
    act = actual[~np.isnan(actual)]
    if len(exp) == 0 or len(act) == 0:
        return float("nan")
    try:
        # bucket edges from expected quantiles
        quantiles = np.linspace(0, 1, buckets + 1)
        bins = np.unique(np.quantile(exp, quantiles))
        if len(bins) <= 1:
            return 0.0
        exp_counts, _ = np.histogram(exp, bins=bins)
        act_counts, _ = np.histogram(act, bins=bins)
        # convert to proportions, avoid zeros by small epsilon
        exp_props = exp_counts / exp_counts.sum()
        act_props = act_counts / act_counts.sum()
        eps = 1e-8
        exp_props = np.where(exp_props == 0, eps, exp_props)
        act_props = np.where(act_props == 0, eps, act_props)
        # psi sum
        psi_val = np.sum((exp_props - act_props) * np.log(exp_props / act_props))
        return float(psi_val)
    except Exception:
        return float("nan")


def precision_at_k(df_joined: pd.DataFrame, k: int) -> Tuple[float, int, int]:
    """
    Compute precision@k: fraction of top-k final_score rows with label_success==1.
    Returns (precision, true_positives, k).
    """
    if df_joined is None or df_joined.empty:
        return float("nan"), 0, k
    df_sorted = df_joined.sort_values("final_score", ascending=False).reset_index(drop=True)
    df_topk = df_sorted.head(k)
    if 'label_success' not in df_topk.columns:
        return float("nan"), 0, k
    tp = int(df_topk['label_success'].sum())
    prec = tp / k if k > 0 else float("nan")
    return float(prec), tp, k


# -------------------------
# Monitoring orchestration
# -------------------------
def run_monitor(program: str, watchlist: str, predictions_path: Optional[str],
                lookback_days_for_baseline: int = 7, top_k: int = 10,
                psi_threshold: float = 0.1, precision_drop_threshold: float = 0.2) -> Dict[str, Any]:
    """
    Main monitor run:
    - Load predictions (predictions_path if supplied else auto-detect)
    - Load labeled outcomes and join
    - Compute metrics & drift
    - Save monitor report + summary artifacts
    Returns report dict.
    """
    log = setup_logger()
    report: Dict[str, Any] = {
        "program": program,
        "watchlist": watchlist,
        "run_at": pd.Timestamp.utcnow().isoformat(),
        "predictions_file": predictions_path,
        "status": "ok",
        "notes": []
    }

    try:
        # Discover predictions artifact if not provided
        if not predictions_path:
            predictions_path = find_latest_predictions(program, watchlist, days_back=lookback_days_for_baseline)
            report["predictions_auto_found"] = bool(predictions_path)
        if not predictions_path:
            report["status"] = "no_predictions"
            report["notes"].append("No predictions artifact found in artifact store.")
            log.warning("[monitor] No predictions artifact to evaluate.")
            # persist minimal report
            artifact_store.save_json(report, f"monitor_report_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
                                     program, watchlist, ext=".json")
            return report

        # Load predictions
        log.info(f"[monitor] Loading predictions from {predictions_path}")
        df_preds = artifact_store.load_dataframe(predictions_path)
        # normalize
        if 'ScanDate' in df_preds.columns:
            df_preds['ScanDate'] = pd.to_datetime(df_preds['ScanDate']).dt.normalize()
        if 'final_score' not in df_preds.columns and 'online_prob' in df_preds.columns:
            df_preds['final_score'] = df_preds.get('online_prob', 0.0)
    except Exception as e:
        log.error(f"[monitor] failed to load predictions: {e}\n{traceback.format_exc()}")
        report["status"] = "error_loading_predictions"
        report["notes"].append(str(e))
        artifact_store.save_json(report, f"monitor_report_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}", program, watchlist, ext=".json")
        return report

    # load labeled data (all) and filter to relevant scan dates
    try:
        df_labels = gather_labeled_data(program, watchlist)
        if df_labels.empty:
            log.info("[monitor] No labeled data artifacts found.")
            report["status"] = "no_labels"
            report["notes"].append("No labeled data artifacts found to evaluate predictions.")
            # still save predictions summary but cannot compute metrics
        else:
            # normalize ScanDate
            if 'ScanDate' in df_labels.columns:
                df_labels['ScanDate'] = pd.to_datetime(df_labels['ScanDate']).dt.normalize()
    except Exception as e:
        log.error(f"[monitor] failed to load labeled data: {e}\n{traceback.format_exc()}")
        df_labels = pd.DataFrame()
        report["notes"].append("Error loading labeled data: " + str(e))

    # Join predictions to labels on Symbol + ScanDate
    df_joined = df_preds.copy()
    if not df_labels.empty:
        # merge labels: find label rows for same (Symbol, ScanDate)
        df_joined = df_joined.merge(df_labels[['Symbol', 'ScanDate', 'label_success', 'realized_return']],
                                    on=['Symbol', 'ScanDate'], how='left', suffixes=('', '_label'))
    else:
        df_joined['label_success'] = np.nan
        df_joined['realized_return'] = np.nan

    # compute evaluation metrics if labels available
    labeled_mask = df_joined['label_success'].notna()
    n_labelled = int(labeled_mask.sum())
    report["n_predictions_total"] = int(len(df_joined))
    report["n_predictions_labeled"] = n_labelled

    if n_labelled == 0:
        log.info("[monitor] No labeled predictions available to compute metrics yet.")
        report["status"] = "no_labels_for_predictions"
    else:
        # compute precision@k for top_k
        prec, tp, k = precision_at_k(df_joined[df_joined['label_success'].notna()], top_k)
        report["precision_at_k"] = prec
        report["top_k"] = k
        report["true_positives_topk"] = tp

        # overall metrics on labeled subset
        y_true = df_joined.loc[labeled_mask, 'label_success'].astype(int).to_numpy()
        # if predicted probability present, use it; otherwise use final_score
        if 'final_score' in df_joined.columns and df_joined['final_score'].notna().sum() > 0:
            y_score = df_joined.loc[labeled_mask, 'final_score'].astype(float).to_numpy()
        elif 'online_prob' in df_joined.columns and df_joined['online_prob'].notna().sum() > 0:
            y_score = df_joined.loc[labeled_mask, 'online_prob'].astype(float).to_numpy()
        else:
            y_score = None

        # compute simple metrics
        if y_score is not None:
            try:
                auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
            except Exception:
                auc = None
        else:
            auc = None
        preds_bin = (y_score >= 0.5).astype(int) if y_score is not None else (df_joined.loc[labeled_mask, 'final_score'].fillna(0) >= 0.5).astype(int)
        try:
            prec_overall = float(precision_score(y_true, preds_bin, zero_division=0))
            rec_overall = float(recall_score(y_true, preds_bin, zero_division=0))
            f1 = float(f1_score(y_true, preds_bin, zero_division=0))
        except Exception:
            prec_overall, rec_overall, f1 = (None, None, None)

        report["metrics"] = {
            "n_labeled": n_labelled,
            "precision_overall": prec_overall,
            "recall_overall": rec_overall,
            "f1_overall": f1,
            "auc": auc
        }

        # realized returns for top-K picks (from labeled set)
        df_labeled_preds = df_joined[labeled_mask].sort_values("final_score", ascending=False)
        topk_labeled = df_labeled_preds.head(top_k)
        # average realized return of top-K (only for those with realized_return available)
        rr = topk_labeled['realized_return'].dropna()
        if not rr.empty:
            report["realized_return_topk_mean"] = float(rr.mean())
            report["realized_return_topk_median"] = float(rr.median())
        else:
            report["realized_return_topk_mean"] = None

    # Drift detection: compare distribution of numeric features between predictions and labeled baseline
    numeric_cols = ['ADX', 'RSI', 'price_over_ema20', 'Volume', 'LastTrendDays']
    psi_results = {}
    for col in numeric_cols:
        if col in df_joined.columns:
            try:
                # baseline: past labeled_data values (last lookback_days for baseline)
                if not df_labels.empty:
                    baseline_vals = pd.to_numeric(df_labels[col], errors='coerce').dropna().to_numpy()
                else:
                    baseline_vals = pd.to_numeric(df_joined[col], errors='coerce').dropna().to_numpy()
                current_vals = pd.to_numeric(df_joined[col], errors='coerce').dropna().to_numpy()
                if baseline_vals.size == 0 or current_vals.size == 0:
                    psi_val = float('nan')
                else:
                    psi_val = psi(baseline_vals, current_vals, buckets=10)
                psi_results[col] = psi_val
            except Exception as e:
                psi_results[col] = float('nan')
        else:
            psi_results[col] = None
    report["psi"] = psi_results

    # Predicted probability distribution shift (psi on probs)
    try:
        if 'final_score' in df_joined.columns:
            cur_probs = df_joined['final_score'].dropna().to_numpy()
            # baseline probs from previous predictions (collect predictions within last lookback_days)
            # naive baseline: look for previous predictions artifacts and concat, else use labeled probabilities
            prev_probs = []
            manifest = artifact_store.read_manifest(program, watchlist)
            # collect up to 7 previous predictions files (excluding current)
            for ent in manifest:
                fn = ent.get("filename", "")
                if fn and "predictions_" in fn:
                    p = artifact_store.find_artifact_by_filename(program, watchlist, fn)
                    if p and str(p) != predictions_path:
                        try:
                            d = artifact_store.load_dataframe(p)
                            if 'final_score' in d.columns:
                                prev_probs.append(d['final_score'].dropna().to_numpy())
                        except Exception:
                            continue
            if prev_probs:
                prev_probs = np.concatenate(prev_probs)
            else:
                # fallback baseline to labeled final_score if available
                if 'final_score' in df_labels.columns:
                    prev_probs = df_labels['final_score'].dropna().to_numpy()
                else:
                    prev_probs = np.array([])
            if prev_probs.size == 0:
                prob_psi = float('nan')
            else:
                prob_psi = psi(prev_probs, cur_probs, buckets=10)
            report["probability_psi"] = prob_psi
        else:
            report["probability_psi"] = None
    except Exception as e:
        report["probability_psi"] = None

    # Determine alerts
    alerts: List[str] = []
    # Precision drop alert vs baseline if baseline available (here we look at last batch metrics in manifest)
    try:
        # attempt to find last batch_train_metrics or last_monitor_report metrics as baseline
        manifest = artifact_store.read_manifest(program, watchlist)
        baseline_precision = None
        # search metrics entries
        for ent in reversed(manifest):
            fn = ent.get("filename", "")
            if fn and ("batch_train_metrics" in fn or "online_update_metrics" in fn or "monitor_report" in fn):
                p = artifact_store.find_artifact_by_filename(program, watchlist, fn)
                if p:
                    try:
                        j = artifact_store.load_json(p.with_suffix(".json") if str(p).endswith(".parquet") else p)
                    except Exception:
                        try:
                            j = artifact_store.load_json(p)
                        except Exception:
                            j = None
                    if j and isinstance(j, dict):
                        # try common keys
                        metrics = j.get("metrics") or j.get("train_metrics") or j.get("online_metrics") or j
                        if isinstance(metrics, dict) and metrics.get("precision") is not None:
                            baseline_precision = metrics.get("precision")
                            break
    except Exception:
        baseline_precision = None

    # Generate alerts based on thresholds
    if report.get("precision_at_k") is not None and baseline_precision is not None:
        try:
            if baseline_precision > 0 and (baseline_precision - report["precision_at_k"]) / baseline_precision >= precision_drop_threshold:
                alerts.append(f"Precision@K dropped by >= {precision_drop_threshold*100:.0f}% from baseline ({baseline_precision} -> {report['precision_at_k']}).")
        except Exception:
            pass

    # PSI alerts
    for col, val in psi_results.items():
        try:
            if val is not None and not math.isnan(val) and val >= psi_threshold:
                alerts.append(f"PSI for {col} = {val:.4f} >= threshold {psi_threshold}")
        except Exception:
            continue

    report["alerts"] = alerts
    if alerts:
        report["status"] = "alerts"
        for a in alerts:
            LOG = setup_logger()
            LOG.warning(f"[monitor ALERT] {a}")

    # Save report and a summary table of joined predictions + labels
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_name = f"monitor_report_{ts}"
    try:
        artifact_store.save_json(report, report_name, program, watchlist, ext=".json", metadata={"generated_at": pd.Timestamp.utcnow().isoformat()})
        # Save joined sample (limited) as Parquet for inspection
        preview = df_joined.copy()
        preview_name = f"monitor_preview_{ts}"
        artifact_store.save_dataframe(preview.head(1000), preview_name, program, watchlist, ext=".parquet", metadata={"rows_in_full": int(len(df_joined))})
    except Exception as e:
        log = setup_logger()
        log.error(f"[monitor] Failed to save monitor artifacts: {e}\n{traceback.format_exc()}")

    return report


def parse_args():
    p = argparse.ArgumentParser(description="Monitoring job for swing trading pipeline")
    p.add_argument("--program", default="swing_buy_recommender", help="Program name (artifact store folder)")
    p.add_argument("--watchlist", required=True, help="Watchlist code (artifact subfolder)")
    p.add_argument("--predictions-path", default=None, help="Optional explicit predictions artifact path")
    p.add_argument("--predictions-days-back", type=int, default=3, help="How many days back to search for predictions")
    p.add_argument("--top-k", type=int, default=10, help="Top-K used for precision@K calculation")
    p.add_argument("--psi-threshold", type=float, default=0.1, help="PSI threshold to raise alert")
    p.add_argument("--precision-drop-threshold", type=float, default=0.2, help="Relative precision drop threshold to alert (e.g., 0.2 = 20%)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()
    try:
        report = run_monitor(program=args.program,
                             watchlist=args.watchlist,
                             predictions_path=args.predictions_path,
                             lookback_days_for_baseline=args.predictions_days_back,
                             top_k=args.top_k,
                             psi_threshold=args.psi_threshold,
                             precision_drop_threshold=args.precision_drop_threshold)
        print("Monitor report summary:")
        print(json.dumps({k: report.get(k) for k in ["status", "n_predictions_total", "n_predictions_labeled", "precision_at_k", "alerts"]}, indent=2, default=str))
    except Exception as e:
        LOG = setup_logger()
        LOG.error(f"[monitor main] fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

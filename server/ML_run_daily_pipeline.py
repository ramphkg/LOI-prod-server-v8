#!/usr/bin/env python3
"""
ML_run_daily_pipeline.py
Python orchestration for daily ML pipeline phases (replaces run_daily_ml_pipeline.sh).

Phases:
  score                -> run primary scorer (ML_scorer.py) and alternate (ML_swing_buy_recommender.py)
  incremental-update   -> run incremental model update (ML_incremental_update.py)
  monitor              -> run monitoring & drift checks (ML_monitor.py)
  bootstrap-retrain    -> full retrain (ML_bootstrap.py)
  all / full           -> score -> incremental-update -> monitor

Exit: non-zero on first failing phase.
"""

import argparse
import subprocess
import sys
import shlex
import time
import datetime as dt
import pathlib

# ---------------- Configuration ----------------
BASE_DIR = pathlib.Path("/home/ram/dev/LOI/LOI-prod-server-v8")
SERVER_DIR = BASE_DIR / "server"
VENV_DIR = pathlib.Path("/home/ram/dev/LOI/venv312")          # adjust if needed
PYTHON = str(VENV_DIR / "bin" / "python")                     # cron should already use venv; keep explicit
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_SOURCE = "FINNHUB_LOCAL"
WATCHLIST = "US-GEMS100"
COUNTRY_FILTER = "USA"          # set "" to disable
TOP_K_SCORE = 20
HORIZON_DAYS = 5
TARGET_RETURN = 0.05
STOP_LOSS = 0.04
INCREMENTAL_MAX_SYMBOLS = ""    # e.g. "100" to limit, empty = all
SCORER_LIMIT = ""               # optional debug limit
ALPHA_ONLINE = 0.7
MONITOR_TOP_K = 10
MONITOR_PSI_THRESHOLD = 0.10
MONITOR_PRECISION_DROP = 0.20
SLEEP_BETWEEN = 5               # seconds between phases for all/full

DAY_TS = dt.datetime.utcnow().strftime("%Y%m%d")
PIPELINE_LOG = LOG_DIR / f"pipeline_{DAY_TS}.log"

# ---------------- Logging ----------------
def utc_now():
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def log(level: str, msg: str):
    line = f"{utc_now()} [{level}] {msg}"
    print(line)
    PIPELINE_LOG.write_text(PIPELINE_LOG.read_text() + line + "\n" if PIPELINE_LOG.exists() else line + "\n")

def info(msg): log("INFO", msg)
def warn(msg): log("WARN", msg)
def error(msg): log("ERROR", msg)

# ---------------- Runner ----------------
def run_phase(name: str, cmd: list[str]) -> None:
    info(f"Starting phase: {name} | cmd={' '.join(shlex.quote(x) for x in cmd)}")
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    PIPELINE_LOG.write_text(PIPELINE_LOG.read_text() + proc.stdout if PIPELINE_LOG.exists() else proc.stdout)
    dur = time.time() - start
    if proc.returncode == 0:
        info(f"Completed phase: {name} (duration={dur:.1f}s)")
    else:
        error(f"Phase failed: {name} (rc={proc.returncode})")
        sys.exit(proc.returncode)

def build_args(base: list[str], optional: dict[str, str | None]) -> list[str]:
    out = base[:]
    for flag, val in optional.items():
        if val is None or val == "":
            continue
        out.extend([flag, str(val)])
    return out

# ---------------- Phase Builders ----------------
def phase_score_both():
    country_arg = COUNTRY_FILTER if COUNTRY_FILTER else ""
    limit_arg = SCORER_LIMIT if SCORER_LIMIT else ""
    # Primary
    run_phase("SCORER_PRIMARY",
              build_args([PYTHON, str(SERVER_DIR / "ML_scorer.py"),
                          "--source", DATA_SOURCE,
                          "--watchlist", WATCHLIST,
                          "--top-k", str(TOP_K_SCORE),
                          "--alpha-online", str(ALPHA_ONLINE)],
                         {"--country": country_arg, "--limit": limit_arg}))
    # Alternate
    run_phase("SCORER_ALT",
              build_args([PYTHON, str(SERVER_DIR / "ML_swing_buy_recommender.py"),
                          "--source", DATA_SOURCE,
                          "--watchlist", WATCHLIST,
                          "--top-k", str(TOP_K_SCORE)],
                         {"--country": country_arg, "--limit": limit_arg}))

def phase_incremental_update():
    max_sym = INCREMENTAL_MAX_SYMBOLS if INCREMENTAL_MAX_SYMBOLS else ""
    run_phase("INCREMENTAL_UPDATE",
              build_args([PYTHON, str(SERVER_DIR / "ML_incremental_update.py"),
                          "--source", DATA_SOURCE,
                          "--watchlist", WATCHLIST,
                          "--horizon", str(HORIZON_DAYS),
                          "--target", str(TARGET_RETURN),
                          "--stop", str(STOP_LOSS)],
                         {"--max-symbols": max_sym}))

def phase_monitor():
    run_phase("MONITOR",
              [PYTHON, str(SERVER_DIR / "ML_monitor.py"),
               "--watchlist", WATCHLIST,
               "--top-k", str(MONITOR_TOP_K),
               "--psi-threshold", str(MONITOR_PSI_THRESHOLD),
               "--precision-drop-threshold", str(MONITOR_PRECISION_DROP)])

def phase_bootstrap_retrain():
    run_phase("BOOTSTRAP_RETRAIN",
              [PYTHON, str(SERVER_DIR / "ML_bootstrap.py"),
               "--source", DATA_SOURCE,
               "--watchlist", WATCHLIST,
               "--horizon", str(HORIZON_DAYS),
               "--target", str(TARGET_RETURN),
               "--stop", str(STOP_LOSS),
               "--train-years", "1",
               "--max-symbols", "200"])

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Daily ML pipeline (Python)")
    parser.add_argument("phase",
                        help="score | incremental-update | monitor | bootstrap-retrain | all | full")
    parser.add_argument("--sleep-between", type=int, default=SLEEP_BETWEEN,
                        help="Seconds to sleep between phases (all/full)")
    args = parser.parse_args()

    info(f"Pipeline start phase={args.phase} watchlist={WATCHLIST} source={DATA_SOURCE}")

    if args.phase == "score":
        phase_score_both()
    elif args.phase == "incremental-update":
        phase_incremental_update()
    elif args.phase == "monitor":
        phase_monitor()
    elif args.phase == "bootstrap-retrain":
        phase_bootstrap_retrain()
    elif args.phase in ("all", "full"):
        phase_score_both()
        if args.sleep_between > 0:
            info(f"Sleeping {args.sleep_between}s")
            time.sleep(args.sleep_between)
        phase_incremental_update()
        if args.sleep_between > 0:
            info(f"Sleeping {args.sleep_between}s")
            time.sleep(args.sleep_between)
        phase_monitor()
    else:
        error("Unknown phase")
        sys.exit(2)

    info(f"Pipeline finished phase={args.phase}")

if __name__ == "__main__":
    main()
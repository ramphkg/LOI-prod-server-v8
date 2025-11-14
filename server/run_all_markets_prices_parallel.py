# run_all_markets_prices_parallel.py

import concurrent.futures
import subprocess
import sys

# Define your market commands
# Note: use the correct python binary and .py file paths in your environment.
commands = [
    ["python3", "reload_prices_mc_parallel.py", "-s", "EOD", "-w", "HK_ALL"],
    ["python3", "reload_prices_mc_parallel.py", "-s", "EOD", "-w", "IN_ALL"],
    ["python3", "reload_prices_mc_parallel.py", "-s", "FINNHUB", "-w", "US_ALL"],
]

def safe_decode(b: bytes) -> str:
    """Decode bytes safely: try utf-8, then cp1252, then utf-8 with replacement."""
    if b is None:
        return ""
    if isinstance(b, str):
        return b
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return b.decode('cp1252')
        except Exception:
            return b.decode('utf-8', errors='replace')

def run_command(cmd):
    """Run a single command and safely decode stdout/stderr to avoid UnicodeDecodeError."""
    cmd_str = ' '.join(cmd)
    print(f"Running: {cmd_str}")
    # Capture output as bytes (text=False) and decode manually with fallback
    try:
        proc = subprocess.run(cmd, capture_output=True, text=False)
    except Exception as e:
        # If subprocess.run itself fails (e.g., file not found), surface error and return non-zero code
        print(f"Failed to start command: {cmd_str}\nError: {e}", file=sys.stderr)
        return 1

    stdout = safe_decode(proc.stdout)
    stderr = safe_decode(proc.stderr)

    if stdout:
        print(f"[{cmd[1] if len(cmd) > 1 else cmd_str}] STDOUT:\n{stdout}")
    if stderr:
        print(f"[{cmd[1] if len(cmd) > 1 else cmd_str}] STDERR:\n{stderr}", file=sys.stderr)

    return proc.returncode

if __name__ == "__main__":
    # Use as many workers as commands (or tune as needed)
    max_workers = min(len(commands), 8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_command, commands))
    print("All markets processed. Return codes:", results)

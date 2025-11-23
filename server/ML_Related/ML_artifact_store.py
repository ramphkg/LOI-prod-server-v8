#!/usr/bin/env python3
"""
ML_artifact_store.py

Utility module to centralize artifact storage and retrieval for the swing trading pipeline.

Purpose
- Provide simple, consistent functions that A-D (bootstrap trainer, incremental updater,
  scorer, monitor) can import and use to save/load artifacts (models, dataframes, JSON metadata)
  under a deterministic tree:
      ../data/<program_name>/<watchlist_name>/

Design goals
- Deterministic paths (sanitized program/watchlist names)
- Atomic writes (write to temporary file then os.replace)
- Per-artifact metadata (sidecar .meta.json) + lightweight manifest (manifest.jsonl)
- Convenience helpers: save/load model (joblib/pickle), DataFrame (parquet/csv), JSON, bytes
- Helpers: list, find latest, manifest entries
- Minimal dependencies: stdlib + pandas + joblib (joblib optional; falls back to pickle)

Usage (examples)
- from artifact_store import save_model, load_model, latest_artifact_path
- save_model(model, "lgbm_v1", program="swing_buy_recommender", watchlist="US01", metadata={...})
- path = latest_artifact_path("swing_buy_recommender", "US01", name_contains="lgbm", ext=".pkl")
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import joblib
except Exception:
    joblib = None

try:
    import pandas as pd
except Exception:
    pd = None

# Base directory for artifacts (relative to project root). Adjust if needed.
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data"


# -------------------------
# Utilities
# -------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize(name: str) -> str:
    # allow letters/numbers/.-_ ; replace other chars with underscore and collapse multiple underscores
    s = re.sub(r"[^A-Za-z0-9\-\._]+", "_", name or "")
    s = re.sub(r"_+", "_", s)
    return s.strip(" _") or "unnamed"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest_parent = dest.parent
    _ensure_dir(dest_parent)
    fd, tmp = tempfile.mkstemp(dir=str(dest_parent), prefix=".tmp_write_")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(data)
        os.replace(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _atomic_write_text(dest: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(dest, text.encode(encoding))


def _atomic_write_json(dest: Path, obj: Any, encoding: str = "utf-8") -> None:
    text = json.dumps(obj, default=str, ensure_ascii=False, indent=2)
    _atomic_write_text(dest, text, encoding=encoding)


@dataclass
class ArtifactMeta:
    program: str
    watchlist: str
    name: str
    filename: str
    ext: str
    created_at: str  # ISO UTC
    size_bytes: int
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Path helpers
# -------------------------
def base_data_dir(base: Optional[Path] = None) -> Path:
    return (Path(base) if base is not None else DEFAULT_BASE_DIR).resolve()


def artifact_dir(program: str, watchlist: str, base: Optional[Path] = None, create: bool = True) -> Path:
    p = base_data_dir(base) / _sanitize(program) / _sanitize(watchlist)
    if create:
        _ensure_dir(p)
    return p


def make_artifact_filename(name: str, ext: str = "", timestamp: Optional[str] = None) -> str:
    name_s = _sanitize(name)
    ts = timestamp or _utc_now_iso()
    ext_pref = ext if ext.startswith(".") or ext == "" else f".{ext}"
    return f"{name_s}_{ts}{ext_pref}"


# -------------------------
# Manifest helpers
# -------------------------
def _manifest_path(prog: str, watch: str, base: Optional[Path] = None) -> Path:
    return artifact_dir(prog, watch, base=base, create=True) / "manifest.jsonl"


def _append_manifest_entry(prog: str, watch: str, meta: ArtifactMeta, base: Optional[Path] = None) -> None:
    path = _manifest_path(prog, watch, base=base)
    entry = meta.to_dict()
    text = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
    _atomic_write_append(path, text.encode("utf-8"))


def _atomic_write_append(path: Path, data: bytes) -> None:
    _ensure_dir(path.parent)
    # open in append-binary mode. This is atomic for single process writes; not guaranteed atomic across processes,
    # but manifest is append-only; for stronger guarantees use file locking (not implemented here).
    with open(path, "ab") as f:
        f.write(data)


def read_manifest(prog: str, watch: str, base: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = _manifest_path(prog, watch, base=base)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip corrupt lines
                continue
    return out


# -------------------------
# Save / load helpers
# -------------------------
def save_bytes(data: bytes, name: str, program: str, watchlist: str, ext: str = ".bin",
               base: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
    d = artifact_dir(program, watchlist, base=base, create=True)
    fname = make_artifact_filename(name, ext)
    dest = d / fname
    _atomic_write_bytes(dest, data)
    meta = ArtifactMeta(program=program, watchlist=watchlist, name=name, filename=fname, ext=ext,
                        created_at=_utc_now_iso(), size_bytes=len(data), extra=metadata or {})
    # write sidecar metadata
    side = dest.with_suffix(dest.suffix + ".meta.json")
    _atomic_write_json(side, meta.to_dict())
    _append_manifest_entry(program, watchlist, meta, base=base)
    return dest


def load_bytes(path: Path) -> bytes:
    p = Path(path)
    return p.read_bytes()


def save_text(text: str, name: str, program: str, watchlist: str, ext: str = ".txt",
              base: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
    b = text.encode("utf-8")
    return save_bytes(b, name, program, watchlist, ext=ext, base=base, metadata=metadata)


def save_json(obj: Any, name: str, program: str, watchlist: str, ext: str = ".json",
              base: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
    text = json.dumps(obj, default=str, ensure_ascii=False, indent=2)
    return save_text(text, name, program, watchlist, ext=ext, base=base, metadata=metadata)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pickle(obj: Any, name: str, program: str, watchlist: str, ext: str = ".pkl",
                base: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
    d = artifact_dir(program, watchlist, base=base, create=True)
    fname = make_artifact_filename(name, ext)
    dest = d / fname
    # use joblib if available for better compression/performance; else use pickle via joblib.dump fallback
    try:
        if joblib is not None:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=str(d), prefix=".tmp_")
            tmp_file.close()
            joblib.dump(obj, tmp_file.name)
            os.replace(tmp_file.name, str(dest))
        else:
            import pickle
            tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=str(d), prefix=".tmp_")
            tmp_file.close()
            with open(tmp_file.name, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_file.name, str(dest))
    except Exception:
        # fallback atomic bytes write of pickle bytes
        import pickle
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write_bytes(dest, b)
    meta = ArtifactMeta(program=program, watchlist=watchlist, name=name, filename=fname, ext=ext,
                        created_at=_utc_now_iso(), size_bytes=dest.stat().st_size, extra=metadata or {})
    side = dest.with_suffix(dest.suffix + ".meta.json")
    _atomic_write_json(side, meta.to_dict())
    _append_manifest_entry(program, watchlist, meta, base=base)
    return dest


def load_pickle(path: Path) -> Any:
    p = Path(path)
    if joblib is not None:
        try:
            return joblib.load(str(p))
        except Exception:
            pass
    # fallback to pickle
    import pickle
    with open(p, "rb") as fh:
        return pickle.load(fh)


def save_dataframe(df: "pd.DataFrame", name: str, program: str, watchlist: str,
                   ext: str = ".parquet", base: Optional[Path] = None,
                   metadata: Optional[Dict[str, Any]] = None, **parquet_kwargs) -> Path:
    if pd is None:
        raise RuntimeError("pandas is required to save DataFrame")
    d = artifact_dir(program, watchlist, base=base, create=True)
    fname = make_artifact_filename(name, ext)
    dest = d / fname
    # write atomic: write to temp then replace
    fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".tmp_df_")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        if ext.lower().endswith(".parquet"):
            df.to_parquet(tmp_path, **parquet_kwargs)
        elif ext.lower().endswith(".csv"):
            df.to_csv(tmp_path, index=False)
        else:
            # default to parquet
            df.to_parquet(tmp_path, **parquet_kwargs)
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    size = dest.stat().st_size if dest.exists() else 0
    meta = ArtifactMeta(program=program, watchlist=watchlist, name=name, filename=fname, ext=ext,
                        created_at=_utc_now_iso(), size_bytes=size, extra=metadata or {})
    side = dest.with_suffix(dest.suffix + ".meta.json")
    _atomic_write_json(side, meta.to_dict())
    _append_manifest_entry(program, watchlist, meta, base=base)
    return dest


def load_dataframe(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required to load DataFrame")
    p = Path(path)
    _ext = p.suffix.lower()
    if _ext == ".parquet":
        return pd.read_parquet(p)
    if _ext == ".csv":
        return pd.read_csv(p)
    # fallback try parquet
    return pd.read_parquet(p)


# -------------------------
# Convenience wrappers (models, metadata)
# -------------------------
def save_model(obj: Any, name: str, program: str, watchlist: str,
               base: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Save model artifact (joblib/pickle) and record metadata."""
    return save_pickle(obj, name, program, watchlist, ext=".pkl", base=base, metadata=metadata)


def load_model(path: Path) -> Any:
    return load_pickle(path)


def save_metadata_json(meta_obj: Dict[str, Any], name: str, program: str, watchlist: str,
                       base: Optional[Path] = None) -> Path:
    return save_json(meta_obj, name, program, watchlist, ext=".meta.json", base=base)


def load_artifact_meta(path: Path) -> Optional[Dict[str, Any]]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            return load_json(meta_path)
        except Exception:
            return None
    # try manifest lookup
    return None


# -------------------------
# Listing & lookup functions
# -------------------------
def list_artifacts(program: str, watchlist: str, base: Optional[Path] = None,
                   pattern: Optional[str] = None, ext: Optional[str] = None) -> List[Path]:
    d = artifact_dir(program, watchlist, base=base, create=False)
    if not d.exists():
        return []
    res = []
    for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            if p.name == "manifest.jsonl":
                continue
            if pattern and pattern not in p.name:
                continue
            if ext and not p.name.endswith(ext):
                continue
            res.append(p)
    return res


def latest_artifact_path(program: str, watchlist: str, base: Optional[Path] = None,
                         name_contains: Optional[str] = None, ext: Optional[str] = None) -> Optional[Path]:
    items = list_artifacts(program, watchlist, base=base, pattern=name_contains, ext=ext)
    return items[0] if items else None


def find_artifact_by_filename(program: str, watchlist: str, filename: str, base: Optional[Path] = None) -> Optional[Path]:
    d = artifact_dir(program, watchlist, base=base, create=False)
    if not d.exists():
        return None
    p = d / filename
    return p if p.exists() else None


# -------------------------
# Small convenience CLI/testing
# -------------------------
def _demo():
    prog = "swing_buy_recommender"
    watch = "US01"
    print("artifact dir:", artifact_dir(prog, watch))
    print("saving sample text...")
    p = save_text("hello world", "greeting", prog, watch, ext=".txt", metadata={"note": "demo"})
    print("wrote:", p)
    print("manifest entries:", read_manifest(prog, watch))
    print("latest:", latest_artifact_path(prog, watch))


# -------------------------
# Exports
# -------------------------
__all__ = [
    "artifact_dir",
    "base_data_dir",
    "make_artifact_filename",
    "save_bytes",
    "load_bytes",
    "save_text",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_dataframe",
    "load_dataframe",
    "save_model",
    "load_model",
    "save_metadata_json",
    "load_artifact_meta",
    "list_artifacts",
    "latest_artifact_path",
    "find_artifact_by_filename",
    "read_manifest",
]

if __name__ == "__main__":
    _demo()

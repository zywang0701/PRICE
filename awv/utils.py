"""Shared helpers: seeding, parquet io, bootstrap confidence intervals."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

POOL_COLUMNS = [
    # identity
    "query_id", "rollout_id", "shard", "seed",
    # rollout observables
    "ell_tokens", "finish_reason", "answer_raw", "answer_canonical", "correct", "text",
    # score vector (every inference-computable score, logged once)
    "phi_lik", "phi_selfcert",
    "conf_mean", "conf_window_min", "conf_window_p10", "conf_tail", "phi_deepconf",
    "phi_conf",
    "phi_prm_prod", "phi_prm_last", "phi_prm_min",
]


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: pathlib.Path, obj) -> None:
    ensure_dir(pathlib.Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def read_pool(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = {"query_id", "ell_tokens", "answer_canonical", "correct"} - set(df.columns)
    if missing:
        raise ValueError(f"pool at {path} missing columns {missing}")
    return df


def append_columns(path: pathlib.Path, new_cols: pd.DataFrame, on=("query_id", "rollout_id")):
    """Merge new score columns into an existing pool parquet, idempotently and
    atomically (write to a temp file, then os.replace), so a crash mid-write
    can never corrupt the pool."""
    import os
    path = pathlib.Path(path)
    df = pd.read_parquet(path)
    df = df.drop(columns=[c for c in new_cols.columns if c not in on and c in df.columns])
    df = df.merge(new_cols, on=list(on), how="left", validate="one_to_one")
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return df


def bootstrap_ci(per_query_values: np.ndarray, stat_fn=np.mean, B: int = 1000,
                 level: float = 0.95, seed: int = 0):
    """Percentile bootstrap over queries. per_query_values: (Q,) or (Q, D)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(per_query_values)
    Q = v.shape[0]
    stats = np.empty((B,) + np.shape(stat_fn(v, axis=0)) if v.ndim > 1 else (B,))
    for b in range(B):
        idx = rng.integers(0, Q, size=Q)
        stats[b] = stat_fn(v[idx], axis=0) if v.ndim > 1 else stat_fn(v[idx])
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    point = stat_fn(v, axis=0) if v.ndim > 1 else stat_fn(v)
    return point, lo, hi

"""Score-calibrated eta-grid (plan §5, amendment 2026-07-08).

The Boltzmann weights e^{eta * phi} are scale-coupled to the score: sweeping eta
on a rescaled score s*phi is the same as sweeping s*eta on phi. A *fixed* numeric
eta-grid therefore has score-dependent resolution -- for a compressed-spread
score the low rungs all reproduce SC, while for phi in [0, 1] the top rungs
duplicate best-of-n. This module pins the grid in effective-tilt units u and
maps eta_k = u_k / s per (cell, score), where s is the score's within-query
spread on the calibration split.

phi is the score the *vote* consumes, i.e. the globally rank-normalized column
the engine feeds to e^{eta * phi} (see engine.normalize_score / build_tables).
The spread must be measured on that same transform, so score_spread() applies
engine.normalize_score by default.

Rung anchors (effective-tilt units u):
  u = 0            SC corner (plain majority vote)
  u = 0.05         smallest weight tilt across one IQR that the S=512 tables
                   resolve above Monte-Carlo noise
  u = log(T_bar)   single-rollout-dominance point (was 4.85 at T_bar=128;
                   4.16 at T_bar=64 after D9 -- Decision A, 2026-07-10)
  u = log(T_bar) + SHOULDER_MARGIN
                   best-of-n saturation shoulder (>~97% agreement with BoN at
                   the counts the policy actually commits)
  u = inf          best-of-n corner
Denser between 0 and log(T_bar), exactly where the SC <-> weighted <-> BoN
crossings live. 13 rungs.

Continuity: SHOULDER_MARGIN is the plan's own 4.85 -> 6.5 spacing, so
u_grid(128) reproduces the pre-amendment grid {..., 4.85, 6.5, inf} exactly,
and u_grid(64) re-anchors it to {..., 4.16, 5.81, inf}.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import engine

# Effective-tilt rungs below log(T_bar). These resolve the SC<->weighted<->BoN
# crossings and are T_bar-independent (they live in absolute tilt units; the
# 0.05 floor is set by the S=512 Monte-Carlo noise, not by the rollout cap).
U_LOW = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.2]

# Gap (in u) from log(T_bar) up to the best-of-n saturation shoulder. Chosen to
# match the plan's original 4.85 -> 6.5 spacing so u_grid(128) is exact.
SHOULDER_MARGIN = 1.65

# s is floored here; hitting the floor flags the score as degenerate (no usable
# within-query spread) for that cell.
SPREAD_FLOOR = 0.01

# Fraction of queries reserved to pin the grid (and gamma) before any frontier
# is computed (plan §5).
CALIB_FRAC = 0.20


def u_grid(T_bar: int) -> list[float]:
    """Pre-registered effective-tilt grid (plan §5), re-anchored to T_bar.

    Returns 13 rungs: the crossing-resolution rungs U_LOW, then the
    single-rollout-dominance point log(T_bar), the best-of-n saturation
    shoulder, and inf. Strictly increasing and closed with inf, matching the
    contract config asserts on cfg.engine.eta_grid.
    """
    if T_bar < 2:
        raise ValueError(f"T_bar must be >= 2, got {T_bar}")
    u_log = round(math.log(T_bar), 2)          # single-rollout dominance
    u_sat = round(u_log + SHOULDER_MARGIN, 2)  # best-of-n saturation shoulder
    grid = [*U_LOW, u_log, u_sat, math.inf]
    if not all(b > a for a, b in zip(grid, grid[1:])):
        raise ValueError(f"u-grid not strictly increasing for T_bar={T_bar}: {grid}")
    return grid


def eta_grid(s: float, T_bar: int) -> list[float]:
    """Concrete eta-grid for a cell: eta_k = u_k / s (inf maps to inf).

    Sorted ascending, starts at 0.0 (SC), ends with inf (best-of-n) -- the exact
    shape cfg.engine.eta_grid must have, so this can be dropped straight onto
    the config the engine reads.
    """
    if s <= 0:
        raise ValueError(f"spread s must be > 0, got {s}")
    return [(u / s if math.isfinite(u) else math.inf) for u in u_grid(T_bar)]


def _within_query_iqr(phi: np.ndarray, qid: np.ndarray) -> np.ndarray:
    """IQR (p75 - p25) of phi within each query. Returns one value per query."""
    df = pd.DataFrame({"q": qid, "phi": phi})
    return df.groupby("q")["phi"].agg(
        lambda x: float(np.subtract(*np.percentile(x.to_numpy(), [75, 25])))
    ).to_numpy()


def score_spread(pool: pd.DataFrame, score_column: str,
                 calib_query_ids=None, *, normalize: bool = True,
                 floor: float = SPREAD_FLOOR):
    """s(cell, score) = median over calibration-split queries of the within-query
    IQR of phi, floored.

    normalize=True applies engine.normalize_score (the GLOBAL rank transform the
    engine uses) over the whole pool first, then restricts to the calibration
    queries -- so s is measured on exactly the phi the vote consumes. Pass
    normalize=False if the column is already in [0, 1] and consumed as-is.

    Returns (s, degenerate); degenerate=True when s hits the floor.
    """
    phi = pool[score_column].to_numpy(dtype=float)
    if normalize:
        phi = engine.normalize_score(phi)                      # global, matches engine
    else:
        phi = np.clip(np.nan_to_num(phi, nan=0.0), 0.0, 1.0)
    qid = pool["query_id"].to_numpy()

    if calib_query_ids is not None:
        keep = np.isin(qid, np.asarray(list(calib_query_ids)))
        phi, qid = phi[keep], qid[keep]
    if len(qid) == 0:
        raise ValueError("no rows in the calibration split")

    iqr = _within_query_iqr(phi, qid)
    s = float(np.median(iqr)) if len(iqr) else 0.0
    degenerate = s <= floor
    return max(s, floor), degenerate


def calibration_split(query_ids, frac: float = CALIB_FRAC, seed: int = 0):
    """Deterministic (calib, readout) split of query ids.

    A fixed `frac` of the unique query ids, chosen by a seeded permutation, is
    the calibration split used to pin the grid and gamma; the rest are held out
    for frontiers/readouts. Reproducible: same (ids, frac, seed) -> same split.
    Returns (calib_ids, readout_ids) as sorted int arrays.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    qids = np.unique(np.asarray(query_ids, dtype=int))
    n_calib = max(1, int(round(frac * len(qids))))
    perm = np.random.default_rng((int(seed), 20250708)).permutation(len(qids))
    calib = np.sort(qids[perm[:n_calib]])
    readout = np.sort(qids[perm[n_calib:]])
    return calib, readout


def calibrate_cell(pool: pd.DataFrame, score_column: str, T_bar: int,
                   *, frac: float = CALIB_FRAC, seed: int = 0,
                   normalize: bool = True) -> dict:
    """One-call grid calibration for a (cell, score): split queries, measure the
    spread, and build the concrete eta-grid.

    Returns a dict ready to drop onto cfg.engine (`eta_grid`) plus the diagnostic
    fields the §5 record needs.
    """
    calib, readout = calibration_split(pool["query_id"].to_numpy(), frac, seed)
    s, degenerate = score_spread(pool, score_column, calib, normalize=normalize)
    us = u_grid(T_bar)
    etas = eta_grid(s, T_bar)
    return {
        "score_column": score_column,
        "T_bar": int(T_bar),
        "spread": s,
        "degenerate": degenerate,
        "u_grid": us,
        "eta_grid": etas,
        "eta_max": etas[-2],                 # log-shoulder rung, the compact cutoff
        "n_calib": int(len(calib)),
        "n_readout": int(len(readout)),
        "calib_query_ids": calib,
        "readout_query_ids": readout,
    }


# --------------------------------------------------------- diagnostics ---

def _committed_answers(phi: np.ndarray, cluster: np.ndarray,
                       etas_finite: np.ndarray, T_bar: int, S: int,
                       rng: np.random.Generator, none_cluster: int = -1):
    """Committed answer per (rung, perm, n) for one query, shape (K+1, S, T).

    Mirrors engine.query_tables' vote exactly but returns the argmax cluster id
    (rather than accuracy), so adjacent rungs can be compared. Last row is
    eta = inf (best-of-n). -1 marks a prefix that commits no answer.
    """
    m = len(phi)
    T = min(T_bar, m)
    A = int(cluster.max()) + 1
    K = len(etas_finite)

    idx = np.stack([rng.permutation(m)[:T] for _ in range(S)])
    phi_p, ans_p = phi[idx], cluster[idx]
    valid_p = ans_p != none_cluster

    out = np.full((K + 1, S, T), -1, dtype=int)
    W = np.zeros((S, K, A))
    rows = np.arange(S)
    best_phi = np.full(S, -np.inf)
    best_ans = np.full(S, -1, dtype=int)
    for n in range(T):
        w = np.exp(np.outer(phi_p[:, n], etas_finite)) * valid_p[:, n][:, None]
        W[rows, :, ans_p[:, n]] += w
        any_vote = W.max(axis=2) > 0                       # (S, K)
        arg = np.argmax(W, axis=2)                         # (S, K)
        out[:K, :, n] = np.where(any_vote, arg, -1).T

        p, a, ok = phi_p[:, n], ans_p[:, n], valid_p[:, n]
        take = ok & ((p > best_phi) | ((p == best_phi) & (a < best_ans)))
        best_phi = np.where(take, p, best_phi)
        best_ans = np.where(take, a, best_ans)
        out[K, :, n] = best_ans
    return out


def rung_redundancy(pool: pd.DataFrame, etas, T_bar: int, S: int,
                    score_column: str, *, calib_query_ids=None, seed: int = 0,
                    normalize: bool = True) -> pd.DataFrame:
    """Adjacent-rung redundancy diagnostic (plan §5).

    For every (query, permutation, n), does rung k commit the same answer as
    rung k-1? Returns a DataFrame with one row per adjacent rung pair and the
    mean agreement fraction. redundancy > 0.995 marks a wasted rung -- the lever
    for re-tuning the upper rungs after Decision A.
    """
    etas = np.asarray(etas, dtype=float)
    etas_finite = etas[np.isfinite(etas)]

    p = pool.copy()
    if "cluster" not in p.columns:
        p, meta = engine.cluster_answers(p)
    else:
        meta = None
    phi_all = (engine.normalize_score(p[score_column].to_numpy())
               if normalize
               else np.clip(np.nan_to_num(p[score_column].to_numpy(), nan=0.0), 0.0, 1.0))
    p = p.assign(_phi=phi_all)

    qids = np.sort(p["query_id"].unique())
    if calib_query_ids is not None:
        qids = qids[np.isin(qids, np.asarray(list(calib_query_ids)))]

    K1 = len(etas)                                  # rungs incl. inf
    agree = np.zeros(K1 - 1)
    denom = np.zeros(K1 - 1)
    groups = dict(tuple(p.groupby("query_id")))
    for qid in qids:
        grp = groups[qid]
        phi = np.clip(grp["_phi"].to_numpy(dtype=float), 0.0, 1.0)
        cluster = grp["cluster"].to_numpy(dtype=int)
        none_c = -1
        if meta is not None:
            none_c = meta[qid].get("none_cluster", -1)
        rng = np.random.default_rng((int(seed), int(qid)))
        ca = _committed_answers(phi, cluster, etas_finite, T_bar, S, rng, none_c)
        # compare each rung to the one below (both committing an answer)
        for k in range(1, K1):
            a, b = ca[k - 1], ca[k]
            both = (a >= 0) & (b >= 0)
            agree[k - 1] += ((a == b) & both).sum()
            denom[k - 1] += both.sum()

    frac = np.divide(agree, denom, out=np.full(K1 - 1, np.nan), where=denom > 0)
    return pd.DataFrame({
        "eta_lo": [f"{etas[k-1]:.3g}" for k in range(1, K1)],
        "eta_hi": [f"{etas[k]:.3g}" for k in range(1, K1)],
        "redundancy": frac,
        "wasted": frac > 0.995,
    })

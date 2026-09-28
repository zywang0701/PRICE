#!/usr/bin/env python3
"""Cross-corpus primitives: build the probe/target tables for ANY cell.

`lp_prep.py` was written for one cell whose 128-rollout pools serve as both the
feature source and the regression target, scored by 5-fold CV inside the same
500 queries.  That protocol conflates two things the learning-curve study has
to separate: the CALIBRATION corpus (MATH-train) and the EVALUATION corpus
(MATH-500).  This script builds the same artifacts for an arbitrary cell and
pool depth, so a corpus generated at m=64 can serve as training data for
primitives that are then evaluated, once, on MATH-500.

Two things are deliberately FROZEN to the evaluation cell rather than
recalibrated per corpus:

  the temperature grid  {tau_k}   and   the risk parameter  gamma

Recalibrating them on MATH-train would silently change what the estimator is
predicting between train and test, and the deployed pipeline pins both ex ante
anyway (Appendix: grid spreads and gamma come from a held-out calibration
split, then are frozen).

Usage:
  PYTHONPATH=.:delivery/scripts python delivery/scripts/lp3_prep.py \
      --cell qwen25-1.5b_mathtrain --n-probe 16
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from awv import engine

ROOT = pathlib.Path(__file__).resolve().parents[2]
GRID_CELL = "qwen25-1.5b_math500"      # the cell whose frozen grid we reuse
GAMMA_TAG = "g0p0029262"
S_PERM = 512
T_BAR = 64
SPLIT_SEED = 20260803


def frozen_grid(score: str):
    """The evaluation cell's temperature grid and risk parameter."""
    t = np.load(ROOT / "outputs" / "cells" / GRID_CELL / "phase3" / "tables"
                / f"{score}__{GAMMA_TAG}__s1.npz", allow_pickle=True)
    return (np.asarray(t["etas"], dtype=float), float(t["gamma"]))


def cached_clusters(cell: str, score: str) -> pd.DataFrame:
    """Cluster answers once per cell (math-verify is expensive) and cache."""
    cache = ROOT / "outputs" / "cells" / cell / "lp3" / "clusters.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cols = ["query_id", "rollout_id", "ell_tokens", "correct", "answer_canonical"]
    pool = pd.read_parquet(ROOT / "outputs" / "cells" / cell / "pools" / "pool.parquet",
                           columns=cols + [score])
    if cache.exists():
        return pool.merge(pd.read_parquet(cache), on=["query_id", "rollout_id"],
                          how="left")
    t0 = time.time()
    clustered, meta = engine.cluster_answers(pool)
    print(f"    clustered {len(pool)} rollouts in {time.time()-t0:.0f}s")
    keep = clustered[["query_id", "rollout_id", "cluster"]].copy()
    md = pd.DataFrame([{"query_id": q, "target_cluster": m["target"],
                        "none_cluster": m["none_cluster"]}
                       for q, m in meta.items()])
    keep = keep.merge(md, on="query_id", how="left")
    keep.to_parquet(cache, index=False)
    return pool.merge(keep, on=["query_id", "rollout_id"], how="left")


OVERLAP = False

def build(cell: str, score: str, n_probe: int):
    pool = cached_clusters(cell, score)
    # rank-normalize on THIS corpus's own pool, exactly as the engine does
    # before building tables; the transform is a global empirical CDF, so the
    # grid's effective-tilt units stay comparable across corpora
    pool[score] = engine.normalize_score(pool[score].to_numpy())
    etas, gamma = frozen_grid(score)
    etas_finite = etas[np.isfinite(etas)]
    qids = np.sort(pool["query_id"].unique())
    Q, K = len(qids), len(etas)
    print(f"  {cell}: {Q} queries, {len(pool)} rollouts, "
          f"{n_probe} probe / rest target, K={K}")

    V_tgt = np.zeros((Q, K, T_BAR), dtype=np.float32)
    logM_tgt = np.zeros(Q)
    phi_pr = np.zeros((Q, n_probe), dtype=np.float32)
    clu_pr = np.zeros((Q, n_probe), dtype=np.int32)
    ell_pr = np.zeros((Q, n_probe), dtype=np.int32)
    cor_pr = np.zeros((Q, n_probe), dtype=bool)
    ell_tgt_mean = np.zeros(Q)
    tgt_cluster = np.zeros(Q, dtype=int)
    none_cluster = np.zeros(Q, dtype=int)
    n_tgt = np.zeros(Q, dtype=int)

    groups = dict(tuple(pool.groupby("query_id")))
    t0 = time.time()
    for qi, qid in enumerate(qids):
        g = groups[qid].sort_values("rollout_id")
        m = len(g)
        perm = np.random.default_rng((SPLIT_SEED, int(qid))).permutation(m)
        pi, ti = (perm, perm) if OVERLAP else (perm[:n_probe], perm[n_probe:])

        phi = np.clip(np.nan_to_num(g[score].to_numpy(float), nan=0.0), 0, 1)
        clu = g["cluster"].to_numpy(int)
        ell = g["ell_tokens"].to_numpy(float)
        tc = int(g["target_cluster"].iloc[0])
        nc = int(g["none_cluster"].iloc[0])
        tgt_cluster[qi], none_cluster[qi] = tc, nc
        phi_pr[qi], clu_pr[qi] = phi[pi], clu[pi]
        ell_pr[qi], cor_pr[qi] = ell[pi], g["correct"].to_numpy(bool)[pi]

        rng = np.random.default_rng((SPLIT_SEED + 1, int(qid)))
        Vq = engine.query_tables(phi[ti], clu[ti], tc, etas_finite, T_BAR,
                                 S_PERM, rng, none_cluster=nc)
        V_tgt[qi, :, :Vq.shape[1]] = Vq
        if Vq.shape[1] < T_BAR:
            # fewer target rollouts than the replay cap: hold the curve flat
            # past the pool's size rather than inventing a continuation
            V_tgt[qi, :, Vq.shape[1]:] = Vq[:, -1:]
        gl = gamma * ell[ti]
        logM_tgt[qi] = np.log(np.mean(np.exp(gl - gl.max()))) + gl.max()
        ell_tgt_mean[qi] = float(ell[ti].mean())
        n_tgt[qi] = len(ti)
        if qi % 500 == 0:
            print(f"    q{qi}/{Q}  {time.time()-t0:.0f}s")

    out = ROOT / "outputs" / "cells" / cell / "lp3" / (f"prep_ov__{score}.npz" if OVERLAP else f"prep__{score}.npz")
    np.savez_compressed(
        out, V_tgt=V_tgt, logM_tgt=logM_tgt, phi_probe=phi_pr, clu_probe=clu_pr,
        ell_probe=ell_pr, ok_probe=(clu_pr != none_cluster[:, None]),
        cor_probe=cor_pr, ell_tgt_mean=ell_tgt_mean, target=tgt_cluster,
        none_cluster=none_cluster, n_target=n_tgt, query_id=qids, etas=etas,
        gamma=gamma, score=score, n_probe=n_probe, split_seed=SPLIT_SEED)
    print(f"  wrote {out.relative_to(ROOT)}  ({time.time()-t0:.0f}s)")


def main():
    global GRID_CELL
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="qwen25-1.5b_mathtrain")
    ap.add_argument("--score", default="phi_deepconf2")
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--overlap", action="store_true",
                    help="E003: probe = target = ALL rollouts (no split); writes prep_ov__*.npz")
    ap.add_argument("--grid-cell", default=GRID_CELL,
                    help="cell whose frozen tau grid / gamma to reuse (the model's own math500 cell)")
    args = ap.parse_args()
    GRID_CELL = args.grid_cell
    global OVERLAP; OVERLAP = args.overlap
    if OVERLAP: args.n_probe = 64
    build(args.cell, args.score, args.n_probe)


if __name__ == "__main__":
    main()

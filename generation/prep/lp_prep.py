#!/usr/bin/env python3
"""Learned-primitives study, step 1: clean probe/target split of every pool.

Motivation (2026-07-27): the CV study showed the whole gap between the
in-sample bound and the deployable frontier is PRIMITIVE ESTIMATION ERROR.
To study estimators of (Delta V, log M) honestly we need targets that are
statistically independent of whatever features an estimator is allowed to
look at.  Each query has 128 logged rollouts; we split them once, per query,
with a fixed seed:

    probe pool   32 rollouts   -> may be used as FEATURES (and, at deployment,
                                  they are paid for and re-used in the vote)
    target pool  96 rollouts   -> used ONLY to build the regression target
                                  V[k, n] and log M

so a probe-conditioned estimator can never "see" its own label noise.  The
target tables are rebuilt with the same engine primitives (S=512 permutations,
T_bar=64, odd counts at eta=0 handled downstream, abstentions voteless) from
the 96 target rollouts alone.

The score is rank-normalized ONCE on the full pool (engine.normalize_score is a
global empirical-CDF transform, so subsetting after normalizing keeps every
number identical to the published tables); answer clustering is also done once
on the full pool and cached, because math-verify equivalence is expensive.

Writes outputs/cells/<cell>/lp/prep__<score>.npz with
    V_tgt      (Q, K, T)   target reward table from the 96 target rollouts
    logM_tgt   (Q,)        target cost MGF from the 96 target rollouts
    phi_probe  (Q, 32)     rank-normalized score of the probe rollouts
    clu_probe  (Q, 32)     answer-cluster code of the probe rollouts
    ell_probe  (Q, 32)     token length of the probe rollouts
    ok_probe   (Q, 32)     probe rollout is not an abstention
    cor_probe  (Q, 32)     probe rollout graded correct (diagnostics only)
    ell_tgt_mean (Q,)      mean length over target rollouts (cost accounting)
    target/none_cluster/query_id/etas/gamma

Usage:  PYTHONPATH=. python delivery/scripts/lp_prep.py [--score phi_deepconf2]
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from awv import engine

ROOT = pathlib.Path(__file__).resolve().parents[2]
CELLS = ["qwen25-1.5b_math500", "llama32-3b_math500", "llama31-8b_math500"]
GAMMA_TAG = "g0p0029262"
N_PROBE = 32                 # rollouts reserved for features
S_PERM = 512                 # permutations, as in the published tables
T_BAR = 64
SPLIT_SEED = 20260727


def cached_clusters(cell: str) -> pd.DataFrame:
    """Cluster answers once per cell (math-verify merge) and cache to disk."""
    cache = ROOT / "outputs" / "cells" / cell / "lp" / "clusters.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cols = ["query_id", "rollout_id", "ell_tokens", "correct",
            "answer_canonical"]
    scores = ["phi_deepconf2", "phi_conf", "phi_prm_last", "phi_selfcert",
              "phi_lik", "conf_mean", "conf_tail"]
    pool = pd.read_parquet(ROOT / "outputs" / "cells" / cell / "pools" / "pool.parquet",
                           columns=cols + scores)
    if cache.exists():
        cl = pd.read_parquet(cache)
        pool = pool.merge(cl, on=["query_id", "rollout_id"], how="left")
        return pool
    t0 = time.time()
    clustered, meta = engine.cluster_answers(pool)
    print(f"    clustered in {time.time()-t0:.0f}s")
    keep = clustered[["query_id", "rollout_id", "cluster"]].copy()
    md = pd.DataFrame([{"query_id": q, "target_cluster": m["target"],
                        "none_cluster": m["none_cluster"]}
                       for q, m in meta.items()])
    keep = keep.merge(md, on="query_id", how="left")
    keep.to_parquet(cache, index=False)
    return pool.merge(keep, on=["query_id", "rollout_id"], how="left")


def build(cell: str, score: str):
    pool = cached_clusters(cell)
    # global rank transform, exactly as the engine does before building tables
    pool[score] = engine.normalize_score(pool[score].to_numpy())
    etas = np.asarray(np.load(ROOT / "outputs" / "cells" / cell / "phase3"
                              / "tables" / f"{score}__{GAMMA_TAG}__s1.npz",
                              allow_pickle=True)["etas"], dtype=float)
    gamma = float(np.load(ROOT / "outputs" / "cells" / cell / "phase3" / "tables"
                          / f"{score}__{GAMMA_TAG}__s1.npz",
                          allow_pickle=True)["gamma"])
    etas_finite = etas[np.isfinite(etas)]
    qids = np.sort(pool["query_id"].unique())
    Q, K = len(qids), len(etas)

    V_tgt = np.zeros((Q, K, T_BAR), dtype=np.float32)
    logM_tgt = np.zeros(Q)
    phi_pr = np.zeros((Q, N_PROBE), dtype=np.float32)
    clu_pr = np.zeros((Q, N_PROBE), dtype=np.int32)
    ell_pr = np.zeros((Q, N_PROBE), dtype=np.int32)
    cor_pr = np.zeros((Q, N_PROBE), dtype=bool)
    ell_tgt_mean = np.zeros(Q)
    tgt_cluster = np.zeros(Q, dtype=int)
    none_cluster = np.zeros(Q, dtype=int)
    n_tgt = np.zeros(Q, dtype=int)

    groups = dict(tuple(pool.groupby("query_id")))
    t0 = time.time()
    for qi, qid in enumerate(qids):
        g = groups[qid].sort_values("rollout_id")
        m = len(g)
        rs = np.random.default_rng((SPLIT_SEED, int(qid)))
        perm = rs.permutation(m)
        pi, ti = perm[:N_PROBE], perm[N_PROBE:]

        phi = np.clip(np.nan_to_num(g[score].to_numpy(float), nan=0.0), 0, 1)
        clu = g["cluster"].to_numpy(int)
        ell = g["ell_tokens"].to_numpy(float)
        cor = g["correct"].to_numpy(bool)
        tc = int(g["target_cluster"].iloc[0])
        nc = int(g["none_cluster"].iloc[0])
        tgt_cluster[qi], none_cluster[qi] = tc, nc

        phi_pr[qi], clu_pr[qi], ell_pr[qi], cor_pr[qi] = \
            phi[pi], clu[pi], ell[pi], cor[pi]

        # ---- target primitives from the DISJOINT target rollouts only ----
        rng = np.random.default_rng((SPLIT_SEED + 1, int(qid)))
        Vq = engine.query_tables(phi[ti], clu[ti], tc, etas_finite, T_BAR,
                                 S_PERM, rng, none_cluster=nc)
        V_tgt[qi, :, :Vq.shape[1]] = Vq
        if Vq.shape[1] < T_BAR:
            V_tgt[qi, :, Vq.shape[1]:] = Vq[:, -1:]
        gl = gamma * ell[ti]
        logM_tgt[qi] = np.log(np.mean(np.exp(gl - gl.max()))) + gl.max()
        ell_tgt_mean[qi] = float(ell[ti].mean())
        n_tgt[qi] = len(ti)
        if qi % 100 == 0:
            print(f"    q{qi}/{Q}  {time.time()-t0:.0f}s")

    out = ROOT / "outputs" / "cells" / cell / "lp" / f"prep__{score}.npz"
    np.savez_compressed(
        out, V_tgt=V_tgt, logM_tgt=logM_tgt, phi_probe=phi_pr,
        clu_probe=clu_pr, ell_probe=ell_pr, ok_probe=(clu_pr != none_cluster[:, None]),
        cor_probe=cor_pr, ell_tgt_mean=ell_tgt_mean, target=tgt_cluster,
        none_cluster=none_cluster, n_target=n_tgt, query_id=qids, etas=etas,
        gamma=gamma, score=score, n_probe=N_PROBE, split_seed=SPLIT_SEED)
    print(f"  wrote {out.relative_to(ROOT)}  ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="phi_deepconf2")
    ap.add_argument("--cells", nargs="*", default=CELLS)
    args = ap.parse_args()
    for cell in args.cells:
        print(f"{cell} ...")
        build(cell, args.score)


if __name__ == "__main__":
    main()

"""Offline engine (CPU): per-query reward tables and cost MGFs.

From one logged pool it computes, for every query q:
  V[k, n] = \\hat V_{eta_k}(n; q)  on the full (eta, n)-grid, eta=inf included,
            by S random permutations of the m logged rollouts (each prefix of a
            uniform permutation is an i.i.d. n-sample without replacement);
  log_Mhat = log( mean_i exp(gamma * ell_i) ),  the single-rollout cost MGF.

Votes are taken over *answer clusters*: unique canonical answer strings merged
by mathematical equivalence (math-verify when available), mirroring standard
self-consistency practice. Ties broken by the lexicographically smallest
cluster representative (a fixed deterministic rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import answers


# ------------------------------------------------------------- clustering ---

def normalize_score(values: np.ndarray) -> np.ndarray:
    """Map a score column to [0,1] by a GLOBAL empirical-CDF rank transform:
    a monotone rescaling (it preserves every within-query ordering the vote
    uses) that spreads compressed scores, e.g. step-reward products, across
    the unit interval so the shared eta grid has uniform resolution. Must be
    applied identically wherever the score is consumed (tables AND the online
    serving path)."""
    raw = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    order = pd.Series(raw).rank(method="average").to_numpy()
    return order / (len(order) + 1.0)


_PARSE_CACHE: dict = {}


def _parsed(s: str):
    if s not in _PARSE_CACHE:
        try:
            from math_verify import parse
            _PARSE_CACHE[s] = parse(f"${s}$")
        except Exception:
            _PARSE_CACHE[s] = None
    return _PARSE_CACHE[s]


def _equivalent(a: str, b: str) -> bool:
    if a == b:
        return True
    pa, pb = _parsed(a), _parsed(b)
    if pa is None or pb is None:
        return False
    try:
        from math_verify import verify
        return bool(verify(pa, pb))
    except Exception:
        return False


TOP_F_MERGE = 20      # pairwise equivalence attempted only among the top-F
                      # most frequent answers; rarer answers stay string-keyed
                      # (their vote weight is negligible)


def cluster_answers(pool: pd.DataFrame) -> pd.DataFrame:
    """Add a per-query 'cluster' column (int codes) and return cluster metadata.

    Cluster code order follows the lexicographic order of each cluster's
    representative (its lexicographically smallest member), so code 0 is the
    tie-break winner among equal weights via np.argmax's first-occurrence rule.
    The abstention key '<none>' gets a cluster code like any other (so the
    cost side still sees those rollouts) but is excluded from voting by the
    engine via the none_cluster index recorded here.
    """
    pool = pool.copy()
    pool["cluster"] = -1
    meta = {}
    for qid, grp in pool.groupby("query_id"):
        counts = grp["answer_canonical"].value_counts()
        uniq = sorted(counts.index)
        top = set(counts.index[:TOP_F_MERGE]) - {"<none>"}
        parent = list(range(len(uniq)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        idx_of = {s: i for i, s in enumerate(uniq)}
        top_list = sorted(top)
        for i in range(len(top_list)):
            for j in range(i + 1, len(top_list)):
                if _equivalent(top_list[i], top_list[j]):
                    a, b = idx_of[top_list[i]], idx_of[top_list[j]]
                    parent[find(b)] = find(a)
        roots = {}
        for i in range(len(uniq)):
            roots.setdefault(find(i), []).append(uniq[i])
        reps = sorted((min(v), tuple(v)) for v in roots.values())
        code_of = {}
        for code, (_, members) in enumerate(reps):
            for s in members:
                code_of[s] = code
        pool.loc[grp.index, "cluster"] = grp["answer_canonical"].map(code_of).to_numpy()

        # target cluster: the one containing any rollout graded correct
        correct_clusters = set(
            code_of[s] for s in grp.loc[grp["correct"], "answer_canonical"].unique())
        target = min(correct_clusters) if correct_clusters else -1
        meta[qid] = {"n_clusters": len(reps), "target": target,
                     "none_cluster": code_of.get("<none>", -1)}
    return pool, meta


# ------------------------------------------------------------------ tables ---

def query_tables(phi: np.ndarray, cluster: np.ndarray, target: int,
                 etas_finite: np.ndarray, T_bar: int, S: int,
                 rng: np.random.Generator, none_cluster: int = -1):
    """V table (K_f + 1, T) for one query; last row is eta = inf.

    Abstentions (the none_cluster) carry no vote weight and are skipped by the
    score-argmax, the standard self-consistency treatment; they still cost
    tokens, which enters through log_Mhat, not here. A prefix with only
    abstentions commits no answer and counts as incorrect.
    """
    m = len(phi)
    T = min(T_bar, m)
    A = int(cluster.max()) + 1
    K = len(etas_finite)

    idx = np.stack([rng.permutation(m)[:T] for _ in range(S)])      # (S, T)
    phi_p = phi[idx]                                                # (S, T)
    ans_p = cluster[idx]                                            # (S, T)
    valid_p = ans_p != none_cluster                                 # (S, T)

    V = np.zeros((K + 1, T), dtype=np.float32)
    if target < 0:
        return V

    W = np.zeros((S, K, A))
    best_phi = np.full(S, -np.inf)
    best_ans = np.full(S, -1, dtype=int)
    rows = np.arange(S)
    for n in range(T):
        w = np.exp(np.outer(phi_p[:, n], etas_finite))              # (S, K)
        w *= valid_p[:, n][:, None]                                 # abstain = 0 weight
        W[rows, :, ans_p[:, n]] += w
        any_vote = W.max(axis=2) > 0                                # (S, K)
        V[:K, n] = ((np.argmax(W, axis=2) == target) & any_vote).mean(axis=0)

        # eta = inf: running score-argmax over valid rollouts, lexicographic ties
        p, a, ok = phi_p[:, n], ans_p[:, n], valid_p[:, n]
        take = ok & ((p > best_phi) | ((p == best_phi) & (a < best_ans)))
        best_phi = np.where(take, p, best_phi)
        best_ans = np.where(take, a, best_ans)
        V[K, n] = (best_ans == target).mean()
    return V


def build_tables(pool: pd.DataFrame, cfg, score_column: str | None = None,
                 seed: int | None = None):
    """Tables for all queries. Returns dict of arrays (save with np.savez)."""
    eng = cfg.engine
    score_column = score_column or eng.score_column
    seed = eng.seed if seed is None else seed
    gamma = cfg.budget.gamma
    etas = np.asarray(eng.eta_grid, dtype=float)
    etas_finite = etas[np.isfinite(etas)]

    if "cluster" not in pool.columns:
        pool, meta = cluster_answers(pool)
    else:
        meta = {}
        for qid, grp in pool.groupby("query_id"):
            none_rows = grp.loc[grp["answer_canonical"] == "<none>", "cluster"] \
                if "answer_canonical" in grp.columns else pd.Series([], dtype=int)
            meta[qid] = {
                "target": int(grp.loc[grp["correct"], "cluster"].min())
                if grp["correct"].any() else -1,
                "none_cluster": int(none_rows.iloc[0]) if len(none_rows) else -1,
            }

    pool = pool.copy()
    pool[score_column] = normalize_score(pool[score_column].to_numpy())

    qids = np.sort(pool["query_id"].unique())
    Q, K, T = len(qids), len(etas), int(eng.T_bar)
    V = np.zeros((Q, K, T), dtype=np.float32)
    log_Mhat = np.zeros(Q)
    target_present = np.zeros(Q, dtype=bool)
    target_arr = np.full(Q, -1, dtype=int)

    groups = dict(tuple(pool.groupby("query_id")))
    for qi, qid in enumerate(qids):
        grp = groups[qid]
        phi = grp[score_column].to_numpy(dtype=float)
        phi = np.clip(np.nan_to_num(phi, nan=0.0), 0.0, 1.0)
        cluster = grp["cluster"].to_numpy(dtype=int)
        target = meta[qid]["target"]
        target_present[qi] = target >= 0
        target_arr[qi] = target
        rng = np.random.default_rng((seed, int(qid)))
        Vq = query_tables(phi, cluster, target, etas_finite, T, eng.S_permutations,
                          rng, none_cluster=meta[qid].get("none_cluster", -1))
        V[qi, :, :Vq.shape[1]] = Vq
        if Vq.shape[1] < T:                       # pool smaller than cap: pad flat
            V[qi, :, Vq.shape[1]:] = Vq[:, -1:]
        g = gamma * grp["ell_tokens"].to_numpy(dtype=float)
        log_Mhat[qi] = np.log(np.mean(np.exp(g - g.max()))) + g.max()

    return {"V": V, "log_Mhat": log_Mhat, "query_id": qids, "etas": etas,
            "target_present": target_present, "target": target_arr,
            "gamma": np.array(gamma),
            "score_column": np.array(score_column), "seed": np.array(seed)}

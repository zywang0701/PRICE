"""Deployable plug-in pipeline (CPU): kNN retrieval over h(q), neighbour-
averaged primitives, bisection initialization of the dual price, and the
multiplicative dual update, served on a held-out query stream.

Served queries never see their own tables: their primitives come only from
calibration neighbours; their realized rollouts are sampled from their own
logged pool (fresh random subsets), mirroring live generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .solver import _net_value, allowed_counts, first_crossing


# ----------------------------------------------------------------- helpers ---

def cosine_knn(H_query: np.ndarray, H_calib: np.ndarray, k: int) -> np.ndarray:
    """Indices (into calib) of the top-k cosine neighbours for each query row."""
    a = H_query / np.linalg.norm(H_query, axis=1, keepdims=True)
    b = H_calib / np.linalg.norm(H_calib, axis=1, keepdims=True)
    sims = a @ b.T
    return np.argsort(-sims, axis=1)[:, :k]


def retrieve_tables(neighbors: np.ndarray, V_cal: np.ndarray, logM_cal: np.ndarray):
    """Neighbour-average primitives: mean V table, mean log-MGF."""
    return V_cal[neighbors].mean(axis=0), float(logM_cal[neighbors].mean())


def solve_plug_in(V: np.ndarray, log_M: float, lam: float, etas: np.ndarray):
    best = (-np.inf, -1, -1)
    for k in range(len(etas)):
        counts = allowed_counts(etas[k], V.shape[1])
        n = first_crossing(V[k], log_M, lam, counts)
        val = _net_value(V[k, n - 1], lam, n, log_M)   # overflow-safe
        if val > best[0]:
            best = (val, k, n)
    assert best[1] >= 0, "plug-in solver found no candidate"
    return best[1], best[2]


def _log_mean_exp(logx: np.ndarray) -> float:
    mx = float(np.max(logx))
    return float(np.log(np.mean(np.exp(logx - mx))) + mx)


def calibrate_lambda1(V_cal, logM_cal, etas, B_target, iters=60):
    """Bisection on log-lambda: corpus-average MGF cost meets the target B.

    The bracket is derived from the tables themselves (a price high enough to
    commit one rollout everywhere, low enough that the cap binds everywhere),
    and the average cost is aggregated in the log domain, so neither extreme
    budgets nor cap-length queries overflow."""
    T = V_cal.shape[2]
    lhi = np.log(2.0) - float(np.min(logM_cal + np.log(np.expm1(np.maximum(logM_cal, 1e-9)))))
    # clamp so np.exp(llo) never underflows to an exact 0.0, which would
    # freeze the multiplicative dual update for the whole stream
    llo = max(-float((T + 1) * np.max(logM_cal)) + np.log(1e-4), np.log(1e-300))
    log_B = np.log(B_target)

    def log_avg_cost(loglam):
        logs = np.empty(V_cal.shape[0])
        for qi in range(V_cal.shape[0]):
            _, n = solve_plug_in(V_cal[qi], logM_cal[qi], np.exp(loglam), etas)
            logs[qi] = n * logM_cal[qi]
        return _log_mean_exp(logs)

    if log_avg_cost(llo) < log_B:          # cap binds before the budget is met
        return float(np.exp(llo))
    for _ in range(iters):
        mid = 0.5 * (llo + lhi)
        if log_avg_cost(mid) > log_B:      # too cheap a price -> overspending
            llo = mid
        else:
            lhi = mid
    return float(np.exp(0.5 * (llo + lhi)))


# -------------------------------------------------------------- the stream ---

def realize_vote(grp: pd.DataFrame, score_column: str, eta: float, n: int,
                 target: int, rng: np.random.Generator):
    """Draw n fresh rollouts from the query's logged pool and commit the vote.

    Abstentions (canonical answer '<none>') cost their tokens but carry no
    vote; a draw with only abstentions commits no answer (counted incorrect)."""
    take = rng.choice(len(grp), size=min(n, len(grp)), replace=False)
    phi = np.clip(np.nan_to_num(grp[score_column].to_numpy(float)[take]), 0.0, 1.0)
    cl = grp["cluster"].to_numpy(int)[take]
    ell = float(grp["ell_tokens"].to_numpy(float)[take].sum())
    if "answer_canonical" in grp.columns:
        ok = grp["answer_canonical"].to_numpy()[take] != "<none>"
    else:
        ok = np.ones(len(take), dtype=bool)
    phi, cl = phi[ok], cl[ok]
    if len(cl) == 0:
        return False, ell
    if np.isinf(eta):
        order = np.lexsort((cl, -phi))         # max phi, ties to smaller cluster
        winner = cl[order[0]]
    else:
        A = cl.max() + 1
        w = np.zeros(A)
        np.add.at(w, cl, np.exp(eta * phi))
        winner = int(np.argmax(w))
    return bool(winner == target), ell


def run_stream(cfg, tables: dict, pool: pd.DataFrame, H: np.ndarray,
               targets: dict, B_target: float, seed: int):
    """One pass over a held-out stream at MGF budget target B_target.

    tables: full oracle tables (used as the calibration store);
    targets: query_id -> target cluster id."""
    onl = cfg.online
    rng = np.random.default_rng(seed)
    qids = tables["query_id"]
    Q = len(qids)
    perm = rng.permutation(Q)
    n_cal = int(round(onl.calib_frac * Q))
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

    V_cal, logM_cal = tables["V"][cal_idx], tables["log_Mhat"][cal_idx]
    etas = tables["etas"]
    k = onl.k_neighbors or int(np.ceil(np.sqrt(n_cal)))
    neighbors = cosine_knn(H[test_idx], H[cal_idx], k)

    lam = calibrate_lambda1(V_cal, logM_cal, etas, B_target)
    score_col = str(tables["score_column"])
    # serve with the SAME global score normalization the tables were built on
    from .engine import normalize_score
    pool = pool.copy()
    pool[score_col] = normalize_score(pool[score_col].to_numpy())
    groups = dict(tuple(pool.groupby("query_id")))
    gamma, kappa = cfg.budget.gamma, cfg.budget.clip_kappa

    records, loglam_avg, served = [], 0.0, 0
    order = rng.permutation(len(test_idx))
    for t, ti in enumerate(order, start=1):
        qi = test_idx[ti]
        Vq, logMq = retrieve_tables(neighbors[ti], V_cal, logM_cal)
        k_star, n_star = solve_plug_in(Vq, logMq, lam, etas)
        qid = int(qids[qi])
        correct, ell = realize_vote(groups[qid], score_col, etas[k_star],
                                    n_star, targets[qid], rng)
        violation = min(np.exp(gamma * ell) / B_target - 1.0, kappa)
        a_t = onl.step_c / np.sqrt(t)
        lam = float(lam * np.exp(a_t * violation))
        loglam_avg += (np.log(lam) - loglam_avg) / t
        served += 1
        records.append({"t": t, "query_id": qid, "eta": float(etas[k_star]),
                        "count": n_star, "correct": correct, "ell": ell,
                        "cost_mgf": float(np.exp(gamma * ell)), "lam": lam})
    df = pd.DataFrame(records)
    summary = {"B_target": B_target, "seed": seed, "served": served,
               "accuracy": float(df["correct"].mean()),
               "mean_count": float(df["count"].mean()),
               "mean_cost_mgf": float(df["cost_mgf"].mean()),
               "mean_b_tokens": float(np.log(df["cost_mgf"].mean()) / gamma),
               "lambda_final": float(lam),
               "lambda_polyak": float(np.exp(loglam_avg))}
    return df, summary

"""Engine and solver unit tests against brute-force ground truth."""

import itertools

import numpy as np
import pandas as pd
import pytest

from awv.config import load_config
from awv.engine import build_tables, query_tables
from awv.solver import allowed_counts, first_crossing, grid_mask_for, solve_query


def brute_force_V(phi, cluster, target, eta, n):
    """Exact E[1{vote correct}] over all ordered n-prefixes (without replacement)."""
    m = len(phi)
    wins = 0
    total = 0
    for prefix in itertools.permutations(range(m), n):
        idx = list(prefix)
        A = cluster.max() + 1
        if np.isinf(eta):
            order = sorted(idx, key=lambda i: (-phi[i], cluster[i]))
            winner = cluster[order[0]]
        else:
            w = np.zeros(A)
            for i in idx:
                w[cluster[i]] += np.exp(eta * phi[i])
            winner = int(np.argmax(w))
        wins += winner == target
        total += 1
    return wins / total


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    m = 6
    phi = rng.uniform(0, 1, m).round(3)
    cluster = np.array([0, 1, 0, 1, 1, 0])
    return phi, cluster


def test_permutation_estimator_unbiased(tiny):
    phi, cluster = tiny
    etas_f = np.array([0.0, 1.0, 4.0])
    rng = np.random.default_rng(1)
    V = query_tables(phi, cluster, target=0, etas_finite=etas_f,
                     T_bar=3, S=20000, rng=rng)
    for k, eta in enumerate(etas_f):
        for n in (1, 2, 3):
            exact = brute_force_V(phi, cluster, 0, eta, n)
            assert abs(V[k, n - 1] - exact) < 0.02, (eta, n, V[k, n - 1], exact)


def test_eta_inf_is_score_argmax(tiny):
    phi, cluster = tiny
    rng = np.random.default_rng(2)
    V = query_tables(phi, cluster, target=0, etas_finite=np.array([0.0]),
                     T_bar=3, S=20000, rng=rng)
    for n in (1, 2, 3):
        exact = brute_force_V(phi, cluster, 0, np.inf, n)
        assert abs(V[-1, n - 1] - exact) < 0.02


def test_allowed_counts_lattice_rule():
    assert allowed_counts(0.0, 8).tolist() == [1, 3, 5, 7]
    assert allowed_counts(1.0, 4).tolist() == [1, 2, 3, 4]


def test_first_crossing_optimal_under_concavity():
    rng = np.random.default_rng(3)
    for _ in range(50):
        T = 40
        gains = np.sort(rng.uniform(0, 0.1, T - 1))[::-1]       # non-increasing
        V_row = np.concatenate([[0.4], 0.4 + np.cumsum(gains)]).clip(max=1.0)
        log_M = rng.uniform(0.01, 0.5)
        lam = 10 ** rng.uniform(-8, -1)
        counts = np.arange(1, T + 1)
        n_fc = first_crossing(V_row, log_M, lam, counts)
        vals = V_row - lam * np.exp(counts * log_M)
        assert np.isclose(vals[n_fc - 1], vals.max(), atol=1e-12)


def test_adaptive_dominates_every_fixed_per_lambda():
    rng = np.random.default_rng(4)
    etas = np.array([0.0, 1.0, np.inf])
    V = rng.uniform(0.2, 1.0, (len(etas), 30))
    V.sort(axis=1)                                              # monotone curves
    log_M = 0.05
    for lam in (1e-2, 1e-4, 1e-6):
        full = grid_mask_for("adaptive", etas)
        _, _, r_ad, c_ad = solve_query(V, log_M, lam, etas, full)
        v_ad = r_ad - lam * c_ad
        for k in range(len(etas)):
            mask = np.zeros(len(etas), dtype=bool); mask[k] = True
            _, _, r, c = solve_query(V, log_M, lam, etas, mask)
            assert v_ad >= r - lam * c - 1e-12


def test_build_tables_mhat_and_shapes():
    cfg = load_config()
    cfg.engine.T_bar = 8
    cfg.engine.S_permutations = 50
    cfg.engine.score_column = "phi_prm_prod"   # the synthetic pool below logs this score
    rng = np.random.default_rng(5)
    rows = []
    for qid in range(3):
        for j in range(16):
            correct = rng.random() < 0.5
            rows.append({
                "query_id": qid, "rollout_id": j,
                "ell_tokens": int(rng.integers(50, 400)),
                "answer_canonical": "1" if correct else "2",
                "correct": bool(correct),
                "phi_prm_prod": float(rng.uniform()),
                "cluster": 0 if correct else 1,
            })
    pool = pd.DataFrame(rows)
    tab = build_tables(pool, cfg)
    K = len(cfg.engine.eta_grid)
    assert tab["V"].shape == (3, K, 8)
    assert (tab["target"] == 0).all()
    g = cfg.budget.gamma
    for qi, qid in enumerate(tab["query_id"]):
        ell = pool.loc[pool.query_id == qid, "ell_tokens"].to_numpy(float)
        assert np.isclose(tab["log_Mhat"][qi], np.log(np.mean(np.exp(g * ell))))
    assert ((tab["V"] >= 0) & (tab["V"] <= 1)).all()


def test_abstentions_carry_no_vote():
    """A '<none>' bloc must never win the vote; all-abstention prefixes count
    as incorrect even when the target cluster is code 0."""
    rng = np.random.default_rng(7)
    m = 200
    # cluster 0 = target (correct), cluster 1 = wrong, cluster 2 = '<none>'
    cluster = rng.choice([0, 1, 2], size=m, p=[0.2, 0.2, 0.6])
    phi = np.where(cluster == 2, 0.99, 0.5)      # abstentions carry top scores
    V = query_tables(phi.astype(float), cluster, target=0,
                     etas_finite=np.array([0.0, 2.0]), T_bar=16, S=2000,
                     rng=rng, none_cluster=2)
    # without exclusion the high-phi abstention bloc would dominate eta=inf
    # and the counts; with exclusion the vote is a 50/50 race between 0 and 1
    assert 0.3 < V[0, 15] < 0.7
    # eta=inf, n=1: the draw is an abstention w.p. ~0.6 (no vote, incorrect),
    # so accuracy ~ the target's pool fraction ~0.2
    assert 0.10 < V[-1, 0] < 0.30
    # all-abstention prefixes at n=1 must not count as correct: P(valid at n=1)
    # is 0.4, so V at n=1 is at most ~0.25
    assert V[0, 0] < 0.3

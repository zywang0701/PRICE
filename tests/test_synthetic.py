"""Theory validation on the paper's Example 1 (two-type population with
deterministic class scores), where the reward curves have closed forms, plus
frontier utilities and an online smoke test. These must pass before any GPU
money is spent."""

import numpy as np
import pandas as pd
from scipy.stats import binom

from awv.config import load_config
from awv.engine import query_tables
from awv.frontier import crossing_pairs, interp_reward, upper_envelope
from awv.online import run_stream

ETA_C = np.log(1.5) / 0.8        # consistency threshold of Example 1


def closed_form_V(eta, n, p_star=0.4, phi_c=0.9, phi_w=0.1):
    """P[vote correct] for two answers, deterministic scores, ties to wrong."""
    if np.isinf(eta):
        return 1.0 - (1.0 - p_star) ** n
    c = np.exp(eta * phi_w) / (np.exp(eta * phi_c) + np.exp(eta * phi_w))
    # correct iff X > n*c, X ~ Binom(n, p_star); equality (tie) goes to wrong
    return float(1.0 - binom.cdf(np.floor(n * c + 1e-12), n, p_star))


def synthetic_query(rng, m=4000, p_star=0.4, phi_c=0.9, phi_w=0.1):
    """Pool draws for one Example-1 query. Correct answer 'b' > 'a' wrong, so
    clusters: wrong -> 0, correct -> 1 and ties break against the target."""
    correct = rng.random(m) < p_star
    phi = np.where(correct, phi_c, phi_w)
    cluster = np.where(correct, 1, 0)
    return phi, cluster, 1


def test_engine_matches_closed_form():
    rng = np.random.default_rng(0)
    phi, cluster, target = synthetic_query(rng)
    etas_f = np.array([0.0, 0.25, 1.0, 4.0])
    V = query_tables(phi, cluster, target, etas_f, T_bar=32, S=4000,
                     rng=np.random.default_rng(1))
    for k, eta in enumerate(list(etas_f) + [np.inf]):
        for n in (1, 3, 8, 21, 32):
            exact = closed_form_V(eta, n)
            got = V[k, n - 1]
            assert abs(got - exact) < 0.04, (eta, n, got, exact)


def test_consistency_direction():
    """Above the threshold the vote converges to the correct answer, below it
    to the wrong one (q_A geometry); mirrored for q_B. Convergence near the
    threshold is slow (the rate vanishes at eta_c), so we test at eta = 0.1
    and 4.0 with n = 128 and also pin the engine to the closed form."""
    n = 128
    etas_f = np.array([0.1, 4.0])           # well below / above eta_c ~ 0.507
    rng = np.random.default_rng(2)
    phi, cluster, target = synthetic_query(rng, m=8000)
    V = query_tables(phi, cluster, target, etas_f, T_bar=n, S=2000,
                     rng=np.random.default_rng(3))
    for k, eta in enumerate(etas_f):
        exact = closed_form_V(eta, n)
        assert abs(V[k, n - 1] - exact) < 0.05, (eta, V[k, n - 1], exact)
    assert V[0, n - 1] < 0.10                # eta < eta_c: inconsistent on q_A
    assert V[1, n - 1] > 0.90                # eta > eta_c: consistent on q_A

    # q_B mirrors the construction: correct mass 0.6 at the low score
    correct = np.random.default_rng(4).random(8000) < 0.6
    phi_b = np.where(correct, 0.1, 0.9)
    cluster_b = np.where(correct, 1, 0)
    Vb = query_tables(phi_b, cluster_b, 1, etas_f, T_bar=n, S=2000,
                      rng=np.random.default_rng(5))
    for k, eta in enumerate(etas_f):
        exact = closed_form_V(eta, n, p_star=0.6, phi_c=0.1, phi_w=0.9)
        assert abs(Vb[k, n - 1] - exact) < 0.05, (eta, Vb[k, n - 1], exact)
    assert Vb[0, n - 1] > 0.90               # eta < eta_c: consistent on q_B
    assert Vb[1, n - 1] < 0.10               # eta > eta_c: inconsistent on q_B


def test_frontier_utilities():
    B = np.array([1.0, 2.0, 4.0, 8.0])
    R = np.array([0.3, 0.5, 0.45, 0.7])     # envelope flattens the dip
    Bs, Rs = upper_envelope(B, R)
    assert (np.diff(Rs) >= 0).all()
    assert np.isclose(interp_reward(np.array([4.0]), B, R)[0], 0.5)

    curves = {
        "lo": {"B": B, "R": np.array([0.40, 0.50, 0.55, 0.56])},
        "hi": {"B": B, "R": np.array([0.20, 0.45, 0.60, 0.80])},
    }
    assert ("lo", "hi") in crossing_pairs(None, curves)
    no_cross = {
        "lo": curves["lo"],
        "flat": {"B": B, "R": np.array([0.1, 0.2, 0.3, 0.4])},
    }
    assert not crossing_pairs(None, no_cross)


def test_online_smoke():
    cfg = load_config()
    cfg.online.n_stream_passes = 1
    cfg.online.calib_frac = 0.5
    rng = np.random.default_rng(6)

    Q, K, T = 40, 3, 16
    etas = np.array([0.0, 1.0, np.inf])
    V = np.zeros((Q, K, T), dtype=np.float32)
    ceilings = rng.uniform(0.5, 1.0, Q)
    for qi in range(Q):
        V[qi] = ceilings[qi] * (1 - np.exp(-0.3 * np.arange(1, T + 1)))[None, :]
    tables = {
        "V": V, "log_Mhat": rng.uniform(0.05, 0.2, Q),
        "query_id": np.arange(Q), "etas": etas,
        "target": np.zeros(Q, dtype=int), "gamma": np.array(cfg.budget.gamma),
        "score_column": "phi",
    }
    H = rng.normal(size=(Q, 8))
    rows = []
    for qid in range(Q):
        for j in range(64):
            good = rng.random() < 0.6
            rows.append({"query_id": qid, "rollout_id": j,
                         "phi": rng.uniform(0.5, 1.0) if good else rng.uniform(0, 0.5),
                         "cluster": 0 if good else 1,
                         "ell_tokens": int(rng.integers(50, 300))})
    pool = pd.DataFrame(rows)
    targets = {qid: 0 for qid in range(Q)}

    df, summary = run_stream(cfg, tables, pool, H, targets,
                             B_target=2.0, seed=0)
    assert summary["served"] == Q - int(0.5 * Q)
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["mean_count"] >= 1
    assert np.isfinite(summary["lambda_final"])

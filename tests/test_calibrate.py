import numpy as np
import pandas as pd

from awv.calibrate import (budget_readouts, ell_gamma, gamma_table,
                           recommend_gamma)


def _pool(rng, Q=50, m=8):
    rows = []
    for qid in range(Q):
        mu = rng.uniform(300, 900)
        ell = np.minimum(rng.lognormal(np.log(mu), 0.5, m), 2048)
        for j in range(m):
            rows.append({"query_id": qid, "ell_tokens": float(ell[j])})
    return pd.DataFrame(rows)


def test_ell_gamma_monotone_and_above_mean():
    rng = np.random.default_rng(0)
    ell = rng.lognormal(np.log(500), 0.6, 5000).clip(max=2048)
    vals = [ell_gamma(ell, g) for g in (1e-4, 1e-3, 3e-3, 1e-2)]
    assert vals[0] >= ell.mean() - 1e-6
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] <= 2048 + 1e-6


def test_recommendation_rule():
    rng = np.random.default_rng(1)
    pool = _pool(rng)
    grid = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
    table = gamma_table(pool, grid, m_target=256)
    rec = recommend_gamma(table, max_risk_premium=1.5, max_mgf_rel_se=0.10)
    assert rec["gamma"] in grid
    if rec["feasible"]:
        row = table[table["gamma"] == rec["gamma"]].iloc[0]
        assert row["risk_premium_median"] <= 1.5
        assert row["mgf_rel_se_at_m"] <= 0.10
        # no larger grid point should also be feasible
        bigger = table[table["gamma"] > rec["gamma"]]
        assert not ((bigger["risk_premium_median"] <= 1.5)
                    & (bigger["mgf_rel_se_at_m"] <= 0.10)).any()


def test_budget_readouts_monotone():
    rng = np.random.default_rng(2)
    pool = _pool(rng)
    rd = budget_readouts(pool, 1e-3, [4, 16, 64])
    bs = [r["b_tokens"] for r in rd]
    assert all(b >= a for a, b in zip(bs, bs[1:]))
    assert bs[0] > 4 * 250          # at least counts x rough mean length

"""Unit tests for the score-calibrated eta-grid (awv/grids.py, plan §5)."""

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from awv import grids

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- u_grid ---

def test_u_grid_shape_and_contract():
    g = grids.u_grid(64)
    assert len(g) == 13
    assert g[0] == 0.0
    assert math.isinf(g[-1])
    assert all(b > a for a, b in zip(g, g[1:])), "strictly increasing"


def test_log_tbar_anchor_moves_with_cap():
    # Decision A: the single-rollout-dominance rung is log(T_bar).
    assert grids.u_grid(64)[-3] == pytest.approx(4.16, abs=0.005)   # log 64
    assert grids.u_grid(128)[-3] == pytest.approx(4.85, abs=0.005)  # log 128


def test_continuity_with_plan_grid_at_128():
    # SHOULDER_MARGIN is the plan's own 4.85 -> 6.5 spacing, so the pre-amendment
    # top rungs must reappear exactly at T_bar=128.
    g = grids.u_grid(128)
    assert g[-3:] == pytest.approx([4.85, 6.5, math.inf], abs=0.005, nan_ok=True)


def test_u_grid_rejects_tiny_cap():
    with pytest.raises(ValueError):
        grids.u_grid(1)


# ------------------------------------------------------------ eta_grid ---

def test_eta_grid_maps_by_inverse_spread():
    s = 0.25
    us = grids.u_grid(64)
    etas = grids.eta_grid(s, 64)
    assert etas[0] == 0.0
    assert math.isinf(etas[-1])
    assert all(b > a for a, b in zip(etas, etas[1:]))
    # eta_k = u_k / s on the finite rungs
    for u, e in zip(us[:-1], etas[:-1]):
        assert e == pytest.approx(u / s)


def test_smaller_spread_gives_larger_eta():
    # scale coupling: halving the spread doubles every finite eta.
    e_wide = np.array(grids.eta_grid(0.4, 64)[:-1])
    e_narrow = np.array(grids.eta_grid(0.2, 64)[:-1])
    assert np.allclose(e_narrow, 2 * e_wide)


def test_eta_grid_rejects_nonpositive_spread():
    with pytest.raises(ValueError):
        grids.eta_grid(0.0, 64)


# --------------------------------------------------------- score_spread ---

def _pool_with_iqr(target_iqr, Q=200, m=128, seed=0):
    """A synthetic pool whose within-query IQR of the raw score is ~target_iqr.
    A uniform on [c-iqr, c+iqr] has IQR = iqr, so draw that per query."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        c = rng.uniform(0.3, 0.7)
        v = rng.uniform(c - target_iqr, c + target_iqr, m)
        for x in v:
            rows.append({"query_id": q, "raw": float(x)})
    return pd.DataFrame(rows)


def test_spread_measured_without_normalization():
    pool = _pool_with_iqr(0.2)
    s, degen = grids.score_spread(pool, "raw", normalize=False)
    assert not degen
    assert s == pytest.approx(0.2, abs=0.03)


def test_degenerate_score_hits_floor():
    pool = pd.DataFrame({"query_id": np.repeat(np.arange(50), 128),
                         "raw": np.full(50 * 128, 0.5)})   # zero within-query spread
    s, degen = grids.score_spread(pool, "raw", normalize=False)
    assert degen
    assert s == grids.SPREAD_FLOOR


def test_spread_restricted_to_calib_split():
    pool = _pool_with_iqr(0.2, Q=100)
    calib, readout = grids.calibration_split(pool["query_id"].to_numpy(),
                                             frac=0.2, seed=3)
    assert len(calib) == 20 and len(readout) == 80
    assert set(calib).isdisjoint(set(readout))
    # spread on the calib subset must equal spread computed over just those qs
    s_all, _ = grids.score_spread(pool, "raw", calib, normalize=False)
    sub = pool[pool["query_id"].isin(calib)]
    s_sub, _ = grids.score_spread(sub, "raw", normalize=False)
    assert s_all == pytest.approx(s_sub)


# ---------------------------------------------------- calibration_split ---

def test_split_is_deterministic_and_covers_all():
    ids = np.arange(500)
    c1, r1 = grids.calibration_split(ids, 0.2, seed=1)
    c2, r2 = grids.calibration_split(ids, 0.2, seed=1)
    assert np.array_equal(c1, c2) and np.array_equal(r1, r2)
    assert np.array_equal(np.union1d(c1, r1), np.unique(ids))
    # different seed -> different split
    c3, _ = grids.calibration_split(ids, 0.2, seed=2)
    assert not np.array_equal(c1, c3)


# ---------------------------------------------------- real-pool smoke ---

_POOL = _ROOT / "outputs/cells/llama31-8b_math500/pools/pool.parquet"


@pytest.mark.skipif(not _POOL.exists(), reason="downloaded pool not present")
@pytest.mark.parametrize("score", ["phi_prm_last", "phi_conf", "phi_deepconf"])
def test_calibrate_cell_on_real_pool(score):
    pool = pd.read_parquet(_POOL, columns=["query_id", score])
    out = grids.calibrate_cell(pool, score, T_bar=64, seed=0)
    assert 0 < out["spread"] <= 1.0
    assert not out["degenerate"], f"{score} should have usable spread on this cell"
    assert out["eta_grid"][0] == 0.0
    assert math.isinf(out["eta_grid"][-1])
    assert all(b > a for a, b in zip(out["eta_grid"], out["eta_grid"][1:]))
    assert out["n_calib"] + out["n_readout"] == pool["query_id"].nunique()

"""Per-query solver and the lambda sweep (CPU).

Implements the paper's priced problem (eq:oracle-commit) on the estimated
tables: at dual price lambda, for each grid temperature take the count that
maximizes V - lambda M^n over the allowed counts (odd counts only at eta = 0,
the lattice/self-consistency rule), then pick the temperature with the best
net value. Methods differ only in which subset of the grid they are allowed
to use. Until 2026-09-16 the count was chosen by the first-crossing rule
(`first_crossing`, kept for reference); see awv/budget.py for the history.
"""

from __future__ import annotations

import numpy as np


def allowed_counts(eta: float, T: int) -> np.ndarray:
    """1-based counts the solver may commit at this temperature."""
    if eta == 0.0:
        return np.arange(1, T + 1, 2)
    return np.arange(1, T + 1)


_EXP_CLIP = 690.0     # exp(690) ~ 1e299, safely below float64 overflow


def _log_expm1(x: np.ndarray) -> np.ndarray:
    """log(e^x - 1) for x > 0, stable for both small and large x."""
    x = np.asarray(x, float)
    return np.where(x > 30.0, x, np.log(np.expm1(np.minimum(x, 30.0))))


def first_crossing(V_row: np.ndarray, log_M: float, lam: float,
                   counts: np.ndarray) -> int:
    """LEGACY (pre 2026-09-16): first-crossing count on the allowed ladder.

    Not used by the paper artifacts any more; `best_count` replaces it.

    The crossing test gain <= lam * M^n (M^{step} - 1) is evaluated in the log
    domain so that cap-length queries (large n * log_M) cannot overflow."""
    v = V_row[counts - 1].astype(float)
    gains = v[1:] - v[:-1]
    steps = (counts[1:] - counts[:-1]).astype(float)
    log_marginal = np.log(lam) + counts[:-1] * log_M + _log_expm1(steps * log_M)
    with np.errstate(divide="ignore"):
        log_gains = np.where(gains > 0, np.log(np.maximum(gains, 1e-300)), -np.inf)
    hit = np.nonzero(log_gains <= log_marginal)[0]
    return int(counts[hit[0]]) if len(hit) else int(counts[-1])


def best_count(V_row: np.ndarray, log_M: float, lam: float,
               counts: np.ndarray) -> int:
    """Exact maximizer of V - lam * M^n over the allowed counts (1-based).

    Ties go to the smallest count. The penalty exponent is clipped so that
    cap-length queries (large n * log_M) cannot overflow."""
    v = V_row[counts - 1].astype(float)
    log_pen = np.log(lam) + counts * log_M
    net = v - np.exp(np.minimum(log_pen, _EXP_CLIP))
    return int(counts[int(np.argmax(net))])


def _net_value(V_kn: float, lam: float, n: int, log_M: float) -> float:
    """V - lam * M^n with the penalty exponent clipped: any penalty beyond
    e^_EXP_CLIP already dwarfs the [0,1] reward, so ordering is preserved."""
    log_pen = np.log(lam) + n * log_M
    return float(V_kn - np.exp(min(log_pen, _EXP_CLIP)))


def solve_query(V: np.ndarray, log_M: float, lam: float, etas: np.ndarray,
                grid_mask: np.ndarray):
    """Best (eta index, count) at price lam over the allowed grid subset.

    Returns (k_star, T_star, reward, cost_mgf)."""
    T = V.shape[1]
    best = (-np.inf, -1, -1)
    for k in np.nonzero(grid_mask)[0]:
        counts = allowed_counts(etas[k], T)
        n_k = best_count(V[k], log_M, lam, counts)
        val = _net_value(V[k, n_k - 1], lam, n_k, log_M)
        if val > best[0]:
            best = (val, k, n_k)
    _, k_star, T_star = best
    assert k_star >= 0, "solver found no candidate; check grid_mask and tables"
    cost_mgf = float(np.exp(min(T_star * log_M, _EXP_CLIP)))
    return k_star, T_star, float(V[k_star, T_star - 1]), cost_mgf


def lambda_grid(tables: dict, n_lambdas: int) -> np.ndarray:
    """Log-spaced prices spanning 'commit one rollout everywhere' down to
    'rollout cap binds': bracket from the tables' own marginal scales."""
    V, log_M = tables["V"], tables["log_Mhat"]
    max_gain = 1.0
    # cheapest possible marginal cost across queries at n = 1
    min_margin = np.min(np.exp(log_M) * (np.exp(log_M) - 1.0))
    lam_hi = 2.0 * max_gain / max(min_margin, 1e-12)
    T = V.shape[2]
    # lower bracket from the p90 cap cost (log domain): deep enough that the
    # cap binds on 90% of queries (so large-budget overtakings stay inside the
    # sweep) without stretching the grid over outlier decades
    log_cap_p90 = float(np.quantile(T * log_M, 0.90))
    lam_lo = float(np.exp(np.log(1e-4) - min(log_cap_p90, 690.0)))
    return np.geomspace(lam_hi, max(lam_lo, 1e-300), n_lambdas)


def sweep(tables: dict, lambdas: np.ndarray, grid_mask: np.ndarray):
    """Run the solver at every price. Returns per-(lambda, query) commitments.

    Output dict of arrays with shape (L, Q): reward, cost_mgf, count, eta_idx."""
    V, log_M, etas = tables["V"], tables["log_Mhat"], tables["etas"]
    L, Q = len(lambdas), V.shape[0]
    out = {key: np.zeros((L, Q)) for key in ("reward", "cost_mgf", "count", "eta_idx")}
    for li, lam in enumerate(lambdas):
        for qi in range(Q):
            k, n, r, c = solve_query(V[qi], log_M[qi], lam, etas, grid_mask)
            out["reward"][li, qi] = r
            out["cost_mgf"][li, qi] = c
            out["count"][li, qi] = n
            out["eta_idx"][li, qi] = k
    return out


def grid_mask_for(method: str, etas: np.ndarray, wsc_eta: float = 1.0) -> np.ndarray:
    """SC = {0}; WSC = one intermediate eta; BoN = {inf}; adaptive = full grid."""
    mask = np.zeros(len(etas), dtype=bool)
    if method == "sc":
        mask[etas == 0.0] = True
    elif method == "wsc":
        k = int(np.argmin(np.abs(etas[np.isfinite(etas)] - wsc_eta)))
        mask[k] = True
    elif method == "bon":
        mask[np.isinf(etas)] = True
    elif method == "adaptive":
        mask[:] = True
    else:
        raise ValueError(method)
    assert mask.any()
    return mask

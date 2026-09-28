"""Fixed-budget frontier evaluation — the paper's dual-price calibration.

Algorithm (main_arxiv.tex, eq:oracle-commit): given a token budget b with
MGF target B = e^{gamma b}, find the dual price lambda by bisection so that the
population mean cost equals the budget,

    (1/Q) sum_q  M(q)^{n*(lambda; q)}  =  B ,      C(lambda) is monotone decreasing,

then read off the reward via the per-query maximizer of the priced problem
max_{tau, n} V_tau(n;q) - lambda M(q)^n  over the finite grid (odd counts only
at eta = 0, all counts elsewhere). Because C(lambda) is a step function, the
budget is met exactly by mixing the two priced policies adjacent to B at the
supporting price (`policy_at_budget`, 2026-09-16), the randomization of the
paper's sec. 3.3. This is the fixed-budget readout: R(b) is evaluated at the
*same* b for every method, so frontiers are directly comparable.

History (2026-09-16): until this date the per-temperature count was chosen by
the FIRST-CROSSING rule (stop at the first count whose marginal gain no longer
pays its marginal cost), which under-spends at small prices on queries whose
estimated reward curve is not concave in the count. The exact maximizer is now
used everywhere; the legacy rule is kept as `_single_eta_vec_first_crossing`
for reference only.
"""

from __future__ import annotations

import math

import numpy as np

from . import solver


_EXP_CLIP = 690.0


def _single_eta_vec(Vk, log_M, eta, lam):
    """Exact per-query maximizer of V - lam * M^n for one temperature.

    Vk is (Q, T) for the chosen rung; returns (count, reward, net_value), each
    (Q,). The candidate counts are solver.allowed_counts (odd counts at eta=0,
    all counts otherwise); the penalty exponent is clipped like solver._net_value
    so cap-length queries cannot overflow. Ties in net value go to the smallest
    count (np.argmax first occurrence)."""
    Q, T = Vk.shape
    counts = solver.allowed_counts(float(eta), T)                  # 1-based
    Vc = Vk[:, counts - 1]                                         # (Q, m)
    log_pen = np.log(lam) + counts[None, :] * log_M[:, None]       # (Q, m)
    net = Vc - np.exp(np.minimum(log_pen, _EXP_CLIP))
    j = np.argmax(net, axis=1)
    rows = np.arange(Q)
    n_star = counts[j].astype(int)
    return n_star, Vc[rows, j], net[rows, j]


def _single_eta_vec_first_crossing(Vk, log_M, eta, lam):
    """LEGACY (pre 2026-09-16): vectorized first-crossing for one temperature.
    Not used by the paper artifacts any more; see the module docstring.

    Vk is (Q, T) for the chosen rung; returns (count, reward, net_value), each
    (Q,). Mirrors solver.first_crossing / _net_value exactly (odd-count lattice
    at eta=0, log-domain crossing test)."""
    Q, T = Vk.shape
    cols = np.arange(0, T, 2) if eta == 0.0 else np.arange(0, T)   # 0-based
    counts = (cols + 1).astype(float)                              # 1-based ladder
    Vc = Vk[:, cols]                                               # (Q, m)
    gains = Vc[:, 1:] - Vc[:, :-1]                                 # (Q, m-1)
    steps = (counts[1:] - counts[:-1])                            # (m-1,)
    n_prev = counts[:-1]                                           # (m-1,)
    log_marg = (np.log(lam) + n_prev[None, :] * log_M[:, None]
                + solver._log_expm1(steps[None, :] * log_M[:, None]))
    with np.errstate(divide="ignore"):
        log_gains = np.where(gains > 0.0, np.log(np.maximum(gains, 1e-300)), -np.inf)
    hit = log_gains <= log_marg                                   # (Q, m-1)
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)                                    # first True (0 if none)
    n_star = np.where(any_hit, counts[first], counts[-1]).astype(int)
    reward = Vk[np.arange(Q), n_star - 1]
    pen = np.exp(np.minimum(np.log(lam) + n_star * log_M, _EXP_CLIP))
    return n_star, reward, reward - pen


def per_query_cost_reward(tables, grid_mask, lam):
    """Per-query (cost_mgf, reward, count) at dual price lam over the grid subset.

    Vectorized over queries; loops only the (<=13) active rungs and takes the
    per-query argmax net value across them (the eta* choice)."""
    V, log_M, etas = tables["V"], tables["log_Mhat"], np.asarray(tables["etas"])
    Q = V.shape[0]
    best_val = np.full(Q, -np.inf)
    best_n = np.ones(Q, dtype=int); best_r = np.zeros(Q)
    for k in np.nonzero(grid_mask)[0]:
        n_k, r_k, val_k = _single_eta_vec(V[:, k, :], log_M, float(etas[k]), lam)
        take = val_k > best_val
        best_val = np.where(take, val_k, best_val)
        best_n = np.where(take, n_k, best_n)
        best_r = np.where(take, r_k, best_r)
    cost = np.exp(np.minimum(best_n * log_M, _EXP_CLIP))
    return cost, best_r, best_n.astype(float)


def _mean_cost(tables, grid_mask, lam):
    return float(per_query_cost_reward(tables, grid_mask, lam)[0].mean())


def lambda_for_budget(tables, grid_mask, B, lam_lo, lam_hi,
                      rtol=3e-3, iters=34):
    """Approximate scalar price with C(lambda) ~ B (legacy helper, tolerance rtol).

    Kept for callers that need one deterministic price. The paper's fixed-budget
    readouts use `policy_at_budget`, which recovers the budget exactly by mixing
    the two priced policies adjacent to B."""
    cL = _mean_cost(tables, grid_mask, lam_lo)          # max cost
    cH = _mean_cost(tables, grid_mask, lam_hi)          # min cost (~1 rollout)
    if B >= cL:
        return lam_lo
    if B <= cH:
        return lam_hi
    lo, hi = lam_lo, lam_hi
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        cm = _mean_cost(tables, grid_mask, mid)
        if abs(cm - B) <= rtol * B:
            return mid
        if cm > B:          # cost too high -> raise the price
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def policy_at_budget(tables, grid_mask, B, lam_lo, lam_hi, iters=64):
    """Exact fixed-budget readout at MGF budget B (paper sec. 3.3, randomization).

    Bisects in log-lambda until the bracket [lo, hi] (cost(lo) > B >= cost(hi))
    collapses onto the supporting price, then mixes the two adjacent priced
    policies with weight theta so that the mean plug-in cost equals B exactly.
    The mixture is a randomized committed policy (every query commits the
    lo-price action with probability theta), so its reward theta*R_lo +
    (1-theta)*R_hi lies on the upper concave hull of the achievable points.
    Outside the feasible range the nearest endpoint policy is returned and the
    realized budget is reported instead of B.

    Returns dict(b, R, Rq, count, lam, theta, C_lo, C_hi) with lam = hi (the
    feasible-side price) and Rq the per-query expected reward under the mixture."""
    gamma = float(tables["gamma"])
    cL, rL, nL = per_query_cost_reward(tables, grid_mask, lam_lo)     # max-cost end
    cH, rH, nH = per_query_cost_reward(tables, grid_mask, lam_hi)     # min-cost end
    if B >= cL.mean():
        return {"b": math.log(cL.mean()) / gamma, "R": float(rL.mean()), "Rq": rL,
                "count": float(nL.mean()), "lam": lam_lo, "theta": 1.0,
                "C_lo": float(cL.mean()), "C_hi": float(cL.mean())}
    if B <= cH.mean():
        return {"b": math.log(cH.mean()) / gamma, "R": float(rH.mean()), "Rq": rH,
                "count": float(nH.mean()), "lam": lam_hi, "theta": 0.0,
                "C_lo": float(cH.mean()), "C_hi": float(cH.mean())}
    # bisection in log-lambda (products lo*hi underflow for tiny prices at very large budgets)
    llo, lhi = math.log(lam_lo), math.log(lam_hi)
    for _ in range(iters):
        if lhi - llo < 1e-12:
            break
        lmid = 0.5 * (llo + lhi)
        if _mean_cost(tables, grid_mask, math.exp(lmid)) > B:
            llo = lmid
        else:
            lhi = lmid
    lo, hi = math.exp(llo), math.exp(lhi)
    c_lo, r_lo, n_lo = per_query_cost_reward(tables, grid_mask, lo)
    c_hi, r_hi, n_hi = per_query_cost_reward(tables, grid_mask, hi)
    C_lo, C_hi = float(c_lo.mean()), float(c_hi.mean())
    theta = 0.0 if C_lo <= C_hi else min(1.0, max(0.0, (B - C_hi) / (C_lo - C_hi)))
    Rq = theta * r_lo + (1.0 - theta) * r_hi
    return {"b": math.log(theta * C_lo + (1.0 - theta) * C_hi) / gamma, "R": float(Rq.mean()),
            "Rq": Rq, "count": float(theta * n_lo.mean() + (1.0 - theta) * n_hi.mean()),
            "lam": hi, "theta": theta, "C_lo": C_lo, "C_hi": C_hi}


def frontier_at_budgets(tables, grid_mask, b_grid, lam_lo, lam_hi):
    """Evaluate R(b) at each target token budget b (paper's fixed-budget readout).

    Each point is the exact budget-B mixture of `policy_at_budget`, so the
    reported budget equals the target inside the feasible range and the reward
    is the constrained optimum over the grid policies under the plug-in cost.

    Returns dict with:
      b        realized token budget  = log(mean cost)/gamma  (= the target when feasible)
      R        population mean reward
      lam      the feasible-side supporting price at each budget
      theta    mixing weight on the lower-price (higher-cost) policy
      Rq       per-query expected reward matrix (len(b_grid), Q) for bootstrap CIs
      count    mean committed rollout count
    """
    Q = tables["V"].shape[0]
    gamma = float(tables["gamma"])
    n = len(b_grid)
    b_out = np.empty(n); R = np.empty(n); lams = np.empty(n); cnt = np.empty(n)
    thetas = np.empty(n); Rq = np.empty((n, Q))
    for i, b in enumerate(b_grid):
        p = policy_at_budget(tables, grid_mask, math.exp(gamma * b), lam_lo, lam_hi)
        b_out[i] = p["b"]; R[i] = p["R"]; lams[i] = p["lam"]; cnt[i] = p["count"]
        thetas[i] = p["theta"]; Rq[i] = p["Rq"]
    return {"b": b_out, "R": R, "lam": lams, "theta": thetas, "Rq": Rq, "count": cnt}


def concave_frontier(b, R, gamma, n_out=80):
    """Exact frontier = upper concave hull of the achievable (B, R) points.

    Strong duality makes R(B) concave in the MGF budget B = e^{gamma b}; budgets
    between two achievable deterministic policies are attained by randomizing
    between them (the paper's auxiliary-uniform device), i.e. the straight edge
    of the hull in (B, R). Taking the hull removes the under-sampled polyline
    corners a raw point sequence shows, and densifying along the hull in B
    renders each randomized edge as the smooth curve it is on the log-b axis.

    Returns (b_out, R_out) sampled at n_out points from b_min to b_max.
    """
    b = np.asarray(b, float); R = np.asarray(R, float)
    B = np.exp(gamma * b)
    o = np.argsort(B); B, R = B[o], np.maximum.accumulate(R[o])
    keep = [0]
    for i in range(1, len(B)):
        while len(keep) >= 2:
            x0, y0 = B[keep[-2]], R[keep[-2]]
            x1, y1 = B[keep[-1]], R[keep[-1]]
            x2, y2 = B[i], R[i]
            # drop the middle vertex if the slope does not strictly decrease
            if (y2 - y1) * (x1 - x0) >= (y1 - y0) * (x2 - x1):
                keep.pop()
            else:
                break
        keep.append(i)
    Bh, Rh = B[keep], R[keep]
    # sample the output evenly on the log token-budget axis (dense at low b,
    # where the frontier is steepest), then read R linearly in B off the hull
    # edges (= the randomized policy between adjacent achievable vertices).
    b_lo, b_hi = np.log(Bh[0]) / gamma, np.log(Bh[-1]) / gamma
    b_out = np.geomspace(b_lo, b_hi, n_out)
    Rd = np.interp(np.exp(gamma * b_out), Bh, Rh)
    return b_out, Rd


def feasible_budget_range(tables, grid_mask, lam_lo, lam_hi, pad=0.999):
    """(b_min, b_max) reachable by this method: one rollout up to saturation."""
    gamma = float(tables["gamma"])
    cH = _mean_cost(tables, grid_mask, lam_hi)          # ~1 rollout
    cL = _mean_cost(tables, grid_mask, lam_lo)          # saturated
    return math.log(cH) / gamma / pad, math.log(cL) / gamma * pad

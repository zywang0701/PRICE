"""Population frontiers, readout budgets, adaptive gaps with bootstrap CIs."""

from __future__ import annotations

import numpy as np


def population_curve(sweep_out: dict, gamma: float):
    """Aggregate a sweep into the population frontier.

    Returns dict with per-lambda arrays: B (mean MGF cost), b_tokens
    (= log B / gamma, the interpretable axis), R (mean reward), mean_count."""
    B = sweep_out["cost_mgf"].mean(axis=1)
    R = sweep_out["reward"].mean(axis=1)
    return {"B": B, "b_tokens": np.log(B) / gamma, "R": R,
            "mean_count": sweep_out["count"].mean(axis=1)}


def upper_envelope(B: np.ndarray, R: np.ndarray):
    """Sort by budget and take the running best reward (a frontier is monotone)."""
    order = np.argsort(B)
    return B[order], np.maximum.accumulate(R[order])


def interp_reward(B_query: np.ndarray, B: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Frontier reward at budgets B_query (log-B linear interpolation, clamped)."""
    Bs, Rs = upper_envelope(B, R)
    return np.interp(np.log(B_query), np.log(Bs), Rs, left=Rs[0], right=Rs[-1])


def readout_lambda_indices(curve: dict, targets) -> list[int]:
    """Indices of the prices whose adaptive mean rollout count is closest to
    the pre-registered readout targets (e.g. 4, 16, 64). Warns when the grid
    cannot reach a target (clamped readout), so a too-shallow lambda grid is
    caught at run time rather than discovered in the plots."""
    out = []
    for t in targets:
        li = int(np.argmin(np.abs(curve["mean_count"] - t)))
        got = curve["mean_count"][li]
        if abs(got - t) > 0.25 * t:
            print(f"[frontier] WARNING: readout target {t} clamped to mean "
                  f"count {got:.1f}; widen/refine the lambda grid")
        out.append(li)
    return out


def ceiling_gap_ci(V: np.ndarray, B_boot: int = 1000, level: float = 0.95,
                   seed: int = 0):
    """Ceiling separation: E_q[max_{eta,n} V] - max_eta E_q[max_n V], with a
    query bootstrap. V has shape (Q, K, T)."""
    rng = np.random.default_rng(seed)
    per_q_ad = V.max(axis=(1, 2))                 # (Q,)
    per_q_eta = V.max(axis=2)                     # (Q, K)

    def gap(idx):
        return float(per_q_ad[idx].mean() - per_q_eta[idx].mean(axis=0).max())

    Q = V.shape[0]
    full = gap(np.arange(Q))
    boots = np.array([gap(rng.integers(0, Q, Q)) for _ in range(B_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return full, float(lo), float(hi)


def adaptive_gap_ci(adaptive: dict, fixed: dict[str, dict], li: int,
                    gamma: float, B_boot: int = 1000, level: float = 0.95,
                    seed: int = 0):
    """Gap = R_adaptive - max over fixed methods of R_fixed at the adaptive
    policy's realized budget, bootstrapped over queries.

    adaptive/fixed values are sweep outputs (dicts of (L, Q) arrays)."""
    rng = np.random.default_rng(seed)
    Q = adaptive["reward"].shape[1]

    def gap_for(idx):
        B_star = adaptive["cost_mgf"][li, idx].mean()
        R_star = adaptive["reward"][li, idx].mean()
        best_fixed = -np.inf
        for sw in fixed.values():
            Bf = sw["cost_mgf"][:, idx].mean(axis=1)
            Rf = sw["reward"][:, idx].mean(axis=1)
            best_fixed = max(best_fixed, float(interp_reward(
                np.array([B_star]), Bf, Rf)[0]))
        return R_star - best_fixed

    full = gap_for(np.arange(Q))
    boots = np.array([gap_for(rng.integers(0, Q, Q)) for _ in range(B_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return full, float(lo), float(hi)


def crossing_pairs(fixed_sweeps: dict[str, dict] | None, curves: dict[str, dict]):
    """Detect frontier crossings between fixed-eta methods: pairs whose reward
    ordering flips somewhere along the (shared) budget axis (point estimate)."""
    names = list(curves.keys())
    found = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = curves[names[i]], curves[names[j]]
            B_common = np.geomspace(max(a["B"].min(), b["B"].min()),
                                    min(a["B"].max(), b["B"].max()), 200)
            Ra = interp_reward(B_common, a["B"], a["R"])
            Rb = interp_reward(B_common, b["B"], b["R"])
            d = Ra - Rb
            if (d > 1e-6).any() and (d < -1e-6).any():
                found.append((names[i], names[j]))
    return found


def crossing_ci(sweep_a: dict, sweep_b: dict, gamma: float, B_boot: int = 500,
                level: float = 0.95, seed: int = 0):
    """CI-separated crossing witness for one pair: bootstrap the interpolated
    reward difference at the two budgets where the point estimate is most
    positive / most negative. Returns (separated, witness dict)."""
    rng = np.random.default_rng(seed)
    Q = sweep_a["reward"].shape[1]

    def diff_at(idx):
        Ba = sweep_a["cost_mgf"][:, idx].mean(axis=1)
        Ra = sweep_a["reward"][:, idx].mean(axis=1)
        Bb = sweep_b["cost_mgf"][:, idx].mean(axis=1)
        Rb = sweep_b["reward"][:, idx].mean(axis=1)
        B_common = np.geomspace(max(Ba.min(), Bb.min()),
                                min(Ba.max(), Bb.max()), 200)
        return B_common, (interp_reward(B_common, Ba, Ra)
                          - interp_reward(B_common, Bb, Rb))

    B_common, d0 = diff_at(np.arange(Q))
    i_pos, i_neg = int(np.argmax(d0)), int(np.argmin(d0))
    if d0[i_pos] <= 0 or d0[i_neg] >= 0:
        return False, {}
    boots = np.empty((B_boot, 2))
    for b in range(B_boot):
        idx = rng.integers(0, Q, Q)
        _, d = diff_at(idx)
        boots[b] = d[i_pos], d[i_neg]
    alpha = (1 - level) / 2
    lo_pos = float(np.quantile(boots[:, 0], alpha))
    hi_neg = float(np.quantile(boots[:, 1], 1 - alpha))
    witness = {"b_pos": float(np.log(B_common[i_pos]) / gamma),
               "d_pos": float(d0[i_pos]), "d_pos_ci_lo": lo_pos,
               "b_neg": float(np.log(B_common[i_neg]) / gamma),
               "d_neg": float(d0[i_neg]), "d_neg_ci_hi": hi_neg}
    return bool(lo_pos > 0 and hi_neg < 0), witness

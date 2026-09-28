#!/usr/bin/env python3
"""Learned-primitives study: estimator zoo + metrics (shared library).

Everything here obeys one rule: an estimator for a TEST query may use
  * h(q), the question embedding (free, no rollouts), and
  * at most m rollouts drawn from that query's PROBE pool (paid for, and
    re-used in the final vote at deployment),
and may be FIT only on calibration queries.  The target primitives live on a
disjoint set of 96 rollouts (see lp_prep.py), so no estimator can see its own
label noise.

Estimator interface
-------------------
    est.fit(train_idx, data)                 # calibration only
    Vhat, logMhat = est.predict(test_idx, data)
where Vhat is (n_test, K, T) and logMhat is (n_test,).

`data` is the per-cell dict from load_cell().
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------- utilities ---


def lcm_1d(v):
    """Least concave majorant of a 1-D curve (matches the deployed estimator)."""
    T = len(v)
    hull = [0]
    for i in range(1, T):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            if (a - o) * (v[i] - v[o]) - (v[a] - v[o]) * (i - o) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    out = np.empty(T)
    for j in range(len(hull) - 1):
        a, b = hull[j], hull[j + 1]
        out[a:b + 1] = np.interp(np.arange(a, b + 1), [a, b], [v[a], v[b]])
    return out


def apply_lcm(V):
    """Concave-project every (query, eta) curve of a (Q, K, T) block."""
    V = np.array(V, dtype=float, copy=True)
    for q in range(V.shape[0]):
        for k in range(V.shape[1]):
            V[q, k] = lcm_1d(V[q, k])
    return V


def folds_of(Q, n_folds=5, seed=20260726):
    """The CV study's fold structure, reproduced exactly."""
    perm = np.random.default_rng(seed).permutation(Q)
    return [np.sort(f) for f in np.array_split(perm, n_folds)], perm


# ------------------------------------------------------- probe simulation ---


def simulate_vote_dist(clu, phi, etas_finite, T, S, rng, weights=None):
    """P[k, n, a] = probability the Boltzmann vote at temperature k with n
    rollouts elects answer cluster `a`, bootstrapped from an empirical
    rollout distribution.

    THIS USES NO LABELS.  It is the label-free half of the primitive: the vote
    DYNAMICS.  A caller turns it into V by weighting with a correctness head,
        V[k, n] = sum_a P[k, n, a] * q(a),
    where q(a) = P(cluster a is the target) must be predicted separately
    (lp_models.CorrectnessHead), because at deployment we observe which answers
    the model produced but not which one is right.

    Convention: cluster code -1 marks abstentions ('<none>'), which carry no
    vote weight; a prefix in which every rollout abstained elects nothing and
    is accumulated into the last column (index A), whose correctness is 0.
    Draws are WITH replacement, so unlike engine.query_tables (which permutes a
    fixed pool and stops at its size) this extends to any n -- what a probe of
    m << T requires.
    """
    clu = np.asarray(clu, int)
    phi = np.asarray(phi, float)
    w = np.ones(len(clu)) if weights is None else np.asarray(weights, float)
    p = w / w.sum()
    A = int(max(clu.max(), 0)) + 1              # answer clusters 0..A-1
    K = len(etas_finite)
    idx = rng.choice(len(clu), size=(S, T), p=p)
    phi_p, ans_p = phi[idx], clu[idx]
    valid_p = ans_p >= 0
    safe_ans = np.where(valid_p, ans_p, 0)

    winners = np.full((S, K + 1, T), A, dtype=np.int64)   # A = "no answer"
    W = np.zeros((S, K, A))
    best_phi = np.full(S, -np.inf)
    best_ans = np.full(S, A, dtype=np.int64)
    rows = np.arange(S)
    for n in range(T):
        ww = np.exp(np.outer(phi_p[:, n], etas_finite)) * valid_p[:, n][:, None]
        W[rows, :, safe_ans[:, n]] += ww
        any_vote = W.max(axis=2) > 0
        winners[:, :K, n] = np.where(any_vote, np.argmax(W, axis=2), A)
        pp, aa, ok = phi_p[:, n], safe_ans[:, n], valid_p[:, n]
        take = ok & ((pp > best_phi) | ((pp == best_phi) & (aa < best_ans)))
        best_phi = np.where(take, pp, best_phi)
        best_ans = np.where(take, aa, best_ans)
        winners[:, K, n] = best_ans

    # counts -> P[k, n, a] with one flat bincount
    kk = np.arange(K + 1)[None, :, None]
    nn = np.arange(T)[None, None, :]
    flat = ((kk * T + nn) * (A + 1) + winners).ravel()
    cnt = np.bincount(flat, minlength=(K + 1) * T * (A + 1))
    return cnt.reshape(K + 1, T, A + 1) / float(S)


def relabel_probe(clu_row, none_cluster):
    """Re-index the observed clusters of one probe by DESCENDING frequency.

    LABEL-FREE: codes come from what the probe saw, not from correctness --
    code 0 is the probe's plurality answer, 1 the runner-up, etc.; '<none>'
    becomes -1 (voteless).  Ties broken by first appearance.  Returns the new
    codes and, for each new code, the original cluster id (so a caller can look
    up the truth on CALIBRATION queries when fitting a correctness head).
    """
    clu_row = np.asarray(clu_row, int)
    keep = clu_row[clu_row != none_cluster]
    vals, counts = np.unique(keep, return_counts=True)
    order = vals[np.argsort(-counts, kind="stable")]
    code = {int(c): i for i, c in enumerate(order)}
    out = np.array([-1 if int(c) == none_cluster else code[int(c)]
                    for c in clu_row], dtype=int)
    return out, order.astype(int)


# ------------------------------------------------------------------ metrics ---


def primitive_errors(Vhat, logMhat, V_tgt, logM_tgt):
    """Estimation-error summary. Vhat/(Q,K,T), logMhat/(Q,).

    R^2 rows are measured against the POPULATION-MEAN predictor (the estimator
    that uses no per-query information at all), so R^2 = 0 means "learned
    nothing beyond the average query" and R^2 = 1 means "exact".  This is the
    honest yardstick here: raw RMSE on V is dominated by the fact that V(n)
    converges to 1{the modal answer is correct}, an almost-Bernoulli quantity.
    """
    d = Vhat - V_tgt
    dV_hat = np.diff(Vhat, axis=2)
    dV_tgt = np.diff(V_tgt, axis=2)
    band = slice(0, 16)          # decision band: crossings live at small n
    V_pop = V_tgt.mean(axis=0, keepdims=True)
    dV_pop = np.diff(V_pop, axis=2)
    lm_pop = logM_tgt.mean()

    def r2(err, base):
        return float(1.0 - (err ** 2).mean() / max((base ** 2).mean(), 1e-12))

    # --- shape vs level -------------------------------------------------
    # The decision only reads MARGINAL gains, so an estimator can have a
    # superb RMSE (it nailed the almost-Bernoulli level V(inf)) and still make
    # bad commitments (it flattened the curve).  Split the two: "level" is the
    # curve's value at the cap, "shape" is the curve minus its own level,
    # rescaled by its own total rise.
    def split(V):
        lev = V[:, :, -1]
        rise = V[:, :, -1] - V[:, :, 0]
        sh = (V - V[:, :, :1]) / np.where(np.abs(rise[:, :, None]) < 1e-3,
                                          np.nan, rise[:, :, None])
        return lev, sh
    lev_h, sh_h = split(Vhat)
    lev_t, sh_t = split(V_tgt)
    ok = np.isfinite(sh_h) & np.isfinite(sh_t)
    shape_rmse = float(np.sqrt(((sh_h - sh_t) ** 2)[ok].mean())) if ok.any() else float("nan")
    rise_h = Vhat[:, :, -1] - Vhat[:, :, 0]
    rise_t = V_tgt[:, :, -1] - V_tgt[:, :, 0]

    return {
        "V_rmse": float(np.sqrt((d ** 2).mean())),
        "level_rmse": float(np.sqrt(((lev_h - lev_t) ** 2).mean())),
        "shape_rmse": shape_rmse,
        "rise_mean_hat": float(rise_h.mean()),
        "rise_mean_tgt": float(rise_t.mean()),
        "rise_corr": float(np.corrcoef(rise_h.ravel(), rise_t.ravel())[0, 1]),
        "V_mae": float(np.abs(d).mean()),
        "V_r2": r2(d, V_tgt - V_pop),
        "dV_rmse": float(np.sqrt(((dV_hat - dV_tgt) ** 2).mean())),
        "dV_rmse_lo": float(np.sqrt(((dV_hat[:, :, band]
                                      - dV_tgt[:, :, band]) ** 2).mean())),
        "dV_r2_lo": r2(dV_hat[:, :, band] - dV_tgt[:, :, band],
                       dV_tgt[:, :, band] - dV_pop[:, :, band]),
        "logM_mae": float(np.abs(logMhat - logM_tgt).mean()),
        "logM_rmse": float(np.sqrt(((logMhat - logM_tgt) ** 2).mean())),
        "logM_r2": r2(logMhat - logM_tgt, logM_tgt - lm_pop),
        "logM_bias": float((logMhat - logM_tgt).mean()),
        "V_bias": float(d.mean()),
    }

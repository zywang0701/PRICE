"""Pilot calibration of the risk parameter gamma and the budget readouts.

gamma trades off three things:
  - tail guarantee: P[overspend by > t] <= e^{-gamma t}, so bigger gamma means
    a stronger guarantee at fixed t (equivalently a smaller t at fixed 5%);
  - interpretability: the risk-adjusted rollout length
    ell_gamma(q) = (1/gamma) log E[e^{gamma ell} | q] grows with gamma toward
    the max length; the *risk premium* ell_gamma / mean-length should stay
    moderate or the budget axis stops meaning "tokens";
  - estimability: M_hat(q) is an empirical MGF; its relative standard error
    at the pool size m grows with gamma (tail domination).

Pre-registered rule: choose the LARGEST gamma on the grid satisfying
  median_q risk premium <= max_risk_premium  AND
  projected rel-SE of M_hat at m_rollouts <= max_mgf_rel_se.
Report the full table so the choice is auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _logmeanexp(x: np.ndarray) -> float:
    mx = x.max()
    return float(np.log(np.mean(np.exp(x - mx))) + mx)


def ell_gamma(lengths: np.ndarray, gamma: float) -> float:
    """Risk-adjusted rollout length (1/gamma) log E[e^{gamma ell}]."""
    return _logmeanexp(gamma * np.asarray(lengths, float)) / gamma


def mgf_rel_se(lengths: np.ndarray, gamma: float, m: int) -> float:
    """Projected relative standard error of M_hat(q) from m i.i.d. rollouts of
    one query, using that query's empirical length distribution as the law."""
    z = np.exp(gamma * (np.asarray(lengths, float) - lengths.max()))
    mean, std = z.mean(), z.std()
    return float(std / (np.sqrt(m) * mean))


def gamma_table(pool: pd.DataFrame, gamma_grid, m_target: int,
                tail_prob: float = 0.05) -> pd.DataFrame:
    """One row per candidate gamma; risk premia AND rel-SEs are per-query
    (the plan gates on the median over queries; p90 reported for the tail)."""
    lengths_all = pool["ell_tokens"].to_numpy(float)
    per_query = [g["ell_tokens"].to_numpy(float)
                 for _, g in pool.groupby("query_id")]
    rows = []
    for gamma in gamma_grid:
        premia = np.array([ell_gamma(l, gamma) / l.mean() for l in per_query])
        rel_ses = np.array([mgf_rel_se(l, gamma, m_target) for l in per_query])
        rows.append({
            "gamma": gamma,
            "tail_t_for_guarantee": float(np.log(1.0 / tail_prob) / gamma),
            "ell_gamma_pooled": ell_gamma(lengths_all, gamma),
            "risk_premium_median": float(np.median(premia)),
            "risk_premium_p90": float(np.quantile(premia, 0.90)),
            "mgf_rel_se_at_m": float(np.median(rel_ses)),
            "mgf_rel_se_p90": float(np.quantile(rel_ses, 0.90)),
        })
    return pd.DataFrame(rows)


def recommend_gamma(table: pd.DataFrame, max_risk_premium: float,
                    max_mgf_rel_se: float) -> dict:
    ok = table[(table["risk_premium_median"] <= max_risk_premium)
               & (table["mgf_rel_se_at_m"] <= max_mgf_rel_se)]
    if len(ok) == 0:                       # fall back to the safest candidate
        row = table.sort_values("gamma").iloc[0]
        feasible = False
    else:
        row = ok.sort_values("gamma").iloc[-1]
        feasible = True
    return {"gamma": float(row["gamma"]), "feasible": feasible,
            "tail_t_for_guarantee": float(row["tail_t_for_guarantee"]),
            "risk_premium_median": float(row["risk_premium_median"]),
            "mgf_rel_se_at_m": float(row["mgf_rel_se_at_m"])}


def budget_readouts(pool: pd.DataFrame, gamma: float, mean_counts) -> list[dict]:
    """Provisional token budgets b for the pre-registered readout counts:
    b ~ count x population-average risk-adjusted rollout length. The final
    readouts are still pinned on the oracle sweep's realized mean counts."""
    per_query = [ell_gamma(g["ell_tokens"].to_numpy(float), gamma)
                 for _, g in pool.groupby("query_id")]
    lg = float(np.mean(per_query))
    return [{"mean_count": int(c), "b_tokens": c * lg,
             "B_mgf": float(np.exp(gamma * c * lg))} for c in mean_counts]

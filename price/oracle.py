"""PRICE-oracle (Algorithm 1): estimate each query's primitives from its own labeled rollout pool,
price the rollouts, and pick the count and the voting temperature that maximize accuracy minus priced cost."""
from dataclasses import dataclass
import numpy as np
from .vote import boltzmann_vote


@dataclass
class QueryPrimitives:
    """What PRICE needs to know about one query.

    V[k, n-1]        accuracy of the vote at temperature taus[k] over the first n rollouts   (eq. Vhat-oracle)
    log_mgf          log M(q) = log E[exp(gamma * length)] of one rollout                    (eq. Mhat-oracle)
    prefix_tokens    [S, T] tokens spent by the first n rollouts of each of the S orderings (for the budget readout)
    """
    V: np.ndarray
    log_mgf: float
    prefix_tokens: np.ndarray


def estimate_primitives(answers, scores, lengths, correct, taus, gamma, horizon=None, n_orderings=64, seed=0):
    """Resample a query's pool: S random orderings, vote on every prefix at every temperature (eq. Vhat-oracle)."""
    rng = np.random.default_rng(seed)
    m = len(answers); T = m if horizon is None else min(horizon, m); K = len(taus)
    hits = np.zeros((K, T)); prefix = np.zeros((n_orderings, T))
    for s in range(n_orderings):
        order = rng.permutation(m)[:T]
        prefix[s] = np.cumsum(np.asarray(lengths, float)[order])
        for n in range(1, T + 1):
            idx = order[:n]
            for k, tau in enumerate(taus):
                hits[k, n - 1] += boltzmann_vote([answers[i] for i in idx], [scores[i] for i in idx], tau) == correct
    log_mgf = float(np.log(np.mean(np.exp(gamma * np.asarray(lengths, float)))))
    return QueryPrimitives(hits / n_orderings, log_mgf, prefix)


def priced_choice(V, log_mgf, price):
    """The per-query decision at price lambda (eq. oracle-commit):
         (k, n) = argmax_{k, n}  V[k, n-1] - lambda * M^n
    The price decides the count, the count decides the voting rule: for each n the best temperature is
    argmax_k V[k, n-1], and the count then trades that accuracy against the priced cost lambda * M^n."""
    K, T = V.shape
    n_grid = np.arange(1, T + 1)
    net = V.max(0) - price * np.exp(n_grid * log_mgf)   # V*(n) - lambda M^n
    n = int(np.argmax(net)) + 1                          # ties resolve to the smaller count
    k = int(np.argmax(V[:, n - 1]))
    return k, n


def realized_budget(primitives, price, gamma):
    """Entropic-risk cost of the population at price lambda (eq. budget-price-search, C-hat)."""
    exps = []
    for p in primitives:
        _, n = priced_choice(p.V, p.log_mgf, price)
        exps.append(np.exp(gamma * p.prefix_tokens[:, n - 1]))
    return float(np.log(np.mean(np.concatenate(exps))) / gamma)


def price_for_budget(primitives, budget, gamma, price_grid):
    """Smallest grid price whose realized cost meets the budget, by bisection (C-hat is non-increasing in lambda)."""
    grid = np.sort(np.asarray(price_grid, float)); lo, hi = 0, len(grid) - 1
    if realized_budget(primitives, grid[hi], gamma) > budget:
        raise ValueError("price grid too narrow: even the largest price overspends the budget")
    while lo < hi:
        mid = (lo + hi) // 2
        if realized_budget(primitives, grid[mid], gamma) <= budget: hi = mid
        else: lo = mid + 1
    return float(grid[lo])


def frontier(primitives, gamma, price_grid):
    """Sweep the price: one (budget, accuracy) operating point per price, which traces the Pareto frontier."""
    points = []
    for lam in np.sort(np.asarray(price_grid, float))[::-1]:
        acc = np.mean([p.V[priced_choice(p.V, p.log_mgf, lam)[0], priced_choice(p.V, p.log_mgf, lam)[1] - 1] for p in primitives])
        points.append((realized_budget(primitives, lam, gamma), float(acc), float(lam)))
    return points

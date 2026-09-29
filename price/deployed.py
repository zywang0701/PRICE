"""PRICE-deployed (Algorithm 2): predicted primitives, a sequential stopping rule with a look-ahead window,
the estimated best temperature at the stop, and an online price update across the query stream."""
import numpy as np
from .vote import boltzmann_vote


def predicted_gain(V_star, n, window=4):
    """max_{1<=j<=w} [V*(n+j) - V*(n)] / j   with V*(n') = max_tau V_tau(n')   (left side of eq. deployed-stop)."""
    T = len(V_star)
    if n >= T:
        return -np.inf
    future = np.arange(n, min(n + window, T))            # indices of counts n+1 .. n+w
    return float(np.max((V_star[future] - V_star[n - 1]) / (future - n + 1)))


def should_stop(V_hat, n, tokens_so_far, log_mgf_hat, price, gamma, window=4):
    """Stop after rollout n when the predicted gain no longer covers the priced marginal cost (eq. deployed-stop):
         max_j [V*(n+j) - V*(n)]/j  <=  lambda * exp(gamma L_n) * (M_hat - 1)"""
    gain = predicted_gain(V_hat.max(0), n, window)
    marginal_cost = price * np.exp(gamma * tokens_so_far) * np.expm1(log_mgf_hat)
    return gain <= marginal_cost


def blend_log_mgf(prior_log_mgf, observed_lengths, gamma, prior_weight_scale=8.0):
    """Cost prediction refreshed after every rollout: the embedding prior blended with the empirical MGF of the
    lengths seen so far, the prior's weight decaying as n / (n + 8)."""
    n = len(observed_lengths)
    if n == 0:
        return prior_log_mgf
    observed = float(np.log(np.mean(np.exp(gamma * np.asarray(observed_lengths, float)))))
    w = n / (n + prior_weight_scale)
    return (1 - w) * prior_log_mgf + w * observed


def update_price(price, tokens_spent, budget, gamma, step):
    """Online price update after a query finishes (eq. online-price-update):
         lambda <- lambda * exp( eta * [exp(gamma (L - b)) - 1] )
    overspending raises the price for later queries, underspending lowers it."""
    return price * np.exp(step * (np.exp(gamma * (tokens_spent - budget)) - 1.0))


def run_query(draw_rollout, predict_curves, prior_log_mgf, taus, price, gamma, window=4, horizon=64):
    """Serve one query at price lambda.

    draw_rollout()                       -> (answer, score, length) of a fresh rollout
    predict_curves(answers, scores, lengths) -> V_hat [K, horizon], the predicted accuracy curves given what was observed
    Returns the voted answer, the count used, the temperature used and the tokens spent."""
    answers, scores, lengths = [], [], []
    for n in range(1, horizon + 1):
        a, phi, ell = draw_rollout()
        answers.append(a); scores.append(phi); lengths.append(ell)
        V_hat = predict_curves(answers, scores, lengths)
        log_mgf = blend_log_mgf(prior_log_mgf, lengths, gamma)
        if should_stop(V_hat, n, sum(lengths), log_mgf, price, gamma, window):
            break
    k = int(np.argmax(V_hat[:, n - 1]))                   # estimated best temperature at the stopping count
    return dict(answer=boltzmann_vote(answers, scores, taus[k]), count=n, tau=taus[k], tokens=float(sum(lengths)))

"""The Boltzmann weighted vote (Definition 2.1): one family that contains SC, score-weighted voting and BoN."""
import numpy as np


def boltzmann_vote(answers, scores, tau):
    """Return the winning answer among `answers` with per-rollout scores `scores` at inverse temperature `tau`.

    weight of answer a  =  sum over rollouts i with a_i = a of exp(tau * phi_i)
      tau = 0    -> self-consistency (plain majority)
      0 < tau    -> score-weighted voting
      tau = inf  -> best-of-n (the single highest-scored rollout)
    Ties between equal totals go to the answer holding the highest single score, then to the first answer seen.
    Abstentions are passed as answer None and cast no vote.
    """
    answers = list(answers); scores = np.asarray(scores, float)
    valid = [i for i, a in enumerate(answers) if a is not None]
    if not valid:
        return None
    if np.isinf(tau):
        return answers[max(valid, key=lambda i: scores[i])]
    total, best_single, order = {}, {}, []
    for i in valid:
        a = answers[i]
        if a not in total:
            total[a] = 0.0; best_single[a] = -np.inf; order.append(a)
        total[a] += np.exp(tau * scores[i]); best_single[a] = max(best_single[a], scores[i])
    return max(order, key=lambda a: (total[a], best_single[a], -order.index(a)))

"""A three-query world, in the spirit of Figure 1: run  python -m price.demo

  q1  easy    the correct answer dominates; one rollout settles it, SC is the right rule
  q2  worth it  the correct answer is a minority but well scored; more rollouts + score weighting pay off
  q3  hard    the correct answer is rare and the scores barely separate it; spending is soon not worth its price

Everything is synthetic and label-free at decision time; only the *estimation* of the curves uses labels,
exactly as PRICE-oracle does on a labeled pool."""
import numpy as np
from .oracle import estimate_primitives, priced_choice, price_for_budget, realized_budget, frontier
from .deployed import run_query

TAUS = [0.0, 1.0, 3.0, np.inf]          # SC, two score-weighted rules, BoN
GAMMA = np.log(20) / 1024               # the paper's risk parameter
WORLD = {                                # answer distribution, score means (correct / wrong), typical length
    "q1 easy":     dict(p_correct=0.80, score_correct=0.65, score_wrong=0.50, length=250),
    "q2 worth it": dict(p_correct=0.38, score_correct=0.80, score_wrong=0.40, length=400),
    "q3 hard":     dict(p_correct=0.12, score_correct=0.85, score_wrong=0.45, length=700),
}


def sample_rollout(spec, rng):
    correct = rng.random() < spec["p_correct"]
    answer = "A" if correct else rng.choice(["B", "C", "D"])
    score = np.clip(rng.normal(spec["score_correct" if correct else "score_wrong"], 0.12), 0, 1)
    length = spec["length"] * rng.lognormal(0.0, 0.25)      # lengths vary by roughly +-25% around the typical value
    return answer, float(score), float(length)


def make_pool(spec, m, rng):
    rows = [sample_rollout(spec, rng) for _ in range(m)]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def main(seed=0):
    rng = np.random.default_rng(seed)
    print("Estimating each query's accuracy curves V_tau(n) and cost MGF from a labeled pool of 128 rollouts ...")
    prim = {}
    for name, spec in WORLD.items():
        a, s, l = make_pool(spec, 128, rng)
        prim[name] = estimate_primitives(a, s, l, correct="A", taus=TAUS, gamma=GAMMA, horizon=32, n_orderings=32, seed=seed)

    print("\nOne price, one decision per query  (n = rollouts, tau = voting temperature; tau=0 is SC, inf is BoN)")
    for lam in [1e-3, 1e-4, 1e-5, 1e-6]:
        row = "  ".join(f"{name}: n={priced_choice(p.V, p.log_mgf, lam)[1]:2d} tau={TAUS[priced_choice(p.V, p.log_mgf, lam)[0]]:<4}"
                        for name, p in prim.items())
        print(f"  lambda={lam:.0e}   {row}")

    budget = 1500.0
    lam_b = price_for_budget(list(prim.values()), budget, GAMMA, np.logspace(-6, -1, 200))
    print(f"\nBudget b = {budget:.0f} risk-adjusted tokens per query  ->  supporting price lambda_b = {lam_b:.2e}")
    for name, p in prim.items():
        k, n = priced_choice(p.V, p.log_mgf, lam_b)
        print(f"  {name:12s} n={n:2d}  tau={TAUS[k]:<4}  predicted accuracy {p.V[k, n-1]:.2f}")
    print(f"  realized budget {realized_budget(list(prim.values()), lam_b, GAMMA):.0f}")

    print("\nThe frontier: sweep the price, read (budget, accuracy)")
    seen = set()
    for b, acc, lam in frontier(list(prim.values()), GAMMA, np.logspace(-7, -2, 60)):
        if round(b) not in seen:
            seen.add(round(b)); print(f"  budget {b:6.0f}   accuracy {acc:.3f}   (lambda {lam:.1e})")

    print("\nPRICE-deployed on fresh rollouts, using the estimated curves as the predictor and the paper's stopping rule (w=4)")
    for name, spec in WORLD.items():
        p = prim[name]
        out = run_query(draw_rollout=lambda: sample_rollout(spec, rng), predict_curves=lambda a, s, l: p.V,
                        prior_log_mgf=p.log_mgf, taus=TAUS, price=lam_b, gamma=GAMMA, window=4, horizon=32)
        print(f"  {name:12s} stopped at n={out['count']:2d}, voted at tau={out['tau']:<4}, answer {out['answer']} "
              f"({'correct' if out['answer']=='A' else 'wrong'}), {out['tokens']:.0f} tokens")


if __name__ == "__main__":
    main()

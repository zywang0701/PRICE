# Method and theory

[Home](../README.md) · [Results](results.md) · [Reproduction](../experiments/README.md)

PRICE (**P**riced **R**ollouts and **I**nference-time voting-rule **C**hoice under a token budg**E**t) chooses both how many rollouts to draw and how to aggregate their answers. A shared price connects each query's decision to the population's token budget.

## One family of voting rules

Given answers $a_i$ and scores $\phi_i$, the Boltzmann vote returns the answer with the greatest total weight:

$$
\hat a_\tau = \arg\max_a \sum_{i:a_i=a} \exp(\tau \phi_i).
$$

The parameter $\tau$ is an **inverse voting temperature**, distinct from the LLM's sampling temperature. At $\tau=0$, the rule is self-consistency (SC); positive finite values give score-weighted voting; the best-of-$n$ (BoN) endpoint selects the highest-scored rollout. The readable implementation is [`price/vote.py`](../price/vote.py); the experiment implementation and tie conventions are in [`awv/engine.py`](../awv/engine.py).

## The token budget

The experiments use the entropic-risk cost of the total tokens $L$ spent on a query:

$$
C_\gamma(L) = \frac{1}{\gamma} \log \mathbb{E}[\exp(\gamma L)].
$$

The expectation covers queries and rollout randomness. This cost is measured in tokens and penalizes expensive tails; it is not simply the mean number of tokens or a hard cap on every query. The paper uses $\gamma=\log(20)/1024$. The figures and table columns should be read using this risk-adjusted budget.

## Theory

### 1. Adaptive voting raises the accuracy ceiling

Under self-consistency, a query is solvable only when the correct answer is the most frequent one. Under best-of-n, only when its rollouts carry the highest scores. These sets differ, and the set of queries solvable by an adaptive rule is their union over temperatures $\tau$.

Let $p_\tau$ be the fraction of queries solvable by the fixed rule $\tau$, and $p_{\mathrm{adaptive}}$ the fraction solvable by some rule. Then

$$
p_{\mathrm{adaptive}} \ge \sup_\tau p_\tau.
$$

Drawing more rollouts under a fixed rule can never solve a query outside that rule's set; adapting the rule can. On MATH-500 the gap is 5.3 points on Qwen2.5-1.5B and 2.8 on Llama-3.2-3B over the best single temperature, and best-of-n on its own solves fewer than a fifth of the queries. Appendix A of the paper characterizes the consistency sets and when the inequality is strict.

<p align="center">
  <a href="../assets/fig_solvable.png"><img src="../assets/fig_solvable.png" width="100%" alt="Fraction of MATH-500 queries solvable by SC, BoN, the best fixed temperature, and adaptive voting"></a>
  <br><sub>Adaptive voting covers more queries than any single fixed voting rule in the evaluated grid.</sub>
</p>

A query counts as solvable by τ when the vote at τ over its full pool of 128 rollouts is correct, with the P(True) score and the 13-temperature grid of the paper; adaptive voting counts a query when some temperature on the grid solves it ([`readme_solvable_fraction.py`](../paper_assets/readme_solvable_fraction.py)).

### 2. No fixed rule is best at every budget, so the rule should adapt

Two theorems make this precise. For any two temperatures there is a query population on which their cost-accuracy frontiers cross, so no fixed temperature is optimal across all populations and budgets. The frontier of adaptive voting, which chooses the temperature per query, dominates every fixed-temperature frontier at every budget. The figure shows both on MATH-500: the best fixed temperature changes at the red markers, and the adaptive frontier lies above all of them.

<p align="center">
  <a href="../assets/fig_crossing.png"><img src="../assets/fig_crossing.png" width="100%" alt="Fixed-voting frontiers cross, while the adaptive frontier dominates"></a>
  <br><sub>The best fixed temperature changes with the budget; the adaptive frontier lies above all fixed-rule frontiers.</sub>
</p>

### 3. The adaptive frontier has a closed form at large budgets

As the budget grows, accuracy converges to its ceiling exponentially fast, and the rate of that convergence is set by the hardest queries that are still solvable. Easy queries and unsolvable queries saturate early; every extra token then goes to the hardest solvable ones, and how quickly they turn extra tokens into fewer errors is what the whole population's convergence rate inherits.

The data agree. On MATH-500 the gap between PRICE-oracle and its ceiling decays almost linearly on a log scale, and the measured rate sits at the far left of the per-query rates: 14.5× below the median query on Qwen2.5-1.5B, 17.1× on Llama-3.2-3B.

<p align="center">
  <a href="../assets/fig_ptrue_rate_validation_qwen.png"><img src="../assets/fig_ptrue_rate_validation_qwen.png" width="100%" alt="The large-budget law against the data"></a>
  <br><sub>Empirical rate validation on Qwen2.5-1.5B.</sub>
</p>

## How PRICE decides

**PRICE-oracle** estimates each query's accuracy curves $V_\tau(n)$ and rollout cost MGF $M$ from a labeled pool. At a shared price $\lambda$, it chooses the count and voting rule that maximize $V_\tau(n)-\lambda M^n$. Searching over prices gives a policy whose measured cost fits the target budget.

**PRICE-deployed** predicts these primitives from MATH-train calibration data and label-free statistics of the rollouts observed so far. After each rollout, it refreshes the predictions and compares the best predicted gain over a short look-ahead window with the priced marginal cost. It stops when the gain no longer covers that cost, then uses the predicted best voting rule at the stopping count.

The [`price/`](../price/) package is a small NumPy implementation of Algorithms 1 and 2. Its synthetic demo illustrates the decisions; the full calibration models and the experiment code that produced the paper's results live in [`experiments/`](../experiments/).

```bash
python -m price.demo
```

The demo uses labels to estimate curves from synthetic pools, then uses those curves for decisions on fresh rollouts. It does not train the deployed calibration models or reproduce a paper result.

### A minimal oracle example

Run from the repository root after installing NumPy:

```python
import numpy as np
from price import estimate_primitives, price_for_budget, priced_choice

# A labeled toy pool, used only to estimate the oracle's primitives.
answers = ["A", "B", "A", "C", "B", "A"]
scores = [0.9, 0.4, 0.8, 0.2, 0.5, 0.7]
lengths = [180, 220, 200, 240, 210, 190]
taus = [0.0, 1.0, 3.0, np.inf]
gamma = np.log(20) / 1024

q = estimate_primitives(
    answers, scores, lengths, correct="A", taus=taus, gamma=gamma,
    horizon=6, n_orderings=32, seed=0,
)
price = price_for_budget(
    [q], budget=600, gamma=gamma, price_grid=np.logspace(-6, 0, 200),
)
k, n = priced_choice(q.V, q.log_mgf, price)
print(f"rollouts={n}, voting temperature={taus[k]}")
```

### Code map

| Component | Readable implementation | Paper experiments |
| --- | --- | --- |
| Boltzmann vote | [`price/vote.py`](../price/vote.py) | [`awv/engine.py`](../awv/engine.py) |
| Curve estimation, priced choice and budget search | [`price/oracle.py`](../price/oracle.py) | [`allocation.py`](../experiments/E17_split_joint_oracle/allocation.py), [`price_sweep.py`](../experiments/E18_lambda_replay/price_sweep.py) |
| Sequential stopping and cost prediction | [`price/deployed.py`](../price/deployed.py) | [`controller.py`](../experiments/E22_ptrue_deployed/controller.py), [`models.py`](../experiments/E22_ptrue_deployed/models.py) |

See the [reproduction guide](../experiments/README.md) for the experiment order, scoring conventions and release limitations.

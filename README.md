# PRICE: Joint Adaptive Voting and Rollout Allocation for Token-Budgeted LLM Test-time Compute and Its Pareto Frontier

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](#citation)
[![Data](https://img.shields.io/badge/Data-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/zach-wang/PRICE-rollouts)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Zhenyu Wang** (Rutgers University, zw425@stat.rutgers.edu) · **Xiaozhi Zhu** (Meta) · **Yifan Hu** (Rutgers University, yifan.hu@rutgers.edu)

Given a token budget that has to cover a stream of queries, how many rollouts should each query get, and how should its rollouts be aggregated into an answer?
PRICE (**P**riced **R**ollouts and **I**nference-time voting-rule **C**hoice under a token budg**E**t) answers both at once.
It turns the budget into a shadow price on rollouts and, query by query, picks the rollout count *and* the voting rule that maximize accuracy minus priced cost.
Self-consistency, score-weighted voting and best-of-n are one family, the Boltzmann weighted vote at temperature τ, so choosing the rule is choosing τ.

<p align="center"><img src="assets/fig_illustration.png" width="92%" alt="PRICE decides how to spend; PRICE yields the best Pareto frontier"></p>

## Takeaways

### 1. No fixed voting rule is best at every budget. Fixed-rule frontiers cross; the adaptive frontier sits above all of them.

Which rule wins depends on how much you can spend. Two theorems make this precise: the cost-accuracy frontiers of any two fixed temperatures can cross, so no fixed rule is universally best over all budgets, while the frontier of adaptive voting, which chooses the rule per query, dominates every fixed rule at every budget. The figure shows both on MATH-500: the fixed-rule frontiers swap order (red dots), the adaptive frontier stays on top.

<p align="center"><img src="assets/fig_crossing.png" width="85%" alt="Fixed-voting frontiers cross, while the adaptive frontier dominates"></p>

### 2. Adapt the count *and* the rule. The rule adds a gain on top of the count.

On MATH-500, letting the number of rollouts vary across queries is worth 2 to 9 accuracy points at matched budget. Letting the voting rule vary too adds another 2 to 3 points, on top of the hindsight-best fixed rule. Accuracy in % at matched realized budget b; PRICE-oracle estimates each query's primitives from a labeled half of its own rollout pool and is evaluated on the other half.

| | Qwen2.5-1.5B ||||||| Llama-3.2-3B |||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **budget b →** | 2k | 5k | 8k | 12k | 20k | 30k | 2k | 5k | 8k | 12k | 20k | 30k |
| SC, fixed count | 48.4 | 55.8 | 59.6 | 62.6 | 65.3 | 66.6 | 40.9 | 47.4 | 51.7 | 54.5 | 57.3 | 59.0 |
| SC, adaptive count | 56.3 | 61.6 | 64.0 | 65.8 | 67.5 | 68.4 | 49.7 | 54.6 | 57.0 | 58.5 | 60.4 | 61.5 |
| Best fixed τ (hindsight), adaptive count | 56.9 | 62.2 | 64.5 | 66.2 | 67.8 | 68.5 | 49.8 | 54.6 | 57.0 | 58.5 | 60.4 | 61.5 |
| **PRICE-oracle** (adaptive count and rule) | **58.9** | **64.8** | **67.4** | **69.1** | **70.9** | **71.7** | **51.5** | **56.6** | **58.7** | **60.3** | **62.3** | **63.5** |
| *gain over the best fixed rule* | +2.0 | +2.6 | +2.9 | +2.9 | +3.1 | +3.2 | +1.7 | +2.0 | +1.7 | +1.8 | +1.9 | +2.0 |

### 3. Deployed, PRICE beats every baseline at every budget, and the lead widens as the budget shrinks.

PRICE-deployed predicts each query's accuracy curves and cost from a labeled calibration corpus (MATH-train) plus label-free statistics of the rollouts drawn so far, and stops sequentially. Against six deployed methods on the same rollout pools, orderings and horizon, it is the best entry in every column: +6.4 and +7.3 points over the best baseline at the tightest budgets, +0.5 and +1.0 at the loosest. At matched accuracy it needs up to 2.8× (Qwen) and 3.5× (Llama) fewer tokens than self-consistency.

| | Qwen2.5-1.5B ||||||| Llama-3.2-3B |||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **budget b →** | 3.7k | 5.3k | 6.7k | 11.6k | 14.5k | 32.0k | 4.4k | 6.2k | 8.1k | 13.8k | 17.4k | 40.4k |
| Self-consistency | 54.9 | 58.0 | 59.8 | 63.2 | 64.2 | 67.1 | 47.3 | 49.7 | 51.7 | 55.2 | 56.5 | 59.9 |
| Best-of-n | 49.8 | 48.1 | 46.0 | 41.2 | 38.9 | 30.4 | 42.7 | 41.3 | 39.8 | 35.9 | 33.9 | 27.1 |
| CISC | 54.9 | 55.0 | 59.8 | 62.3 | 64.3 | 66.8 | 47.3 | 47.3 | 51.8 | 54.3 | 56.5 | 59.7 |
| Adaptive-Consistency | 46.9 | 47.3 | 48.9 | 48.9 | 48.9 | 48.9 | 40.7 | 40.7 | 40.7 | 40.7 | 44.4 | 44.4 |
| ESC | 53.9 | 56.7 | 56.7 | 62.6 | 63.7 | 65.9 | 47.4 | 50.2 | 50.2 | 55.5 | 56.9 | 59.2 |
| DeepConf | 52.2 | 58.4 | 58.4 | 63.9 | 63.9 | 66.9 | 43.9 | 50.2 | 50.2 | 55.8 | 55.8 | 59.5 |
| **PRICE-deployed** | **61.3** | **63.3** | **64.3** | **66.2** | **66.6** | **67.6** | **54.6** | **56.3** | **57.5** | **59.2** | **59.7** | **61.0** |
| *gain over the best baseline* | +6.4 | +4.9 | +4.5 | +2.2 | +2.2 | +0.5 | +7.3 | +6.1 | +5.7 | +3.3 | +2.8 | +1.0 |

<p align="center"><img src="assets/fig_ptrue_deployed_frontier.png" width="85%" alt="The frontier of PRICE-deployed against SC and ESC"></p>

Budgets are entropic-risk token budgets, b = (1/γ) log E[exp(γ L)] with γ = log(20)/1024, so that the fraction of queries overspending b by more than 1,024 tokens stays below 5%. The fixed-count rows are read at the count that spends the budget; the adaptive methods at the best feasible point of their own sweep.

### 4. The frontier has a closed form at large budgets: the ceiling is set by which queries some voting rule can solve, the speed by the slowest of them.

For the class of adaptive committed policies we derive the large-budget asymptotics of the Pareto frontier. The accuracy ceiling is the coverage of queries that adaptive voting solves with probability approaching one, plus the residual accuracy on the rest, and adaptive voting covers at least as many queries as any fixed rule. The gap to the ceiling closes exponentially in the budget at the worst-case exchange rate among the solvable queries, because at large budgets the remaining tokens flow to the hardest queries that can still be solved. The data agree: the measured terminal slope sits at the far left of the fitted per-query exchange rates, 14.5× below the median on Qwen and 17.1× on Llama.

<p align="center"><img src="assets/fig_ptrue_rate_validation_qwen.png" width="85%" alt="The large-budget law against the data"></p>

## How PRICE decides

The whole decision is a few lines. `price/` is a readable numpy implementation of Algorithms 1 and 2 of the paper, and `python -m price.demo` runs it on a three-query toy world: an easy query that settles at once, a query worth more compute, and a hard one whose rollouts stop covering their price.

```python
import numpy as np
from price import estimate_primitives, priced_choice, price_for_budget, should_stop

# PRICE-oracle: primitives from a labeled pool, then one argmax per query at the supporting price
taus = [0.0, 1.0, 3.0, np.inf]                                   # SC, score-weighted, BoN
q = estimate_primitives(answers, scores, lengths, correct="A", taus=taus, gamma=np.log(20) / 1024)
lam = price_for_budget([q, *others], budget=1500, gamma=np.log(20) / 1024, price_grid=np.logspace(-6, -1, 200))
k, n = priced_choice(q.V, q.log_mgf, lam)                        # argmax_{tau, n}  V_tau(n) - lam * M^n

# PRICE-deployed: after each rollout, stop when the predicted gain over the next w rollouts
# no longer covers the priced marginal cost  lam * exp(gamma L_n) * (M_hat - 1)
stop = should_stop(V_hat, n, tokens_so_far, log_mgf_hat, lam, gamma=np.log(20) / 1024, window=4)
```

| File | What it holds | The code that produced the paper's numbers |
|---|---|---|
| `price/vote.py` | the Boltzmann weighted vote (Definition 2.1) | `awv/engine.py` |
| `price/oracle.py` | curve estimation, the priced per-query choice, bisection for the supporting price, the frontier sweep | `experiments/E17_split_joint_oracle/allocation.py`, `experiments/E18_lambda_replay/price_sweep.py` |
| `price/deployed.py` | the look-ahead stopping rule, the cost blend, the online price update | `experiments/E22_ptrue_deployed/controller.py`, `models.py` |

## Reproduce

Every result is a CPU replay over released rollout pools (four cells, 0.8 GB); generation and scoring are the only GPU stages and were run once.

```bash
pip install -r requirements.txt && python scripts/download_data.py
bash experiments/run_all.sh oracle       # PRICE-oracle branch, ~35 min
bash experiments/run_all.sh deployed     # PRICE-deployed branch
bash experiments/run_all.sh paper        # tables and figures into build/
```

[`experiments/README.md`](experiments/README.md) maps every table and figure to the step that produces it, lists the pre-registered protocol of each step, and states what was and was not re-executed after packaging. The oracle branch reproduces the paper's run exactly from the released data.

## Citation

```bibtex
@article{wang2026price,
  title   = {PRICE: Joint Adaptive Voting and Rollout Allocation for Token-Budgeted LLM Test-time Compute and Its Pareto Frontier},
  author  = {Wang, Zhenyu and Zhu, Xiaozhi and Hu, Yifan},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv identifier to follow}
}
```

## License

Code under the MIT License. Data under CC BY 4.0, with the upstream model and dataset terms in [`DATA_LICENSE.md`](DATA_LICENSE.md).

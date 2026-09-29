# Results and evaluation

[Home](../README.md) · [Method and theory](method.md) · [Reproduction](../experiments/README.md)

The reported experiments use **MATH-500** with **Qwen2.5-1.5B** and **Llama-3.2-3B**, and P(True) self-evaluation scores. Accuracy gains are percentage points. Budgets use the paper's [entropic-risk token cost](method.md#the-token-budget), which accounts for expensive tails.

## At a glance

| PRICE-deployed | Qwen2.5-1.5B | Llama-3.2-3B |
| --- | --- | --- |
| Accuracy at the tightest reported budget | **61.3%** | **54.6%** |
| Gain over the best baseline at that budget | **+6.4 points** | **+7.3 points** |
| Largest matched-accuracy token reduction vs self-consistency | **2.8× fewer** | **3.5× fewer** |

The six deployed baselines are self-consistency, best-of-$n$, CISC, Adaptive-Consistency, ESC and DeepConf (offline). The complete budget-by-budget results are shown below.

## PRICE-oracle

PRICE-oracle estimates each query's accuracy curves and cost from a labeled half of its own rollout pool, solves the priced problem, and is evaluated on the other half. It is a benchmark for what joint adaptation can give, not a deployable method. Green numbers are the gain of an adaptive count over a fixed count under the same rule; red numbers are the gain of adapting the rule as well, over the hindsight-best fixed temperature with an adaptive count.

<p align="center">
  <a href="../assets/tab_oracle.png"><img src="../assets/tab_oracle.png" width="100%" alt="PRICE-oracle against fixed rules on MATH-500"></a>
  <br><sub>Oracle comparison: gains from adapting the count (green) and adapting the voting rule (red).</sub>
</p>

## PRICE-deployed

PRICE-deployed predicts each query's accuracy curves and cost from a labeled calibration corpus (MATH-train) plus label-free statistics of the rollouts drawn so far, refreshes the predictions after every rollout, and stops when the predicted gain over the next few rollouts no longer covers the priced marginal cost. All methods are replayed on the same rollout pools, orderings and horizon with their original voting and stopping conventions. Red numbers are PRICE-deployed minus the best baseline in the column.

<p align="center">
  <a href="../assets/tab_deployed.png"><img src="../assets/tab_deployed.png" width="100%" alt="PRICE-deployed against six baselines on MATH-500"></a>
  <br><sub>Deployed comparison: accuracy at matched budgets. Red numbers show the gain over the best baseline in each column.</sub>
</p>

The lead widens as the budget shrinks, from +0.5 / +1.0 points at the loosest budgets to +6.4 / +7.3 at the tightest. On the frontier, PRICE-deployed lies above self-consistency and ESC at every budget and reaches the same accuracy with up to 2.8× (Qwen) and 3.5× (Llama) fewer tokens than self-consistency.

<p align="center">
  <a href="../assets/fig_ptrue_deployed_frontier.png"><img src="../assets/fig_ptrue_deployed_frontier.png" width="100%" alt="The frontier of PRICE-deployed against SC and ESC"></a>
  <br><sub>The deployed cost–accuracy frontier on MATH-500.</sub>
</p>

## Evaluation protocol

| | PRICE-oracle | PRICE-deployed |
| --- | --- | --- |
| Information used to make decisions | Curves and costs estimated from a labeled half of each evaluation query's own rollout pool | Predictors trained on a labeled MATH-train calibration corpus, plus label-free statistics of observed evaluation rollouts |
| Evaluation | The other half of the pool, in both split directions | Sequential replay on MATH-500, with predictors frozen before test scoring |
| Purpose | Measure the gains available from jointly adapting count and rule | Evaluate decisions without access to the test answer label |

All deployed methods use the same rollout pools, orderings and horizon, with their original voting and stopping conventions. The oracle and deployed protocols use different information; their numbers should be interpreted within their respective comparisons.

The solvable-query fractions in the [theory notes](method.md#theory) use the vote over all 128 rollouts and the 13-temperature grid. They are a finite-pool diagnostic of the accuracy ceiling, not an additional deployed benchmark.

## Reproduce these exhibits

- **Oracle table:** E21 and E27, assembled by [`ptrue_paper_assets.py`](../paper_assets/ptrue_paper_assets.py).
- **Deployed table:** E23 and E25, assembled by the same script.
- **Deployed frontier:** E22, E23 and E25, plotted by [`ptrue_figures.py`](../paper_assets/ptrue_figures.py).
- **Rate validation:** E26, plotted by the same figure script.

The [reproduction guide](../experiments/README.md) maps all exhibits to their inputs and records what was re-executed after packaging. The oracle branch was verified end to end from the released data. The deployed branch is included as run for the paper, but was not re-executed after packaging.

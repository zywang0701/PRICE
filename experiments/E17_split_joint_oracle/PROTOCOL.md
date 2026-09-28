# Updated joint-oracle split-pool diagnostic

Design saved before new result calculation. Exploratory study on the existing
MATH-500 pools, 2026-09-16. No new generation or external compute. Do not overwrite
the manuscript, old E01/E03 results, staged images, or deployed controller.

## Scope

Joint oracle chooses both a temperature and a committed rollout count for every
query. This extends E16's voting-only half-pool check. Use E16's repaired target
sets, frozen calibration CDF, 13-temperature grid and deployed cohort (489 Qwen,
496 Llama). Cached label-free answer clustering remains unchanged. The reference
answer is used for oracle estimation on the selection half and evaluation only.

Each query has 128 logged rollouts. Four deterministic splits (seed 20260922,
query ID, split number) yield disjoint halves of 64; use 256 independent random
orderings within each half. All counts 1..64 are allowed for every temperature,
with the deployed tie rule. Evaluate both A→B and B→A, retaining each direction.
Permutations and partitions share queries and rollouts; they are not independent
samples. Also recompute a corrected full-128-pool in-sample reference under the
same score, count, tie and cost conventions.

## Cost definitions

Primary: exact finite-pool expected exp(gamma * total tokens) under uniform
sampling without replacement. Compute elementary symmetric polynomials of the
single-rollout exponentiated costs in log space and divide by choose(m,n).
This is the same sampling law as the reward permutations, evaluated analytically
rather than via noisy sampled exponentials. gamma=0.0029262.

Secondary: the paper's iid plug-in estimator (mean exp(gamma*ell))**n, on the
same halves and reward tables. Keep this separate, never mix cost definitions
on an axis. In both views, actual mean tokens are n * half-pool mean length.

## Selection and frozen evaluation

For each nominal A-side budget, solve the finite action LP exactly using each
query's upper concave reward-versus-MGF-cost hull and the sorted marginal slopes
of those hulls. Randomize along the one marginal edge needed to meet the budget.
All actions, the global fixed-temperature choice, and mixture weights depend on
A only. Evaluate the frozen choices on B. Do not refit the price, temperature,
count, or mixture using B, and do not monotonicize or optimize the B rewards.

Families:
- Joint per-query count + temperature oracle.
- A-selected best global temperature + per-query count oracle (strong fixed-vote).
- Calibration-fixed temperature + per-query count oracle.
- Calibration-fixed temperature + a common count, randomized if necessary.
- Common-count SC.
- Joint oracle's exact count policies, replacing its vote by the best one global
  temperature selected on A at those same counts (isolates voting at equal cost).
- The same count policies with the calibration-fixed vote (second voting control).

Nominal grid includes the six paper budget columns, legacy 2/3/5/8/12/20/30k
columns and a dense geometric grid 1.5k..60k. Mark infeasible or saturated
endpoints explicitly. Source-side dominance and budget recovery are checks;
transferred policies need not dominate.

## Aggregation, plots, uncertainty

Average reward and MGF costs over queries and the eight split directions first,
then take log / gamma for aggregate risk cost. Do NOT average log-risk budgets.
Plot frozen policies against B's actual measured cost, alongside A-side in-sample
curves and the recomputed full-pool in-sample reference. Cross-method readings at
the same nominal A budget can have different B costs and must be labeled so.
Same-count voting contrasts have identical A/B costs by construction.

Bootstrap query IDs jointly across every split/direction and method (1000
replicates), conditional on the fitted oracle policies. Report accuracy optimism,
actual B/source budget ratios, and paired equal-count voting gaps. No causal
attribution of the independent count-oracle contrast at mismatched B cost.

Show raw B policy curves without constructing a new B-side upper envelope.
The curves remain labeled split-pool oracle diagnostics, not true population
frontiers or deployable policies. Grading, existing clustering, finite-pool,
query-cohort exclusions, and repeated-test-use limitations remain explicit.

## Verification

Compare the allocation-hull solver against an independent scipy.linprog solution
on synthetic finite problems, including nonmonotone rewards. Verify analytic
finite-pool MGFs against enumerated subsets, budget recovery, source dominance,
multi-correct/absent-gold scoring, disjoint rollout IDs, source-only selection,
same-count identical costs, aggregate-MGF arithmetic, and source artifact hashes.
Render and inspect all published figures. Publish code, action arrays, complete
tables, validation record, and report with source links and known limitations.

# A-estimated primitives, lambda sweep executed and costed on B

2026-09-16. Supersedes E17's A-budget transfer diagnostic as the main frontier
readout, following the user's explicit clarification. Preserve E17 as a diagnostic.

For each query reuse the four 64+64 partitions and both directions from E17.
Use the same repaired scoring, calibration-frozen CDF, vote/tie conventions,
13 temperatures, all counts 1..64, and 256 orderings per half. Queries: 489 Qwen,
496 Llama. No extra rollout generation or change to deployed controllers.

1. Estimate V_A(tau,n;q) and M_A(q) from A only. For each fixed lambda, select the
   committed action maximizing V_A - lambda * M_A**n. Ties favor smaller count,
   then smaller temperature. Do not solve an A-side target-budget constraint.
2. Sweep a predetermined grid: lambda=0 and 16,385 geometric values from 1e-180
   to 1e6. Test numerical readout sensitivity against its nested half grid.
   Each action remains a function of A primitives and lambda, never B labels.
3. On B, use precisely the E17 random orderings, vote on the selected prefix,
   and charge that prefix's actual cumulative logged tokens L_B. Average accuracy
   and exp(gamma * L_B) across queries, paths, and directions before taking
   log/gamma. Gamma=.0029262. The B cost is neither M_B**n nor analytic finite-pool
   expected cost. Also report mean actual B tokens.
4. Construct retrospective cost-accuracy envelopes of the resulting lambda
   policies, with randomization between two complete policies on the MGF scale
   where needed. Read all main-table methods at matched B risk budgets
   2k/3k/5k/8k/12k/20k/30k. Save raw sweep points as well as envelope endpoints
   and interpolation weights. Do not refit per-query actions using B.
5. Fixed-temperature methods use the same A-only count selection for each lambda.
   Report SC, BoN, the existing calibration-fixed temperature, and hindsight-best
   fixed temperature. The last is explicitly the best *single* global temperature
   on the B retrospective envelope at each budget; mixtures across temperatures
   are not used to construct that benchmark. This is not a deployable selector.
6. For a matched-count voting diagnostic, at each lambda keep joint counts and
   choose one global temperature on A to maximize reward at those counts. Execute
   on B. Read this control at the joint frontier's same two lambda policies and
   mixing weight, guaranteeing identical path costs. A randomized mixture can
   use its two corresponding A-selected global votes; it does not choose per-query
   temperatures using B labels. Include the calibration-fixed vote as a control.
7. Query-bootstrap paired equal-count gaps (1,000 draws) conditional on the fitted
   primitives and the retrospectively chosen operating points. State that these
   intervals do not refit/reselect envelopes or cover fresh rollout uncertainty.

Interpretation: this is a retrospective frontier of A-estimated policies evaluated
on B. Lambda and the hindsight global comparator are retrospectively read on B;
there is no prospective budget guarantee. This is not the true population oracle
frontier, nor a sequential stopping controller. Expected primitives and actual
execution costs have distinct roles. B labels are used for scoring and global
frontier readout, not for per-query action fitting.

Validation: fixed-lambda actions against direct grid argmax; replay token sums
against reconstructed permutations; independent formula checks for B MGF
aggregation and budget mixtures; same-count path-cost identity; main-table values
from saved curves; source hashes; dense-grid sensitivity; LaTeX build and visual QA.

Numerical refinement: the initial 8,193-point sweep differed from its nested half
grid by at most 0.026 pp (Qwen) / 0.035 pp (Llama) over the fixed-temperature and
joint readouts. The final grid is doubled to 16,385 positive prices; preserve
`results_grid8193.json` and compare the final sweep to that nested grid.

# Non-PRM scores at fixed rollout count

User request: test other voting scores, excluding PRM. Compare P(True), self-certainty (the logged mean top-1 token probability), likelihood (the logged geometric mean sampled-token probability), and DeepConf as the reference. No PRM score is loaded or evaluated.

The target is voting quality at identical generated-rollout cost, not joint count allocation or deployment compute efficiency. For each n=1,...,64 all arms share identical B rollout prefixes. Reuse E17/E18's 489 Qwen / 496 Llama cohort, corrected grading, label-free clusters, four 64+64 partitions, both directions, 256 orderings per half and 13-temperature grid. Retain n=4,8,16,32,64 as summary columns, with n=32 as the predeclared illustrative comparison. No lambda sweep, count allocation or budget interpolation.

Data audit: all four non-PRM scores are present for every evaluation rollout. Llama's calibration-pool P(True) is entirely absent. Therefore use one normalization protocol for all four scores and both models: fit a pooled empirical CDF using only A's rollout scores in each direction, then freeze it for A and B. Do not fit a CDF on B, fill missing calibration P(True) with zero, or mix normalization protocols across scores. This is an A-estimated oracle comparison, not a calibration-only deployed experiment. It intentionally differs from E19's calibration-frozen CDF, so DeepConf is recomputed too.

Primary tie convention: exact voting-weight ties use the smallest existing label-free answer-cluster ID; abstention clusters do not vote. The SC baseline is consequently independent of score and must be identical across scores. BoN returns the cluster containing the highest-scored valid rollout, with tied scores resolved by cluster ID. This differs from the old score-based tie convention. Also compute the E19 score-native convention (the original 1e-9 score-max perturbation on finite-temperature votes) as a separate sensitivity, with its own SC baseline. Do not mix the two in one contrast.

At each fixed n and fitting direction:

- SC: tau=0.
- BoN: tau=infinity.
- A-global: a single temperature maximizes mean A reward across queries.
- A-per-query: each query selects argmax_tau V_A(tau,n;q), with the smallest-temperature tie convention for selecting a rule.
- B-global hindsight: one global temperature at n maximizes pooled B accuracy across all directions and queries.
- B-per-query hindsight: each B query/half selects and evaluates on the same empirical reward curve. This optimistic diagnostic is not independently evaluated or a population upper bound.

Primary comparisons are adaptive minus A-global and adaptive minus B-global, plus absolute adaptive accuracy relative to recomputed DeepConf. Report adaptive minus SC as a secondary comparison. A stronger globally fixed score need not have a larger per-query-adaptivity gap.

Report all 64 counts for all scores, the predetermined summary counts, and paired query-bootstrap pointwise 95% intervals from 1,000 draws. Keep all directions of a query together. Freeze selected temperatures within bootstrap; do not claim refitting, retrospective-selection or simultaneous uncertainty is covered. All generated-token costs are identical between scores and arms at a fixed n. P(True) requires an additional scoring forward pass in a real deployment; that overhead is absent from the logged generation-token cost and is not claimed to be free.

Validation: exact query/rollout ID alignment; unchanged rollout lengths and labels; nonmissing scores; A-only CDF; disjoint pools; fast vote kernel versus independent vectorized reference and existing implementation; all primary SC outcomes identical across scores; n=1 agreement; raw-prefix reconstruction and B reward lookup; A/B same-pool symmetry; common generated-token costs. Do not overwrite the manuscript or E19 while assessing this score comparison. Existing caches and local CPU only.

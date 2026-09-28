# Answer-set voting experiment

Frozen design before new model fitting or test scoring, 2026-09-16.

## Scope and data

Use the existing Qwen/Llama 64-rollout MATH-train pools and frozen MATH-500
replays. No generation, external inference, or modification of old artifacts.
Use all 7,425 calibration queries, including queries with no correct rollout;
split by query 70/15/15 with seed 20260921 into fit/tune/audit. Query ID, gold,
correctness, future rollouts and budget never enter feature construction.
This is exploratory development following E15 test diagnostics. The existing
MATH-500 workload is not a newly untouched confirmatory test set.

## Scoring repair

Keep the existing, label-free answer clusters and vote tie rules. Accept every
correct cluster, not only the lowest numbered one. For queries with any conflict
between raw grader flags and the old target, regrade every cluster's canonical
representative against the reference gold with the installed math-verify grader.
Keep a second stricter-numeric (12 rather than 6 rounding digits) scoring view.
Log all changes and internal cluster disagreements. Do not use correctness to
merge answer clusters. Canonicalization/grader limitations are not claimed solved.
Qwen train metadata uses the shared MATH-train query table saved with Llama;
validate query IDs and sample gold alignment explicitly before using it.

## Split-pool diagnostic

Within each calibration query split the 64 distinct rollout IDs into 32+32,
using four seeded partitions. Select one temperature on A across counts
2/4/8/16/32, evaluate on disjoint B using the same counts, then reverse A/B.
Use 16 permutations per half; report query-bootstrap uncertainty. Compare
same-half hindsight gain with transferred gain over the same fixed temperature.
These are partitions of one logged pool, not freshly generated independent pools.
Also train a predictor from A prefixes to the B utility curve, with two partitions
and both directions. It never reads B during inference and freezes after count 32.

## Models

Build one feature vector per observed valid answer cluster: frequency; score
quantiles and top three order statistics; high-score support; recent versus early
support; removal-of-top-score robustness; all 13 weighted vote shares; and
prefix-wide context. Treat absent/abstaining answers explicitly. Compare:

1. shared per-answer MLP (rich-feature control);
2. Deep Sets equivariant comparison, with pooled information from every answer;
3. Deep Sets encoder predicting the disjoint-half temperature utility curve.

The first two use multi-label correctness BCE, balanced per state, plus pairwise
correct/incorrect comparisons on vote-reachable answers. No unique oracle-tau
label is required. At inference compare distinct reachable answers, then choose
a supporting grid temperature closest to the fixed rung. Only switch if its
predicted correctness advantage exceeds a calibration-selected threshold.
No forced switching when all temperatures agree. The third uses utility
regression, sharing a query-level future-pool target across prefix lengths.

Width 48, AdamW, fixed seed, 60 epochs; tune checkpoints 15/30/60 and thresholds
0/.02/.05/.1/.2. All three families retained regardless of results. Each epoch
samples two states per query. Tune settings on tune; select family versus a
fixed-temperature fallback on audit; refit chosen family settings on all
calibration before testing. Select a global fixed comparator from fit queries
under repaired scoring; also retain the original fixed rung for reference.

## Evaluation and verification

First compare candidates with the fixed and old deployed votes at exactly the
same E15 six operating-point stopping paths. All rollout tokens are identical;
report actual risk budgets, six accuracy differences, rescue/harm, query-bootstrap
intervals, temperature usage, and original six diagnostic queries. Also report
fixed counts 2/4/8/16/32/64. Selectors update at counts 1/2/4/8/16/32/48/64 and hold
their temperature between updates; no budget input. CPU inference time is
measured separately, not counted as generation tokens.

Verify raw replay reconstruction, missing-gold/all-abstention behavior, target-set
scoring, prefix causality, answer permutation equivariance, disjoint rollout
halves, query splits, train-only score normalization, and saved selection hashes.
Bootstrap queries, not permutations; intervals condition on the fitted model and
do not account for repeated research on this workload or model selection.
No default-controller or manuscript update solely from exploratory improvement.

## Post-evaluation diagnostic added after frozen scoring

The existing test pools contain 128 rollouts per query. After all models and
their evaluation were complete, add a 64+64 transfer check using four seeded
partitions, 16 permutations per half, both directions, and counts 2/4/8/16/32/64.
Select a single hindsight temperature on A, evaluate it on disjoint B; compare
with the same calibration-frozen global temperature. This diagnostic does not
alter training, selection, thresholds, or test scores, and is reported separately
from the prespecified calibration 32+32 diagnostic and stopping-path comparison.

# E24: query-only committed P(True) deployment

The prespecified local protocol is in PROTOCOL.md. Results are conditional on the existing E22 scoring/replay artifacts and use the six E23 baseline families unchanged. This is a strict pre-sampling deployed policy, not a split-pool oracle. Models and temperature choices were frozen using calibration before evaluating test. See the paper workspace's committed_policy_comparison.md for the result and limitations.

From this directory, reproduce for each cell (qwen or llama):

```bash
python run.py --cell qwen
python analyze.py --cell qwen
python verify.py --cell qwen
python run.py --cell llama
python analyze.py --cell llama
python verify.py --cell llama
```


Runtime API: `CommittedController(model_directory, etas).decide(normalized_query_embedding, log_price, fixed_temperature_index)` returns a pre-sampling count and voting temperature. Use `None` for the joint arm. The embedding must use the E22 fit-time embedding model and normalization. This API has no rollout-observation update. Execute exactly the returned count and then aggregate using the E22 P(True) score transform and vote convention.

No GPU work or new scoring was needed. This implementation does not quantify the best possible committed learner. Models were not retuned after examining test results. The main frontier operating points use retrospective test-workload readouts; audit-calibrated prices are reported separately with actual achieved test risks. Cost excludes scorer, embedding, and controller overhead.

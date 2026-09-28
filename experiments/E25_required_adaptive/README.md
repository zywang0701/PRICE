# E25: required adaptive voting

Read PROTOCOL.md for the scope correction and explicit post-audit presentation amendment. The primary sequential controller uses E22 PTrueController with arm regression_w4 on both models. The primary committed controller is E24 CommittedController with fixed_temperature_index=None. No fixed-temperature fallback is allowed in these main policies.

setup25.query_locked_actions adds the stricter budget-invariant query-temperature variant. It takes only predicted reward/cost curves, chooses a single temperature per query, and allocates counts by price. For a new query, obtain these predictions from the frozen E24 model using only its normalized query embedding. All input preprocessing and vote conventions remain those of E22.

Reproduce from this directory for each cell (qwen, llama):

```bash
python analyze.py --cell qwen
python verify.py --cell qwen
python analyze.py --cell llama
python verify.py --cell llama
```

Compile table3_ptrue_preview.tex and table3_committed_preview.tex twice with pdflatex from the paper workspace. All table content is blue. Previous fixed-primary files are preserved in previous_fixed_previews/. Model and original experiment artifacts are unchanged; only current standalone reports/previews are updated. Reuse of an examined benchmark makes this an exploratory follow-up. Temperature diversity and overall baseline gains do not establish positive incremental voting gain.

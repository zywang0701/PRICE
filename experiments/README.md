# Reproducing the paper

This is the reproduction guide for the experiment chain. The project page, results and the readable algorithm are in the [top-level README](../README.md). Companion data: [`zach-wang/PRICE-rollouts`](https://huggingface.co/datasets/zach-wang/PRICE-rollouts) on Hugging Face. All commands below run from the repository root.

PRICE (**P**riced **R**ollouts and **I**nference-time voting-rule **C**hoice under a token budg**E**t)
allocates a token budget across queries by pricing rollouts, and chooses the Boltzmann voting
temperature per query. This repository contains the code for both versions evaluated in the paper:

- **PRICE-oracle** estimates each query's accuracy curves and cost MGF from a labeled half of its own
  rollout pool and solves the priced per-query problem; it is a benchmark, not a deployable method.
- **PRICE-deployed** predicts both primitives from a labeled calibration corpus (MATH-train) and the
  label-free statistics of the rollouts generated so far, and stops sequentially.

Every experiment is a CPU replay over logged rollout pools. Generation and scoring (the only GPU
stages) were run once; the pools are released so that the chain can be re-run without a GPU.

## Setup

```bash
git clone https://github.com/zywang0701/PRICE.git && cd PRICE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU chain; Python >= 3.10
python scripts/download_data.py          # ~0.8 GB into outputs/cells/, verified against the dataset manifest
pytest tests/                            # CPU unit tests of the shared library
```

`PRICE_ROOT` (default: the repository) is where `outputs/cells/` lives; `PRICE_PAPER_DIR`
(default: `build/`) is where the paper tables and figures are written.

## Reproducing the paper

```bash
bash experiments/run_all.sh oracle       # PRICE-oracle branch    E02 → E16 → E17 → E18 → E20 → E21 → E27
bash experiments/run_all.sh deployed     # PRICE-deployed branch  E22 → E23 → E24 → E25 → E26
bash experiments/run_all.sh paper        # tables into build/tables, figures into build/images
```

| Paper exhibit | Produced by | From |
|---|---|---|
| Table 1, oracle half (PRICE-oracle vs SC / BoN / best fixed temperature) | `paper_assets/ptrue_paper_assets.py` → `tab_oracle_ptrue.tex` | E21, E27 |
| Table 1, deployed half (PRICE-deployed vs six baselines) | `paper_assets/ptrue_paper_assets.py` → `tab_deployed_ptrue.tex` | E23, E25 |
| Figure: large-budget law (rate validation, Qwen and Llama) | `paper_assets/ptrue_figures.py` | E26 |
| Figure: PRICE-deployed frontier vs SC and ESC | `paper_assets/ptrue_figures.py` | E22, E23, E25 |
| Figure: split-pool oracle frontiers (appendix) | `paper_assets/ptrue_figures.py` | E21 |
| Figures: tokens and counts by MATH level (appendix) | `paper_assets/ptrue_figures.py` | E26 |
| Table: prospective prices (appendix) | `paper_assets/ptrue_paper_assets.py` → `tab_ptrue_cost_prospective.tex` | E22, E25 |
| Table: rate sensitivity (appendix) | `paper_assets/ptrue_rate_allocation.py` | E26 |
| README figure: solvable-query fractions by voting rule (not in the paper) | `paper_assets/readme_solvable_fraction.py` → `assets/fig_solvable.png` | E16, E20, E22 |

Each `experiments/E*/` folder is one step of the chain and keeps its `PROTOCOL.md`, the design
written down before that step was run. The steps, in dependency order:

| Step | What it does |
|---|---|
| `replay_frozen.py` (E02) | 64 fixed orderings of every MATH-500 query's 128 rollouts, with the score transform frozen on the calibration corpus |
| E16 | Verified answer sets: label-free answer clusters, repaired grading (`math-verify`), fit/tune/audit split of MATH-train |
| E17 | Split-pool joint oracle (count + temperature) with exact finite-pool costs; both split directions |
| E18 | Actual path costs on the cached permutations and the price sweep utilities |
| E20 | Voting scores at fixed count (P(True), self-certainty, likelihood, DeepConf) under one normalization protocol |
| E21 | PRICE-oracle with the P(True) score: adaptive count and temperature, common-tie and score-native-tie conventions |
| E27 | Fixed-count SC and BoN controls on exactly E21's replay (`paper_assets/ptrue_oracle_fixed_count.py`) |
| E22 | PRICE-deployed: calibration-trained reward-curve and cost predictors, sequential stopping, frozen before any test scoring |
| E23 | The six deployed baselines (SC, BoN, CISC, Adaptive-Consistency, ESC, DeepConf) on the same pools, orderings and horizon; Table 1 columns |
| E24 | Committed per-query count/temperature optimizer with the frozen E22 predictors |
| E25 | Query-adaptive voting as the deployed policy; comparison against the calibration-selected fixed temperature |
| E26 | Rate law and budget allocation by MATH level (`paper_assets/ptrue_rate_allocation.py`) |

## What is and is not in this release

- The chain above is the code as it was run for the paper, with hard-coded local paths replaced. It
  was packaged after the fact. The oracle branch was re-executed end to end from the released data
  in a clean environment and reproduced the paper's run exactly (every numeric field of the E21 and
  E27 result files, and byte-identical E17/E18/E20 tables). The deployed branch was not re-executed
  after packaging; its scripts are shipped as run.
- `config/calibration_fixed_temperature.json` records the calibration-fixed temperature per
  generator (k = 5, τ = 2.59 for Qwen; k = 3, τ = 0.94 for Llama), which was selected by an earlier
  MATH-train calibration experiment that is not part of this release. E17 and E18 read it from there.
- E16's own answer-set selector (`model.py`, `controller.py`, `evaluate.py`) needs `torch` and is not
  on the chain; the chain uses only `E16_answer_sets/data.py` (grading, clusters, splits).
- `replay_frozen.py` carries the corrected grading of queries whose reference answer never appears
  in the pool (all their votes count as wrong). Downstream steps already used the repaired labels,
  so the paper's numbers are unaffected; a replay rebuilt here differs from the paper's cached one
  only on those queries' `corr` entries.
- E23 cross-checks its baseline replay against an earlier, pre-P(True) implementation. That file is
  not released; the check is skipped when it is absent and the fact is recorded in
  `baseline_replay_checks.json`.
- Not included: the in-sample crossing figures, the cost-tail figure and the cross-score appendix
  table, which come from an earlier pipeline on the same pools, and that earlier deployed pipeline itself.
- `config/tab_data_{qwen,llama}.json` hold the six column budgets of Table 1 (the realized budgets of
  SC and BoN at fixed counts), read by E23 and E25.

## Generation and scoring (optional, GPU)

`generation/` holds the code that produced the released pools, for anyone who wants a new
generator or benchmark:

- `phase1_rollouts/`: vLLM pool generation (`01_generate_pool.py --cell <cell>`), one YAML per cell in `config/cells/`.
- `phase2_scores/`: query embeddings (`03_embed_queries.py`), confidence traces (`04_conf_trace_v2.py`, `09_add_deepconf2_column.py`) and P(True) self-evaluation scoring (`02_score_pool.py --skip-prm`).
- `prep/`: label-free answer clustering and the per-query reward-curve tables (`lp_prep.py` for MATH-500, `lp3_prep.py --overlap` for MATH-train).
- `ptrue_llama/`: the pinned FP16 re-scoring of Llama's P(True) used by PRICE-deployed (`score_llama.py`, model revision recorded in the script).

The SLURM files are the ones used on our cluster; partition names and cache paths are marked `## EDIT`.
Install `requirements-gpu.txt` on top of `requirements.txt` for these stages.

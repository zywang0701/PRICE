<div align="center">

# PRICE

### Pareto Frontier of LLM Test-Time Compute:<br>Adaptive Rollouts and Voting Under Token Budgets

[Zhenyu Wang](https://zywang0701.github.io/)<sup>1</sup> · Xiaozhi Zhu<sup>2</sup> · [Yifan Hu](https://sites.google.com/view/yifan-hu)<sup>1</sup>

<sup>1</sup> Rutgers University &nbsp; <sup>2</sup> Meta

<p>
  <a href="https://huggingface.co/datasets/zach-wang/PRICE-rollouts"><img src="assets/badge-data.svg" alt="Download rollout data" height="30"></a>
  <a href="#quick-start"><img src="assets/badge-start.svg" alt="Quick start" height="30"></a>
  <a href="#results"><img src="assets/badge-results.svg" alt="View results" height="30"></a>
</p>

**English** · [中文](README_zh.md)

[Introduction](#introduction) · [Theory](#theory) · [Results](#results) · [Quick Start](#quick-start) · [Reproduction](experiments/README.md) · [Citation](#citation)

</div>

## Introduction

More rollouts can improve an LLM's answer, but every rollout spends tokens. With a budget shared across queries, the decision is both **how much to generate** and **how to vote** on the answers.

**PRICE** (**P**riced **R**ollouts and **I**nference-time voting-rule **C**hoice under a token budg**E**t) jointly adapts the rollout count and voting rule for each query to maximize accuracy under a token budget. A shared price balances the expected accuracy gain against the cost of more generation: easy queries finish early, while queries that benefit from more compute receive more rollouts.

The voting rule ranges from self-consistency to score-weighted voting and best-of-*n*. PRICE chooses where to operate for each query, then stops when the predicted gain no longer covers the priced cost.

<p align="center">
  <a href="assets/fig_illustration.png"><img src="assets/fig_illustration.png" width="100%" alt="PRICE chooses a rollout count and voting rule for each query; a schematic shows the adaptive cost–accuracy frontier above fixed voting rules."></a>
  <br><sub><b>Figure 1.</b> Jointly adapting how much to generate and how to vote. The frontier is schematic.</sub>
</p>

## Theory

- **Adaptive voting raises the accuracy ceiling.** Different rules solve different queries. Choosing the rule per query covers the union of their solvable sets, reaching a ceiling at least as high as any fixed rule.
- **No fixed rule is universally best across budgets.** Fixed-rule frontiers can cross. The adaptive frontier dominates every fixed-rule frontier at every budget.
- **The adaptive frontier has a closed form at large budgets.** Accuracy approaches its ceiling exponentially, at a rate governed by the hardest solvable queries; the empirical rate validation supports this law.

See [theory, supporting figures and algorithm details](docs/method.md).

## Results

On **MATH-500**, using **Qwen2.5-1.5B** and **Llama-3.2-3B**:

- **PRICE-oracle** measures the gains available from joint adaptation using labeled rollout pools. Adapting the count adds **2–9 accuracy points** at matched budget; adapting the voting rule adds **2–3 points** beyond the hindsight-best fixed rule with an adaptive count.
- **PRICE-deployed** uses calibration-trained predictors and label-free rollout statistics at test time. It outperforms all six evaluated deployed baselines at every reported budget, with the largest gains at the tightest budgets.

| PRICE-deployed | Qwen2.5-1.5B | Llama-3.2-3B |
| --- | --- | --- |
| Gain over the best baseline at the tightest budget | **+6.4 points** | **+7.3 points** |
| Largest matched-accuracy token reduction vs self-consistency | **2.8× fewer** | **3.5× fewer** |

<p align="center">
  <a href="assets/fig_ptrue_deployed_frontier.png"><img src="assets/fig_ptrue_deployed_frontier.png" width="100%" alt="PRICE-deployed cost–accuracy frontiers on MATH-500 for Qwen2.5-1.5B and Llama-3.2-3B, compared with self-consistency and ESC."></a>
  <br><sub><b>Figure 2.</b> PRICE-deployed reaches higher accuracy at matched budget and the same accuracy with fewer tokens.</sub>
</p>

Gains are percentage points. Budgets use the paper's **risk-adjusted token cost**, which accounts for expensive tails. See [full tables and evaluation protocol](docs/results.md) and the [budget definition](docs/method.md#the-token-budget).

## Quick Start

### 1. Try the algorithm — no GPU or dataset needed

Use **Python 3.10+**. From a fresh checkout:

```bash
git clone https://github.com/zywang0701/PRICE.git
cd PRICE
python -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=1.26"
python -m price.demo
```

The demo shows three synthetic queries: one that settles early, one worth more compute, and one where further rollouts stop paying off. It prints the chosen counts and voting rules, a budget–accuracy sweep, and sequential stopping decisions. The toy curves are estimated from labeled synthetic pools; the paper's deployed predictors are trained separately on MATH-train.

### 2. Prepare the released rollout pools

The paper experiments replay logged rollouts on CPU. The four released model–dataset cells total approximately **0.8 GB**; the downloader verifies file hashes against the dataset manifest.

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
```

### 3. Reproduce the paper results

Run the branches in order; the deployed branch uses artifacts from the oracle branch:

```bash
bash experiments/run_all.sh oracle       # PRICE-oracle and shared replay artifacts
bash experiments/run_all.sh deployed     # PRICE-deployed and six baselines
bash experiments/run_all.sh paper        # Tables and figures into build/
```

The oracle branch was re-executed from the released data and reproduced the paper's run exactly. The deployed branch is shipped as run for the paper and was not re-executed after packaging. See the [reproduction guide](experiments/README.md) for the complete experiment chain and release scope.

<details>
<summary><b>Tests, output locations and optional GPU stages</b></summary>

```bash
python -m pytest tests/
```

Downloaded pools live in `outputs/cells/`; the paper tables and figures are written to `build/tables/` and `build/images/`. `PRICE_ROOT` and `PRICE_PAPER_DIR` allow custom locations; see the [setup notes](experiments/README.md#setup).

To generate and score new rollout pools, use the scripts in [`generation/`](generation/) and install [`requirements-gpu.txt`](requirements-gpu.txt) on top of the CPU requirements. The supplied SLURM scripts contain cluster settings marked `## EDIT`. See [generation and scoring](experiments/README.md#generation-and-scoring-optional-gpu).

</details>

## Code Guide

[`price/`](price/) contains a readable NumPy implementation of the two algorithms. [`experiments/`](experiments/) contains the calibration, replay and evaluation pipeline that produced the paper's numbers.

| Component | Implementation |
| --- | --- |
| Boltzmann weighted vote | [`price/vote.py`](price/vote.py) |
| Oracle curves, priced choice and budget search | [`price/oracle.py`](price/oracle.py) |
| Sequential stopping, cost updates and online price | [`price/deployed.py`](price/deployed.py) |
| Synthetic example | [`price/demo.py`](price/demo.py) |
| Paper experiment chain | [`experiments/run_all.sh`](experiments/run_all.sh) |

[Method notes](docs/method.md) · [Full results](docs/results.md) · [Reproduction](experiments/README.md) · [Rollout data](https://huggingface.co/datasets/zach-wang/PRICE-rollouts)

## Citation

```bibtex
@misc{wang2026price,
  title  = {Pareto Frontier of LLM Test-Time Compute: Adaptive Rollouts and Voting Under Token Budgets},
  author = {Wang, Zhenyu and Zhu, Xiaozhi and Hu, Yifan},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code is released under the [MIT License](LICENSE). Rollout data is released under **CC BY 4.0**, subject to the upstream model and dataset terms in [DATA_LICENSE.md](DATA_LICENSE.md).

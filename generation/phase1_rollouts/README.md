# Phase 1: rollout pool generation

Implements phase 1 of the campaign plan (`../plan/full_campaign_plan.html`,
§3 + §11): generate m=256 rollout pools for **4 generators x 3 tasks = 12
cells**, with the F0 gates deliberately omitted (decision 2026-07-08).
Full design record with rationale: `../docs/phase1_rollout_generation.html`.

## The 12 cells

| model \ task | math500 (500) | olympiadbench (580*) | mmlupro (500) |
|---|---|---|---|
| `llama32-3b`  (unsloth/Llama-3.2-3B-Instruct†)      | 2048 tok | 4096 tok | 2048 tok |
| `llama31-8b`  (unsloth/Meta-Llama-3.1-8B-Instruct†) | 2048 tok | 4096 tok | 2048 tok |
| `qwen25-1.5b` (Qwen/Qwen2.5-1.5B-Instruct)          | 2048 tok | 4096 tok | 2048 tok |
| `qwen25-7b`   (Qwen/Qwen2.5-7B-Instruct)            | 2048 tok | 4096 tok | 2048 tok |

† Ungated mirrors of the `meta-llama/*` repos (identical weights) — Meta's HF
access gate rejected the account; no HF login is needed for any asset.

\* OlympiadBench `OE_TO_maths_en_COMP` has 674 items; 93 multi-answer items and
1 multi-part gold are dropped. MMLU-Pro is a 500-item subset of the test
split, stratified by category with pinned seed 20260708.

Pinned protocol (plan §13): m=256 rollouts/query, temperature 0.8, top-p 1.0,
seed 20260707, cap 2048 tokens (4096 on OlympiadBench), vLLM, sharded +
resumable. Per-rollout log includes `phi_lik`, `phi_selfcert`, and the
DeepConf-style sliding-window confidence columns (`phi_deepconf` =
window-minimum of the mean top-5 logprob, window 128); `phi_conf`/PRM columns
are NaN placeholders for phase 2.

## Running locally (CPU, no GPU)

```bash
pip install -r ../requirements.txt      # from experiments/: requirements.txt
pytest ../tests                          # 45 tests
python scripts/01_generate_pool.py                                    # list cells
python scripts/01_generate_pool.py --cell qwen25-1.5b_mmlupro --dry-run   # loader check (needs `datasets`)
```

(Any working directory is fine — scripts locate the experiments/ root
themselves.)

## Running on Amarel (SLURM)

One-time setup on a **login node** (internet lives there; compute nodes have
none). From the `experiments/` root:

```bash
bash phase1_rollouts/slurm/setup_env.sh          # conda env "awv" + deps + /scratch symlinks
conda activate awv
export HF_HOME=/scratch/$USER/hf_cache
python phase1_rollouts/scripts/00_prefetch_assets.py             # models + datasets, no login needed
python phase1_rollouts/scripts/01_generate_pool.py --cell all --dry-run   # loaders OK end to end
```

Submit the 12-cell array job **from the experiments/ root**:

```bash
sbatch phase1_rollouts/slurm/phase1_pool.sbatch            # all 12 cells
sbatch --array=2 phase1_rollouts/slurm/phase1_pool.sbatch  # a single cell by index
squeue -u $USER                                            # logs: phase1_rollouts/slurm/logs/
```

When cells finish, copy the paid artifacts off scratch (no backups there):

```bash
bash phase1_rollouts/slurm/archive_pools.sh      # final pools -> ~/awv_pool_archive (backed-up /home)
```

Notes:

- Cell index i = the (i+1)-th name in **sorted** order (the no-arg
  `01_generate_pool.py` listing).
- Jobs are `--requeue`-safe: one parquet shard per query block, finished
  shards and finished cells are skipped on restart.
- bf16 on Ampere+ GPUs, automatic fp16 fallback on older cards (V100); the
  8B/7B cells want a >=24 GB card.
- Storage: repo + conda env in `/home` (backed up, slow), HF cache +
  `outputs/` on `/scratch` (fast, ~50 GB total, far below the 1 TB soft quota
  so the 90-day purge never applies).
- After the first shard of each task type lands, spot-check rows: answers
  extracted, `correct` sensible, `phi_deepconf` finite, `finish_reason` mostly
  `stop` (a large `length` fraction on OlympiadBench means the 4096 cap
  binds).

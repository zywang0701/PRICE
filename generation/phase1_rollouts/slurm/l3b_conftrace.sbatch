#!/bin/bash
#SBATCH --job-name=awv-l3-ct2
#SBATCH --partition=gpu   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%A_%a.out
# conf_trace_v2 for llama32-3b_mathtrain, sharded 4 ways by row residue.
# batch_conf() halves its batch on CUDA OOM, so small-memory GPUs are safe.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate awv
export HF_HOME="$HOME/hf_cache"
export TOKENIZERS_PARALLELISM=false
# 04 imports `awv` directly -- the repo root must be importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
python generation/phase2_scores/scripts/04_conf_trace_v2.py --cell llama32-3b_mathtrain \
    --shard-mod 4 --shard-rem "$SLURM_ARRAY_TASK_ID"

#!/bin/bash
#SBATCH --job-name=awv-mathtrain
#SBATCH --partition=gpu-redhat   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%A_%a.out
#
# MATH-train calibration pool for Qwen2.5-1.5B (the learned-primitives corpus).
# 7,425 queries x 64 rollouts, sharded 50 queries at a time.
#
# The four array tasks split the shard index by residue (--shard-mod 4), so
# they work the same cell concurrently on four GPUs without colliding: each
# writes its own shard files, and whichever task finds all shards present at
# the end merges them into pool.parquet.  Every task is resumable -- finished
# shards are skipped -- so requeues and resubmissions are safe.
#
# Submit from the repository root:
#   sbatch phase1_rollouts/slurm/mathtrain_pool.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -d config/cells ]; then
  echo "ERROR: submit from the repository root (config/cells not found in $PWD)" >&2
  exit 1
fi

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate awv

# /home, not /scratch: gpu-redhat mixes two node classes that mount different
# /scratch filesystems, but /home is mounted identically everywhere
export HF_HOME="$HOME/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

CELL=qwen25-1.5b_mathtrain
echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID}/4 -> $CELL on $(hostname), gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python generation/phase1_rollouts/scripts/01_generate_pool.py --cell "$CELL" \
    --shard-mod 4 --shard-rem "$SLURM_ARRAY_TASK_ID"

echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID} done"

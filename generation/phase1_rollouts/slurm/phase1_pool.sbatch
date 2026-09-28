#!/bin/bash
#SBATCH --job-name=awv-pool
#SBATCH --partition=gpu   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-11                    # one task per (model, task) cell
#SBATCH --requeue                       # generation is shard-resumable, requeue is safe
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%A_%a.out
#
# Phase 1: generate the rollout pools for all 12 cells (4 models x 3 tasks).
# SUBMIT FROM THE repository ROOT (the --output path above is relative to
# the submit dir, and setup_env.sh pre-creates the logs dir):
#   sbatch phase1_rollouts/slurm/phase1_pool.sbatch             # all cells
#   sbatch --array=3 phase1_rollouts/slurm/phase1_pool.sbatch   # one cell
#
# Prereqs (login node): bash phase1_rollouts/slurm/setup_env.sh, then
# python generation/phase1_rollouts/scripts/00_prefetch_assets.py. Cells are indexed by
# the SORTED yaml names in config/cells/ — run
# `python generation/phase1_rollouts/scripts/01_generate_pool.py` (no args) to print the
# list; array index i maps to the (i+1)-th name.
#
# Note: if the assigned node has a pre-Ampere GPU (e.g. V100), the code
# auto-falls back to float16; no config change needed.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -d config/cells ]; then
  echo "ERROR: submit this job from the repository root (config/cells not found in $PWD)" >&2
  exit 1
fi

module purge || true
module load miniconda 2>/dev/null || module load anaconda 2>/dev/null || true
module load cuda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate awv

export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_OFFLINE=1                 # everything was prefetched on the login node
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# array index -> cell name (sorted registry order)
mapfile -t CELLS < <(ls config/cells/*.yaml | sort | xargs -n1 basename | sed 's/\.yaml$//')
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> cell $CELL on $(hostname), gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python generation/phase1_rollouts/scripts/01_generate_pool.py --cell "$CELL"
echo "[$(date)] cell $CELL done"

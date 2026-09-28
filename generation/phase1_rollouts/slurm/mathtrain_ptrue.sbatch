#!/bin/bash
#SBATCH --job-name=awv-mt-ptrue
#SBATCH --partition=cgpu-redhat,gpu-redhat   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%j.out
#
# P(True) for the MATH-train corpus, split out of the post chain on purpose.
#
# It is the main text's score, but the learning-curve study runs on
# phi_deepconf2, which is already installed.  Keeping P(True) as an
# INDEPENDENT job means a failure here cannot block the result -- the earlier
# chained layout cost a full run when a trivial missing-file error in a later
# stage invalidated an otherwise-complete five-hour pass.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate awv
export HF_HOME="$HOME/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CELL=qwen25-1.5b_mathtrain
echo "[$(date)] P(True) for $CELL on $(hostname)"
python generation/phase2_scores/scripts/02_score_pool.py --cell "$CELL" --skip-prm
echo "[$(date)] done"

#!/bin/bash
#SBATCH --job-name=awv-l3-finish
#SBATCH --partition=gpu   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%j.out
# After the sharded conf_trace2 array: merge parts -> 09 phi_deepconf2 -> 02 P(True).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate awv
export HF_HOME="$HOME/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
CELL=llama32-3b_mathtrain
P="outputs/cells/$CELL/pools"
if [ ! -f "$P/conf_trace2.parquet" ]; then
  echo "--- merge conf_trace2 parts"
  python generation/phase2_scores/scripts/04_conf_trace_v2.py --cell "$CELL" --merge-parts
fi
echo "--- 09 install phi_deepconf2"
python generation/phase2_scores/scripts/09_add_deepconf2_column.py --cell "$CELL"
echo "--- 02 P(True) (PRM skipped)"
python generation/phase2_scores/scripts/02_score_pool.py --cell "$CELL" --skip-prm
echo "[$(date)] finish chain done"

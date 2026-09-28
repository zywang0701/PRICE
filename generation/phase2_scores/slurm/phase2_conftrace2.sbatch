#!/bin/bash
#SBATCH --job-name=awv-ctrace2
#SBATCH --partition=gpu-redhat   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|adalovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-11%4
#SBATCH --requeue
#SBATCH --output=generation/phase2_scores/slurm/logs/%x_%A_%a.out
#
# Stage 04: corrected DeepConf confidence trajectories (conf_trace2.parquet),
# fixing the sign/extremum bug of delivery/baselines_published_spec.md §4.4.
# One cell per array task. The three Track-1 (F1) math500 cells are indices
# 0-2, so the minimum rerun for the baselines figure is:
#
#   sbatch --array=0-2 phase2_scores/slurm/phase2_conftrace2.sbatch
#
# Full 12-cell sweep (Track-2 deepconf-column regeneration) is the default
# array. Same teacher-forcing cost as the v1 conf_trace job (~1-2 h/cell).
# DO NOT SUBMIT until the spec has been reviewed (spec §5).
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
[ -d config/cells ] || { echo "submit from the repository root" >&2; exit 1; }

module purge || true
module load cuda 2>/dev/null || true
CONDA_EXE_PATH="$(command -v conda || echo "$HOME/anaconda3/bin/conda")"
eval "$("$CONDA_EXE_PATH" shell.bash hook)"
conda activate awv
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR"

CELLS=(
  qwen25-1.5b_math500      # 0  (F1)
  llama32-3b_math500       # 1  (F1)
  llama31-8b_math500       # 2  (F1)
  qwen25-7b_math500        # 3
  qwen25-1.5b_mmlupro      # 4
  qwen25-7b_mmlupro        # 5
  llama32-3b_mmlupro       # 6
  llama31-8b_mmlupro       # 7
  qwen25-1.5b_olympiadbench # 8
  qwen25-7b_olympiadbench  # 9
  llama32-3b_olympiadbench # 10
  llama31-8b_olympiadbench # 11
)
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"

# --- GPU preflight (same as phase2_score.sbatch) ---
GPU_IDX="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"; GPU_IDX="${GPU_IDX%%,*}"
read -r FREE TOTAL <<<"$(nvidia-smi --query-gpu=memory.free,memory.total \
  --format=csv,noheader,nounits -i "$GPU_IDX" | tr -d ' ' | tr ',' ' ')"
if [ "${FREE:-0}" -lt "$(( TOTAL * 85 / 100 ))" ]; then
  echo "[$(date)] PREFLIGHT FAIL: $(hostname -s) gpu $GPU_IDX ${FREE}/${TOTAL} MiB free; requeue" >&2
  [ "${SLURM_RESTART_COUNT:-0}" -ge 8 ] && exit 1
  EXC="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^ExcNodeList=//p')"
  [ "$EXC" = "(null)" ] && EXC=""
  scontrol update jobid="$SLURM_JOB_ID" ExcNodeList="$(hostname -s)${EXC:+,$EXC}" || true
  scontrol requeue "$SLURM_JOB_ID"; sleep 60; exit 0
fi
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> $CELL on $(hostname), gpu $GPU_IDX ${FREE}/${TOTAL} MiB"

python generation/phase2_scores/scripts/04_conf_trace_v2.py --cell "$CELL" --batch 8
echo "[$(date)] $CELL done"

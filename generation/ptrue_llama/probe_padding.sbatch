#!/bin/bash
#SBATCH --job-name=e22-ptrue-check
#SBATCH --partition=gpu,cgpu   ## EDIT: partition name of your cluster
#SBATCH --constraint=ampere|adalovelace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%A_%a.out   ## EDIT: log directory (relative to the submit directory)
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"   ## model cache; the pinned revision must be present when HF_HUB_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "${PRICE_SCRATCH:-$SLURM_SUBMIT_DIR}"   ## directory holding score_llama.py, scoring_config.json and the key parquet files
python probe_padding.py

#!/bin/bash
# One-time environment setup on Amarel (run on a LOGIN node, has internet).
#
#   cd <repo>/experiments
#   bash phase1_rollouts/slurm/setup_env.sh
#
# Creates a conda env "awv" with python 3.11 + the GPU requirements, puts the
# HuggingFace cache on /scratch, and symlinks outputs/ to /scratch (home
# quotas are small and home I/O is slow; /scratch is per-user, fast, and
# visible from compute nodes). Adjust the module line to whatever
# `module avail` on Amarel currently offers if needed.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
SCRATCH="/scratch/$USER"
HF_HOME="$SCRATCH/hf_cache"

# --- conda (Amarel guide: module load a miniconda, or use your own install) ---
module purge || true
module load miniconda 2>/dev/null || module load anaconda 2>/dev/null || true
if ! command -v conda >/dev/null; then
  echo "conda not found: module-load a conda (check 'module avail') or install"
  echo "miniconda under $SCRATCH, then rerun."
  exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q '^awv '; then
  conda create -y -n awv python=3.11
fi
conda activate awv

pip install -r "$EXP_DIR/requirements.txt" -r "$EXP_DIR/requirements-gpu.txt"

mkdir -p "$HF_HOME"

# SLURM opens the #SBATCH --output file BEFORE the job script runs, so the
# logs dir must exist at submit time
mkdir -p "$EXP_DIR/phase1_rollouts/slurm/logs"

# outputs -> /scratch: pools are active job I/O (fast, big) — the repo (code,
# usually in /home) stays clean and backed up. Amarel guide: /home I/O is slow,
# /scratch is purge-eligible only above the 1TB soft quota (we stay ~50GB).
if [ ! -e "$EXP_DIR/outputs" ]; then
  mkdir -p "$SCRATCH/awv_outputs"
  ln -s "$SCRATCH/awv_outputs" "$EXP_DIR/outputs"
  echo "outputs/ -> $SCRATCH/awv_outputs (scratch is NOT backed up:"
  echo "  run 'bash phase1_rollouts/slurm/archive_pools.sh' after generation to copy pools to /home)"
fi
echo
echo "Environment ready. Next steps (still on the login node, from $EXP_DIR):"
echo "  conda activate awv"
echo "  export HF_HOME=$HF_HOME"
echo "  python generation/phase1_rollouts/scripts/00_prefetch_assets.py    # all repos ungated, no HF login needed"
echo "  python generation/phase1_rollouts/scripts/01_generate_pool.py --cell all --dry-run   # loader check, no GPU"
echo "  sbatch phase1_rollouts/slurm/phase1_pool.sbatch          # 12-cell array job"

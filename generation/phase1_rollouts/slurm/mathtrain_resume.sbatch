#!/bin/bash
#SBATCH --job-name=awv-mt-resume
#SBATCH --partition=gpu-redhat   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%A_%a.out
#
# Resume the MATH-train pool after the first attempt OOM'd at 93/149 shards.
#
# Cause: each shard's vLLM RequestOutputs carry ~8M Logprob objects at m=64,
# and holding the previous shard's graph while building the next grew RSS by
# ~2 GB/shard -- fine for the 10-shard MATH-500 cell, fatal for the ~37 shards
# each worker owns here (MaxRSS 59 GB against a 64 GB request).
#
# Three independent guards, because a leak that is only characterised and not
# understood should not be trusted to a single fix:
#   1. `del outs, rows` + gc.collect() per shard  (awv/generate.py)
#   2. --max-shards 10, so the process is retired and rebuilt before it can
#      accumulate, driven by the retry loop below
#   3. 120 GB instead of 64 GB
#
# Finished shards are skipped, so this simply continues from 93/149.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate awv
export HF_HOME="$HOME/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

CELL=qwen25-1.5b_mathtrain
SHARDS="outputs/cells/$CELL/pools/pool_shards"
# `ls` on a missing directory exits non-zero, and under `set -euo pipefail`
# that kills the script before any work starts. This survived only because the
# directory already existed here; the same line killed the math500deep job.
mkdir -p "$SHARDS"
n_shards() { find "$SHARDS" -name 'shard_*.parquet' 2>/dev/null | wc -l | tr -d ' '; }
echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID}/4 resuming $CELL on $(hostname)"

for attempt in $(seq 1 12); do
  before=$(n_shards)
  echo "[$(date)] pass $attempt (shards on disk: $before/149)"
  # a pass that finds nothing left in its own slice exits 0 having written
  # nothing, which is the loop's termination signal
  python generation/phase1_rollouts/scripts/01_generate_pool.py --cell "$CELL" \
      --shard-mod 4 --shard-rem "$SLURM_ARRAY_TASK_ID" --max-shards 10
  after=$(n_shards)
  if [ "$after" -le "$before" ]; then
    echo "[$(date)] no new shards this pass -- worker slice complete"
    break
  fi
done

echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID} done ($(n_shards)/149 shards)"

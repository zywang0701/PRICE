#!/bin/bash
#SBATCH --job-name=awv-l3-post
#SBATCH --partition=gpu   ## EDIT: partition name of your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=generation/phase1_rollouts/slurm/logs/%x_%j.out
#
# Everything the MATH-train pool needs AFTER generation, chained on one GPU:
#
#   03  h(q) embeddings          -- required by every retrieval-based estimator
#   04  conf_trace_v2            -- the corrected DeepConf statistic
#   09  add phi_deepconf2        -- merges it into pool.parquet, the column the
#                                   round-3 search was run on
#   02  P(True) (--skip-prm)     -- the main text's score; PRM is skipped, it
#                                   needs a separate 7B verifier and is
#                                   quarantined from the token accounting anyway
#
# Ordered cheapest-first so a partial run still leaves something usable: the
# embeddings alone unblock the kNN baselines on the new corpus.
#
# Submit with a dependency on the generation array, e.g.
#   sbatch --dependency=afterany:60017766:60018862 \
#          phase1_rollouts/slurm/mathtrain_post.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate awv
export HF_HOME="$HOME/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# stage 04 imports `awv` directly (unlike 02/03, which go through _common and
# patch sys.path themselves), so the repo root has to be importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CELL=llama32-3b_mathtrain
POOL="outputs/cells/$CELL/pools/pool.parquet"

# afterany (not afterok) so a single flaky worker cannot silently cancel this
# chain -- but then the pool really has to be there before we start
if [ ! -f "$POOL" ]; then
  echo "ERROR: $POOL missing -- generation did not finish merging." >&2
  echo "Shards present: $(ls outputs/cells/$CELL/pools/pool_shards/ 2>/dev/null | wc -l)" >&2
  exit 1
fi
echo "[$(date)] post-generation chain for $CELL on $(hostname)"

# each stage is guarded on its own artifact, so a rerun after a mid-chain
# failure resumes rather than repeating hours of finished GPU work
P="outputs/cells/$CELL/pools"

if [ ! -f "$P/embeddings.npz" ]; then
  echo "--- 03 embeddings"
  python generation/phase2_scores/scripts/03_embed_queries.py --cell "$CELL"
else
  echo "--- 03 embeddings: present, skipping"
fi

if [ ! -f "$P/conf_trace2.parquet" ]; then
  echo "--- 04 conf_trace_v2"
  python generation/phase2_scores/scripts/04_conf_trace_v2.py --cell "$CELL"
else
  echo "--- 04 conf_trace_v2: present, skipping"
fi

echo "--- 09 install phi_deepconf2"
python generation/phase2_scores/scripts/09_add_deepconf2_column.py --cell "$CELL"

echo "--- 02 P(True) (PRM skipped)"
python generation/phase2_scores/scripts/02_score_pool.py --cell "$CELL" --skip-prm

echo "[$(date)] post chain done"

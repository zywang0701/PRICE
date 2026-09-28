#!/bin/bash
# Copy the paid phase-1 artifacts (final pools + query lists, NOT the shard
# working dirs) from scratch to backed-up home storage. Scratch has no
# backups/snapshots; the pools are the only thing phase 1 spends money on.
# Idempotent: rsync only copies what changed. Run on a login node.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
DEST="${1:-$HOME/awv_pool_archive}"

mkdir -p "$DEST"
rsync -av --prune-empty-dirs \
  --include='*/' --include='pool.parquet' --include='queries.parquet' \
  --include='report.json' --exclude='*' \
  "$EXP_DIR/outputs/cells/" "$DEST/cells/"

echo
du -sh "$DEST"
echo "Archived to $DEST (backed-up /home). To pull to your laptop:"
echo "  rsync -avz $USER@amarel.rutgers.edu:$DEST/ ./awv_pool_archive/"

#!/bin/bash
# The experiment chain behind the paper's P(True) results, in dependency order.
# Run from the repository root after `python scripts/download_data.py`.
# Every step is a CPU replay over the downloaded pools; each folder's PROTOCOL.md
# is the pre-registered design of that step.
#
#   bash experiments/run_all.sh oracle     # E02 -> E16 -> E17 -> E18 -> E20 -> E21 -> E27   (PRICE-oracle, Table 1 left, Fig. rate law)
#   bash experiments/run_all.sh deployed   # E22 -> E23 -> E24 -> E25 -> E26              (PRICE-deployed, Table 1 right, Fig. frontier)
#   bash experiments/run_all.sh paper      # tables and figures into build/ (needs both branches)
#   bash experiments/run_all.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
E=experiments
what="${1:-all}"

oracle() {
  for c in qwen llama; do
    echo "== E02 frozen replay ($c)";      python $E/replay_frozen.py $c
    echo "== E16 answer sets / grading ($c)"
    python $E/E16_answer_sets/data.py --cell $c --phase train
    python $E/E16_answer_sets/data.py --cell $c --phase test
    echo "== E17 split-pool joint oracle ($c)"
    python $E/E17_split_joint_oracle/build_tables.py --cell $c
    python $E/E17_split_joint_oracle/run_oracle.py --cell $c --cost exact
    python $E/E17_split_joint_oracle/run_oracle.py --cell $c --cost plugin
    echo "== E18 replay costs and price sweep ($c)"
    python $E/E18_lambda_replay/build_replay_cost.py --cell $c
    python $E/E18_lambda_replay/run_sweep.py --cell $c
    python $E/E18_lambda_replay/read_frontiers.py --cell $c
    echo "== E20 aligned score arrays ($c)"
    python $E/E20_nonprm_scores/prepare.py --cell $c
  done
  echo "== E20 voting tables at fixed count (both cells, four scores)"
  python $E/E20_nonprm_scores/run_all.py
  python $E/E20_nonprm_scores/analyze.py
  echo "== E21 P(True) adaptive-count oracle (both cells, both tie conventions)"
  python $E/E21_ptrue_adaptive_count/run_all.py
  echo "== E27 fixed-count controls on the E21 replay"
  python paper_assets/ptrue_oracle_fixed_count.py
}

deployed() {
  for c in qwen llama; do
    echo "== E22 PRICE-deployed: calibration, training, evaluation ($c)"
    python $E/E22_ptrue_deployed/prepare.py --cell $c --phase train
    python $E/E22_ptrue_deployed/run_local.py --cell $c
    python $E/E22_ptrue_deployed/analyze.py --cell $c
    echo "== E23 baselines and Table 1 columns ($c)"
    python $E/E23_ptrue_table3/run.py --cell $c
    python $E/E23_ptrue_table3/analyze.py --cell $c
    echo "== E24 committed per-query optimizer ($c)"
    python $E/E24_ptrue_committed/run.py --cell $c
    python $E/E24_ptrue_committed/analyze.py --cell $c
    echo "== E25 query-adaptive voting as the deployed policy ($c)"
    python $E/E25_required_adaptive/analyze.py --cell $c
  done
  echo "== E26 rate law and allocation by level"
  python paper_assets/ptrue_rate_allocation.py
}

paper() {
  echo "== tables (build/tables) and figures (build/images)"
  python paper_assets/ptrue_paper_assets.py
  python paper_assets/ptrue_figures.py
}

case "$what" in
  oracle) oracle ;;
  deployed) deployed ;;
  paper) paper ;;
  all) oracle; deployed; paper ;;
  *) echo "usage: $0 {oracle|deployed|paper|all}" >&2; exit 2 ;;
esac

# Phase 2: score passes + query embeddings (NOT YET ADAPTED)

Carried over from the E1 codebase; runs after the phase-1 pools exist.

- `scripts/02_score_pool.py` — P(True) self-evaluation (`phi_conf`) and PRM
  step-reward scores appended to each cell's pool. **Pending work before
  running (plan §4):** swap the PRM to `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`
  (math cells only; same standalone-process discipline as the Qwen PRM), and
  add the Qwen2.5-Math-PRM-72B subset arm on the two headline cells.
- `scripts/03_embed_queries.py` — h(q) hidden-state embeddings for the online
  kNN (F5).

Both take `--cell <name>` like every stage. The pool schema already contains
every phase-2 column (NaN until filled), so phase 2 never regenerates rollouts.

"""Stage 01 (GPU): generate a cell's pool, m=256 rollouts per query.

One (model, task) cell per invocation (`--cell llama31-8b_math500`), or
`--cell all` to run every registered cell sequentially on one GPU. Resumable:
one parquet shard per block of queries; finished shards (and cells whose final
pool parquet exists) are skipped, so SLURM requeues are safe.

`--dry-run` loads the queries and prints example prompts + counts without
importing vLLM — run it on a login node to validate loaders before paying GPU.
"""

import sys

from _common import base_parser, get_cfg


def run_cell(args, cell: str | None):
    args.cell = cell
    cfg = get_cfg(args)

    from awv import data
    from awv.utils import ensure_dir

    label = cell or "(base config)"
    if cfg.paths.pool.exists() and not args.force:
        print(f"[01] {label}: pool exists at {cfg.paths.pool}, skipping "
              f"(--force to regenerate)")
        return

    queries = data.load_queries(cfg)
    print(f"[01] {label}: {len(queries)} queries, model={cfg.generation.model}, "
          f"m={cfg.generation.m_rollouts}, max_new_tokens={cfg.generation.max_new_tokens}")

    if args.dry_run:
        row = queries.iloc[0]
        print("--- system prompt ---")
        print(cfg.generation.system_prompt.strip())
        print("--- first question ---")
        print(row["question"][:1500])
        print("--- gold answer ---")
        print(row["gold_answer"])
        return

    ensure_dir(cfg.paths.pool.parent)
    queries.to_parquet(cfg.paths.pool.parent / "queries.parquet", index=False)
    from awv import generate
    generate.generate_pool(cfg, queries, cfg.generation.m_rollouts, cfg.paths.pool,
                           shard_mod=args.shard_mod, shard_rem=args.shard_rem,
                           max_shards=args.max_shards)


def main():
    p = base_parser(__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="load queries + print sample prompts, no GPU work")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if the final pool parquet exists")
    p.add_argument("--shard-mod", type=int, default=1,
                   help="run several GPUs on one cell: this worker takes the "
                        "shards congruent to --shard-rem modulo --shard-mod")
    p.add_argument("--shard-rem", type=int, default=0)
    p.add_argument("--max-shards", type=int, default=None,
                   help="write at most this many shards then exit, so a long "
                        "run can retire the process before RSS accumulates")
    args = p.parse_args()

    from awv.config import list_cells

    if args.cell is None:
        print("Registered cells:")
        for c in list_cells():
            print(" ", c)
        print("\nPick one with --cell <name>, or run everything with --cell all.")
        sys.exit(1)

    cells = list_cells() if args.cell == "all" else [args.cell]
    for cell in cells:
        run_cell(args, cell)


if __name__ == "__main__":
    main()

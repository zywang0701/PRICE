"""Stage 03 (GPU): h(q) hidden-state embeddings for the online kNN."""

import pandas as pd

from _common import base_parser, get_cfg


def main():
    args = base_parser(__doc__).parse_args()
    cfg = get_cfg(args)

    from awv import embed

    queries = pd.read_parquet(cfg.paths.pool.parent / "queries.parquet")
    embed.embed_queries(cfg, queries, cfg.paths.embeddings)


if __name__ == "__main__":
    main()

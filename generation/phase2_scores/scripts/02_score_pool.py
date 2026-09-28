"""Stage 02 (GPU): extra score passes over the logged pool.

Fills phi_conf (P(True) + nothing else needed: phi_lik and phi_selfcert were
logged in stage 01) and phi_prm_{prod,last,min} (Qwen2.5-Math-PRM-7B).
Idempotent: columns are overwritten in place on the pool parquet.
"""

import pandas as pd

from _common import base_parser, get_cfg


def main():
    p = base_parser(__doc__)
    p.add_argument("--skip-ptrue", action="store_true")
    p.add_argument("--skip-prm", action="store_true")
    args = p.parse_args()
    cfg = get_cfg(args)

    from awv import scores

    queries = pd.read_parquet(cfg.paths.pool.parent / "queries.parquet")
    if not args.skip_ptrue:
        scores.score_ptrue(cfg, cfg.paths.pool, queries)
    if not args.skip_prm:
        scores.score_prm(cfg, cfg.paths.pool, queries)
    df = pd.read_parquet(cfg.paths.pool)
    for c in ["phi_lik", "phi_selfcert", "phi_conf", "phi_prm_prod"]:
        print(c, "nan-fraction:", float(df[c].isna().mean()))


if __name__ == "__main__":
    main()

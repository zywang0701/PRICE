"""Optional (GPU): F0 pilot gate — kept for reference; the F campaign runs
with gates omitted (decision 2026-07-08), so nothing in phase 1 requires this.

Generates a small pilot pool (m_pilot rollouts/query), then reports:
  - single-rollout accuracy, overall and per difficulty level, against the
    pre-registered 25-60% band, with the slice recommendation;
  - realized length distribution and the implied gamma for the tail guarantee;
  - answer-diversity stats (distinct answers per query).
"""

from _common import base_parser, get_cfg


def main():
    p = base_parser(__doc__)
    p.add_argument("--reuse-pool", action="store_true", help="skip generation, reuse pilot pool")
    args = p.parse_args()
    cfg = get_cfg(args)

    import numpy as np
    from awv import data, generate
    from awv.utils import read_pool, write_json, ensure_dir

    queries = data.load_queries(cfg)
    ensure_dir(cfg.paths.pilot_pool.parent)
    queries.to_parquet(cfg.paths.pilot_pool.parent / "queries.parquet", index=False)

    if not args.reuse_pool:
        generate.generate_pool(cfg, queries, cfg.generation.m_pilot, cfg.paths.pilot_pool)

    pool = read_pool(cfg.paths.pilot_pool)
    acc = pool.groupby("query_id")["correct"].mean()
    by_level = pool.merge(queries[["query_id", "level"]], on="query_id") \
                   .groupby("level")["correct"].mean()
    lengths = pool["ell_tokens"].to_numpy()
    overall = float(pool["correct"].mean())
    t0 = cfg.budget.overspend_tokens
    in_band = 0.25 <= overall <= 0.60
    if in_band:
        slice_rec = "all levels"
    elif overall > 0.60:
        slice_rec = "restrict to levels [4, 5]"
    else:
        slice_rec = "restrict to levels [1, 2, 3], else fall back to GSM8K"

    # ---- gamma calibration (pre-registered rule) and budget readouts ----
    from awv import calibrate

    gtable = calibrate.gamma_table(pool, [float(g) for g in cfg.budget.gamma_grid],
                                   m_target=cfg.generation.m_rollouts,
                                   tail_prob=cfg.budget.tail_prob)
    rec = calibrate.recommend_gamma(gtable, cfg.budget.max_risk_premium,
                                    cfg.budget.max_mgf_rel_se)
    readouts = calibrate.budget_readouts(pool, rec["gamma"],
                                         cfg.solver.readout_mean_counts)

    report = {
        "n_queries": int(len(queries)),
        "m_pilot": int(cfg.generation.m_pilot),
        "single_rollout_accuracy": overall,
        "accuracy_by_level": {str(k): float(v) for k, v in by_level.items()},
        "band_25_60": in_band,
        "slice_recommendation": slice_rec,
        "length_mean": float(lengths.mean()),
        "length_p50": float(np.percentile(lengths, 50)),
        "length_p95": float(np.percentile(lengths, 95)),
        "length_max": int(lengths.max()),
        "distinct_answers_per_query_mean": float(
            pool.groupby("query_id")["answer_canonical"].nunique().mean()),
        "queries_with_correct_in_pool": float((acc > 0).mean()),
        "gamma_table": gtable.to_dict(orient="records"),
        "gamma_recommended": rec,
        "gamma_plan_default": float(np.log(1 / cfg.budget.tail_prob * 1.0) / t0),
        "budget_readouts_at_recommended_gamma": readouts,
        "config_updates": {
            "budget.gamma": rec["gamma"],
            "budget.clip_kappa": float(np.exp(rec["gamma"] * t0) - 1.0),
        },
    }
    write_json(cfg.paths.pilot_report, report)
    print(gtable.to_string(index=False))
    print("recommended:", rec)
    print("readouts:", readouts)


if __name__ == "__main__":
    main()

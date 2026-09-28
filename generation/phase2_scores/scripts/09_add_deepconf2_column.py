#!/usr/bin/env python3
"""Phase 3 · step 9 — install the CORRECTED DeepConf score column.

Writes `phi_deepconf2` (= per-rollout Lowest Group Confidence in the paper's
C-convention, from conf_trace2.parquet; see delivery/baselines_published_spec.md
§4.4) into each cell's pool.parquet, and appends its score-calibrated eta-grid
entry to phase3/calibration.json — so the standard pipeline can rebuild the
grid column:

  python generation/phase2_scores/scripts/09_add_deepconf2_column.py --all

Idempotent: rewriting the column and recalibrating produce identical values.
The original (sign-bugged) `phi_deepconf` column and its tables are left in
place for the record; downstream figures/tables should switch to phi_deepconf2.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from awv import grids

_ROOT = pathlib.Path(__file__).resolve().parents[3]
F1_CELLS = ["llama31-8b_math500", "llama32-3b_math500", "qwen25-1.5b_math500"]
COL = "phi_deepconf2"


def add_column(cell: str, seed: int = 0) -> None:
    pdir = _ROOT / "outputs" / "cells" / cell / "pools"
    pool = pd.read_parquet(pdir / "pool.parquet")
    tr = pd.read_parquet(pdir / "conf_trace2.parquet",
                         columns=["lgc", "conf_min_logged"])
    assert len(tr) == len(pool), f"{cell}: conf_trace2/pool length mismatch"
    # row-order check: the trace's copied logged column must equal the pool's
    corr = np.corrcoef(tr["conf_min_logged"], pool["conf_window_min"])[0, 1]
    assert corr > 0.999, f"{cell}: conf_trace2 row order mismatch (corr={corr:.4f})"
    pool[COL] = tr["lgc"].to_numpy()
    pool.to_parquet(pdir / "pool.parquet")

    calib_path = _ROOT / "outputs" / "cells" / cell / "phase3" / "calibration.json"
    rec = json.loads(calib_path.read_text())
    c = grids.calibrate_cell(pool[["query_id", COL]], COL, int(rec["T_bar"]),
                             frac=rec["calib_frac"], seed=rec["seed"])
    rec["scores"][COL] = {
        "spread": c["spread"],
        "degenerate": bool(c["degenerate"]),
        "eta_grid": [None if e == float("inf") else round(e, 4)
                     for e in c["eta_grid"]],
        "eta_max": round(c["eta_max"], 4),
    }
    calib_path.write_text(json.dumps(rec, indent=2))
    print(f"{cell}: {COL} written (corr_check={corr:.4f}, "
          f"spread={c['spread']:.3f}, eta_max={c['eta_max']:.2f}, "
          f"degenerate={c['degenerate']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cells = F1_CELLS if args.all else [args.cell]
    if not cells or cells == [None]:
        ap.error("pass --cell or --all")
    for cell in cells:
        add_column(cell)


if __name__ == "__main__":
    main()

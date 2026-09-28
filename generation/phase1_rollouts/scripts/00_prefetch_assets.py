"""Prefetch every model + dataset into the HF cache (no GPU needed).

Run this ON A LOGIN NODE (or any node with internet), with HF_HOME pointed at
shared storage (e.g. /scratch/$USER/hf_cache on Amarel), so the SLURM GPU jobs
never need to download anything. All repos are ungated: no HF login needed.

    export HF_HOME=/scratch/$USER/hf_cache
    python phase1_rollouts/scripts/00_prefetch_assets.py            # everything
    python phase1_rollouts/scripts/00_prefetch_assets.py --datasets-only
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from awv.config import list_cells, load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets-only", action="store_true")
    p.add_argument("--models-only", action="store_true")
    args = p.parse_args()

    cfgs = [load_config(cell=c) for c in list_cells()]

    if not args.models_only:
        from datasets import load_dataset
        seen = set()
        for cfg in cfgs:
            key = cfg.data.dataset
            if key in seen:
                continue
            seen.add(key)
            print(f"[prefetch] dataset {key}")
            if key == "math500":
                load_dataset(cfg.data.hf_id_math500, split="test")
            elif key == "gsm8k":
                load_dataset(cfg.data.hf_id_gsm8k, "main", split="test")
            elif key == "olympiadbench":
                load_dataset(cfg.data.hf_id_olympiadbench,
                             cfg.data.olympiadbench_subset, split="train")
            elif key == "mmlupro":
                load_dataset(cfg.data.hf_id_mmlupro, split="test")

    if not args.datasets_only:
        from huggingface_hub import snapshot_download
        models = sorted({cfg.generation.model for cfg in cfgs})
        for mid in models:
            print(f"[prefetch] model {mid}")
            snapshot_download(mid)

    print("[prefetch] done — cache is ready for offline GPU jobs")


if __name__ == "__main__":
    main()

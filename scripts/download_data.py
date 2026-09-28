"""Download the companion dataset into the layout the experiment chain expects.

    python scripts/download_data.py            # everything (~0.9 GB)
    python scripts/download_data.py --cells qwen25-1.5b_math500 qwen25-1.5b_mathtrain

Places  cells/<cell>/...   under  <repo>/outputs/cells/<cell>/
and     overrides/llama/*  under  <repo>/experiments/E22_ptrue_deployed/llama/
then verifies every file against manifest.json shipped with the dataset.
"""
import argparse, hashlib, json, os, pathlib, shutil, sys
from huggingface_hub import snapshot_download

REPO = pathlib.Path(__file__).resolve().parents[1]
DATASET = "zach-wang/PRICE-rollouts"
CELLS = ["qwen25-1.5b_math500", "qwen25-1.5b_mathtrain", "llama32-3b_math500", "llama32-3b_mathtrain"]

def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path(os.environ.get("PRICE_ROOT", REPO)))
    a = ap.parse_args()
    patterns = ["manifest.json", "overrides/*"] + [f"cells/{c}/*" for c in a.cells]
    local = pathlib.Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=patterns))
    manifest = json.loads((local / "manifest.json").read_text())
    bad = 0
    for rel, meta in manifest["files"].items():
        src = local / rel
        if not src.exists():
            continue
        if rel.startswith("cells/"):
            dst = a.root / "outputs" / rel
        elif rel.startswith("overrides/llama/"):
            dst = a.root / "experiments" / "E22_ptrue_deployed" / "llama" / pathlib.Path(rel).name
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != meta["sha256"]:
            shutil.copyfile(src, dst)
        ok = sha256(dst) == meta["sha256"]
        bad += not ok
        print(("ok   " if ok else "FAIL ") + str(dst.relative_to(a.root)))
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()

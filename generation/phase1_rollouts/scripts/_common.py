"""Shared CLI plumbing for the numbered scripts (cell-registry aware).

Lives inside a phase folder: the experiments/ root (where awv/, config/ and
outputs/ live) is two levels up.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from awv.config import list_cells, load_config  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None,
                   help="path to base YAML config (default: config/base.yaml)")
    p.add_argument("--cell", default=None,
                   help=f"cell name from config/cells/ (one of: {', '.join(list_cells())})")
    return p


def get_cfg(args):
    return load_config(args.config, cell=getattr(args, "cell", None))

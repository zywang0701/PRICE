"""Config loading: YAML -> attribute-accessible namespace.

F-campaign cell registry: `load_config(cell="llama31-8b_math500")` deep-merges
config/cells/<cell>.yaml over config/base.yaml and scopes every output path
under outputs/cells/<cell>/, so the 12 (model, task) cells share one pinned
base and differ only in the few keys their yaml overrides.
"""

from __future__ import annotations

import math
import pathlib
from types import SimpleNamespace

import yaml

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def list_cells() -> list[str]:
    """Registered cell names = yaml stems in config/cells/, sorted."""
    return sorted(p.stem for p in (_ROOT / "config" / "cells").glob("*.yaml"))


def load_config(path: str | pathlib.Path = None, cell: str | None = None):
    """Load YAML config; optionally overlay a cell config from the registry.

    Relative paths in cfg.paths are resolved against the experiments/ directory
    (the parent of this package). When `cell` is given, output paths are scoped
    under outputs/cells/<cell>/.
    """
    path = pathlib.Path(path) if path else _ROOT / "config" / "base.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    if cell is not None:
        cell_path = _ROOT / "config" / "cells" / f"{cell}.yaml"
        if not cell_path.exists():
            raise FileNotFoundError(
                f"unknown cell {cell!r}; registered cells: {list_cells()}")
        with open(cell_path) as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
        raw.setdefault("cell", {}).setdefault("name", cell)

    cfg = _to_ns(raw)

    # ---- resolve nulls that default to the generator ----
    if getattr(cfg.scores, "ptrue_model", None) is None:
        cfg.scores.ptrue_model = cfg.generation.model
    if getattr(cfg.embedding, "model", None) is None:
        cfg.embedding.model = cfg.generation.model
    # per-task default system prompt (cfg.prompts.<dataset>)
    if getattr(cfg.generation, "system_prompt", None) is None:
        cfg.generation.system_prompt = getattr(cfg.prompts, cfg.data.dataset)

    # YAML's .inf already parses to float('inf'); coerce grid to floats anyway.
    cfg.engine.eta_grid = [float(x) for x in cfg.engine.eta_grid]
    assert all(b >= a for a, b in zip(cfg.engine.eta_grid, cfg.engine.eta_grid[1:])), \
        "eta_grid must be sorted"
    assert math.isinf(cfg.engine.eta_grid[-1]), "grid must end with .inf (best-of-n corner)"

    # ---- path scoping ----
    cell_name = getattr(getattr(cfg, "cell", None), "name", None)
    for name, val in vars(cfg.paths).items():
        p = pathlib.Path(val)
        if not p.is_absolute():
            if cell_name is not None and p.parts and p.parts[0] == "outputs":
                p = pathlib.Path("outputs") / "cells" / cell_name / pathlib.Path(*p.parts[1:])
            p = _ROOT / p
        setattr(cfg.paths, name, p)

    # clip level is DERIVED from (gamma, overspend_tokens) so the two can never
    # drift apart when the pilot updates gamma
    if hasattr(cfg, "budget"):
        cfg.budget.clip_kappa = float(
            math.exp(cfg.budget.gamma * cfg.budget.overspend_tokens) - 1.0)
    return cfg

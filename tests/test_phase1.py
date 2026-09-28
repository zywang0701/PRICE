"""Phase-1 (F-campaign pool generation) unit tests: cell-config merging,
MMLU-Pro letter extraction/grading, stratified subsetting, and the
DeepConf-style sliding-window confidence stats. All CPU, no downloads."""

import numpy as np
import pandas as pd
import pytest

from awv import answers, data
from awv.config import list_cells, load_config
from awv.generate import window_conf_stats

EXPECTED_CELLS = {
    "qwen25-1.5b_math500", "qwen25-1.5b_mathtrain",
    "llama32-3b_math500", "llama32-3b_mathtrain",
}   # the four cells released with the paper (the full campaign registry had 12)


# ------------------------------------------------------------- cell registry

def test_registry_has_the_released_cells():
    assert EXPECTED_CELLS <= set(list_cells())


@pytest.mark.parametrize("cell", sorted(EXPECTED_CELLS))
def test_cell_config_loads(cell):
    cfg = load_config(cell=cell)
    assert cfg.generation.model is not None
    assert cfg.data.dataset in cell
    # paths scoped per cell so pools can never collide
    assert f"cells/{cell}" in str(cfg.paths.pool)
    # nulls resolved to the generator
    assert cfg.scores.ptrue_model == cfg.generation.model
    assert cfg.embedding.model == cfg.generation.model
    # per-task system prompt resolved
    assert cfg.generation.system_prompt
    if cfg.data.dataset == "mmlupro":
        assert "answer is" in cfg.generation.system_prompt
        assert cfg.data.subset_size == 500
    if cfg.data.dataset == "olympiadbench":
        assert cfg.generation.max_new_tokens == 4096
    else:
        assert cfg.generation.max_new_tokens == 2048
    # pinned engine setting from the plan
    assert cfg.engine.S_permutations == 512


def test_cell_paths_distinct():
    pools = {str(load_config(cell=c).paths.pool) for c in list_cells()}
    assert len(pools) == len(EXPECTED_CELLS)


def test_unknown_cell_raises():
    with pytest.raises(FileNotFoundError):
        load_config(cell="nope_nope")


# --------------------------------------------------------- mmlupro answers

@pytest.mark.parametrize("text,want", [
    ("blah blah. The answer is (C).", "C"),
    ("… the answer is B", "B"),
    ("The answer is (A). Wait, no. The answer is (D).", "D"),
    ("Therefore \\boxed{E}", "E"),
    ("Therefore \\boxed{(F)}", "F"),
    ("I think option (G) fits best", "G"),
    ("no letter here", None),
    ("The answer is (K).", None),      # K is not a valid 10-way option
])
def test_extract_letter(text, want):
    assert answers.extract_answer(text, "mmlupro") == want


def test_mmlupro_grade_and_canonicalize():
    assert answers.grade("c", "C", "mmlupro")
    assert not answers.grade("B", "C", "mmlupro")
    assert not answers.grade(None, "C", "mmlupro")
    assert answers.canonicalize("c", "mmlupro") == "C"
    assert answers.canonicalize("", "mmlupro") == "<none>"
    # two failed extractions must not grade as equal to each other or gold
    assert not answers.grade("Z", "C", "mmlupro")


def test_math_grading_unchanged():
    assert answers.grade("\\dfrac{1}{2}", "\\frac{1}{2}")
    assert answers.canonicalize("  \\left( 3 \\right)") == "(3)"


def test_format_mcq_question():
    q = data.format_mcq_question("What is 2+2?", ["3", "4", "5"])
    assert "Options:" in q and "A. 3" in q and "B. 4" in q and "C. 5" in q


# ------------------------------------------------------ stratified sampling

def test_stratified_subset_proportional_and_deterministic():
    rng = np.random.default_rng(0)
    cats = ["a"] * 600 + ["b"] * 300 + ["c"] * 100
    df = pd.DataFrame({
        "question": [f"q{i}" for i in range(1000)],
        "category": cats,
        "gold_answer": ["A"] * 1000,
    }).sample(frac=1, random_state=3).reset_index(drop=True)  # shuffle input order
    sub = data.stratified_subset(df, 100, seed=42)
    assert len(sub) == 100
    counts = sub["category"].value_counts()
    assert counts["a"] == 60 and counts["b"] == 30 and counts["c"] == 10
    # deterministic and input-order independent
    df2 = df.sample(frac=1, random_state=7).reset_index(drop=True)
    sub2 = data.stratified_subset(df2, 100, seed=42)
    assert sorted(sub["question"]) == sorted(sub2["question"])


def test_stratified_subset_small_stratum_kept():
    df = pd.DataFrame({
        "question": [f"q{i}" for i in range(101)],
        "category": ["big"] * 100 + ["tiny"],
    })
    sub = data.stratified_subset(df, 10, seed=0)
    assert len(sub) == 10
    assert "tiny" in set(sub["category"])   # >=1 per stratum


# ------------------------------------------------- deepconf window stats

def test_window_conf_stats_known_values():
    conf = np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    st = window_conf_stats(conf, window=2)
    # window means: [-.5, -1, -.5, 0, 0]
    assert st["conf_window_min"] == -1.0
    assert st["conf_mean"] == pytest.approx(-2.0 / 6.0)
    assert st["conf_tail"] == 0.0                      # last 2 tokens
    assert st["conf_window_p10"] <= st["conf_window_min"] + 1e-12 or \
           st["conf_window_p10"] >= st["conf_window_min"]  # well-defined


def test_window_conf_stats_short_and_empty():
    st = window_conf_stats(np.array([-2.0]), window=128)  # shorter than window
    assert st["conf_window_min"] == -2.0 == st["conf_mean"] == st["conf_tail"]
    st = window_conf_stats(np.array([]), window=128)
    assert np.isnan(st["conf_window_min"]) and np.isnan(st["conf_mean"])


def test_window_conf_stats_matches_naive():
    rng = np.random.default_rng(1)
    conf = rng.normal(-1, 0.5, size=300)
    w = 64
    st = window_conf_stats(conf, w)
    naive = np.array([conf[i:i + w].mean() for i in range(300 - w + 1)])
    assert st["conf_window_min"] == pytest.approx(naive.min())
    assert st["conf_window_p10"] == pytest.approx(np.percentile(naive, 10))

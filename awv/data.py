"""Dataset loading for the F campaign (plan §3.2).

Tasks: MATH-500, OlympiadBench (OE_TO_maths_en_COMP), MMLU-Pro (stratified 500),
GSM8K (pre-registered fallback), and MATH-train (the calibration corpus for the
learned primitives). Every loader returns the same schema:
query_id, question, gold_answer, level, category — where `question` is the full
user-message text (options are already formatted in for MCQ), `gold_answer` is
the reference answer in the same space the extractor produces (boxed content
for math, an option letter for MMLU-Pro), and level/category carry the task's
difficulty/stratification metadata (0 / "" when absent).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from awv.answers import extract_boxed

MMLU_LETTERS = "ABCDEFGHIJ"


def format_mcq_question(question: str, options: list[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    lines += [f"{MMLU_LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


def stratified_subset(df: pd.DataFrame, n: int, seed: int,
                      by: str = "category") -> pd.DataFrame:
    """Deterministic proportional stratified sample (largest-remainder
    apportionment, >=1 item per stratum). df is sorted before sampling so the
    result is independent of the input row order."""
    df = df.sort_values([by, "question"], kind="stable").reset_index(drop=True)
    if n >= len(df):
        return df
    sizes = df.groupby(by).size()
    quotas = sizes / len(df) * n
    take = np.maximum(np.floor(quotas).astype(int), 1)
    rem = n - int(take.sum())
    if rem > 0:
        order = (quotas - np.floor(quotas)).sort_values(ascending=False).index
        for cat in order[:rem]:
            take[cat] += 1
    elif rem < 0:
        order = take.sort_values(ascending=False).index
        for cat in order[:-rem]:
            take[cat] -= 1
    rng = np.random.default_rng(seed)
    parts = []
    for cat, group in df.groupby(by, sort=True):
        k = min(int(take[cat]), len(group))
        idx = rng.choice(len(group), size=k, replace=False)
        parts.append(group.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


_WORD = re.compile(r"[a-z0-9]+")


def _norm_question(text: str) -> str:
    """Whitespace/case/LaTeX-spacing-insensitive form, for duplicate matching."""
    s = str(text).lower()
    s = re.sub(r"\\[,;:!]|\\quad|\\qquad|~", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _near_duplicate_mask(questions, reference, thresh: float = 0.8):
    """Word-Jaccard >= thresh against any reference question.

    An inverted index on the rarer words keeps this linear in practice: a MATH
    train item is only compared against references that share vocabulary with
    it. Exact matches after normalization are caught by the same test at
    Jaccard 1.0.
    """
    ref_sets = [set(_WORD.findall(_norm_question(r))) for r in reference]
    index: dict[str, list[int]] = {}
    for j, s in enumerate(ref_sets):
        for w in s:
            index.setdefault(w, []).append(j)
    out = np.zeros(len(questions), dtype=bool)
    for i, q in enumerate(questions):
        qs = set(_WORD.findall(_norm_question(q)))
        if not qs:
            continue
        cand: dict[int, int] = {}
        for w in qs:
            for j in index.get(w, ()):
                cand[j] = cand.get(j, 0) + 1
        for j, inter in cand.items():
            union = len(qs) + len(ref_sets[j]) - inter
            if union and inter / union >= thresh:
                out[i] = True
                break
    return out


def stratified_interleave(df: pd.DataFrame, seed: int,
                          by=("category", "level")) -> pd.DataFrame:
    """Round-robin the strata so any prefix is balanced across (category, level).

    Generation is sharded in order and resumable, so an interleaved corpus makes
    every prefix a usable stratified pilot: the learning-curve study can read
    N = 1k, 2k, ... off the shards that have finished rather than waiting for
    the whole corpus.
    """
    rng = np.random.default_rng(seed)
    buckets = []
    for _, g in df.groupby(list(by), sort=True):
        idx = g.index.to_numpy()
        buckets.append(list(rng.permutation(idx)))
    order = []
    while any(buckets):
        for b in buckets:
            if b:
                order.append(b.pop())
    return df.loc[order].reset_index(drop=True)


def _strip_math_delims(ans: str) -> str:
    """Strip $ delimiters plus stray trailing punctuation, to a fixed point
    (some OlympiadBench golds end in '...$.')."""
    s = str(ans).strip()
    while True:
        t = re.sub(r"^\$+|\$+$", "", s).strip().rstrip(".").strip()
        if t == s:
            return s
        s = t


def load_queries(cfg) -> pd.DataFrame:
    from datasets import load_dataset  # heavy import kept local

    name = cfg.data.dataset
    if name == "math500":
        ds = load_dataset(cfg.data.hf_id_math500, split="test")
        # 'level' may be an int or a string like "Level 3" depending on version
        levels = [int(m.group()) if (m := re.search(r"\d+", str(x))) else 0
                  for x in ds["level"]]
        df = pd.DataFrame({
            "question": ds["problem"],
            "gold_answer": ds["answer"],
            "level": levels,
            "category": [str(x) for x in ds["subject"]] if "subject" in ds.column_names
                        else [""] * len(levels),
        })
    elif name == "mathtrain":
        # The MATH training split (Hendrycks et al.), all seven subjects: the
        # calibration corpus the learned primitives are fit on, evaluated on
        # MATH-500. Deduplicated against MATH-500 so retrieval can never reach
        # a near-copy of an evaluation query, and interleaved by (subject,
        # level) so any shard prefix is a balanced pilot.
        subjects = ["algebra", "counting_and_probability", "geometry",
                    "intermediate_algebra", "number_theory", "prealgebra",
                    "precalculus"]
        parts = []
        for subj in subjects:
            ds = load_dataset(cfg.data.hf_id_mathtrain, subj, split="train")
            lv = [int(m.group()) if (m := re.search(r"\d+", str(x))) else 0
                  for x in ds["level"]]
            parts.append(pd.DataFrame({
                "question": ds["problem"],
                "gold_answer": [extract_boxed(s) for s in ds["solution"]],
                "level": lv,
                "category": subj,
            }))
        df = pd.concat(parts, ignore_index=True)
        n0 = len(df)
        # the reference solution is the only source of a gold answer here, so
        # items whose solution has no \boxed{} are ungradable and are dropped
        df = df[df["gold_answer"].notna()].reset_index(drop=True)
        print(f"[data] mathtrain: {len(df)}/{n0} items carry a boxed gold")
        ref = load_dataset(cfg.data.hf_id_math500, split="test")["problem"]
        dup = _near_duplicate_mask(df["question"].tolist(), ref)
        if dup.any():
            print(f"[data] mathtrain: dropping {int(dup.sum())} items that "
                  f"near-duplicate a MATH-500 evaluation query")
            df = df[~dup].reset_index(drop=True)
        df = stratified_interleave(df, int(cfg.data.subset_seed))
    elif name == "gsm8k":
        ds = load_dataset(cfg.data.hf_id_gsm8k, "main", split="test")
        answers = [a.split("####")[-1].strip().replace(",", "")
                   for a in ds["answer"]]
        df = pd.DataFrame({
            "question": ds["question"],
            "gold_answer": answers,
            "level": [0] * len(answers),
            "category": [""] * len(answers),
        })
    elif name == "olympiadbench":
        ds = load_dataset(cfg.data.hf_id_olympiadbench,
                          cfg.data.olympiadbench_subset, split="train")
        df = pd.DataFrame({
            "question": ds["question"],
            "gold_list": ds["final_answer"],
            "multi": ds["is_multiple_answer"] if "is_multiple_answer" in ds.column_names
                     else [False] * len(ds),
            "category": [str(x) for x in ds["subfield"]] if "subfield" in ds.column_names
                        else [""] * len(ds),
        })
        # single-answer items only: multi-answer grading is ill-posed for a
        # vote over one extracted answer (logged; expected to drop only a few)
        n0 = len(df)
        df = df[(~df["multi"].astype(bool))
                & (df["gold_list"].map(lambda g: g is not None and len(g) == 1))]
        print(f"[data] olympiadbench: kept {len(df)}/{n0} single-answer items")
        df = df.assign(
            gold_answer=[_strip_math_delims(g[0]) for g in df["gold_list"]],
            level=0,
        )[["question", "gold_answer", "level", "category"]].reset_index(drop=True)
        # a '$' surviving the strip marks a multi-part gold that slipped past
        # the is_multiple_answer flag; grading it as one answer is ill-posed
        leftover = df["gold_answer"].str.contains(r"\$")
        if leftover.any():
            print(f"[data] olympiadbench: dropping {int(leftover.sum())} "
                  f"items with multi-part golds")
            df = df[~leftover].reset_index(drop=True)
    elif name == "mmlupro":
        ds = load_dataset(cfg.data.hf_id_mmlupro, split="test")
        df = pd.DataFrame({
            "question": [format_mcq_question(q, o)
                         for q, o in zip(ds["question"], ds["options"])],
            "gold_answer": [str(a).strip().upper() for a in ds["answer"]],
            "level": [0] * len(ds),
            "category": [str(c) for c in ds["category"]],
        })
        bad = ~df["gold_answer"].isin(set(MMLU_LETTERS))
        if bad.any():
            print(f"[data] mmlupro: dropping {int(bad.sum())} rows with non-letter gold")
            df = df[~bad].reset_index(drop=True)
        if cfg.data.subset_size is not None:
            df = stratified_subset(df, int(cfg.data.subset_size),
                                   int(cfg.data.subset_seed))
    else:
        raise ValueError(f"unknown dataset {name}")

    if cfg.data.difficulty_levels is not None:
        keep = set(int(x) for x in cfg.data.difficulty_levels)
        df = df[df["level"].astype(int).isin(keep)].reset_index(drop=True)
    if cfg.data.max_queries is not None:
        df = df.iloc[: int(cfg.data.max_queries)].reset_index(drop=True)
    df.insert(0, "query_id", range(len(df)))
    return df

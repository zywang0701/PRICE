"""Answer extraction, canonicalization, and grading (task-aware).

Math tasks (math500, olympiadbench, gsm8k): primary path uses the math-verify
package (the released MATH grader family); falls back to a normalized string
match when math-verify is unavailable or fails to parse. The same
canonicalization is applied to rollout answers and to the gold answer, so the
answer space A is consistent.

MMLU-Pro: the answer space is the option letter A-J. Extraction follows the
standard harness ("The answer is (X)"), with \\boxed{X} and a trailing "(X)"
as fallbacks; grading is exact letter match.
"""

from __future__ import annotations

import re

MMLU_LETTERS = "ABCDEFGHIJ"

_BOXED = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} content with balanced braces; None if absent."""
    last = None
    for m in _BOXED.finditer(text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[start:i - 1]
    return last


_LAST_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANSWER_IS = re.compile(r"answer\s+is:?\s*\**\s*\(?([A-Ja-j])\)?", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([A-Ja-j])\)")


def extract_letter(text: str) -> str | None:
    """MMLU-Pro option letter: 'The answer is (X)' (last occurrence), then a
    boxed letter, then the last parenthesized letter."""
    hits = _ANSWER_IS.findall(text)
    if hits:
        return hits[-1].upper()
    boxed = extract_boxed(text)
    if boxed:
        s = re.sub(r"[\s()\\.$]|\\text\{|\}", "", boxed)
        if len(s) == 1 and s.upper() in MMLU_LETTERS:
            return s.upper()
    hits = _PAREN_LETTER.findall(text)
    if hits:
        return hits[-1].upper()
    return None


def extract_answer(text: str, dataset: str = "math500") -> str | None:
    if dataset == "gsm8k":
        boxed = extract_boxed(text)
        cands = _LAST_NUMBER.findall(boxed if boxed else text)
        return cands[-1].replace(",", "") if cands else None
    if dataset == "mmlupro":
        return extract_letter(text)
    return extract_boxed(text)   # math500, olympiadbench


def canonicalize(ans: str | None, dataset: str = "math500") -> str:
    """Normalized string form used as the vote's answer key."""
    if ans is None:
        return "<none>"
    if dataset == "mmlupro":
        s = ans.strip().upper()
        return s if s in set(MMLU_LETTERS) else "<none>"
    s = ans.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        s = str(int(float(s)))
    return s if s else "<none>"


def grade(pred: str | None, gold: str, dataset: str = "math500") -> bool:
    """True iff pred matches gold in the task's answer space."""
    if pred is None:
        return False
    if dataset == "mmlupro":
        return canonicalize(pred, dataset) == canonicalize(gold, dataset) != "<none>"
    try:
        from math_verify import parse, verify  # optional dependency
        g, p = parse(f"${gold}$"), parse(f"${pred}$")
        if verify(g, p):
            return True
    except Exception:
        pass
    return canonicalize(pred) == canonicalize(gold)

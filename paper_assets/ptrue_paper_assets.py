"""Build manuscript P(True) assets from frozen E21/E25 results; no fitting.

Run with python paper_assets/ptrue_paper_assets.py.
--experiment-root can point to another copy of the lab_review artifacts.
Checks table values independently against saved per-query replay arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import os
REPO = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("PRICE_PAPER_DIR", REPO / "build"))
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-root", type=Path, default=REPO / "experiments")
ROOT = parser.parse_args().experiment_root
GAMMA = .0029262
BLUE = "#0052af"
LABELS = {"qwen": "Qwen2.5-1.5B", "llama": "Llama-3.2-3B"}
BASELINES = {"sc": "Self-consistency", "bon": "Best-of-$n$", "cisc": "CISC", "ac": "Adaptive-Consistency", "esc": "ESC", "dcoff": "DeepConf (offline)"}
sources = {}


def read(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def mix(values, point):
    return (1 - point["theta"]) * values[point["lo"]] + point["theta"] * values[point["hi"]]


def hull(cost, reward):
    """Independent increasing upper concave hull in MGF coordinates."""
    order = sorted(range(len(cost)), key=lambda i: (cost[i], -reward[i]))
    keep = []
    for i in order:
        if keep and (cost[i] == cost[keep[-1]] or reward[i] <= reward[keep[-1]]):
            continue
        while len(keep) > 1:
            a, b = keep[-2:]
            if (reward[b] - reward[a]) / (cost[b] - cost[a]) > (reward[i] - reward[b]) / (cost[i] - cost[b]):
                break
            keep.pop()
        keep.append(i)
    return np.array(keep)


def adjacent_mix(cost, reward, budget):
    """Read a fixed-count sweep at `budget` by mixing the two adjacent counts whose realized
    MGF costs bracket it (no hull, dominated counts are kept), so the reading follows the
    actual accuracy of each count. Returns (accuracy, lo, hi, theta) with 0-based indices."""
    o = np.argsort(cost); c, r = cost[o], reward[o]; target = np.exp(GAMMA * budget)
    assert target >= c[0]
    if target >= c[-1]:
        return float(r[-1]), int(o[-1]), int(o[-1]), 0.
    j = int(np.searchsorted(c, target)); lo, hi = max(0, j - 1), j
    theta = float((target - c[lo]) / (c[hi] - c[lo])) if hi != lo else 0.
    return float((1 - theta) * r[lo] + theta * r[hi]), int(o[lo]), int(o[hi]), theta


def values_at(data, budgets):
    c, r = (data["C"].mean(1), data["R"].mean(1)) if "C" in data else (data["Cbar"], data["Rbar"])
    h = hull(c, r)
    return 100 * np.interp(np.exp(GAMMA * np.asarray(budgets)), c[h], r[h])


def table(path, columns, rows, caption, label, split=None, header=r"budget $b$ $\rightarrow$", width="0.95", head="Method"):
    r"""Overleaf-era table layout (2026-09-17): resizebox, 3.5pt columns, generator blocks
    separated by 12pt, `Method\hfill budget ->` header, ourshl row for PRICE, green/red
    subscript deltas via \up{}/\dn{}. Rows are (name, cells, bold) or ("group", label),
    ("rule",), ("space",); cells are preformatted strings."""
    n = len(columns)
    spec = "@{}l" + ("c" * split + "@{\\hspace{12pt}}" + "c" * (n - split) if split else "c" * n) + "@{}"
    lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{" + width + r"\textwidth}{!}{%",
             r"\setlength{\tabcolsep}{3.5pt}\begin{tabular}{" + spec + "}", r"\toprule"]
    if split:
        lines.extend([f" & \\multicolumn{{{split}}}{{c}}{{Qwen2.5-1.5B}} & \\multicolumn{{{n-split}}}{{c}}{{Llama-3.2-3B}} " + r"\\", f"\\cmidrule(lr){{2-{split+1}}}\\cmidrule(lr){{{split+2}-{n+1}}}"])
    lines += [head + "\\hfill " + header + " & " + " & ".join(columns) + r" \\", r"\midrule"]
    for idx, row in enumerate(rows):
        nxt = rows[idx + 1][0] if idx + 1 < len(rows) else None
        if row[0] == "group":
            lines.append(f"\\multicolumn{{{n+1}}}{{@{{}}l}}{{\\emph{{{row[1]}}}}}\\\\[1pt]")
        elif row[0] == "rule":
            lines.append(r"\midrule")
        elif row[0] == "space":
            lines.append(r"\addlinespace[3pt]")
        elif row[0] == "delta":          # ("delta", numeric gains, shaded[, label]): green/red numbers under the row above
            # PRICE rows (shaded) carry their gain in the accent red of the figures; count gains stay green
            cells = [r"{\scriptsize\textcolor{" + ("deltaprice" if row[2] else ("deltagood" if g >= 0 else "deltabad")) + "}{(" + format(g, "+.1f") + ")}}" for g in row[1]]
            row_label = row[3] if len(row) > 3 else ""      # (do not shadow the table label)
            lines.append((r"\rowcolor{ourshl} " if row[2] else "") + row_label + " & " + " & ".join(cells) + r" \\")
        else:
            name, values, bold = row
            end = r" \\[-1.5pt]" if nxt == "delta" else r" \\"
            lines.append((r"\rowcolor{ourshl} " if bold else "") + name + " & " + " & ".join(values) + end)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    (PAPER / "tables" / path).write_text("\n".join(lines) + "\n")


def cell(v, delta=None, bold=False, dec=1):
    r"""`58.9`, `\textbf{58.9}\up{2.0}`: one-decimal accuracy with an optional subscript delta."""
    s = format(v, f".{dec}f")
    if bold:
        s = r"\textbf{" + s + "}"
    if delta is not None:
        s += (r"\up{" if delta >= 0 else r"\dn{") + format(abs(delta), f".{dec}f") + "}"
    return s


E21 = {tag: read(ROOT / "E21_ptrue_adaptive_count" / tag / "common_tie/results.json") for tag in LABELS}
E27 = {tag: read(ROOT / "E27_ptrue_oracle_fixed_count" / tag / "results.json") for tag in LABELS}
NATIVE = {tag: read(ROOT / "E21_ptrue_adaptive_count" / tag / "score_native_tie/results.json") for tag in LABELS}
E25 = {tag: read(ROOT / "E25_required_adaptive" / tag / "results.json") for tag in LABELS}
points = {tag: read(ROOT / "E25_required_adaptive" / tag / "readout_points.json") for tag in LABELS}
seq = {tag: dict(np.load(ROOT / "E22_ptrue_deployed" / tag / "test/regression_w4.npz")) for tag in LABELS}
check = {}

for tag, d in E25.items():
    p = d["policies"]["sequential_adaptive"]
    assert d["policy_config"]["sequential_primary"] == "regression_w4"
    assert not d["policy_config"]["fixed_temperature_fallback_allowed"]
    np.testing.assert_allclose(values_at(seq[tag], d["budgets"]), p["accuracy_pct"], atol=1e-10)
    for j, point in enumerate(points[tag]["sequential_adaptive"]):
        np.testing.assert_allclose(100 * mix(seq[tag]["R"], point).mean(), p["accuracy_pct"][j], atol=1e-10)
        np.testing.assert_allclose(np.log(mix(seq[tag]["C"], point).mean()) / GAMMA, d["budgets"][j], atol=1e-8)
        np.testing.assert_allclose(mix(seq[tag]["M"], point).mean(), p["mean_tokens"][j], atol=1e-8)
        np.testing.assert_allclose(sum(p["usage"][j]["temperature_share"]), 1, atol=1e-10)
    for key in BASELINES:
        base = dict(np.load(ROOT / "E23_ptrue_table3" / tag / "baselines" / (key + ".npz")))
        np.testing.assert_array_equal(base["qid"], seq[tag]["qid"])
        np.testing.assert_allclose(values_at(base, d["budgets"]), d["baseline_accuracy_pct"][key], atol=1e-10)
    selection = read(ROOT / "E23_ptrue_table3" / tag / "selection.json")
    for j, arm in enumerate(selection["fixed_arms"]):
        raw = np.load(ROOT / "E22_ptrue_deployed" / tag / "test" / (arm + ".npz"))
        np.testing.assert_allclose(values_at(raw, [d["budgets"][j]])[0], d["fixed_sequential_accuracy_pct"][j], atol=1e-10)
    for key, rel in [("committed_joint", "E24_ptrue_committed"), ("committed_query", "E25_required_adaptive")]:
        filename = "joint.npz" if key == "committed_joint" else "query_locked.npz"
        raw = np.load(ROOT / rel / tag / "test" / filename)
        np.testing.assert_allclose(values_at(raw, d["budgets"]), d["policies"][key]["accuracy_pct"], atol=1e-10)
    for mode in ["common_tie", "score_native_tie"]:
        source = E21[tag] if mode == "common_tie" else NATIVE[tag]
        for key, arm in [("joint", "joint"), ("sc", "fixed_0"), ("bon", "fixed_12"), ("best_fixed_hindsight", None)]:
            for b, point in zip(source["budgets"], source["methods"][key]):
                name = arm or f'fixed_{point["temperature_index"]}'
                raw = np.load(ROOT / "E21_ptrue_adaptive_count" / tag / mode / (name + ".npz"))
                np.testing.assert_allclose(100 * mix(raw["R_B"], point).mean(), point["accuracy_B_pct"], atol=1e-10)
                risk = np.log(mix(raw["C_B"], point).mean()) / GAMMA
                np.testing.assert_allclose(risk, point["b_actual"], atol=1e-8)
                assert risk <= b + 1e-8  # Saturated frontiers may leave budget unused.
    # Invert the feasible MGF hull at the SC accuracy in the fourth budget column.
    target = d["baseline_accuracy_pct"]["sc"][3] / 100
    c, r = seq[tag]["C"].mean(1), seq[tag]["R"].mean(1)
    h = hull(c, r)
    required = np.log(np.interp(target, r[h], c[h])) / GAMMA
    check[tag] = dict(queries=d["queries"], primary="regression_w4", main_cells_verified=48,
                      oracle_cells_verified=56, sc_anchor_accuracy_pct=100 * target,
                      sc_anchor_risk=d["budgets"][3], price_risk_at_anchor=required,
                      risk_ratio=d["budgets"][3] / required,
                      adaptive_mean_counts=[float(mix(seq[tag]["N"], p).mean()) for p in points[tag]["sequential_adaptive"]])

fmt = lambda values, signed=False: [format(v, "+.2f" if signed else ".2f") for v in values]
both = lambda get: [x for tag in LABELS for x in get(tag)]
ORACLE_COLS = [0, 2, 3, 4, 5, 6]   # 2k, 5k, 8k, 12k, 20k, 30k of the seven oracle budgets (Table 2 drops 3k)
DEPLOYED_COLS = [0, 1, 2, 3, 4, 5]  # all six deployed budgets, n = 3, 4, 5, 8, 10, 22 (Table 3)
pick = lambda values, cols, per: [values[i] for i in cols] + [values[per + i] for i in cols]
def subset(rows, cols, per):
    """Keep only the selected budget columns of every (name, cells, bold) row."""
    return [(r[0], pick(r[1], cols, per)) + tuple(r[2:]) if len(r) >= 3 else r for r in rows]
blocks, fixed_read, gains = [], {}, {}
for key, name in [("sc", r"SC ($\tau=0$)"), ("bon", r"BoN ($\tau=\infty$)")]:
    for tag in LABELS:
        raw = np.load(ROOT / "E27_ptrue_oracle_fixed_count" / tag / (key + "_fixed_count.npz"))
        np.testing.assert_allclose(values_at(dict(R=raw["R_B"], C=raw["C_B"]), E21[tag]["budgets"]),
            [p["accuracy_B_pct"] for p in E27[tag]["methods"][key]], atol=1e-10)
        for p in E27[tag]["methods"][key]:
            np.testing.assert_allclose(np.log(mix(raw["C_B"], p).mean())/GAMMA, p["b_actual"], atol=1e-8)
        check[tag]["oracle_fixed_count_cells_verified"] = check[tag].get("oracle_fixed_count_cells_verified", 0) + 7
        # 2026-09-17 (Zhenyu): fixed-count rows are read as swept, not as the feasible envelope.
        reads = [adjacent_mix(raw["C_B"].mean(1), raw["R_B"].mean(1), b) for b in E21[tag]["budgets"]]
        fixed_read[(tag, key)] = [100 * x[0] for x in reads]
        if key == "sc":   # SC is monotone here, so the swept reading equals the E27 envelope reading
            np.testing.assert_allclose(fixed_read[(tag, key)], [p["accuracy_B_pct"] for p in E27[tag]["methods"][key]], atol=1e-6)
        check[tag].setdefault("oracle_fixed_count_as_swept", {})[key] = dict(
            accuracy_B_pct=[round(v, 4) for v in fixed_read[(tag, key)]],
            counts=[(x[1] + 1, x[2] + 1, round(x[3], 4)) for x in reads])
    fixed = both(lambda t: fixed_read[(t, key)])
    adaptive = both(lambda t: [p["accuracy_B_pct"] for p in E21[t]["methods"][key]])
    gains[key] = [a - f for a, f in zip(adaptive, fixed)]
    if key == "sc":
        np.testing.assert_allclose(gains[key], both(lambda t: E27[t]["contrasts"]["sc_count_gain_pp"]), atol=1e-6)
    check["count_gain_pp_" + key] = [round(g, 3) for g in gains[key]]
    blocks += [("group", name), (r"\quad fixed count", [cell(v) for v in fixed], False),
               (r"\quad adaptive count", [cell(v) for v in adaptive], False), ("delta", gains[key], False), ("space",)]
# hindsight-best fixed temperature with one common count (all 13 temperatures, read as swept)
for tag in LABELS:
    allz = np.load(ROOT / "E27_ptrue_oracle_fixed_count" / tag / "all_fixed_count.npz")
    c = allz["C_B"].mean(1)
    per_k = np.array([[100 * adjacent_mix(c, allz["R_B"][k].mean(1), b)[0] for b in E21[tag]["budgets"]] for k in range(allz["R_B"].shape[0])])
    np.testing.assert_allclose(per_k[0], fixed_read[(tag, "sc")], atol=1e-9)
    np.testing.assert_allclose(per_k[-1], fixed_read[(tag, "bon")], atol=1e-9)
    fixed_read[(tag, "best")] = per_k.max(0).tolist()
    check[tag]["oracle_best_fixed_tau_fixed_count"] = dict(accuracy_B_pct=[round(v, 4) for v in fixed_read[(tag, "best")]],
                                                          temperature_index=per_k.argmax(0).tolist())
fixed = both(lambda t: fixed_read[(t, "best")])
adaptive = both(lambda t: [x["accuracy_B_pct"] for x in E21[t]["methods"]["best_fixed_hindsight"]])
gains["best"] = [a - f for a, f in zip(adaptive, fixed)]
check["count_gain_pp_best"] = [round(g, 3) for g in gains["best"]]
blocks += [("group", r"Best fixed $\tau$ (hindsight)"), (r"\quad fixed count", [cell(v) for v in fixed], False),
           (r"\quad adaptive count", [cell(v) for v in adaptive], False), ("delta", gains["best"], False)]
joint = both(lambda t: [x["accuracy_B_pct"] for x in E21[t]["methods"]["joint"]])
rule_gain = both(lambda t: E21[t]["contrasts"]["best_fixed"]["mean"])
assert all(g > 0 for g in rule_gain)   # bold marks the column best, which is PRICE-oracle everywhere
rows = blocks + [("rule",), (r"PRICE-oracle", [cell(v, bold=True) for v in joint], True), ("delta", rule_gain, True, r"\quad adaptive count \& vote")]
oracle_headers = ["2k", "3k", "5k", "8k", "12k", "20k", "30k"]
check["table2_columns"] = [oracle_headers[i] for i in ORACLE_COLS]
for key in ("sc", "bon", "best"):
    g = check["count_gain_pp_" + key]
    check["shown_count_gain_range_" + key] = dict(qwen=[round(min(g[i] for i in ORACLE_COLS), 2), round(max(g[i] for i in ORACLE_COLS), 2)],
                                                   llama=[round(min(g[7 + i] for i in ORACLE_COLS), 2), round(max(g[7 + i] for i in ORACLE_COLS), 2)])
    print("shown_count_gain_range", key, check["shown_count_gain_range_" + key])
print("best_tau fixed-count row", {t: [round(v, 1) for v in fixed_read[(t, "best")]] for t in LABELS})
table("tab_oracle_ptrue.tex", pick(oracle_headers * 2, ORACLE_COLS, 7), subset(rows, ORACLE_COLS, 7),
      r"PRICE-oracle against fixed voting rules on MATH-500 with the self-evaluation score (accuracy in \%, at matched realized entropic-risk budget). Fixed count uses one rollout count for all queries, swept over $n=1,\ldots,64$ and read at the count whose realized budget matches the column, and adaptive count chooses the count per query. The best fixed $\tau$ is the single temperature on the grid that gives the highest accuracy at that budget, chosen separately for every budget and row after seeing the results. Green numbers in parentheses are the gain of the adaptive count over the fixed count with the same voting rule, and red numbers are the gain of PRICE-oracle over the best fixed $\tau$ with adaptive count. Appendix~\ref{app:exp-split-pool} gives the protocol and the omitted $3$k budget.", "tab:oracle-vs-fixed", len(ORACLE_COLS), width="0.98")

# 2026-09-17 (Zhenyu): fixed-count rows are read as swept. The column budgets are the realized
# budgets of SC and BoN at n = 3, 4, 5, 8, 10, 22 on both generators, so SC's envelope reading is
# already its exact operating point, while BoN (which peaks at n=2 with P(True)) is re-read at
# those counts from the replay tensors instead of at its saturated envelope.
exact = {}
for tag in LABELS:
    z = np.load(ROOT / "E22_ptrue_deployed" / tag / "test.npz")
    np.testing.assert_array_equal(z["qid"], seq[tag]["qid"])
    tok = z["tok"].astype(float); corr = z["corr"]
    bn = np.log(np.exp(GAMMA * tok).mean(1).mean(0)) / GAMMA          # realized budget of count n
    cols = [int(np.argmin(np.abs(bn - b))) + 1 for b in E25[tag]["budgets"]]
    np.testing.assert_allclose([bn[n - 1] for n in cols], E25[tag]["budgets"], atol=1e-6)
    assert cols == [3, 4, 5, 8, 10, 22], cols
    sc_exact = [100 * corr[:, :, n - 1, 0].mean() for n in cols]
    np.testing.assert_allclose(sc_exact, E25[tag]["baseline_accuracy_pct"]["sc"], atol=1e-9)
    exact[tag] = dict(counts=cols, bon=[100 * corr[:, :, n - 1, 12].mean() for n in cols])
    assert max(exact[tag]["bon"]) < min(E25[tag]["best_baseline_accuracy_pct"])   # leads vs best baseline unchanged
    check[tag]["deployed_column_counts"] = cols
    check[tag]["bon_exact_at_column_counts_pct"] = [round(v, 4) for v in exact[tag]["bon"]]
baseline_row = lambda key: (r"\quad " + BASELINES[key], [cell(v) for v in both(lambda t: exact[t]["bon"] if key == "bon" else E25[t]["baseline_accuracy_pct"][key])], False)
price = both(lambda t: E25[t]["policies"]["sequential_adaptive"]["accuracy_pct"])
lead = both(lambda t: E25[t]["policies"]["sequential_adaptive"]["gain_vs_best_pp"])
assert all(g > 0 for g in lead)   # bold marks the column best, which is PRICE-deployed everywhere
rows = [("group", "Fixed-count")] + [baseline_row(k) for k in ("sc", "bon", "cisc")] + [("space",),
        ("group", "Adaptive-count")] + [baseline_row(k) for k in ("ac", "esc", "dcoff")] + [("rule",),
        ("PRICE-deployed", [cell(v, bold=True) for v in price], True), ("delta", lead, True)]
check["table3_columns"] = pick(both(lambda t: E25[t]["headers"]), DEPLOYED_COLS, 6)
table("tab_deployed_ptrue.tex", pick(both(lambda t: E25[t]["headers"]), DEPLOYED_COLS, 6), subset(rows, DEPLOYED_COLS, 6),
      r"Accuracy (\%) at matched realized entropic-risk budget $b$ on MATH-500 with the self-evaluation score. Bold marks the column best, and the red numbers in parentheses are PRICE-deployed minus the best baseline in the column. Every method is replayed on the same rollout pools and paths. The column budgets are the realized budgets of $n=3,4,5,8,10,22$ rollouts, and fixed-count methods are read at those counts, so BoN's row follows its actual decline with $n$. Adaptive-count methods and PRICE-deployed are read at the column budget from the feasible frontiers of their sweeps. Appendix~\ref{app:exp-baselines-impl} gives the baseline sweeps.", "tab:deployed", len(DEPLOYED_COLS), width="0.95")

fixed = both(lambda t: E25[t]["fixed_sequential_accuracy_pct"])
diff = both(lambda t: E25[t]["policies"]["sequential_adaptive"]["gain_vs_fixed_pp"])
np.testing.assert_allclose(diff, np.array(price) - np.array(fixed), atol=5e-3)
rows = [("PRICE-deployed (adaptive vote)", [cell(v, bold=True) for v in price], True),
        ("Fixed-temperature controller", [cell(v) for v in fixed], False),
        ("PRICE-deployed $-$ fixed temperature (pp)", fmt(diff, True), False)]
table("tab_ptrue_fixed_ablation.tex", both(lambda t: E25[t]["headers"]), rows,
      r"Fixed-temperature ablation of PRICE-deployed with the self-evaluation score (accuracy in \%, at the six budgets of Table~\ref{tab:deployed}). The fixed-temperature controller shares the score CDF, embeddings, reward predictor, cost prior, horizon and replay paths of PRICE-deployed and still adapts its rollout count per query, but votes at one temperature per budget selected on the calibration tune queries. The last row is PRICE-deployed minus this controller, and Table~\ref{tab:ci} gives its paired-bootstrap intervals.", "tab:fixed-ablation", 6)

rows = []
for tag in LABELS:
    p = E25[tag]["policies"]["sequential_adaptive"]
    rows.append((LABELS[tag] + ": budget", E25[tag]["headers"], False))
    for key, name in [("gain_vs_best", "PRICE $-$ best baseline")]:
        ci = p[key + "_ci"]
        rows.append((name + " (pp)", fmt(p[key + "_pp"], True), False))
        rows.append((r"95\% interval", [f'[{a:+.2f}, {b:+.2f}]' for a, b in zip(ci["lo"], ci["hi"])], False))
table("tab_ptrue_ci.tex", ["1", "2", "3", "4", "5", "6"], rows,
      r"Pointwise $95\%$ paired query-bootstrap intervals for the lead of PRICE-deployed over the best baseline in Table~\ref{tab:deployed} (800 draws). Each draw keeps the 64 orderings of a sampled query together, rebuilds every frontier and reselects the best baseline. The calibrated components stay frozen.", "tab:ci", header=r"budget column $\rightarrow$")

rows = []
for tag in LABELS:
    d = E21[tag]
    rows.append((LABELS[tag] + r": PRICE-oracle $-$ best fixed $\tau$ (pp)", fmt(d["contrasts"]["best_fixed"]["mean"], True), False))
    rows.append((r"\quad 95\% interval", [f'[{a:+.2f}, {b:+.2f}]' for a, b in zip(d["contrasts"]["best_fixed"]["lo"], d["contrasts"]["best_fixed"]["hi"])], False))
    for key, name in [("best_fixed_hindsight", r"\quad Best fixed $\tau$"), ("joint", r"\quad PRICE-oracle")]:
        rows.append((name + ": mean rollout count", fmt([p["mean_n"] for p in d["methods"][key]]), False))
        rows.append((name + ": mean generated tokens", [f'{p["mean_tokens"]:,.0f}' for p in d["methods"][key]], False))
table("tab_ptrue_oracle_details.tex", ["2k", "3k", "5k", "8k", "12k", "20k", "30k"], rows,
      r"Split-pool oracle of Table~\ref{tab:oracle-vs-fixed}. The gain of PRICE-oracle over the hindsight-best fixed temperature with its pointwise $95\%$ paired query-bootstrap interval from 1,000 draws, and the mean rollout count and mean generated tokens of both arms at each budget on the held-out half B.", "tab:split-pool", width="0.98")

requested = both(lambda t: E25[t]["budgets"])
achieved = both(lambda t: [x["actual_risk"] for x in E25[t]["prospective"]["sequential_adaptive"]])
overspend = [100 * (a - r) / r for a, r in zip(achieved, requested)]
rows = [("Requested budget (tokens)", [f"{x:,.0f}" for x in requested], False),
        ("Achieved budget (tokens)", [f"{x:,.0f}" for x in achieved], False),
        (r"\quad overspend (\%)", [format(x, "+.1f") for x in overspend], False),
        (r"Accuracy (\%), price chosen in advance", [cell(x["accuracy_pct"]) for x in both(lambda t: E25[t]["prospective"]["sequential_adaptive"])], False),
        (r"Accuracy (\%), price read retrospectively", [cell(v) for v in price], False)]
check["prospective_overspend_pct"] = [round(x, 2) for x in overspend]
table("tab_ptrue_cost_prospective.tex", both(lambda t: E25[t]["headers"]), rows,
      r"Prices chosen in advance. For each budget of Table~\ref{tab:deployed}, the price of PRICE-deployed is selected on the audit queries of the calibration corpus and applied unchanged to MATH-500. The achieved entropic-risk budget overshoots the request by at most $0.9\%$ on Qwen2.5-1.5B and $3.8\%$ on Llama-3.2-3B, and the accuracy matches the retrospective readout of Table~\ref{tab:deployed}.", "tab:prospective", 6, head="")

rows = []
for key, name in [("committed_joint", r"Committed joint: $\tau(q,b)$"), ("committed_query", r"Committed query-only vote: $\tau(q)$")]:
    vals = fmt(both(lambda t: E25[t]["policies"][key]["accuracy_pct"]))
    rows.append((name, [r"\textbf{" + v + "}" for v in vals] if key == "committed_joint" else vals, key == "committed_joint"))
rows += [("Fixed-temperature committed (ablation)", fmt(both(lambda t: E25[t]["fixed_committed_accuracy_pct"])), False),
         ("Best external baseline", fmt(both(lambda t: E25[t]["best_baseline_accuracy_pct"])), False)]
table("tab_ptrue_committed.tex", both(lambda t: E25[t]["headers"]), rows,
      r"self-evaluation committed implementations: accuracy (\%) at the six retrospective risk budgets of Table~\ref{tab:deployed}. All actions are chosen before sampling from query-only predictions. Bold identifies the joint adaptive implementation. Query-only voting fixes each query's temperature across budgets, but still adapts its count. These learned committed variants do not reproduce the broad lead of the sequential controller.", "tab:committed-ptrue", 6)

import sys
sys.path.insert(0, str(REPO / "paper_assets"))
import ptrue_figures  # house-style figures (2026-09-17), read the same saved artifacts
figure_stats = ptrue_figures.draw_all(ROOT)
check["figures"] = figure_stats

audit=dict(experiment_root=str(ROOT),source_sha256=sources,checks=check,
           status="passed",scope="Independent saved-array reconstruction; no retraining, new rollout generation or bootstrap rerun.")
(PAPER/"ptrue_manuscript_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(check,indent=2))

#!/usr/bin/env python3
"""Stage 04 (GPU): DeepConf group-confidence trajectory, CORRECTED convention.

Fixes the sign/extremum bug documented in delivery/baselines_published_spec.md
§4.4: DeepConf's token confidence is C_t = -(1/k)·Σ log P over the top-k
tokens (positive, higher = more confident; Fu et al. 2025, Eq. 2), and the
online statistic is the LOWEST group confidence LGC = min_j G_j (Eq. 6).
The generation pass logged the un-negated mean top-5 logprob and took the
min window — i.e. -max_j G_j, the wrong extremum; `conf_trace.parquet` (v1)
stored the same convention.

This pass re-scores each rollout's logged text with one teacher-forcing
forward (window 128, top-5, temperature 0.8 — the pool's own scale; their
2048/top-20 is matched proportionally, ≤2048-token traces vs their 32-64k)
and writes, per rollout, in the CORRECT C-convention (G_j = -W_j(m)):

  lgc        min_j G_j            — DeepConf's online filter/weight statistic
  c_bot10    mean of the bottom-10% G_j windows (their Eq. 5, offline)
  rmin_pos, rmin_val
             running-min staircase of G_j: token position at which the
             running minimum first drops to each level. DeepConf-online
             terminates a trace at the first window with G_j < s, so the
             abort token count = first staircase position with value < s.
  conf_min_logged_check
             min_j W_j(m) recomputed (v1/logged convention) — validation
             column, must match pool `conf_window_min` (corr > 0.999).

Writes outputs/cells/<cell>/pools/conf_trace2.parquet (row-aligned to
pool.parquet via the same shard order as v1).

Run on a GPU node:  python 04_conf_trace_v2.py --cell <cell> [--limit N]
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from awv import config
from awv.generate import _build_prompts

WINDOW = 128
TOPK = 5
TEMP = 0.8
_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _staircase(g):
    """(token_pos, running_min) staircase of the C-convention window means g."""
    if len(g) == 0:
        return [0], [0.0]
    rmin = np.minimum.accumulate(g)
    drop = np.concatenate([[True], rmin[1:] < rmin[:-1] - 1e-9])
    idx = np.where(drop)[0]
    pos = (idx + WINDOW).astype(int).tolist()          # token count at that window end
    val = [round(float(v), 4) for v in rmin[idx]]
    return pos, val


def trace_stats(means_m):
    """From the m-convention window means, return the corrected-record dict.

    means_m = sliding-window means of the mean top-k LOGPROB (the logged
    convention); G = -means_m is DeepConf's group confidence."""
    if len(means_m) == 0:
        return {"lgc": 0.0, "c_bot10": 0.0, "rmin_pos": [0], "rmin_val": [0.0],
                "conf_min_logged_check": 0.0}
    g = -np.asarray(means_m, dtype=float)
    nb = max(1, int(len(g) * 0.10))
    pos, val = _staircase(g)
    return {"lgc": float(g.min()),
            "c_bot10": float(np.sort(g)[:nb].mean()),
            "rmin_pos": pos, "rmin_val": val,
            "conf_min_logged_check": float(np.asarray(means_m).min())}


@torch.no_grad()
def _forward(model, tok, ids, plen, device):
    T = max(len(x) for x in ids)
    inp = torch.full((len(ids), T), tok.pad_token_id or 0, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    out = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
    res = []
    for i, x in enumerate(ids):
        Lp, Lf = plen[i], len(x)
        if Lf - Lp < 1:
            res.append(trace_stats(np.array([]))); continue
        lg = out[i, Lp - 1:Lf - 1].float() / TEMP      # only the completion positions
        lse = torch.logsumexp(lg, dim=-1)
        top = lg.topk(TOPK, dim=-1).values
        conf = (top - lse[:, None]).mean(dim=-1).cpu().numpy()   # m_t (logprob scale)
        w = min(WINDOW, len(conf))
        c = np.concatenate([[0.0], np.cumsum(conf)])
        means = (c[w:] - c[:-w]) / w
        res.append(trace_stats(means))
    del out
    return res


def batch_conf(model, tok, prompts, texts, device, maxlen=3072):
    """Memory-safe: recursively halves the batch on CUDA OOM (down to 1, then
    truncates an over-long sequence)."""
    ids, plen = [], []
    for pr, tx in zip(prompts, texts):
        pi = tok(pr, add_special_tokens=False).input_ids
        ci = tok(tx, add_special_tokens=False).input_ids
        seq = (pi + ci)[:maxlen]
        ids.append(seq); plen.append(min(len(pi), len(seq)))
    try:
        return _forward(model, tok, ids, plen, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(ids) == 1:
            return batch_conf(model, tok, prompts, texts, device, maxlen=maxlen // 2)
        h = len(ids) // 2
        return (batch_conf(model, tok, prompts[:h], texts[:h], device, maxlen)
                + batch_conf(model, tok, prompts[h:], texts[h:], device, maxlen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard-mod", type=int, default=0,
                    help=">0: process only rows at positions ≡ shard-rem (mod shard-mod), write a part file")
    ap.add_argument("--shard-rem", type=int, default=0)
    ap.add_argument("--merge-parts", action="store_true",
                    help="CPU: merge conf_trace2_part*.parquet into conf_trace2.parquet and exit")
    args = ap.parse_args()

    cfg = config.load_config(cell=args.cell)
    cell_dir = _ROOT / "outputs" / "cells" / args.cell / "pools"
    if args.merge_parts:
        parts = sorted(glob.glob(str(cell_dir / "conf_trace2_part*.parquet")))
        out = pd.concat([pd.read_parquet(p) for p in parts]).sort_values("_pos").reset_index(drop=True)
        assert (out["_pos"].to_numpy() == np.arange(len(out))).all(), "part files do not tile the pool"
        out = out.drop(columns=["_pos"])
        corr = np.corrcoef(out["conf_min_logged_check"], out["conf_min_logged"])[0, 1]
        print(f"merged {len(parts)} parts, {len(out)} rows; validation corr={corr:.4f}", flush=True)
        out.to_parquet(cell_dir / "conf_trace2.parquet")
        print(f"wrote {cell_dir / 'conf_trace2.parquet'}", flush=True)
        if corr <= 0.95:
            raise SystemExit(f"likely row misalignment (corr={corr:.4f})")
        return
    tok = AutoTokenizer.from_pretrained(cfg.generation.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.generation.model, torch_dtype=dtype, device_map="cuda").eval()
    device = model.device

    q = pd.read_parquet(cell_dir / "queries.parquet", columns=["query_id", "question"])
    qmap = dict(zip(q["query_id"], q["question"]))
    sysp = cfg.generation.system_prompt
    shards = sorted(glob.glob(str(cell_dir / "pool_shards" / "*.parquet")))
    df = pd.concat([pd.read_parquet(s, columns=["query_id", "text", "conf_window_min"])
                    for s in shards], ignore_index=True)
    df["_pos"] = np.arange(len(df))
    if args.limit:
        df = df.iloc[:args.limit].copy()
    if args.shard_mod:
        df = df.iloc[args.shard_rem::args.shard_mod].copy()
    print(f"{args.cell}: {len(df)} rollouts"
          + (f" (residue {args.shard_rem} mod {args.shard_mod})" if args.shard_mod else "")
          + f", model {cfg.generation.model}", flush=True)

    rows = []
    for b in range(0, len(df), args.batch):
        sub = df.iloc[b:b + args.batch]
        prompts = _build_prompts(tok, [qmap[int(x)] for x in sub["query_id"]], sysp)
        rows += batch_conf(model, tok, prompts, sub["text"].tolist(), device)
        if b % (args.batch * 50) == 0:
            print(f"  {b}/{len(df)}", flush=True)

    out = pd.DataFrame({"query_id": df["query_id"].to_numpy(),
                        "lgc": [r["lgc"] for r in rows],
                        "c_bot10": [r["c_bot10"] for r in rows],
                        "rmin_pos": [r["rmin_pos"] for r in rows],
                        "rmin_val": [r["rmin_val"] for r in rows],
                        "conf_min_logged_check": [r["conf_min_logged_check"] for r in rows],
                        "conf_min_logged": df["conf_window_min"].to_numpy()})
    if args.shard_mod:
        out["_pos"] = df["_pos"].to_numpy()
    corr = np.corrcoef(out["conf_min_logged_check"], out["conf_min_logged"])[0, 1]
    mae = np.abs(out["conf_min_logged_check"] - out["conf_min_logged"]).mean()
    print(f"validation vs logged conf_window_min: corr={corr:.4f}  MAE={mae:.4f}", flush=True)
    # Write BEFORE the gate — the recompute is ~1 h of GPU; never discard it.
    # Gate = row-MISALIGNMENT detection only (misalignment gives corr ~ 0).
    # Teacher-forcing vs vLLM-generation logprobs reproduce at corr ~ 0.9999 on
    # Qwen but only ~ 0.988 on the Llama models (measured; v1 conf_trace logs
    # show the same 0.988-0.989), so 0.999 would reject healthy Llama runs.
    if args.shard_mod:
        outp = cell_dir / f"conf_trace2_part{args.shard_rem}of{args.shard_mod}.parquet"
    else:
        outp = cell_dir / ("conf_trace2.parquet" if not args.limit else "conf_trace2_sample.parquet")
    out.to_parquet(outp)
    print(f"wrote {outp}", flush=True)
    if corr <= 0.95:
        bad = outp.with_name(outp.stem + "_SUSPECT.parquet")
        outp.rename(bad)
        raise SystemExit(f"likely row misalignment (corr={corr:.4f}); kept as {bad.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute the per-token DeepConf group-confidence TRAJECTORY for each logged
rollout, so DeepConf-online's early-termination token count is exact.

The generation pass logged only the sliding-window-minimum confidence
(conf_window_min). Here we re-score each rollout's logged text with one
teacher-forcing forward pass (no generation), reproducing the exact per-token
confidence c_t = mean of the top-5 logprobs at temperature 0.8, then the
128-token sliding-window group confidence g_j, and store the running-minimum
staircase (the token position at which the running-min first drops to each
level). DeepConf terminates a trace at the first window whose group confidence
falls below its threshold s -> abort token = interpolate on this staircase.

Writes outputs/cells/<cell>/pools/conf_trace.parquet (row-aligned to pool.parquet).
Validate: conf_min_recomp should match the logged conf_window_min.

Run on a GPU node:  python score_conf_trace.py --cell <cell> [--limit N]
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
_ROOT = pathlib.Path(__file__).resolve().parents[0]


def _staircase(means):
    """(token_pos, running_min) staircase of the window-mean confidence."""
    if len(means) == 0:
        return [0], [0.0]
    rmin = np.minimum.accumulate(means)
    drop = np.concatenate([[True], rmin[1:] < rmin[:-1] - 1e-9])
    idx = np.where(drop)[0]
    pos = (idx + WINDOW).astype(int).tolist()          # token count at that window end
    val = [round(float(v), 4) for v in rmin[idx]]
    return pos, val


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
            res.append(([0], [0.0])); continue
        lg = out[i, Lp - 1:Lf - 1].float() / TEMP      # only the completion positions
        lse = torch.logsumexp(lg, dim=-1)
        top = lg.topk(TOPK, dim=-1).values
        conf = (top - lse[:, None]).mean(dim=-1).cpu().numpy()
        w = min(WINDOW, len(conf))
        c = np.concatenate([[0.0], np.cumsum(conf)])
        means = (c[w:] - c[:-w]) / w
        res.append(_staircase(means))
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
    args = ap.parse_args()

    cfg = config.load_config(cell=args.cell)
    root = pathlib.Path(cfg.paths.pool).parent if hasattr(cfg.paths, "pool") else None
    cell_dir = _ROOT / "outputs" / "cells" / args.cell / "pools"
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
    if args.limit:
        df = df.iloc[:args.limit].copy()
    print(f"{args.cell}: {len(df)} rollouts, model {cfg.generation.model}", flush=True)

    pos_all, val_all, minr = [], [], []
    for b in range(0, len(df), args.batch):
        sub = df.iloc[b:b + args.batch]
        prompts = _build_prompts(tok, [qmap[int(x)] for x in sub["query_id"]], sysp)
        res = batch_conf(model, tok, prompts, sub["text"].tolist(), device)
        for pos, val in res:
            pos_all.append(pos); val_all.append(val); minr.append(min(val))
        if b % (args.batch * 50) == 0:
            print(f"  {b}/{len(df)}", flush=True)

    out = pd.DataFrame({"query_id": df["query_id"].to_numpy(),
                        "rmin_pos": pos_all, "rmin_val": val_all,
                        "conf_min_recomp": minr,
                        "conf_min_logged": df["conf_window_min"].to_numpy()})
    corr = np.corrcoef(out["conf_min_recomp"], out["conf_min_logged"])[0, 1]
    mae = np.abs(out["conf_min_recomp"] - out["conf_min_logged"]).mean()
    print(f"validation vs logged conf_window_min: corr={corr:.4f}  MAE={mae:.4f}", flush=True)
    outp = cell_dir / ("conf_trace.parquet" if not args.limit else "conf_trace_sample.parquet")
    out.to_parquet(outp)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()

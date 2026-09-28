"""Pool generation with vLLM (GPU stage 01).

Writes one parquet shard per block of queries; rerunning skips finished shards,
so the stage is resumable (and safe under SLURM requeue). Logged per rollout,
on top of the E1 schema:

  phi_lik        exp(mean logprob of the sampled tokens)   — free from the pass
  phi_selfcert   mean top-1 probability over positions     — free
  conf_*         DeepConf-style group-confidence summaries (plan §3.3, §4):
                 per-token confidence c_i = mean of the top-k logprobs at
                 position i (higher = more confident); sliding-window means of
                 c over `deepconf_window` tokens, reduced to min / p10 / tail.
  phi_deepconf   = conf_window_min (the sliding-window-minimum group
                 confidence; the paper's bottom-group signal, sign flipped so
                 higher is better). Logged here so F3's DeepConf replay never
                 touches the GPU again.
  finish_reason  'stop' | 'length' — audit column for the token cap.

phi_conf and the PRM columns are filled by scores.py in stage 02.
"""

from __future__ import annotations

import gc
import math
import os
import pathlib

import numpy as np
import pandas as pd

from . import answers
from .utils import ensure_dir


def window_conf_stats(conf: np.ndarray, window: int) -> dict:
    """Sliding-window summaries of a per-token confidence trace.

    conf: (T,) per-token confidence (mean top-k logprob; higher = confident).
    Windows are all contiguous spans of length min(window, T). Pure function,
    unit-tested; O(T) via cumulative sums.
    """
    conf = np.asarray(conf, dtype=float)
    if conf.size == 0:
        return {"conf_mean": np.nan, "conf_window_min": np.nan,
                "conf_window_p10": np.nan, "conf_tail": np.nan}
    w = min(int(window), conf.size)
    c = np.concatenate([[0.0], np.cumsum(conf)])
    means = (c[w:] - c[:-w]) / w
    return {
        "conf_mean": float(conf.mean()),
        "conf_window_min": float(means.min()),
        "conf_window_p10": float(np.percentile(means, 10)),
        "conf_tail": float(conf[-w:].mean()),
    }


def _pick_dtype():
    """bfloat16 on Ampere+ (SM >= 80), float16 otherwise (e.g. V100/Turing,
    which have no bf16 units) — matters on shared clusters like Amarel where
    the GPU model varies by node."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return "bfloat16"
        return "float16"
    except Exception:
        return "bfloat16"


def _build_prompts(tokenizer, questions, system_prompt):
    prompts = []
    for q in questions:
        msgs = [{"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": q}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def generate_pool(cfg, queries: pd.DataFrame, m: int, out_path: pathlib.Path,
                  shard_size: int | None = None, shard_mod: int = 1,
                  shard_rem: int = 0, max_shards: int | None = None) -> pathlib.Path:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = cfg.generation
    dataset = cfg.data.dataset
    shard_size = shard_size or gen.shard_size
    out_path = pathlib.Path(out_path)
    shard_dir = ensure_dir(out_path.parent / (out_path.stem + "_shards"))

    tokenizer = AutoTokenizer.from_pretrained(gen.model)
    # max_model_len caps the KV-cache sizing: without it vLLM reserves for the
    # model's full context window (128k on Llama-3.1), which OOMs mid-size GPUs
    # AWV_GPU_MEM_UTIL: lower this on shared nodes where stray neighbor
    # processes squat GPU memory and the 0.90 default cannot be satisfied
    llm = LLM(model=gen.model, dtype=_pick_dtype(),
              gpu_memory_utilization=float(os.environ.get("AWV_GPU_MEM_UTIL", "0.90")),
              max_model_len=int(gen.max_model_len))

    n_shards = math.ceil(len(queries) / shard_size)
    written = 0
    for shard in range(n_shards):
        # shard striding lets several GPUs work one cell concurrently: worker r
        # of K takes the shards congruent to r mod K, and since each writes a
        # distinct file they never collide
        if shard % shard_mod != shard_rem:
            continue
        shard_file = shard_dir / f"shard_{shard:04d}.parquet"
        if shard_file.exists():
            print(f"[generate] shard {shard} exists, skipping")
            continue
        block = queries.iloc[shard * shard_size:(shard + 1) * shard_size]
        prompts = _build_prompts(tokenizer, block["question"].tolist(), gen.system_prompt)
        params = SamplingParams(
            n=m, temperature=gen.temperature, top_p=gen.top_p,
            max_tokens=gen.max_new_tokens, logprobs=gen.logprobs_topk,
            seed=gen.seed + shard,
        )
        outs = llm.generate(prompts, params)
        rows = []
        for (_, qrow), out in zip(block.iterrows(), outs):
            for j, comp in enumerate(out.outputs):
                text = comp.text
                # phi_lik: mean logprob of the SAMPLED tokens, indexed by token
                # id (the logprobs dict holds both the sampled and the top-k
                # entries; insertion order is not contractual). These are the
                # sampling-processed (temperature 0.8) logprobs, the score
                # available at inference time; documented in the README.
                tok_lps, top1_ps, topk_means = [], [], []
                lp_dicts = comp.logprobs or []
                for tid, d in zip(comp.token_ids, lp_dicts):
                    if not d:
                        continue
                    if tid in d:
                        tok_lps.append(d[tid].logprob)
                    lps = [x.logprob for x in d.values()]
                    top1_ps.append(math.exp(max(lps)))
                    topk_means.append(float(np.mean(lps)))
                conf = window_conf_stats(np.array(topk_means), gen.deepconf_window)
                raw = answers.extract_answer(text, dataset)
                rows.append({
                    "query_id": int(qrow["query_id"]),
                    "rollout_id": j,
                    "shard": shard,
                    "seed": gen.seed + shard,
                    "ell_tokens": len(comp.token_ids),
                    "finish_reason": str(comp.finish_reason),
                    "answer_raw": raw if raw is not None else "",
                    "answer_canonical": answers.canonicalize(raw, dataset),
                    "correct": bool(answers.grade(raw, str(qrow["gold_answer"]), dataset)),
                    "text": text,
                    "phi_lik": float(np.exp(np.mean(tok_lps))) if tok_lps else 0.0,
                    "phi_selfcert": float(np.mean(top1_ps)) if top1_ps else 0.0,
                    "conf_mean": conf["conf_mean"],
                    "conf_window_min": conf["conf_window_min"],
                    "conf_window_p10": conf["conf_window_p10"],
                    "conf_tail": conf["conf_tail"],
                    "phi_deepconf": conf["conf_window_min"],
                    "phi_conf": np.nan,
                    "phi_prm_prod": np.nan,
                    "phi_prm_last": np.nan,
                    "phi_prm_min": np.nan,
                })
        pd.DataFrame(rows).to_parquet(shard_file, index=False)
        print(f"[generate] wrote shard {shard} ({len(rows)} rollouts)", flush=True)

        # A shard's RequestOutputs carry one Logprob object per (sequence,
        # token, top-k) -- roughly 8M of them at m=64 -- and holding the last
        # shard's graph while the next one is built grew RSS by ~2 GB per
        # shard, which OOM'd the 64 GB job after ~25 shards. Drop them
        # explicitly; `max_shards` additionally lets the caller retire the
        # process periodically so nothing can accumulate across a long run.
        del outs, rows
        gc.collect()
        written += 1
        if max_shards is not None and written >= max_shards:
            print(f"[generate] hit --max-shards {max_shards}, exiting so the "
                  f"process can be restarted clean")
            break

    shards = sorted(shard_dir.glob("shard_*.parquet"))
    if len(shards) < n_shards:
        # a strided worker (or a requeued job) finished its own slice only;
        # merging now would publish a pool that silently omits queries
        print(f"[generate] {len(shards)}/{n_shards} shards present, "
              f"not merging yet")
        return out_path
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df.to_parquet(out_path, index=False)
    print(f"[generate] pool: {len(df)} rollouts, {df['query_id'].nunique()} queries -> {out_path}")
    return out_path

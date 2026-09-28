"""Extra score passes over a logged pool (GPU stage 02).

phi_conf      P(True) self-evaluation (Kadavath et al., 2022): one forward pass
              per rollout on a True/False prompt; score = p(True)/(p(True)+p(False)),
              with the leading-space token variants folded in for robustness.
phi_prm_*     Qwen2.5-Math-PRM-7B step rewards, reduced by product / last / min.

Both passes are CHECKPOINTED: scores are written per chunk to a side directory
and merged into the pool only at the end via an atomic replace, so a crash
never corrupts the paid pool parquet and reruns skip finished chunks.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .utils import append_columns, ensure_dir

CHUNK = 2048      # rollouts per checkpoint file


def _ensure_pad(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # right padding is what the PRM path's per-step indexing assumes; the
    # P(True) path overrides this to left padding locally, because it reads a
    # single last-token logit and wants that index to be position -1
    tok.padding_side = "right"
    return tok


def _chunk_dir(pool_path: pathlib.Path, name: str) -> pathlib.Path:
    return ensure_dir(pool_path.parent / f"scores_{name}")


def _assemble(pool_path, name, n_expected, columns):
    cdir = _chunk_dir(pool_path, name)
    parts = sorted(cdir.glob("chunk_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if len(df) != n_expected:
        raise RuntimeError(f"{name}: {len(df)} scored rows != {n_expected} pool rows; "
                           f"remove stale chunks in {cdir} and rerun")
    return append_columns(pool_path, df[["query_id", "rollout_id"] + columns])


def _token_variants(tok, word: str) -> list[int]:
    """Single-token ids for `word` and ` word` (chat-template drift insurance)."""
    ids = []
    for s in (word, " " + word):
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:
        raise ValueError(f"{word!r} is not a single token for this tokenizer")
    return ids


# ---------------------------------------------------------------- P(True) ---

def score_ptrue(cfg, pool_path: pathlib.Path, queries: pd.DataFrame) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sc = cfg.scores
    tok = _ensure_pad(AutoTokenizer.from_pretrained(sc.ptrue_model))
    # torch_dtype, not dtype: the `dtype` alias only exists from transformers
    # 4.56 and this env pins 4.50.3, while torch_dtype is accepted by both
    model = AutoModelForCausalLM.from_pretrained(
        sc.ptrue_model, torch_dtype=torch.bfloat16, device_map="auto").eval()

    true_ids = _token_variants(tok, "True")
    false_ids = _token_variants(tok, "False")
    qmap = dict(zip(queries["query_id"], queries["question"]))

    df = pd.read_parquet(pool_path, columns=["query_id", "rollout_id", "answer_raw"])
    cdir = _chunk_dir(pool_path, "ptrue")
    bs = sc.ptrue_batch_size
    for c0 in range(0, len(df), CHUNK):
        cfile = cdir / f"chunk_{c0:08d}.parquet"
        if cfile.exists():
            continue
        block = df.iloc[c0:c0 + CHUNK]
        prompts = []
        for _, r in block.iterrows():
            msg = sc.ptrue_template.format(
                question=qmap[r["query_id"]],
                answer=r["answer_raw"] if r["answer_raw"] else "(no final answer)")
            chat = [{"role": "user", "content": msg}]
            prompts.append(tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True))
        vals = np.empty(len(prompts))
        # P(True) needs the logits at ONE position, but the default forward
        # materialises them at every position: (batch, seq_len, 152k vocab) in
        # bf16 is ~28 GB at batch 64 x ~1.5k tokens, which is what OOM'd the
        # first MATH-train attempt. `logits_to_keep=1` computes the last
        # position only -- and that position is the last REAL token only under
        # left padding, hence the switch below.
        tok.padding_side = "left"
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                            truncation=True, max_length=3072).to(model.device)
                logits = model(**batch, logits_to_keep=1).logits
                final = logits[:, -1, :].float()
                lt = torch.logsumexp(final[:, true_ids], dim=-1)
                lf = torch.logsumexp(final[:, false_ids], dim=-1)
                vals[i:i + bs] = torch.sigmoid(lt - lf).cpu().numpy()
        out = block[["query_id", "rollout_id"]].copy()
        out["phi_conf"] = vals
        out.to_parquet(cfile, index=False)
        print(f"[ptrue] chunk {c0}/{len(df)}")
    return _assemble(pool_path, "ptrue", len(df), ["phi_conf"])


# -------------------------------------------------------------------- PRM ---

def _prm_step_rewards(logits, token_masks):
    import torch
    probs = torch.softmax(logits, dim=-1) * token_masks.unsqueeze(-1)
    rewards = []
    for sample in probs:
        pos = sample[sample.sum(dim=-1) != 0]
        rewards.append(pos[:, 1].cpu().float().numpy() if len(pos) else np.array([0.0]))
    return rewards


def _fit_steps(tok, qtext, steps, sep, max_length, sys_prompt):
    """Drop middle steps (keeping the first and final ones) until the chat fits
    max_length, so the final-step separator is never truncated away."""
    dropped = False
    while True:
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": qtext},
                {"role": "assistant", "content": sep.join(steps) + sep}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        if len(tok.encode(text)) <= max_length or len(steps) <= 2:
            return text, dropped
        steps = steps[:len(steps) // 2] + steps[len(steps) // 2 + 1:]
        dropped = True


def score_prm(cfg, pool_path: pathlib.Path, queries: pd.DataFrame) -> pd.DataFrame:
    """Qwen2.5-Math-PRM-7B per its model-card protocol: steps joined by the
    <extra_0> separator token; the classification head emits one reward per step.

    Run this as a STANDALONE process (fresh CUDA context). Loading another model
    earlier in the same process leaves GPU state that drives this 7B's bf16
    forward to NaN; eager attention plus a fail-fast NaN guard below make a
    silent recurrence impossible.
    """
    import gc

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    torch.cuda.empty_cache(); gc.collect()
    sc = cfg.scores
    tok = _ensure_pad(AutoTokenizer.from_pretrained(sc.prm_model, trust_remote_code=True))
    # The PRM's remote modeling code targets transformers 4.x; shim the two
    # 5.x removals it trips over: implicit config attribute defaults
    # (pad_token_id) and DynamicCache.from_legacy_cache. We never pass a
    # cache, so use_cache=False plus an empty-cache shim is sufficient.
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "from_legacy_cache"):
        DynamicCache.from_legacy_cache = classmethod(
            lambda cls, past_key_values=None, *a, **k: cls())
    mcfg = AutoConfig.from_pretrained(sc.prm_model, trust_remote_code=True)
    if getattr(mcfg, "pad_token_id", None) is None:
        mcfg.pad_token_id = tok.pad_token_id
    mcfg.use_cache = False
    model = AutoModel.from_pretrained(
        sc.prm_model, config=mcfg, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", trust_remote_code=True).eval()
    sep_id = tok.encode("<extra_0>")[0]
    sys_prompt = "Please reason step by step."
    max_length = 4096
    qmap = dict(zip(queries["query_id"], queries["question"]))

    df = pd.read_parquet(pool_path, columns=["query_id", "rollout_id", "text"])
    cdir = _chunk_dir(pool_path, "prm")
    bs = sc.prm_batch_size
    n_dropped = 0
    for c0 in range(0, len(df), CHUNK):
        cfile = cdir / f"chunk_{c0:08d}.parquet"
        if cfile.exists():
            continue
        block = df.iloc[c0:c0 + CHUNK].reset_index(drop=True)
        # Build the chat strings, then sort the chunk by length so each batch
        # is length-homogeneous: only the few longest rollouts ever form a long
        # batch, capping the eager attention's O(seq^2) memory and keeping the
        # short majority fast. Output is keyed by (query_id, rollout_id), so the
        # reordering is invisible to the downstream merge.
        convs = []
        for q, t in zip(block["query_id"], block["text"]):
            steps = [s for s in t.split(sc.prm_step_separator) if s.strip()] or [t]
            text, dropped = _fit_steps(tok, qmap[q], steps, "<extra_0>",
                                       max_length, sys_prompt)
            n_dropped += dropped
            convs.append(text)
        lengths = np.array([len(tok.encode(c)) for c in convs])
        order = np.argsort(lengths)
        prod = np.empty(len(block)); last = np.empty(len(block)); mn = np.empty(len(block))
        with torch.no_grad():
            for i in range(0, len(order), bs):
                sel = order[i:i + bs]
                batch = tok([convs[k] for k in sel], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(model.device)
                outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
                logits = outputs[0]
                if c0 == 0 and i == 0 and not torch.isfinite(logits).all():
                    raise RuntimeError(
                        "PRM forward produced non-finite logits on the first "
                        "batch; run stage 02 PRM as a standalone process "
                        "(--skip-ptrue) so no prior model state corrupts it.")
                masks = (batch["input_ids"] == sep_id)
                rewards = _prm_step_rewards(logits, masks)
                for j, k in enumerate(sel):
                    r = np.clip(rewards[j], 1e-6, 1.0)
                    prod[k] = float(np.exp(np.log(r).sum()))
                    last[k] = float(r[-1])
                    mn[k] = float(r.min())
                del batch, outputs, logits
        if not np.isfinite(prod).all():
            raise RuntimeError(f"PRM chunk {c0} has non-finite rewards; aborting "
                               f"before writing garbage (do not retry blind).")
        out = block[["query_id", "rollout_id"]].copy()
        out["phi_prm_prod"], out["phi_prm_last"], out["phi_prm_min"] = prod, last, mn
        out.to_parquet(cfile, index=False)
        print(f"[prm] chunk {c0}/{len(df)} (step-dropped so far: {n_dropped})")
    print(f"[prm] rollouts with dropped middle steps: {n_dropped}/{len(df)}")
    return _assemble(pool_path, "prm", len(df),
                     ["phi_prm_prod", "phi_prm_last", "phi_prm_min"])

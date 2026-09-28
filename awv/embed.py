"""Query embeddings h(q): last-layer hidden state at the final prompt token
of the generator (GPU stage 03). Used only by the online kNN retrieval."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

import os

from .utils import ensure_dir


def embed_queries(cfg, queries: pd.DataFrame, out_path: pathlib.Path) -> pathlib.Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emb = cfg.embedding
    tok = AutoTokenizer.from_pretrained(emb.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # last-token indexing assumes right padding
    # torch_dtype, not dtype: the `dtype` alias only exists from transformers
    # 4.56 and this env pins 4.50.3, while torch_dtype is accepted by both
    model = AutoModelForCausalLM.from_pretrained(
        emb.model, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True).eval()

    prompts = []
    for q in queries["question"]:
        msgs = [{"role": "system", "content": cfg.generation.system_prompt.strip()},
                {"role": "user", "content": q}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    vecs = []
    # AWV_EMBED_BS: shrink on small-memory GPUs (the CausalLM forward also
    # materializes lm_head logits, which OOMs 24GB cards at the default batch)
    bs = int(os.environ.get("AWV_EMBED_BS", emb.batch_size))
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            batch = tok(prompts[i:i + bs], return_tensors="pt",
                        padding=True, truncation=True, max_length=2048).to(model.device)
            hidden = model(**batch).hidden_states[-1]
            lastpos = batch["attention_mask"].sum(1) - 1
            h = hidden[torch.arange(hidden.shape[0]), lastpos]
            vecs.append(h.float().cpu().numpy())

    H = np.concatenate(vecs, axis=0)
    out_path = pathlib.Path(out_path)
    ensure_dir(out_path.parent)
    np.savez(out_path, query_id=queries["query_id"].to_numpy(), H=H)
    print(f"[embed] {H.shape} -> {out_path}")
    return out_path

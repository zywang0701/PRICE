# Data license

The rollout pools, scores and embeddings in the companion dataset
`zach-wang/PRICE-rollouts` (Hugging Face) are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

They were generated from public models and public benchmarks, whose own terms
continue to apply to the derived content:

- Rollouts of `Qwen/Qwen2.5-1.5B-Instruct` — Apache License 2.0.
- Rollouts of `unsloth/Llama-3.2-3B-Instruct` (a re-upload of Meta's
  Llama-3.2-3B-Instruct) — subject to the
  [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).
  Built with Llama.
- Questions and reference answers come from the MATH dataset
  (Hendrycks et al., 2021, MIT License) and its MATH-500 split
  (Lightman et al., 2023).

The pool files also carry columns from scores that the paper does not use
(`phi_prm_*` from a process reward model and the DeepConf-family columns);
they are kept so that the released files are byte-identical to the ones whose
SHA-256 hashes the experiment manifests record.

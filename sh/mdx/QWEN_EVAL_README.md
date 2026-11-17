# Qwen3-8B / Qwen3.5-9B evaluation on MDX

`sh/mdx/eval_qwen3.sh` submits **two independent model groups**, with editable
`QWEN3_EXP_NAMES` and `QWEN35_EXP_NAMES` arrays. Model names are directory names
under `MODEL_ROOT` from your private `.env`. Set an array to `()` to skip it.

```bash
bash sh/mdx/eval_qwen3.sh
```

All four families (Qwen, LLM-jp, Phi, DSDT) use the same uv environment and
Slurm accounting. The Qwen entrypoint is `qwen_eval.py`; it shares checkpoint,
scoring, and LiveCodeBench export implementation with `llmjp_eval.py` but uses
Qwen's own chat template, `enable_thinking=True`, and `<think>...</think>` parser.
Unlike the LLM-jp profile it **does not** use Harmony tokens or `reasoning_effort`.
Answers and checkpoint files are written directly to
`$EVAL_OUTPUTS/<model>_16/<dataset>/`, not to a hash subdirectory.

## Defaults / caveats

- Qwen3-8B: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`,
  32,768 output tokens, 40,960 model context. Official Qwen3 documentation
  advises YaRN for evaluation budgets exceeding the available context.
- Qwen3.5-9B: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`,
  `presence_penalty=1.5`, 38,912 output tokens, 65,536 model context.
  For precise coding tasks the official card suggests temperature 0.6 and
  presence_penalty 0, so the bundled **single-pass** suite uses the general
  thinking protocol instead of claiming to reproduce that specialized setting.
- `N=16` across the nine standard datasets. Existing evaluation output paths
  and `analysis_results/run.sh` can consume the Qwen generation metrics.
  The code score still requires the separate LiveCodeBench CPU evaluator.
- Qwen3.5 uses a different architecture. vLLM's official Qwen3.5-9B recipe
  lists vLLM >= 0.17.0, so the project's pinned vLLM 0.30.0 meets the stated
  minimum. The project's existing version is deliberately **not changed** by
  this patch. A one-question GPU smoke test is still needed for the exact
  CUDA/model combination; model support cannot be proved by static tests.
  Do not blindly upgrade the shared uv environment and break LLM-jp.
- Model checkpoints must match `Qwen3-8B` and `Qwen3.5-9B` as written in your
  `MODEL_ROOT`, or edit the experiment arrays.

## Datasets

The default suite is:

- Math: math500, aime24, minerva_math, olympiadbench, amc23, aime26
- MC: gpqa_diamond, mmlu_stem
- Code: livecodebench_v6

`data_loader.py` also has loader branches for `math`, `gsm8k`, `svamp`,
`asdiv`, `mawps`, `mmlu_pro` and `carp_en`; several depend on already-prepared
local files or external datasets. The default reports only summarize the nine
listed above: to add a new dataset to the report, update its categories as well
as its loader/parser/benchmark pinning. A dataset spelled `prof`, `proof` or
`gsmturb` is **not implemented**; don't insert those strings into `DATA_NAMES`
without identifying an upstream dataset version and an appropriate scorer.

Official Qwen references:
- https://huggingface.co/Qwen/Qwen3-8B
- https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://recipes.vllm.ai/Qwen/Qwen3.5-9B

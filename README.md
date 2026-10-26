# eval_math_lcb_direct — unified uv runtime on MDX

One `pyproject.toml`, one Python 3.13.5 uv environment and one vLLM 0.30.0
installation for **all** three evaluation profiles. No Conda evaluator and no
second environment/lock file. The original Phi/DSDT evaluation pipeline
(`math_eval.py`, Qwen/Phi prompts, generation options and scoring) is preserved.
The LLM-jp-specific `llmjp_eval.py` uses its model Harmony template and parses
only complete `final` channel responses for grading.

| Entry point | Evaluator | Model-specific protocol |
| --- | --- | --- |
| `sh/mdx/eval_llmjp.sh` | `llmjp_eval.py` | Harmony, token-ID parsing |
| `sh/mdx/eval_phi.sh` | `math_eval.py` | Original `phi4-mini-reasoning` |
| `sh/mdx/eval.sh` | `math_eval.py` | Original `qwen25-math-cot` |

## Installation using your EXISTING `.env`

The `UV_PROJECT_ENVIRONMENT` in your existing `.env` is the **only**
interpreter location used for LLM-jp, Phi, DSDT and Ray dispatchers. The
tracked `.env.example` documents all supported settings. To remove obsolete
Conda/Ray interpreter variables from an **existing, gitignored** `.env` without
changing its data/model/cache paths, run:

```bash
python3 sh/mdx/migrate_env.py .env
```

This updates `.env` locally, backs up its prior contents to an owner-only
`.env.bak` (or numbered backup), renames `CUDA_MODULE_LLMJP` to the shared
`EVAL_CUDA_MODULE`, and removes `LEGACY_EVAL_PYTHON`, `CUDA_MODULE_LEGACY`,
`RAY_PYTHON` and `RAY_CLI`. The old CUDA name remains accepted as a fallback
until migration. Both `.env` and its backup are ignored by Git. Do not remove
the original Conda environment until migrated GPU evaluations have passed.

```bash
cd /path/to/eval_math_lcb_direct
set -a; source .env; set +a
uv python install 3.13.5
uv venv --python 3.13.5 "${UV_PROJECT_ENVIRONMENT:-.venv}"
uv sync
bash sh/mdx/check_env.sh
```

`uv sync` generates `uv.lock` if absent. Commit **uv.lock** after resolution;
subsequent installations should use `uv sync --locked`. The ZIP did not ship a
pre-resolved lock and this patch does not invent one. All Slurm workers use the
absolute `${UV_PROJECT_ENVIRONMENT}/bin/python` (or checkout `/.venv/bin/python`).

CUDA module: the shared vLLM 0.30.0 environment uses CUDA 13.0, regardless
of model family. On MDX load `cuda/13.0/13.0.3` (or the existing
`EVAL_CUDA_MODULE` in `.env`). Using the old CUDA 12.8 module for Phi/DSDT
would be inconsistent with the new vLLM wheel. MDX GPU jobs must remain within
16 CPU cores and 48 GiB memory per GPU.

## Submission

```bash
bash sh/mdx/eval_llmjp.sh   # LLM-jp, dedicated Harmony scorer
bash sh/mdx/eval_phi.sh     # Phi original evaluator, shared uv interpreter
bash sh/mdx/eval.sh         # DSDT original evaluator, shared uv interpreter
```

Model lists and batch/generation settings remain in the corresponding scripts.
All three scripts now use one shared Python executable, including after sbatch
when worker nodes re-load `.env`. Old non-MDX NICT Ray dispatchers are also
wired to the same uv interpreter and Ray executable; however the **remote Ray
cluster** itself must be started with the matching Python minor version and
Ray version (`2.56.1`). Historical PBS scripts under `sh/nict` and `sh/eval.sh` are also switched to
`${UV_PROJECT_ENVIRONMENT}/bin/python` and do not activate Conda. PBS machines
need a CUDA 13-capable driver/module; when the module has a different name,
set `NICT_CUDA_MODULE` in `.env`. They are not MDX Slurm submitters.

## Validation and comparability

```bash
bash sh/mdx/check_env.sh
uv run --no-sync python -m unittest discover -s tests -v
```

Run a small GPU smoke evaluation *for each model family* before full datasets.
This patch verifies interpreter selection and imports statically but **cannot
prove** that vLLM 0.30.0 loads all checkpoint architectures or that GPU outputs
match vLLM 0.8.5. A shared runtime preserves script and evaluation flow, not
numerical identity across inference engines. Re-evaluate baselines when
comparing model performance under the migrated engine.

LiveCodeBench custom outputs remain separately scored with the official
sandbox runner. Do not execute model-generated code directly in the inference
job. GPQA material and model outputs belong on appropriately permissioned
shared storage.

## Sources

- LLM-jp vLLM cookbook: https://github.com/llm-jp/llm-jp-4-cookbook/tree/main/llmjp4_vllm
- vLLM 0.30.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- uv projects: https://docs.astral.sh/uv/concepts/projects/

The old root-level `environment.yml` and `requirements.txt` were removed to
avoid accidentally reconstructing the obsolete vLLM 0.8.5 runtime. The
`latex2sympy/` vendored package retains its own upstream reference files.

### Mathematical LaTeX parser compatibility

The repository includes a local `latex2sympy2` 1.9.0 package under
`latex2sympy/`. Its generated parser and `setup.py` both require
`antlr4-python3-runtime==4.11.1`. The top-level `pyproject.toml` resolves
this local source with `[tool.uv.sources]` instead of installing PyPI
`latex2sympy2==1.9.1`, which pins incompatible ANTLR 4.7.2. Keep ANTLR
4.11.1: `sympy.parsing.latex.parse_latex` also expects this series.
The grading code in `grader.py` and `parser.py` is unchanged.

After applying the fix to the existing unified-uv checkout:

```bash
set -a; source .env; set +a
uv lock
uv sync
uv run --no-sync python -m unittest discover -s tests -v
```

Commit the generated `uv.lock` so subsequent jobs use the same parser and
dependency versions. A parser change can alter math-equivalence judgments,
so do not silently replace the vendored parser with a different release.

### Mathematical parser regression test (structural versus algebraic equality)

The bundled `latex2sympy2` may produce unevaluated SymPy expressions that
print identically to simplified results, but compare unequal with `==`.
The CPU regression test now uses `simplify(actual - expected) == 0` for
scalar formulas, while still comparing both the bundled and SymPy parsers.
This change affects tests only: `grader.py` and `parser.py` are unchanged.

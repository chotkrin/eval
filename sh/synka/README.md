# SYNKA independent evaluation submitters

Use on `cpusv06` (not `cpusv01`). These scripts do **not** call `sh/mdx/` for Slurm submission.
They run the existing `llmjp_eval.py` (Harmony), `math_eval.py` (Phi/DSDT),
with the SAME uv Python and the SAME cached dataset/output layouts as MDX.

## Setup

The patch creates an ignored local `.env_synka` template and a tracked
`.env_synka.example`. Edit **three** values in `.env_synka`:
`VERL_HOME`, `SYNKA_PROJECT_ID`, and `SYNKA_NODELIST`.
`VERL_HOME` is the root for `data/eval`, `data/raw`, `models`, `eval_outputs`,
`envs`, and `cache`. The project checker refuses an unset account/node list.
Do not put a real ID, token, or private filesystem path in Git.

Use `sinfo -N -p YOUR_PROJECT -o '%N %G %T'` to find your allocated nodes.
`SYNKA_NODELIST` must contain **exactly 8 distinct nodes within your partition**,
e.g. Slurm bracket syntax with your real IDs, not the example from the manual.
`--nodes=1 --nodelist=$SYNKA_NODELIST` strictly confines each job to the list.
`--gres=gpu:H200:8` for LLM-jp/Phi, `gpu:H200:4` for DSDT.

```bash
cd /path/to/eval_math_lcb_direct
set -a; source .env_synka; set +a
mkdir -p "$VERL_HOME" "$DATA_DIR" "$MODEL_ROOT" "$EVAL_OUTPUTS"          "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
uv python install 3.13.5
uv sync --locked
bash sh/synka/download_extra_datasets.sh   # normalize optional benchmarks first
bash sh/synka/eval_llmjp.sh
bash sh/synka/eval_phi.sh
bash sh/synka/eval.sh                   # DSDT
```

Each `eval_*.sh` contains its own editable `EXP_NAMES` and `DATA_NAMES`.
The default still runs the original nine benchmarks. Uncomment the single
`# DATA_NAMES="${DATA_NAMES},gsm_plus,..."` line in any model script
to add seven answer-checkable robustness subsets after downloading them.
Run ProcessBench separately and report its F1, not as mathematics accuracy.
ProofBench has generation-only outputs; OMEGA requires validated generated
exports, so both are listed as opt-in options rather than misleading accuracies.

The SYNKA guide explicitly requires both `--partition` and `--account`, GPU GRES
with `H200`, submission from `cpusv06`, and proxy settings for internet access.
`SYNKA_PARTITION=ungrouped` uses a time limit not exceeding 24 hours (set
`SYNKA_WALLTIME=12:00:00`); dedicated partitions default to `--time=0` and
are still subject to cluster policy.

A lock file under `<exp>_<n>/.evaluation.lock` prevents concurrent writes.
Slurm copies batch scripts to a spool directory; the submitter exports the
absolute `PROJECT_ROOT` and `EVAL_ENV_FILE` so the worker resolves the checkout
correctly. Logs and `jobs.tsv` are stored under
`$EVAL_OUTPUTS/synka_submissions/<timestamp>/`.

**CUDA version caution:** the guide demonstrates `cuda/12.8`, while the uv
project pins newer vLLM/PyTorch wheels. Check `nvidia-smi`, Torch CUDA and
vLLM imports on an allocated H200 before starting full evaluation. A CUDA
module name alone does not prove wheel/driver compatibility.

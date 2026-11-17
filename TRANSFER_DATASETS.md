# Paired mathematical transfer evaluation — project patch

## What was already implemented

Original MATH-500 / AIME / AMC and other existing datasets are already listed in `sh/mdx/eval.sh` and loaded through the original evaluator using cached `${DATA_DIR}/<dataset>/test.jsonl` files. The patch **does not rerun, alter or claim to replace** their results. This repository already stores per-sample `code`, `pred`, `score`, and `completion_tokens` in evaluator JSONL when `--save_outputs` is passed.

## New dataset identifiers (all use `SPLIT="test"`)

| Name | Source | Notes |
|---|---|---|
| `gsm_plus`, `gsm_plus_seed` | `qintongli/GSM-Plus` (HF) | Paired perturbations / distinct seed original. Seed deduplicated by source question hash. |
| `gsm_symbolic_main`, `gsm_symbolic_p1`, `gsm_symbolic_p2` | `apple/GSM-Symbolic` (HF) | Same original GSM8K question group (`original_id`); `p1/p2` change schema difficulty. |
| `math_perturb_simple`, `math_perturb_hard` | `Kaffaljidhmah2/MATH-Perturb` GitHub | Pair by `problem_id`. **Evaluation only, not for training.** |
| `omega_exploratory`, `omega_compositional`, `omega_transformative` | `sunblaze-ucb/omega` GitHub | Upstream is a *generator*: run their scripts, export validated JSONL per axis. Never silently substitutes synthetic labels. |
| `processbench_gsm8k` | `Qwen/ProcessBench` (HF) | Explicit process-error identification task with zero-based first-error index or `-1`. Its accuracy measures **error-detection**, not mathematical answer solving. |
| `proofbench` | `wenjiema02/ProofBench` (HF) | **Generation only**, metric `acc=null` and `proof_grading_required`; at most 145 distinct problems deduplicated by `problem_id` from upstream `train` (435 graded model solutions); conflicting statements use strict-majority selection or are excluded with an audit in `proofbench/proofbench_conflicts.json`. Upstream does not have a `test` split; local evaluation file is still `proofbench/test.jsonl`. Expert rubric scoring is separate. |
| `obligation_transfer` | Local output from `obligation-lab` zip | Interface already enabled; not invoked in shell until uncommented. |

Each normalized row retains `family_id`, `variant`, `transfer_type`, `source_dataset`, `source_id`; local obligation rows also contain `obligation_ids`, `obligation_types`, `schema_id`. Inference outputs retain these keys for later analysis. There is no hidden change of prompt type or answer grader.

## Locations (no hardcoded user directories)

On MDX the submitter sources `<checkout>/.env`, defines `DATA_DIR=${DATA_HOME}/data/eval`, and sets `HF_HOME=${DATA_HOME}/cache/huggingface`. `transfer_prepare.py` uses `DATA_DIR` (fallback `./data`). Upstream GitHub raw downloads go in `TRANSFER_RAW_DIR` if provided, otherwise `${DATA_HOME}/data/raw` (because `DATA_DIR` is `${DATA_HOME}/data/eval`). Set `TRANSFER_RAW_DIR` to an absolute path to override.

```bash
cd /path/to/eval_math_lcb_direct
set -a; source .env; set +a
export HF_HOME="${HF_HOME:-${DATA_HOME}/cache/huggingface}"
# The unified evaluator Python must already exist from uv sync.
"${UV_PROJECT_ENVIRONMENT}/bin/python" transfer_prepare.py \
  --data-dir "${DATA_DIR}" \
  --datasets gsm_plus,gsm_plus_seed,gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2,processbench_gsm8k,proofbench
```

To download the two GitHub repos, on an internet-connected login node (uses public GitHub, not a Hugging Face loader):

```bash
mkdir -p "${DATA_HOME}/data/raw"
git clone --depth 1 https://github.com/Kaffaljidhmah2/MATH-Perturb.git "${DATA_HOME}/data/raw/MATH-Perturb"
git clone --depth 1 https://github.com/sunblaze-ucb/omega.git "${DATA_HOME}/data/raw/omega"
"${UV_PROJECT_ENVIRONMENT}/bin/python" transfer_prepare.py --data-dir "${DATA_DIR}" \
  --datasets math_perturb_simple,math_perturb_hard
```

**OMEGA**: the upstream repo generates problems from difficulty/type parameters. Its output format can depend on which generator you run; run upstream generator according to README, validate the generated answer with its original solver, and export one JSONL per axis into `${DATA_HOME}/data/raw/omega/exports/omega_<axis>.jsonl` (`axis`: `exploratory`, `compositional`, `transformative`). Each row must contain `question` (or `problem`), a solver-verified `answer` (or `final_answer`), and `family_id` or `template_id` or `problem_type`. Can also include `variant`, `difficulty`, `id`, `solution`. Adapter **refuses** incomplete records, and does not pretend an OMEGA axis can be inferred from a file name without you supplying a validated axis export. Then run `transfer_prepare.py --datasets omega_exploratory,omega_compositional,omega_transformative`. This is an import interface, not a replacement implementation of all OMEGA generators.

## Submit LLM-jp, Phi or DSDT on MDX or SYNKA

Edit the active `DATA_NAMES` in `sh/mdx/eval_llmjp.sh`, `sh/mdx/eval_phi.sh` or `sh/mdx/eval.sh` (DSDT). Their SYNKA counterparts live under `sh/synka/`. The original nine benchmarks remain the default; uncomment the `DATA_NAMES="${DATA_NAMES},..."` line to append seven answer-checkable robustness datasets, or replace `DATA_NAMES` for a dedicated run, e.g.:

```bash
DATA_NAMES="gsm_plus,gsm_plus_seed,gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2,math_perturb_simple,math_perturb_hard"
# obligation_transfer can be added later:
# DATA_NAMES="obligation_transfer"
```

Submit: `bash sh/mdx/eval.sh`, `bash sh/mdx/eval_phi.sh`, or `bash sh/mdx/eval_llmjp.sh` (SYNKA: matching `sh/synka/` entrypoints). Same selection may be made in the respective `sh/nict/*.sh` files. `mdx_prepare_shared_caches` now prebuilds all selected datasets before Slurm job submission; the source GitHub repos (when needed) must be downloaded first. For Ray/NICT, pre-run `transfer_prepare.py` on the driver before launching jobs; this prevents multiple workers downloading the same datasets concurrently.

**Important**: identical `DATA_NAMES` does not imply original and transfer datasets have identical scoring semantics. Do not mix ProcessBench (error-identification) or ProofBench (pending) into mathematical answer accuracy averages. Run them separately.

## QA & caveats

* Upstream G-Symbolic answers are extracted from the final `####` line; reject malformed gold rather than guessing the last numeric token.
* The original numeric `math_equal` scorer remains in use for numeric transfer tasks. Symbolic formulas and multi-part outputs may need manual audit or the MATH-Perturb official checker before publication.
* ProofBench results save free-form model traces with no fraudulent integer accuracy.
* For scientific comparisons, use question-family splits and fixed caches; do **not** resample different rows independently for different checkpoints.
* Apple GSM-Symbolic uses a noncommercial, no-derivatives license. Check upstream licensing before redistribution. The patch does not bundle third-party raw datasets.
* MATH-Perturb requests test-only use. Never use it to train the compression model.

## MDX download command and evaluation entrypoint (2026-10)

A convenient downloader now exists; edit the `DATASETS=(...)` array in
`sh/mdx/download_extra_datasets.sh` and run on an internet-connected login node:

```bash
cd /home/zelin/code/eval_math_lcb_direct
bash sh/mdx/download_extra_datasets.sh
```

It sources `.env` but does not commit, print or copy `.env`/credentials. All
normalized rows go to `${DATA_DIR}/<dataset>/test.jsonl`, **the identical
location convention as MATH500 and other existing benchmarks**. GitHub sources
are cloned under `${TRANSFER_RAW_DIR:-$(dirname "${DATA_DIR}")/raw}` and their
commit hashes are printed. Existing normalized data are not overwritten.

`sh/mdx/eval_transfer.sh` is a standalone editable submission script for LLM-jp.
It does not alter the default nine-dataset evaluation or its aggregates. Its
`DATA_NAMES` list may be changed after downloading; `EXP_NAMES` is a normal
editable Bash list.

Additional official ProcessBench splits now supported:
`processbench_math`, `processbench_olympiadbench`, in addition to
`processbench_gsm8k`. The `label` equals the *zero-based index of the first
incorrect paragraph*, or `-1` if all paragraphs are correct. The evaluator
reports `processbench_error_acc`, `processbench_correct_acc` and
`processbench_f1` for candidate 0, the official balanced F1 formulation. The
ordinary pass@k columns are a different success-at-k statistic and must **not**
be called ProcessBench F1. Use the official ProcessBench's own evaluation
command when claiming reproduction of their published numbers (their prompt
and sampling protocol are not automatically identical to this project's).

**ProofBench name collision:** Here `proofbench` refers to the
`wenjiema02/ProofBench` 145-problem natural-language proof dataset with
expert-graded generated proofs, **not** Vals AI `ProofBench` (Lean 4 and tool
use, 100 public/100 private statements). Our `proofbench` saves model proofs
and sets accuracy/pass@k to `null` with `proof_grading_required`. Do not
include it in answer-accuracy aggregates without implementing a validated
proof grader. The Vals private set cannot be downloaded in full from its public
GitHub repository, and merely downloading Lean examples does not reproduce
that benchmark.

**OMEGA:** The project `https://github.com/sunblaze-ucb/omega` contains
problem-generating scripts, not an unambiguous static `test.jsonl`. The
cloning command downloads the source; for evaluation run its official
generators with fixed parameters/seed, verify answers using the source
verifiers, and write `omega_exploratory.jsonl`, `omega_compositional.jsonl`,
`omega_transformative.jsonl` under `${TRANSFER_RAW_DIR}/omega/exports/`.
Each record must contain a question, a verified answer and a family/template
identifier. The loader refuses missing exports rather than inventing one.

Dataset research restrictions apply: MATH-Perturb is test-only, GSM-Plus
publisher forbids training on the test set, and GSM-Symbolic's license is
CC BY-NC-ND 4.0. Keep these out of RL training corpora.

## 2026-11: mathematical-obligation-adjacent datasets

These are **not** direct mathematical-obligation completion metrics.

* `proofflowbench_graph`: ICLR 2026 ProofFlowBench, 184 manually annotated
  proof dependency DAGs, HF `rafaelcabral96/ProofFlowBench`. Input is the theorem,
  reference proof and self-contained annotated statements **without edges**;
  output is `{"edges": [["prerequisite", "dependent"], ...]}`. Report exact-match
  and direct-edge precision/recall/F1. This is a *new graph reconstruction proxy*
  and **not** the paper's official ProofScore nor proof obligation coverage.
* `proofflowbench_proof`: Same 184 theorems, free-form proof generation; grading
  is **pending** (`acc=null`), reference proof/graph saved for independent audit.
* `prmbench_error_steps`: ACL 2025 PRMBench, HF `hitsmy/PRMBench_Preview`
  (published as the single `train` split, 6,216 examples). Input is a labeled
  step sequence, output JSON `{"error_steps": [1,3]}` with **1-based** indices.
  Report exact-set match and step precision/recall/F1; this is **NOT** the
  official PRMBench scoring protocol (which evaluates process reward models).
* `processbench_omnimath`: Fourth official ACL 2025 ProcessBench split.
  Uses the same zero-based *first-error* task and balanced F1 as other splits.

No upstream data are repackaged in this patch. The download shell stores
normalized rows in `${DATA_DIR}/<name>/test.jsonl` as before. `test.jsonl`
is a **local evaluation file name**: ProofFlowBench and PRMBench originally
publish only `train` splits; do not call them official held-out test sets.
ProofFlowBench downloads the original `proofflowbench.jsonl` using the pinned
HF dataset revision `5141fd0`, bypassing incompatible `datasets.Json` metadata.
Both structured tasks use **raw final model output**, not the numeric `\boxed{}`
extractor. Keep them separate from answer `pass@1` headlines.

For SLURM jobs, download first on a login node:

```bash
bash sh/mdx/download_obligation_datasets.sh
# Edit DATA_NAMES in sh/mdx/eval.sh (or eval_phi.sh/eval_llmjp.sh)
# to: proofflowbench_graph,proofflowbench_proof,prmbench_error_steps,processbench_omnimath
bash sh/mdx/eval.sh
```

Run offline tests: `python -m unittest discover -s tests -p test_obligation_adjacent.py -v`
(or `python -m unittest tests.test_obligation_adjacent -v`).

## ProofFlowBench upstream annotation audit (2026-11)

Some records in the pinned upstream ProofFlowBench source violate its own
README's guarantee that dependency IDs belong to the same DAG and that the
DAG is acyclic. They may contain missing node references or self-links. The
loader **does not alter the gold graph** and quarantines invalid *entire*
records from `proofflowbench_graph/test.jsonl`. Inspect
`proofflowbench_graph/proofflowbench_annotation_audit.json` for source count,
valid count, and the exact origin/id/reason of each excluded record. Report
the accepted sample count with experimental results; **do not call the
reduced set the full 184-question graph benchmark**.

`proofflowbench_proof` keeps all source theorems and original proof text,
because that task does not require a trusted dependency DAG. Its
`gold_proof_graph` is preserved **unvalidated** for provenance only, not
accepted as validated proof-quality ground truth. Existing generation-only
status and `acc=null` remain unchanged.

If a cached `proofflowbench_graph/test.jsonl` exists from another version,
back it up and remove that file before rerunning `transfer_prepare.py` to
generate a new audit. Never mix scores from different filtered subsets.


## PRMBench source annotation audit (2026-11)

The `hitsmy/PRMBench_Preview` dataset card documents 1-based step numbers,
while some upstream `error_steps` refer to indexes greater than the number
of strings in `modified_process`. Such rows cannot be scored reliably from
this representation. `prmbench_error_steps` now excludes **whole invalid
records** and writes `prmbench_error_steps/prmbench_annotation_audit.json`
with source IDs, original row indexes, original annotations, counts and
rejection reasons. Do not relabel or clamp errors based on their apparent
nearby position. Report actual retained counts and state clearly that this
is a filtered, generative adaptation of PRMBench, **not an official PRM
benchmark reproduction**. Labels surviving the range check have *not*
received human semantic re-verification.

If `prmbench_error_steps/test.jsonl` already exists, back it up and delete
it before regenerating; preparation does not overwrite cached datasets.

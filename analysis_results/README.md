# MDX evaluation result analysis (read-only)

This analysis uses **saved** generations/metrics; it never loads model weights, re-runs
inference, executes code, or invents scores. The patch is applied to the ORIGINAL
uploaded `eval.tar.gz.2` (no previous analysis patch required).

## Configuration

Edit `analysis_results/run.sh`, not `sh/mdx/eval.sh`:

```bash
EXP_NAMES=(
  "DAPO-DSDT-1.5B-0-3072"
  "DAPO-DSDT-1.5B-LASER-D-LB4096-a0.5"
  "DAPO-DSDT-1.5B-LEASH-Lt4096-llr0.005"
  "DeepSeek-R1-Distill-Qwen-1.5B"
)
DATASETS="gsm_plus,gsm_plus_seed,gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2,math_perturb_simple,math_perturb_hard"
N_SAMPLING=16
PASS_K=(1 5 10)
```

The seven names above are **all uncommented datasets** in the uploaded
`sh/mdx/eval.sh`; the grouping into GSMPlus / GSMSymbolic / MATHPerturb does
not mean that only three datasets are analyzed. Change `DATASETS` to select
other completed datasets; only those datasets are read. Override for one run:

```bash
ANALYSIS_DATASETS='math500,processbench_gsm8k,proofbench' bash analysis_results/run.sh
```

Run normally:

```bash
bash analysis_results/run.sh
```

Results go by default to `analysis_results/reports_selected/` (configurable via
`ANALYSIS_REPORT_DIR`) so existing historical `reports/` are not overwritten.

## Supported datasets and metric meanings

- **Math answers**: `math500,aime24,minerva_math,olympiadbench,amc23,aime26`
  and direct loader datasets `math,gsm8k,svamp,asdiv,mawps,carp_en`.
- **MC**: `gpqa_diamond,mmlu_stem,mmlu_pro`.
- **Code**: `livecodebench_v6` and other explicitly versioned `livecodebench_vN`;
  these need official LiveCodeBench scores, not generated-code guessing.
- **GSMPlus**: `gsm_plus,gsm_plus_seed`.
- **GSMSymbolic**: `gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2`.
- **MATHPerturb**: `math_perturb_simple,math_perturb_hard`.
- **OMEGA**: `omega_exploratory,omega_compositional,omega_transformative`.
- **ObligationTransfer**: `obligation_transfer`.
- **ProcessBench**: `processbench_gsm8k,processbench_math,processbench_olympiadbench`.
- **ProofBench**: `proofbench`.

**Important: incompatible scores are NEVER averaged together.**

- Answer-type datasets: `pass@1`, `pass@5`, `pass@10` (or the values requested
  in `PASS_K`), loaded from existing scoring metrics. Each is a 0–100 score.
- ProcessBench: `processbench_f1`, `processbench_error_acc`, and
  `processbench_correct_acc`, all 0–100, from the existing evaluator's
  **candidate-0** metrics. Separate CSVs are written, plus baseline deltas.
  ProcessBench is not interpreted as mathematical answer accuracy.
- ProofBench: the current evaluator writes `proof_grading_required`, not a
  mathematical proof score. `proofbench_status.csv` shows generation/status;
  no pass@k/F1 is invented. External proof grading must be independently run.
- All selected datasets: mean completion tokens and truncation rate when
  faithfully available. Historical missing finish reasons remain `NA`.

`AllDatasets` and `Overall` in a pass@k table aggregate **only selected
answer-scored datasets** (not ProcessBench/ProofBench). For ProcessBench F1
(and subgroup accuracy), only selected ProcessBench datasets enter the mean.
For length/truncation, all selected datasets are eligible. `NA` means
unavailable or inapplicable; no score is silently treated as zero.

## Output files

- `report.md`: all selected datasets and compatible group summaries.
- `pass_at_*.csv`: answer-task pass@k and model-minus-baseline deltas.
- `processbench_f1.csv`, `processbench_error_acc.csv`,
  `processbench_correct_acc.csv`: created only when ProcessBench selected;
  corresponding `delta_*` files show differences from baseline.
- `proofbench_status.csv`: created when ProofBench selected.
- `mean_completion_tokens.csv`, `truncation_rate_pct.csv`, and the
  baseline delta and length-ratio tables.
- `coverage.csv`: per-experiment/dataset metric kind, score status,
  missing values, selected output source, generations, length and truncation
  status. Inspect this before interpreting partial results.

Missing required metrics/generations produce `INCOMPLETE` and exit code 1
*after saving the partial report*. ProofBench with the expected
`NEEDS_GRADING` status is not a fabricated zero score and need not trigger
failure when its generations exist. `--allow-unavailable-truncation` allows
historical truncation metadata to remain `NA` without failing.

Avoid claiming that macro averages across heterogeneous datasets constitute
an official benchmark score or a causal measure of migration/obligation loss.

---

## Historical hashed result migration

New LLM-jp outputs live directly in `<experiment>_<N>/<dataset>/` (for example, `math500/generations.jsonl` and `math500/metrics.json`). No `llmjp4/` subdirectory or SHA fingerprint is used. Both previously migrated `llmjp4/` directories and historical `llmjp4_<hash>` / `llm-jp4-xxx` directories are recognized for migration. When multiple historical runs
exist, the analysis and migration tools share the following choice rule:

1. Prefer runs with a complete generation checkpoint **and** scoring results.
2. Otherwise prefer complete generation checkpoints.
3. Otherwise prefer the most fully saved questions; break ties by latest file modification time.
4. If multiple runs have completed the same stage, take the newest. Never merge two generation files.

Edit `EXP_NAMES` in `analysis_results/migrate.sh`, then run:

```bash
bash analysis_results/migrate.sh           # preview only, NO file changes
bash analysis_results/migrate.sh --apply   # requires no running evaluation for these experiments
bash analysis_results/run.sh               # regenerate tables
```

The winner's files are moved directly into `<experiment>_<N>/<dataset>/`. Losing subdirectories go to `<experiment>_<N>/.llmjp_archives/<dataset>/`, and replaced dataset-root LLM-jp files go to its `dataset_root/` archive. Unrelated root files from Phi/DSDT are left untouched. Original JSONL and official LCB scorer outputs are retained. The archive may be removed manually **only
once backups and new scores have been verified**. Migration obtains the same
Slurm `.evaluation.lock` as the generation worker; never manually relocate
files while an experiment job is running.

Resume checks `protocol.json` generation settings, candidate count, sample IDs,
question text and ground truth. Changes to `uv.lock`, scorer implementation or
weight-file timestamps no longer cause cache resets, but changing sampling
settings or switching model checkpoints requires an explicit `--overwrite`
(which archives the previous run) rather than silently mixing outputs.

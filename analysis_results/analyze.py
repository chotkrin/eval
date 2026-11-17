#!/usr/bin/env python3
"""Read-only analysis of existing benchmark metrics and generation lengths.

Supports answer-scored pass@k, ProcessBench F1, and ungraded ProofBench
status without conflating incompatible metrics or rerunning model inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

# Reuse the project's single source of truth for dataset names, metric scaling,
# official LiveCodeBench score validation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sh" / "mdx"))
from lcb_results import DATASETS, read_dataset_metric  # noqa: E402
sys.path.insert(0, str(ROOT))
from transfer_datasets import TRANSFER_DATASETS  # noqa: E402
from llmjp_storage import candidates, choose  # noqa: E402

GROUPS = ("Math", "MC", "Code")
DATASET_NAMES = tuple(name for group in GROUPS for name in DATASETS[group])
COLUMNS = ("experiment", *DATASET_NAMES, *GROUPS, "AllDatasets", "Overall")

# Original Math/MC/Code public API remains unchanged. Other datasets use the
# same names as the project loaders. One selected dataset -> one report column.
# Do not aggregate proof/process metrics into answer pass@k.
TRANSFER_GROUPS = {
    "GSMPlus": ("gsm_plus_seed", "gsm_plus"),
    "GSMSymbolic": ("gsm_symbolic_main", "gsm_symbolic_p1", "gsm_symbolic_p2"),
    "MATHPerturb": ("math_perturb_simple", "math_perturb_hard"),
    "OMEGA": ("omega_exploratory", "omega_compositional", "omega_transformative"),
    "ObligationTransfer": ("obligation_transfer",),
    "ProcessBench": ("processbench_gsm8k", "processbench_math", "processbench_olympiadbench"),
    "ProofBench": ("proofbench",),
}
# Additional datasets directly supported by data_loader.py, outside the MDX
# default suite. All other eval transfer datasets come from TRANSFER_DATASETS.
OTHER_GROUPS = {
    "Math": ("math", "gsm8k", "svamp", "asdiv", "mawps", "carp_en"),
    "MC": ("mmlu_pro",),
}
SUPPORTED_GROUPS = {
    **{group: (*DATASETS[group], *OTHER_GROUPS.get(group, ())) for group in GROUPS},
    **TRANSFER_GROUPS,
}
PROCESS_DATASETS = frozenset(TRANSFER_GROUPS["ProcessBench"])
PROOF_DATASETS = frozenset(TRANSFER_GROUPS["ProofBench"])
NON_ANSWER_DATASETS = PROCESS_DATASETS | PROOF_DATASETS
SUPPORTED_DATASETS = frozenset(
    name for names in SUPPORTED_GROUPS.values() for name in names
)
# Guard against the eval loader's evolving registry falling out of sync.
assert set(TRANSFER_DATASETS).issubset(SUPPORTED_DATASETS)


def dataset_group(name: str):
    if name.startswith("livecodebench_") and re.fullmatch(r"livecodebench_v[1-9][0-9]*", name):
        return "Code"
    for group, members in SUPPORTED_GROUPS.items():
        if name in members:
            return group
    return None


def score_kind(name: str):
    if name in PROCESS_DATASETS:
        return "process_f1"
    if name in PROOF_DATASETS:
        return "proof_ungraded"
    return "answer"


def select_layout(datasets=None):
    """Build columns/groups in the exact requested order, as in eval.sh.

    An answer pass@k macro averages answer datasets ONLY. ProcessBench has
    its own F1 metrics; ProofBench is generated but has no proof grader here.
    """
    names = DATASET_NAMES if datasets is None else tuple(datasets)
    if not names or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("dataset list must be nonempty and contain no duplicates or empty names")
    unknown = [name for name in names if dataset_group(name) is None]
    if unknown:
        raise ValueError("unsupported dataset(s): " + ", ".join(unknown))
    groups = {}
    for name in names:
        group = dataset_group(name)
        groups.setdefault(group, []).append(name)
    groups = {group: tuple(members) for group, members in groups.items()}
    columns = ("experiment", *names, *groups, "AllDatasets", "Overall")
    return names, groups, columns


def finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def read_object(path: Path):
    with path.open(encoding="utf-8") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError(f"not a JSON object: {path}")
    return result


def locate_generation(dataset_dir: Path, n: int):
    """Use the same run selection as migration and pass@k reading."""
    if not dataset_dir.is_dir():
        return None, None, "MISSING"
    selected = choose(candidates(dataset_dir, n))
    if selected is not None:
        if not selected.valid:
            return None, None, "INVALID"
        return (selected.directory / "generations.jsonl",
                selected.directory / "metrics.json", "exact_token_ids")
    legacy = list(dataset_dir.glob("*_metrics.json"))
    if not legacy:
        return None, None, "MISSING"
    if len(legacy) != 1:
        return None, None, "AMBIGUOUS"
    m = legacy[0]
    return m.with_name(m.name.removesuffix("_metrics.json") + ".jsonl"), m, "legacy_reencoded"


def generation_measurements(dataset_dir: Path, n: int):
    """Read each selected generation file once for both length and truncation.

    A truncation is a vLLM finish_reason == "length" (token budget reached),
    not an absent final answer or an incorrect solution. Do not infer it from
    re-tokenized text length. Historical metrics.truncated is accepted only
    when complete, fresh, and consistent with the stored generation count.
    """
    jsonl, metrics, basis = locate_generation(dataset_dir, n)
    if jsonl is None:
        return ((None, basis, basis, 0, 0),
                (None, basis, "not_recorded", 0, 0))
    if not metrics.is_file() or not jsonl.is_file():
        return ((None, "PENDING", basis, 0, 0),
                (None, "PENDING", "not_recorded", 0, 0))
    try:
        meta = read_object(metrics)
        expected = meta.get("num_samples")
        if type(expected) is not int or expected <= 0:
            raise ValueError("invalid num_samples")
        questions, generations, tokens = 0, 0, 0
        indices = set()
        valid_length = True
        with_finish, without_finish = 0, 0
        invalid_finish = False
        truncated = 0
        with jsonl.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("non-object generation row")
                codes = row.get("code")
                if not isinstance(codes, list) or len(codes) != n:
                    raise ValueError("wrong candidate count in generation row")
                if "idx" in row:
                    idx = str(row["idx"])
                    if idx in indices:
                        raise ValueError("duplicate question index")
                    indices.add(idx)
                if basis == "exact_token_ids":
                    lengths = row.get("completion_tokens_per_sample")
                    if not isinstance(lengths, list) or len(lengths) != n or not all(
                        type(x) is int and x >= 0 for x in lengths
                    ):
                        valid_length = False
                    else:
                        value = sum(lengths)
                else:
                    value = row.get("completion_tokens")
                    if type(value) is not int or value < 0:
                        valid_length = False
                if valid_length:
                    if "completion_tokens" in row and row["completion_tokens"] != value:
                        valid_length = False
                    else:
                        tokens += value

                if "finish_reason" not in row:
                    without_finish += 1
                else:
                    with_finish += 1
                    reasons = row["finish_reason"]
                    # Do not treat abnormal/unknown termination as a successful
                    # non-truncated completion: its rate cannot be trusted.
                    if (not isinstance(reasons, list) or len(reasons) != n or
                            any(reason not in ("stop", "length") for reason in reasons)):
                        invalid_finish = True
                    else:
                        truncated += sum(reason == "length" for reason in reasons)
                questions += 1
                generations += n
        if questions != expected:
            raise ValueError(f"incomplete generations: {questions}/{expected} questions")
        if "num_generations" in meta and meta["num_generations"] != generations:
            raise ValueError("num_generations mismatch")

        mean = tokens / generations if valid_length else None
        if mean is not None and not finite_nonnegative(mean):
            valid_length = False
        length = (mean if valid_length else None,
                  "OK" if valid_length else "INVALID", basis,
                  questions if valid_length else 0,
                  generations if valid_length else 0)

        fresh_metrics = metrics.stat().st_mtime_ns >= jsonl.stat().st_mtime_ns
        metric_truncated = meta.get("truncated")
        if invalid_finish or (with_finish and without_finish):
            truncation = (None, "INVALID", "invalid_finish_reason", 0, generations)
        elif with_finish == questions:
            if (fresh_metrics and metric_truncated is not None and
                    (type(metric_truncated) is not int or metric_truncated != truncated)):
                truncation = (None, "INVALID", "finish_reason_mismatch", 0, generations)
            else:
                truncation = (100.0 * truncated / generations, "OK", "finish_reason",
                              truncated, generations)
        elif (type(metric_truncated) is int and 0 <= metric_truncated <= generations
              and fresh_metrics):
            truncation = (100.0 * metric_truncated / generations, "OK", "metrics_truncated",
                          metric_truncated, generations)
        elif metric_truncated is not None:
            truncation = (None, "INVALID" if fresh_metrics else "UNAVAILABLE",
                          "invalid_or_stale_metrics", 0, generations)
        else:
            truncation = (None, "UNAVAILABLE", "not_recorded", 0, generations)
        return length, truncation
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError, json.JSONDecodeError):
        return ((None, "INVALID", basis, 0, 0),
                (None, "INVALID", "invalid_generation", 0, 0))


def length_metric(dataset_dir: Path, n: int):
    """Retain the existing API for consumers of the length-only analysis."""
    return generation_measurements(dataset_dir, n)[0]


def truncation_metric(dataset_dir: Path, n: int):
    """Return rate (%) and availability, independently of correctness."""
    return generation_measurements(dataset_dir, n)[1]


def read_special_metric(dataset_dir: Path, n: int, name: str):
    """Read original metric files; NEVER re-score generated outputs.

    ProcessBench's official harmonic-mean F1 is a separate metric. ProofBench
    has no automatic grader; report the explicit proof_grading_required status.
    """
    generation, metrics_path, source = locate_generation(dataset_dir, n)
    if metrics_path is None:
        return {}, source
    if not metrics_path.is_file() or not generation.is_file():
        return {}, "PENDING"
    try:
        obj = read_object(metrics_path)
        count = obj.get("num_samples")
        if type(count) is not int or count <= 0:
            return {}, "INVALID"
        if name in PROCESS_DATASETS:
            # Require full/valid scored generation selection, not just an
            # unrelated or partially persisted metrics.json.
            selected = choose(candidates(dataset_dir, n))
            if selected is not None and (not selected.valid or not selected.generation_complete
                                         or not selected.fully_scored):
                return {}, "PENDING"
            keys = ("processbench_f1", "processbench_error_acc", "processbench_correct_acc")
            metrics = {key: obj.get(key) for key in keys}
            if any(not finite_nonnegative(value) or value > 100 for value in metrics.values()):
                return {}, "PENDING" if any(value is None for value in metrics.values()) else "INVALID"
            if obj.get("processbench_f1_sample_index", 0) != 0:
                return {}, "INVALID"
            return {key: float(value) for key, value in metrics.items()}, "OK"
        if name in PROOF_DATASETS:
            if obj.get("evaluation_status") == "proof_grading_required":
                return {"proof_status": "NEEDS_GRADING", "num_samples": count}, "NEEDS_GRADING"
            # No verified general-purpose proof metric is exposed by this evaluator.
            return {"proof_status": "UNVERIFIED", "num_samples": count}, "UNVERIFIED"
        raise ValueError(f"unsupported specialized dataset {name}")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}, "INVALID"


def mean_complete(values):
    """Macro mean only when all required inputs exist; never hide missing runs."""
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def aggregate_measurements(dataset_values, groups=None, dataset_names=None,
                           metric_kind="all"):
    """Macro average compatible datasets only, with strict missing behavior.

    answer -> excludes process/proof; process -> only ProcessBench; all -> all.
    Inapplicable columns are NA, never interpreted as missing measurements.
    """
    if groups is None:
        groups = {group: DATASETS[group] for group in GROUPS}
    if dataset_names is None:
        dataset_names = DATASET_NAMES
    out = dict(dataset_values)
    allowed = {
        "answer": lambda ds: score_kind(ds) == "answer",
        "process": lambda ds: ds in PROCESS_DATASETS,
        "all": lambda ds: True,
    }[metric_kind]
    active_groups = []
    for group, members in groups.items():
        eligible = [ds for ds in members if allowed(ds)]
        out[group] = mean_complete([dataset_values[ds] for ds in eligible])
        if eligible:
            active_groups.append(group)
    eligible_all = [ds for ds in dataset_names if allowed(ds)]
    out["AllDatasets"] = mean_complete([dataset_values[ds] for ds in eligible_all])
    out["Overall"] = mean_complete([out[group] for group in active_groups])
    return out


def collect(base: Path, experiments: list[str], n: int, ks: list[int],
            dataset_names=None, groups=None):
    if dataset_names is None or groups is None:
        dataset_names, groups, _ = select_layout(dataset_names)
    wide = {f"pass@{k}": [] for k in ks}
    wide.update({name: [] for name in (
        "processbench_f1", "processbench_error_acc", "processbench_correct_acc",
        "length", "truncation")})
    status_rows, proof_rows = [], []
    for exp in experiments:
        exp_root = base / f"{exp}_{n}"
        scores = {k: {} for k in ks}
        process = {key: {} for key in (
            "processbench_f1", "processbench_error_acc", "processbench_correct_acc")}
        lengths, truncations = {}, {}
        for ds in dataset_names:
            kind = score_kind(ds)
            if kind == "answer":
                obj, score_status = read_dataset_metric(exp_root, ds, n)
                specialized = {}
            else:
                specialized, score_status = read_special_metric(exp_root / ds, n, ds)
                obj = None
            length_data, truncation_data = generation_measurements(exp_root / ds, n)
            mean_length, length_status, basis, count, num_gens = length_data
            truncation_rate, truncation_status, truncation_source, truncated_count, truncation_n = truncation_data
            missing_k = []
            for k in ks:
                value = obj.get(f"pass@{k}") if score_status == "OK" and obj else None
                scores[k][ds] = value
                if kind == "answer" and value is None:
                    missing_k.append(str(k))
            for measure in process:
                process[measure][ds] = specialized.get(measure) if score_status == "OK" else None
            lengths[ds] = mean_length if length_status == "OK" else None
            truncations[ds] = truncation_rate if truncation_status == "OK" else None
            historical = candidates(exp_root / ds, n)
            selected = choose(historical)
            status_rows.append(dict(experiment=exp, dataset=ds,
                                    category=dataset_group(ds), score_kind=kind,
                                    score_status=score_status, missing_pass_k=",".join(missing_k),
                                    length_status=length_status, length_basis=basis,
                                    questions=count, generations=num_gens,
                                    truncation_status=truncation_status,
                                    truncation_source=truncation_source,
                                    truncated_generations=truncated_count if truncation_status == "OK" else "",
                                    truncation_denominator=truncation_n,
                                    selected_run=selected.directory.name if selected else "",
                                    alternatives=max(0, len(historical)-1),
                                    generation_complete=("yes" if selected.generation_complete else "no") if selected else ""))
            if kind == "proof_ungraded":
                proof_rows.append(dict(experiment=exp, dataset=ds,
                                       status=score_status,
                                       questions=specialized.get("num_samples", count),
                                       generations=num_gens,
                                       note="No automatic proof score; requires validated grader"))
        for k in ks:
            wide[f"pass@{k}"].append({"experiment": exp, **aggregate_measurements(
                scores[k], groups, dataset_names, metric_kind="answer")})
        for measure in process:
            wide[measure].append({"experiment": exp, **aggregate_measurements(
                process[measure], groups, dataset_names, metric_kind="process")})
        wide["length"].append({"experiment": exp, **aggregate_measurements(
            lengths, groups, dataset_names, metric_kind="all")})
        wide["truncation"].append({"experiment": exp, **aggregate_measurements(
            truncations, groups, dataset_names, metric_kind="all")})
    return wide, status_rows, proof_rows


def render_number(value):
    return "NA" if value is None else f"{value:.2f}"


def write_csv(path: Path, data: list[dict], columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def markdown_table(rows, columns=COLUMNS):
    def cell(item):
        return str(item).replace("|", "\\|").replace("\n", " ")
    formatted = [[cell(r["experiment"]), *(render_number(r[col]) for col in columns[1:])] for r in rows]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
        *("| " + " | ".join(values) + " |" for values in formatted),
    ])


def delta_from_baseline(rows, columns, baseline, ratio=False):
    """Model minus baseline in percentage points/tokens; optional length %."""
    originals = [row for row in rows if row["experiment"] == baseline]
    if len(originals) != 1:
        raise ValueError(f"baseline experiment not present exactly once: {baseline!r}")
    anchor = originals[0]
    answer = []
    for row in rows:
        out = {"experiment": row["experiment"]}
        for key in columns[1:]:
            new, old = row[key], anchor[key]
            if new is None or old is None or (ratio and old <= 0):
                out[key] = None
            else:
                out[key] = (100.0 * new / old) if ratio else (new - old)
        answer.append(out)
    return answer


def produce(base: Path, dest: Path, experiments: list[str], n: int, ks: list[int],
            dataset_names=None, baseline=None):
    if n < 1 or not ks or any(k < 1 or k > n for k in ks):
        raise ValueError("all pass@k must satisfy 1 <= k <= n_sampling")
    if not experiments or len(experiments) != len(set(experiments)) or any(
        not x or x in (".", "..") or "/" in x or "\\" in x for x in experiments
    ):
        raise ValueError("experiments must be unique directory names")
    dataset_names, groups, columns = select_layout(dataset_names)
    if baseline is not None and baseline not in experiments:
        raise ValueError(f"baseline {baseline!r} must be one of experiments")
    dest.mkdir(parents=True, exist_ok=True)
    # Only files named in the previous manifest belong to this analyzer.
    # This ensures selecting a new DATASETS subset removes stale per-dataset
    # reports without deleting unrelated user files in the same directory.
    manifest_path = dest / "analysis_generated_files.json"
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(previous, list) or any(
                    not isinstance(name, str) or Path(name).name != name or
                    not name.endswith((".csv", ".md")) for name in previous):
                raise ValueError("invalid analysis manifest")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid previous report manifest: {manifest_path}") from exc
        for name in previous:
            (dest / name).unlink(missing_ok=True)
    emitted = {"report.md", "coverage.csv", "mean_completion_tokens.csv", "truncation_rate_pct.csv"}
    emitted.update(f"pass_at_{k}.csv" for k in ks)
    if any(ds in PROCESS_DATASETS for ds in dataset_names):
        emitted.update(("processbench_f1.csv", "processbench_error_acc.csv",
                        "processbench_correct_acc.csv"))
    if any(ds in PROOF_DATASETS for ds in dataset_names):
        emitted.add("proofbench_status.csv")
    if baseline is not None:
        emitted.update(("delta_mean_completion_tokens.csv", "delta_truncation_rate_pct.csv",
                        "completion_length_pct_of_baseline.csv"))
        emitted.update(f"delta_pass_at_{k}.csv" for k in ks)
        if any(ds in PROCESS_DATASETS for ds in dataset_names):
            emitted.update(("delta_processbench_f1.csv",
                            "delta_processbench_error_acc.csv",
                            "delta_processbench_correct_acc.csv"))
    wide, status, proof_rows = collect(base, experiments, n, ks, dataset_names, groups)
    lines = ["# Evaluation analysis", "",
             f"Sampling count: N={n}. Scores are pass@k percentages (0–100); lengths are mean generated tokens per candidate.",
             "", "Equal-weight macro summaries are only across compatible datasets.",
             f"Selected datasets: {', '.join(dataset_names)}.",
             "Groups: " + "; ".join(f"{group}=[{', '.join(members)}]" for group, members in groups.items()) + ".",
             "Pass@k aggregates use only answer-scored datasets; ProcessBench is NOT included.",
             "ProcessBench F1 and subgroup accuracies are reported separately and are NOT mixed with pass@k.",
             "ProofBench has no automatic proof scorer; its results are generation-only, never pass@k=0.",
             "AllDatasets: equal weight across compatible datasets for each measure. "
             "Overall: equal weight across compatible groups, NOT across questions.",
             "Aggregates are NA unless every required dataset has that metric; missing scores are never zero.",
             "LLM-jp lengths use original output token IDs; Phi/DSDT legacy lengths are text re-encoding estimates.",
             "Truncation rate (%) is the fraction of candidates ending with finish_reason=length; it is not missing-final rate.",
             "Legacy generations without recorded finish reasons or valid fresh truncated counts show NA, not an inferred 0%.",
             "Truncation is measured independently of code-execution scoring. Aggregates require all datasets.",
             "Historical runs are chosen by completed scoring/generation progress and then newest modification time.", ""]
    for k in ks:
        rows = wide[f"pass@{k}"]
        write_csv(dest / f"pass_at_{k}.csv",
                  [{col: r[col] if col == "experiment" else render_number(r[col]) for col in columns} for r in rows], columns)
        lines.extend((f"## pass@{k} (%)", "", markdown_table(rows, columns), ""))
    if any(ds in PROCESS_DATASETS for ds in dataset_names):
        for measure, filename, heading in (
            ("processbench_f1", "processbench_f1.csv", "ProcessBench F1 (%, candidate 0)"),
            ("processbench_error_acc", "processbench_error_acc.csv", "ProcessBench error-trace accuracy (%)"),
            ("processbench_correct_acc", "processbench_correct_acc.csv", "ProcessBench correct-trace accuracy (%)"),
        ):
            rows = wide[measure]
            write_csv(dest / filename,
                      [{col: r[col] if col == "experiment" else render_number(r[col]) for col in columns}
                       for r in rows], columns)
            lines.extend((f"## {heading}", "", markdown_table(rows, columns), ""))
    if proof_rows:
        proof_columns = ("experiment", "dataset", "status", "questions", "generations", "note")
        write_csv(dest / "proofbench_status.csv", proof_rows, proof_columns)
        lines.extend(("## ProofBench proof-generation status (not an accuracy metric)", "",
                      "| experiment | dataset | status | questions | generations |",
                      "|---|---|---|---|---|",
                      *("| " + " | ".join(str(item[k]) for k in proof_columns[:5]) + " |" for item in proof_rows), ""))
    rows = wide["length"]
    write_csv(dest / "mean_completion_tokens.csv",
              [{col: r[col] if col == "experiment" else render_number(r[col]) for col in columns} for r in rows], columns)
    lines.extend(("## Mean completion length (tokens / candidate)", "", markdown_table(rows, columns), ""))
    rows = wide["truncation"]
    write_csv(dest / "truncation_rate_pct.csv",
              [{col: r[col] if col == "experiment" else render_number(r[col]) for col in columns} for r in rows], columns)
    lines.extend(("## Truncation rate (%)", "", markdown_table(rows, columns), ""))
    if baseline is not None:
        lines.extend((f"## Differences versus {baseline}", "",
                      "Score/truncation differences are percentage points; length differences are tokens. "
                      "Negative scores mean worse performance; negative lengths mean shorter outputs. "
                      "These are marginal differences, not paired per-question causal effects.", ""))
        for measure, filename, heading in [
            *((f"pass@{k}", f"delta_pass_at_{k}.csv", f"Delta pass@{k} (percentage points)") for k in ks),
            ("length", "delta_mean_completion_tokens.csv", "Delta mean completion length (tokens)"),
            ("truncation", "delta_truncation_rate_pct.csv", "Delta truncation (percentage points)"),
        ]:
            diff = delta_from_baseline(wide[measure], columns, baseline)
            write_csv(dest / filename,
                      [{col: row[col] if col == "experiment" else render_number(row[col]) for col in columns}
                       for row in diff], columns)
            lines.extend((f"### {heading}", "", markdown_table(diff, columns), ""))
        if any(ds in PROCESS_DATASETS for ds in dataset_names):
            for measure in ("processbench_f1", "processbench_error_acc", "processbench_correct_acc"):
                delta = delta_from_baseline(wide[measure], columns, baseline)
                write_csv(dest / f"delta_{measure}.csv",
                          [{col: row[col] if col == "experiment" else render_number(row[col])
                            for col in columns} for row in delta], columns)
                lines.extend((f"### Delta {measure} (percentage points)", "",
                              markdown_table(delta, columns), ""))
        ratio = delta_from_baseline(wide["length"], columns, baseline, ratio=True)
        write_csv(dest / "completion_length_pct_of_baseline.csv",
                  [{col: row[col] if col == "experiment" else render_number(row[col]) for col in columns}
                   for row in ratio], columns)
        lines.extend(("### Mean completion length (% of baseline)", "",
                      markdown_table(ratio, columns), ""))
    write_csv(dest / "coverage.csv", status, (
        "experiment", "dataset", "category", "score_kind", "score_status", "missing_pass_k", "length_status", "length_basis", "questions", "generations", "truncation_status", "truncation_source", "truncated_generations", "truncation_denominator", "selected_run", "alternatives", "generation_complete"
    ))
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path.write_text(json.dumps(sorted(emitted), indent=2), encoding="utf-8")
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="EVAL_OUTPUTS")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--ks", default="1,5,10", help="comma-separated values, e.g. 1,5,10")
    parser.add_argument("--datasets", default=None,
                        help="comma-separated eval dataset names, including ProcessBench/ProofBench; omit for legacy suite")
    parser.add_argument("--baseline", default=None,
                        help="experiment name used for difference and token-ratio reports")
    parser.add_argument("--allow-unavailable-truncation", action="store_true",
                        help="accept legacy runs with no finish reasons (still shown as NA)")
    parser.add_argument("experiments", nargs="+")
    args = parser.parse_args(argv)
    try:
        ks = [int(item.strip()) for item in args.ks.split(",")]
        if len(set(ks)) != len(ks):
            raise ValueError("duplicate pass@k values")
        datasets = ([item.strip() for item in args.datasets.split(",")]
                    if args.datasets is not None else None)
        status = produce(args.output_dir, args.report_dir, args.experiments, args.n, ks,
                         dataset_names=datasets, baseline=args.baseline)
    except ValueError as exc:
        parser.error(str(exc))
    print((args.report_dir / "report.md").read_text(encoding="utf-8"))
    print(f"\nCSV and coverage files: {args.report_dir}")
    missing = [s for s in status if
               (s["score_kind"] == "answer" and s["missing_pass_k"]) or
               (s["score_kind"] == "process_f1" and s["score_status"] != "OK") or
               (s["score_kind"] == "proof_ungraded" and s["score_status"] != "NEEDS_GRADING") or
               s["length_status"] != "OK" or
               (s["truncation_status"] != "OK" and not
                (args.allow_unavailable_truncation and s["truncation_status"] == "UNAVAILABLE"))]
    if args.allow_unavailable_truncation:
        unavailable = sum(s["truncation_status"] == "UNAVAILABLE" for s in status)
        if unavailable:
            print(f"NOTE: {unavailable} legacy truncation rates unavailable (NA); not treated as failures.",
                  file=sys.stderr)
    if missing:
        print(f"INCOMPLETE: {len(missing)} experiment/dataset rows have missing or invalid score/length/truncation; see coverage.csv", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

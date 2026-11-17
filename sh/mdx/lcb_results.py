#!/usr/bin/env python3
"""Read existing project metrics and official LiveCodeBench custom-evaluator files.

Never runs generated programs. Only the separate scoring shell invokes the
trusted external LCB runner, which itself executes untrusted model code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Also works when invoked directly: python sh/mdx/lcb_results.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmjp_storage import candidates, choose

DATASETS = {
    "Math": ("math500", "aime24", "minerva_math", "olympiadbench", "amc23", "aime26"),
    "Code": ("livecodebench_v6",),
    "MC": ("gpqa_diamond", "mmlu_stem"),
}
METRICS = ("acc", "pass@1", "pass@5", "pass@10")


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def valid_percent(x, scale=1.0):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    value = float(x) * scale
    return value if 0 <= value <= 100 else None


def custom_paths(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*.json")
        if path.name == "lcb_custom.json" or path.name.endswith("_lcb_custom.json")
    )


def companions(custom: Path):
    basename = custom.with_name(custom.stem + "_codegeneration_output")
    return (
        basename.with_suffix(".json"),
        basename.with_name(basename.name + "_eval.json"),
        basename.with_name(basename.name + "_eval_all.json"),
    )


def inspect_score(custom: Path, expected_n: int):
    """Fail closed on incomplete, stale, malformed or differently sized scores."""
    try:
        original = read_json(custom)
        if not isinstance(original, list) or not original:
            raise ValueError("empty/non-list custom input")
        ids = [str(row["question_id"]) for row in original]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question_id")
        if any(not isinstance(row.get("code_list"), list) or
               len(row["code_list"]) != expected_n or
               not all(isinstance(code, str) for code in row["code_list"])
               for row in original):
            raise ValueError(f"code_list must contain exactly {expected_n} strings per question")
        output, scores, all_results = companions(custom)
        if any(not x.is_file() or x.stat().st_size == 0 for x in (output, scores, all_results)):
            return False, "missing/empty official score files", None
        if any(x.stat().st_mtime_ns < custom.stat().st_mtime_ns for x in (output, scores, all_results)):
            return False, "score file older than custom input", None
        output_data, metrics_data, all_data = map(read_json, (output, scores, all_results))
        if not all(isinstance(x, list) and len(x) == len(original) for x in (output_data, all_data)):
            raise ValueError("per-question score/output lengths do not match input")
        by_id = {str(x["question_id"]): x["code_list"] for x in original}
        if [str(x.get("question_id")) for x in output_data] != sorted(ids):
            raise ValueError("scored question_id order differs from custom input")
        if any(x.get("code_list") != by_id[str(x["question_id"])] for x in output_data):
            raise ValueError("scored code_list differs from custom input (stale score)")
        if [str(x.get("question_id")) for x in all_data] != sorted(ids):
            raise ValueError("eval_all question_id order differs from custom input")
        if any(not isinstance(x.get("graded_list"), list) or len(x["graded_list"]) != expected_n
               or not all(isinstance(grade, bool) for grade in x["graded_list"])
               for x in all_data):
            raise ValueError("eval_all graded_list missing or invalid")
        # Official custom_evaluator stores [pass@k metrics, per-question results, metadata].
        if not isinstance(metrics_data, list) or len(metrics_data) < 2 or not isinstance(metrics_data[0], dict):
            raise ValueError("unexpected official metrics format")
        metrics = metrics_data[0]
        for key, value in metrics.items():
            if key.startswith("pass@") and key != "detail" and valid_percent(value, 100) is None:
                raise ValueError(f"invalid official {key}")
        if valid_percent(metrics.get("pass@1"), 100) is None:
            raise ValueError("missing pass@1")
        return True, "scored", {k: valid_percent(metrics.get(k), 100) for k in METRICS if k != "acc"}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, str(exc), None


def select_custom_input(dataset_dir: Path, n: int) -> Path:
    """Choose the same LCB generation as analysis, even before migration."""
    selected = choose(candidates(dataset_dir, n))
    if selected is not None:
        if not selected.valid:
            raise ValueError(f"Invalid LLM-jp checkpoint: {selected.directory}")
        custom = selected.directory / "lcb_custom.json"
        if not custom.is_file():
            raise FileNotFoundError(f"Missing LCB custom input: {custom}")
        return custom
    paths = custom_paths(dataset_dir)
    if len(paths) != 1:
        raise ValueError(f"Expected one legacy LCB custom input in {dataset_dir}, found {len(paths)}")
    return paths[0]


def read_dataset_metric(directory: Path, data_name: str, n: int):
    dataset_dir = directory / data_name
    selected = choose(candidates(dataset_dir, n))
    if selected is not None and not selected.valid:
        return None, "INVALID"
    if data_name.startswith("livecodebench_"):
        if selected is not None:
            paths = [selected.directory / "lcb_custom.json"]
        else:
            paths = custom_paths(dataset_dir)
        if not paths:
            return None, "MISSING"
        if len(paths) != 1:
            return None, "AMBIGUOUS"
        ok, reason, scores = inspect_score(paths[0], n)
        if not ok:
            return None, "PENDING" if reason == "missing/empty official score files" else "INVALID"
        return {"n": len(read_json(paths[0])), "acc": None, **scores}, "OK"
    if selected is not None:
        paths = [selected.directory / "metrics.json"]
        if not selected.generation_complete or not selected.fully_scored:
            return None, "PENDING"
    else:
        paths = sorted(dataset_dir.glob("*_metrics.json")) if dataset_dir.is_dir() else []
        if not paths:
            return None, "MISSING"
        if len(paths) != 1:
            return None, "AMBIGUOUS"
    try:
        obj = read_json(paths[0])
        if not isinstance(obj, dict) or not isinstance(obj.get("num_samples"), int) or obj["num_samples"] <= 0:
            return None, "INVALID"
        scores = {key: valid_percent(obj.get(key)) for key in METRICS}
        if scores["pass@1"] is None and scores["acc"] is None:
            return None, "PENDING"
        return {"n": obj["num_samples"], **scores}, "OK"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, "INVALID"


def fmt(x):
    return "NA" if x is None else f"{x:.2f}"


def row(exp, level, group, dataset, score, status, coverage):
    return dict(experiment=exp, level=level, category=group, dataset=dataset,
                questions=score["n"] if score else "", coverage=coverage,
                status=status, **{name: fmt(score.get(name)) if score else "NA" for name in METRICS})


def aggregate(items: list[tuple[dict | None, str]], expected: int):
    # The macro average is only valid with complete dataset coverage AND the
    # respective metric reported in every included dataset. No silent partial averages.
    complete = len(items) == expected and all(status == "OK" for _, status in items)
    result = {"n": sum(obj["n"] for obj, status in items if obj is not None and status == "OK")}
    for name in METRICS:
        values = [obj.get(name) for obj, status in items if obj is not None and status == "OK"]
        result[name] = sum(values) / expected if complete and len(values) == expected and all(v is not None for v in values) else None
    return result, "OK" if complete else "PARTIAL", f"{sum(status == 'OK' for _, status in items)}/{expected}"


def create_report(base: Path, names: list[str], n: int):
    table = []
    incomplete = False
    for exp in names:
        root = base / f"{exp}_{n}"
        type_rows = []
        for group, datasets in DATASETS.items():
            measurements = []
            for name in datasets:
                obj, status = read_dataset_metric(root, name, n)
                measurements.append((obj, status))
                table.append(row(exp, "dataset", group, name, obj, status, "1/1" if status == "OK" else "0/1"))
            average, status, coverage = aggregate(measurements, len(datasets))
            if status != "OK":
                incomplete = True
            table.append(row(exp, "category", group, "ALL", average, status, coverage))
            type_rows.append((average, status))
        # Balanced macro: first average the benchmarks within each category,
        # then average the three categories with equal weight.
        overall, status, coverage = aggregate(type_rows, len(DATASETS))
        table.append(row(exp, "overall", "ALL", "ALL", overall, status, coverage))
    return table, incomplete


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    check = sub.add_parser("score-status")
    check.add_argument("custom", type=Path)
    check.add_argument("--n", type=int, default=16)
    selected = sub.add_parser("select-custom", help="Print the selected custom input path")
    selected.add_argument("directory", type=Path)
    selected.add_argument("--n", type=int, default=16)
    summary = sub.add_parser("summary")
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--n", type=int, default=16)
    summary.add_argument("--csv", type=Path, required=True)
    summary.add_argument("experiments", nargs="+")
    args = ap.parse_args()
    if args.command == "score-status":
        ok, reason, _ = inspect_score(args.custom, args.n)
        print(f"{'SCORED' if ok else 'PENDING'}: {args.custom}: {reason}")
        return 0 if ok else 1
    if args.command == "select-custom":
        try:
            print(select_custom_input(args.directory, args.n))
        except (OSError, ValueError) as exc:
            print(f"LCB selection error: {exc}", file=sys.stderr)
            return 2
        return 0
    table, incomplete = create_report(args.output_dir, args.experiments, args.n)
    columns = ("experiment", "level", "category", "dataset", "questions", "coverage", "status") + METRICS
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    # Unambiguous pipe-delimited console table. CSV retains the full data.
    print(" | ".join(columns))
    for item in table:
        print(" | ".join(str(item[column]) for column in columns))
    print(f"\nCSV: {args.csv}")
    print("Scores are percentages (0-100). Category: unweighted mean of datasets; overall: equal-weight Math/Code/MC.")
    if incomplete:
        print("INCOMPLETE: at least one experiment is missing or has invalid/pending dataset scores; incomplete aggregates are NA.", file=sys.stderr)
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())

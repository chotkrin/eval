"""LLM-jp results at <experiment>/<dataset>/, with safe legacy migration.

Old llmjp4/ and llmjp4_<hash>/ folders are recognized only as migration
sources. Results from discarded runs are archived outside the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

# Fields that make a checkpoint safe (or unsafe) to resume with this run.
GENERATION_FIELDS = (
    "protocol", "model", "dataset", "split", "num_test_sample", "start", "end",
    "seed", "temperature", "top_p", "top_k", "n_sampling",
    "max_tokens_per_call", "max_model_len", "reasoning_effort",
)

# Only these files belong to a root-level LLM-jp run. In particular, do not
# move unrelated legacy Phi/DSDT files or arbitrary user artifacts.
ROOT_FILES = frozenset({
    "protocol.json", "generations.jsonl", "scored.jsonl", "metrics.json",
    "lcb_custom.json",
})


def is_result_file(path: Path):
    return path.is_file() and (
        path.name in ROOT_FILES or path.name.startswith("lcb_custom_codegeneration_")
    )


def root_result_files(dataset_dir: Path):
    return sorted((p for p in dataset_dir.iterdir() if is_result_file(p)), key=lambda p: p.name)


def normalize_config(raw):
    if not isinstance(raw, dict):
        return None
    model = raw.get("model")
    if isinstance(model, dict):
        model = model.get("path")
    if not isinstance(model, str) or not model:
        return None
    config = {key: raw.get(key) for key in GENERATION_FIELDS}
    config["model"] = str(Path(model).resolve())
    if any(key not in raw for key in GENERATION_FIELDS):
        return None
    # Qwen-specific generation values must also match before resuming.
    if str(raw.get("protocol", "")).startswith(("qwen3-", "qwen35-")):
        extra = ("prompt_type", "min_p", "presence_penalty")
        if any(key not in raw for key in extra):
            return None
        config.update({key: raw[key] for key in extra})
    return config


def read_protocol(folder):
    try:
        return normalize_config(json.loads((folder / "protocol.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def count_checkpoint(path, n):
    """Count valid, unique complete questions without editing the checkpoint.

    A torn final line is ignored. An invalid interior record is never ignored.
    """
    if not path.exists():
        return 0, True
    indices = set()
    try:
        with path.open("rb") as stream:
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    if stream.read(1):
                        return 0, False
                    break
                record = json.loads(line)
                if not isinstance(record, dict) or "idx" not in record:
                    return 0, False
                if not all(isinstance(record.get(k), list) and len(record[k]) == n
                           for k in ("code", "pred")):
                    return 0, False
                index = str(record["idx"])
                if index in indices:
                    return 0, False
                indices.add(index)
        return len(indices), True
    except (OSError, ValueError, UnicodeError, TypeError):
        return 0, False


@dataclass(frozen=True)
class Run:
    directory: Path
    questions: int
    valid: bool
    generation_complete: bool
    fully_scored: bool
    updated_ns: int
    config: dict | None


def inspect_run(folder, n, dataset):
    count, valid = count_checkpoint(folder / "generations.jsonl", n)
    metrics_path = folder / "metrics.json"
    expected = None
    has_score = False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        num = metrics.get("num_samples")
        if type(num) is int and num > 0:
            expected = num
        if dataset.startswith("livecodebench_"):
            # The presence of score files is insufficient: an interrupted or
            # outdated official evaluator may have left malformed/stale files.
            # Reuse exactly the same validator as reporting/scoring.
            from sh.mdx.lcb_results import inspect_score
            has_score = inspect_score(folder / "lcb_custom.json", n)[0]
        else:
            # A benchmark generation checkpoint newer than its metrics is
            # not fully scored, even when metrics still contain a pass@1.
            generation_file = folder / "generations.jsonl"
            metrics_fresh = (generation_file.is_file() and
                             metrics_path.stat().st_mtime_ns >= generation_file.stat().st_mtime_ns)
            has_score = metrics_fresh and any(
                type(metrics.get(k)) in (int, float) and
                math.isfinite(metrics[k]) and 0 <= metrics[k] <= 100
                for k in ("pass@1", "acc"))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    complete = valid and expected is not None and count == expected
    updated = max((p.stat().st_mtime_ns for p in folder.iterdir() if p.is_file()),
                  default=folder.stat().st_mtime_ns)
    return Run(folder, count, valid, complete, complete and has_score,
               updated, read_protocol(folder))


def is_old_run_dir(name: str):
    return (name in ("llmjp4", "llm-jp4") or
            name.startswith(("llmjp4_", "llmjp4-", "llm-jp4_", "llm-jp4-")))


def candidates(dataset_dir, n):
    if not dataset_dir.is_dir():
        return []
    paths = []
    if root_result_files(dataset_dir):
        paths.append(dataset_dir)
    paths += sorted(p for p in dataset_dir.iterdir() if p.is_dir() and is_old_run_dir(p.name))
    return [inspect_run(path, n, dataset_dir.name) for path in paths]


def choose(runs):
    if not runs:
        return None
    valid = [r for r in runs if r.valid]
    if not valid:
        return max(runs, key=lambda r: (r.updated_ns, str(r.directory)))
    scored = [r for r in valid if r.fully_scored]
    if scored:
        return max(scored, key=lambda r: (r.updated_ns, str(r.directory)))
    complete = [r for r in valid if r.generation_complete]
    if complete:
        return max(complete, key=lambda r: (r.updated_ns, str(r.directory)))
    return max(valid, key=lambda r: (r.questions, r.updated_ns, str(r.directory)))


def archive_destination(folder: Path, *, dataset_root: bool = False):
    """Get a never-overwritten archive path for a whole run or root files."""
    if dataset_root:
        experiment, dataset, label = folder.parent, folder.name, "dataset_root"
    else:
        experiment, dataset, label = folder.parent.parent, folder.parent.name, folder.name
    base = experiment / ".llmjp_archives" / dataset
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate = base / label
    i = 1
    while candidate.exists():
        candidate = base / f"{label}.{i}"
        i += 1
    return candidate


def archive_root_files(dataset_dir: Path):
    """Move only LLM-jp root artifacts out of the way, never legacy files."""
    paths = root_result_files(dataset_dir)
    if not paths:
        return None
    target = archive_destination(dataset_dir, dataset_root=True)
    target.mkdir(mode=0o700)
    for path in paths:
        path.rename(target / path.name)
    return target


def migrate_one(dataset_dir, n, *, apply=False):
    """Flatten one winning run into dataset_dir, archiving competitors.

    Supports both *already migrated* llmjp4/ and unmigrated hashed directories.
    Other files living directly under dataset_dir are never relocated. Caller
    owns <experiment>/.evaluation.lock before apply=True.
    """
    runs = candidates(dataset_dir, n)
    if not runs:
        return {"dataset": dataset_dir.name, "status": "missing"}
    best = choose(runs)
    result = {"dataset": dataset_dir.name, "status": "preview", "chosen": str(best.directory),
              "questions": best.questions, "generation_complete": best.generation_complete,
              "fully_scored": best.fully_scored, "alternatives": len(runs) - 1}
    if not best.valid:
        result["status"] = "invalid"
        return result
    if not apply:
        return result
    if best.directory != dataset_dir:
        # Preflight *all* destination names before modifying data. Unrelated
        # files/dirs must never be overwritten during a flatten migration.
        for source in best.directory.iterdir():
            destination = dataset_dir / source.name
            if destination.exists() and not is_result_file(destination):
                raise FileExistsError(f"Refusing to overwrite unrelated dataset file: {destination}")
        # Existing root-level LLM-jp output, if any, belongs to a losing run.
        archive_root_files(dataset_dir)
    for run in runs:
        if run.directory in (best.directory, dataset_dir):
            continue
        run.directory.rename(archive_destination(run.directory))
    if best.directory != dataset_dir:
        for source in sorted(best.directory.iterdir()):
            source.rename(dataset_dir / source.name)
        best.directory.rmdir()
    dataset_dir.chmod(0o700)
    metrics_file = dataset_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            from llmjp_profile import write_json_atomic
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            original_stat = metrics_file.stat()
            metrics["output_dir"] = str(dataset_dir)
            if "custom_output_file" in metrics:
                metrics["custom_output_file"] = str(dataset_dir / "lcb_custom.json")
            write_json_atomic(metrics_file, metrics)
            # Rewriting embedded absolute paths is NOT rescoring. Preserve
            # freshness metadata so an old score cannot appear newer than an
            # updated generations.jsonl merely because it was migrated.
            import os
            os.utime(metrics_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        except (ValueError, TypeError, OSError) as exc:
            raise RuntimeError(f"Failed to update migrated metrics: {metrics_file}") from exc
    result.update(status="already_flat" if best.directory == dataset_dir else "migrated",
                  target=str(dataset_dir))
    return result

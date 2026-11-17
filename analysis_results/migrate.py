#!/usr/bin/env python3
"""Non-destructive flattening of old LLM-jp runs into dataset directories.

Dry-run by default. --apply obtains the SAME per-experiment flock as the MDX
worker; never run while a job is actively writing that experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sh" / "mdx"))
from lcb_results import DATASETS  # noqa: E402
from llmjp_storage import migrate_one  # noqa: E402


@contextmanager
def experiment_lock(experiment_dir, apply):
    if not apply:
        yield
        return
    if not experiment_dir.is_dir():
        yield
        return
    with (experiment_dir / ".evaluation.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Evaluation is still running; cannot migrate: {experiment_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def migrate(base: Path, experiments: list[str], n: int, apply: bool):
    if n < 1 or not experiments or len(set(experiments)) != len(experiments):
        raise ValueError("Invalid n or duplicate/empty experiment list")
    datasets = [ds for names in DATASETS.values() for ds in names]
    failures = 0
    for exp in experiments:
        if not exp or exp in (".", "..") or "/" in exp or "\\" in exp:
            raise ValueError(f"Invalid experiment name: {exp!r}")
        root = base / f"{exp}_{n}"
        with experiment_lock(root, apply):
            for dataset in datasets:
                result = migrate_one(root / dataset, n, apply=apply)
                if result["status"] == "missing":
                    continue
                print(f"{exp} / {dataset}: {result}", flush=True)
                if result["status"] == "invalid":
                    failures += 1
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("experiments", nargs="+")
    args = ap.parse_args()
    try:
        failures = migrate(args.output_dir, args.experiments, args.n, args.apply)
    except (ValueError, RuntimeError, OSError) as exc:
        ap.error(str(exc))
    if not args.apply:
        print("DRY RUN ONLY: use --apply to migrate after checking the choices.")
    else:
        print("Migration finished. Losing runs remain under .llmjp_archives/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

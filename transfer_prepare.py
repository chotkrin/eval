#!/usr/bin/env python3
"""Pre-download + normalize new benchmarks using the evaluation .env paths.

Run on a networked login node. Standalone and MDX shell share the same layout.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from transfer_datasets import TRANSFER_DATASETS, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="gsm_plus,gsm_plus_seed,gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    args = parser.parse_args()
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for name in names:
        if name not in TRANSFER_DATASETS:
            raise SystemExit(f"unknown {name}; supported: {','.join(TRANSFER_DATASETS)}")
        rows = prepare(name, args.data_dir)
        print(f"{name}: {len(rows)} rows -> {Path(args.data_dir)/name/'test.jsonl'}")

if __name__ == '__main__':
    main()

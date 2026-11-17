"""CPU-only tests for exact finish-reason truncation and legacy NA handling."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_analysis_truncation", ROOT / "analysis_results/analyze.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def generation(root, dataset, reasons, *, legacy=False, score=True):
    dataset_dir = root / dataset
    target = dataset_dir if legacy else dataset_dir / "llmjp4_test"
    n = 2
    rows = []
    for idx, finish in enumerate(reasons):
        row = {"idx": idx, "code": ["partial", "done"], "pred": ["", "42"],
               "completion_tokens": 20}
        if not legacy:
            row["completion_tokens_per_sample"] = [10, 10]
        if finish is not None:
            row["finish_reason"] = finish
        rows.append(row)
    if legacy:
        basename = "test_cot_-1_seed0_t0.6_s0_e-1"
        gen = target / f"{basename}.jsonl"
        met = target / f"{basename}_metrics.json"
    else:
        gen = target / "generations.jsonl"
        met = target / "metrics.json"
    gen.parent.mkdir(parents=True, exist_ok=True)
    gen.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metrics = {"num_samples": len(rows), "num_generations": len(rows) * n,
               "pass@1": 50.0 if score else None}
    write_json(met, metrics)
    return gen, met


class TruncationAnalysisTest(unittest.TestCase):
    def test_exact_length_finishes_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, report = Path(tmp) / "outputs", Path(tmp) / "reports"
            for ds in ANALYSIS.DATASET_NAMES:
                # One of four responses reached max_tokens; score status is
                # irrelevant to truncation, including LiveCodeBench.
                generation(base / "exp_2", ds, [["length", "stop"], ["stop", "stop"]])
            status = ANALYSIS.produce(base, report, ["exp"], 2, [1])
            self.assertEqual(len(status), len(ANALYSIS.DATASET_NAMES))
            with (report / "truncation_rate_pct.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(list(row), list(ANALYSIS.COLUMNS))
            self.assertEqual(row["math500"], "25.00")
            self.assertEqual(row["Code"], "25.00")
            self.assertEqual(row["AllDatasets"], "25.00")
            self.assertEqual(row["Overall"], "25.00")
            self.assertIn("## Truncation rate (%)", (report / "report.md").read_text())
            self.assertTrue(all(item["truncation_source"] == "finish_reason" for item in status))
            self.assertEqual(status[0]["truncated_generations"], 1)
            self.assertEqual(status[0]["truncation_denominator"], 4)

    def test_length_and_truncation_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "math500"
            generation(Path(tmp), "math500", [["length", "stop"], ["stop", "stop"]])
            self.assertEqual(ANALYSIS.length_metric(data, 2)[:2], (10.0, "OK"))
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[:2], (25.0, "OK"))

    def test_legacy_with_no_finish_reason_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "math500"
            generation(Path(tmp), "math500", [None, None], legacy=True)
            self.assertEqual(ANALYSIS.length_metric(data, 2)[:3], (10.0, "OK", "legacy_reencoded"))
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[:3], (None, "UNAVAILABLE", "not_recorded"))

    def test_metrics_fallback_only_if_fresh_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "math500"
            gen, met = generation(Path(tmp), "math500", [None, None])
            metric = json.loads(met.read_text())
            metric["truncated"] = 3
            write_json(met, metric)
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[:4],
                             (75.0, "OK", "metrics_truncated", 3))
            os.utime(met, ns=(1_000_000_000, 1_000_000_000))
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "UNAVAILABLE")
            os.utime(met, None)
            metric["truncated"] = 5  # more than total candidates
            write_json(met, metric)
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "INVALID")

    def test_corrupt_mixed_abort_and_inconsistent_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "math500"
            gen, met = generation(Path(tmp), "math500", [["length", "stop"], None])
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "INVALID")
            generation(Path(tmp), "math500", [["abort", "stop"], ["stop", "stop"]])
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "INVALID")
            generation(Path(tmp), "math500", [["length"], ["stop", "stop"]])
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "INVALID")
            generation(Path(tmp), "math500", [["length", "stop"], ["stop", "stop"]])
            metric = json.loads(met.read_text())
            metric["truncated"] = 2  # conflicts with recorded 1
            write_json(met, metric)
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "INVALID")
            metric["truncated"] = 1
            write_json(met, metric)
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "OK")
            metric["num_samples"] = 3
            write_json(met, metric)
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[1], "INVALID")

    def test_generation_length_invalid_does_not_erase_known_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "math500"
            gen, met = generation(Path(tmp), "math500", [["length", "stop"], ["stop", "stop"]])
            lines = [json.loads(s) for s in gen.read_text().splitlines()]
            lines[0].pop("completion_tokens_per_sample")
            gen.write_text("".join(json.dumps(s) + "\n" for s in lines))
            self.assertEqual(ANALYSIS.length_metric(data, 2)[1], "INVALID")
            self.assertEqual(ANALYSIS.truncation_metric(data, 2)[0], 25.0)

    def test_missing_truncation_makes_macro_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, report = Path(tmp) / "outputs", Path(tmp) / "reports"
            for ds in ANALYSIS.DATASET_NAMES:
                generation(base / "exp_2", ds, [["length", "stop"], ["stop", "stop"]])
            # A single dataset missing finish information invalidates that
            # category and both overall aggregates, not the other categories.
            generation(base / "exp_2", "math500", [None, None])
            statuses = ANALYSIS.produce(base, report, ["exp"], 2, [1])
            with (report / "truncation_rate_pct.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["math500"], "NA")
            self.assertEqual(row["Math"], "NA")
            self.assertEqual(row["MC"], "25.00")
            self.assertEqual(row["Code"], "25.00")
            self.assertEqual(row["Overall"], "NA")
            self.assertEqual(row["AllDatasets"], "NA")
            math_status = next(s for s in statuses if s["dataset"] == "math500")
            self.assertEqual(math_status["truncation_status"], "UNAVAILABLE")
            with (report / "coverage.csv").open() as f:
                self.assertIn("truncation_status", next(csv.DictReader(f)))


if __name__ == "__main__":
    unittest.main()

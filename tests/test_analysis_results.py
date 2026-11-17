"""CPU-only integration tests for read-only wide-format analysis."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_analysis", ROOT / "analysis_results/analyze.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_modern(root, ds, n=2, num=2, length=15):
    target = root / ds / "llmjp4_test"
    rows = [{"idx": i, "code": ["answer"] * n, "pred": ["answer"] * n,
             "completion_tokens_per_sample": [length - 5, length + 5],
             "completion_tokens": 2 * length} for i in range(num)]
    write_jsonl(target / "generations.jsonl", rows)
    write_json(target / "metrics.json", {"num_samples": num, "num_generations": num*n,
             "pass@1": 60, "pass@2": 80, "avg_completion_tokens": length})
    if ds.startswith("livecodebench_"):
        custom = target / "lcb_custom.json"
        raw = [{"question_id": f"q{i}", "code_list": ["print(1)"] * n} for i in range(num)]
        write_json(custom, raw)
        from lcb_results import companions
        output, metrics, all_data = companions(custom)
        write_json(output, list(raw))
        write_json(metrics, [{"pass@1": .50, "pass@2": .75}, {}, {}])
        write_json(all_data, [{**r, "graded_list": [True, False]} for r in raw])
    return target


class WideAnalysisTest(unittest.TestCase):
    def test_dataset_columns_category_and_two_overalls(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            for experiment, length in (("sft", 15), ("rl", 25)):
                for ds in MODULE.DATASET_NAMES:
                    make_modern(base / (experiment + "_2"), ds, length=length)
            statuses = MODULE.produce(base, report, ["sft", "rl"], 2, [1, 2])
            self.assertEqual(len(statuses), 18)
            with (report / "pass_at_1.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), list(MODULE.COLUMNS))
            self.assertEqual([r["experiment"] for r in rows], ["sft", "rl"])
            self.assertEqual(rows[0]["Math"], "60.00")
            self.assertEqual(rows[0]["MC"], "60.00")
            self.assertEqual(rows[0]["Code"], "50.00")
            self.assertEqual(rows[0]["Overall"], "56.67")  # equal 3 types
            self.assertEqual(rows[0]["AllDatasets"], "58.89")  # equal 9 datasets
            with (report / "mean_completion_tokens.csv").open() as f:
                lengths = list(csv.DictReader(f))
            self.assertEqual(lengths[0]["Overall"], "15.00")
            self.assertEqual(lengths[1]["AllDatasets"], "25.00")
            self.assertIn("pass@2", (report / "report.md").read_text())

    def test_missing_code_score_retains_length_and_no_partial_average(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            for ds in MODULE.DATASET_NAMES:
                make_modern(base / "rl_2", ds)
            target = base / "rl_2/livecodebench_v6/llmjp4_test/lcb_custom_codegeneration_output_eval.json"
            target.unlink()
            MODULE.produce(base, report, ["rl"], 2, [1])
            with (report / "pass_at_1.csv").open() as f:
                score = next(csv.DictReader(f))
            self.assertEqual(score["Code"], "NA")
            self.assertEqual(score["Overall"], "NA")
            self.assertEqual(score["AllDatasets"], "NA")
            self.assertEqual(score["Math"], "60.00")
            with (report / "mean_completion_tokens.csv").open() as f:
                length = next(csv.DictReader(f))
            self.assertEqual(length["Code"], "15.00")
            self.assertEqual(length["Overall"], "15.00")

    def test_legacy_length_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = Path(temp) / "math500"
            write_json(ds / "old_metrics.json", {"num_samples": 2, "pass@1": 40})
            write_jsonl(ds / "old.jsonl", [
                {"idx": 1, "code": ["a", "b"], "completion_tokens": 40},
                {"idx": 2, "code": ["a", "b"], "completion_tokens": 60},
            ])
            self.assertEqual(MODULE.length_metric(ds, 2)[:3], (25, "OK", "legacy_reencoded"))
            (ds / "old.jsonl").write_text('{"idx":1,"code":["a","b"],"completion_tokens":40}\n')
            self.assertEqual(MODULE.length_metric(ds, 2)[1], "INVALID")
            write_json(ds / "extra_metrics.json", {"num_samples": 2})
            self.assertEqual(MODULE.length_metric(ds, 2)[1], "AMBIGUOUS")

    def test_rejects_partial_or_mismatched_modern_token_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            target = make_modern(Path(temp), "math500")
            rows = [json.loads(line) for line in (target / "generations.jsonl").read_text().splitlines()]
            rows[0]["completion_tokens_per_sample"] = [10]
            write_jsonl(target / "generations.jsonl", rows)
            self.assertEqual(MODULE.length_metric(target.parent, 2)[1], "INVALID")
            rows[0]["completion_tokens_per_sample"] = [10, 20]
            write_jsonl(target / "generations.jsonl", rows)
            self.assertEqual(MODULE.length_metric(target.parent, 2)[1], "OK")
            # A different hashed directory must not make complete metrics NA.
            second = target.parent / "llmjp4_second"
            second.mkdir()
            (second / "protocol.json").write_text("{}")
            self.assertEqual(MODULE.length_metric(target.parent, 2)[1], "OK")

    def test_shell_uses_unified_uv_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            venv = tmp / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").symlink_to(sys.executable)
            env_file = tmp / ".env"
            env_file.write_text(f'EVAL_OUTPUTS="{tmp / "existing_outputs"}"\n'
                                f'UV_PROJECT_ENVIRONMENT="{venv}"\n')
            destination = tmp / "tables"
            clean_env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp),
                         "EVAL_ENV_FILE": str(env_file), "ANALYSIS_REPORT_DIR": str(destination)}
            result = subprocess.run(["bash", str(ROOT / "analysis_results/run.sh")],
                                    env=clean_env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue((destination / "report.md").exists())
            with (destination / "pass_at_1.csv").open() as stream:
                names = [item["experiment"] for item in csv.DictReader(stream)]
            # The experiment list is user-editable in run.sh. Verify that the
            # report follows it rather than assuming a fixed default list.
            shell_source = (ROOT / "analysis_results/run.sh").read_text(encoding="utf-8")
            match = re.search(r"(?ms)^EXP_NAMES=\(\s*(.*?)^\)\s*$", shell_source)
            self.assertIsNotNone(match, "run.sh must define an EXP_NAMES=(...) array")
            configured = shlex.split(match.group(1), comments=True)
            self.assertTrue(configured, "run.sh should specify at least one experiment")
            self.assertEqual(names, configured)

    def test_missing_requested_pass_k_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            for ds in MODULE.DATASET_NAMES:
                make_modern(base / "rl_2", ds)
            metrics_path = base / "rl_2/math500/llmjp4_test/metrics.json"
            obj = json.loads(metrics_path.read_text())
            obj.pop("pass@2")
            write_json(metrics_path, obj)
            statuses = MODULE.produce(base, report, ["rl"], 2, [1, 2])
            self.assertEqual(next(s for s in statuses if s["dataset"] == "math500")["missing_pass_k"], "2")
            with (report / "pass_at_2.csv").open() as f:
                score = next(csv.DictReader(f))
            self.assertEqual(score["Math"], "NA")
            self.assertEqual(score["Overall"], "NA")

    def test_two_historical_hashes_choose_best_instead_of_all_na(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "outputs", Path(temp) / "report"
            exp = base / "exp001_2"
            for ds in MODULE.DATASET_NAMES:
                make_modern(exp, ds, n=2, num=2)
            first = exp / "math500/llmjp4_test"
            # Make a second fully scored hash with later modification time.
            second = exp / "math500/llmjp4_second"
            second.mkdir()
            for file in first.iterdir():
                (second / file.name).write_bytes(file.read_bytes())
            metric_path = second / "metrics.json"
            data = json.loads(metric_path.read_text())
            data["pass@1"] = 91.0
            metric_path.write_text(json.dumps(data))
            import os
            for file in first.iterdir():
                os.utime(file, ns=(1_000_000_000, 1_000_000_000))
            for file in second.iterdir():
                os.utime(file, ns=(2_000_000_000, 2_000_000_000))
            statuses = MODULE.produce(base, report, ["exp001"], 2, [1])
            with (report / "pass_at_1.csv").open() as f:
                score = next(csv.DictReader(f))
            self.assertEqual(score["math500"], "91.00")
            self.assertNotEqual(score["Math"], "NA")
            selected = next(x for x in statuses if x["dataset"] == "math500")
            self.assertEqual(selected["selected_run"], "llmjp4_second")
            self.assertEqual(selected["alternatives"], 1)

    def test_flat_dataset_directory_scores_and_lengths(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            exp = base / "rl_2"
            for ds in MODULE.DATASET_NAMES:
                child = make_modern(exp, ds)
                root = child.parent
                for item in child.iterdir():
                    item.rename(root / item.name)
                child.rmdir()
            statuses = MODULE.produce(base, report, ["rl"], 2, [1, 2])
            self.assertEqual(len(statuses), len(MODULE.DATASET_NAMES))
            self.assertTrue(all(s["selected_run"] == s["dataset"] for s in statuses))
            with (report / "pass_at_1.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["Math"], "60.00")
            self.assertEqual(row["Code"], "50.00")
            with (report / "mean_completion_tokens.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["Overall"], "15.00")

    def test_transfer_suite_grouping_and_baseline_deltas(self):
        dataset_scores = {
            "gsm_plus_seed": 20, "gsm_plus": 100,
            "gsm_symbolic_main": 60, "gsm_symbolic_p1": 70,
            "gsm_symbolic_p2": 80, "math_perturb_simple": 90,
            "math_perturb_hard": 10,
        }
        names = list(dataset_scores)
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            for experiment, adjustment, length in (("baseline", 0, 100), ("compressed", -5, 40)):
                for ds, score in dataset_scores.items():
                    target = make_modern(base / f"{experiment}_2", ds, length=length)
                    metrics = target / "metrics.json"
                    data = json.loads(metrics.read_text())
                    data["pass@1"] = score + adjustment
                    metrics.write_text(json.dumps(data))
            statuses = MODULE.produce(base, report, ["compressed", "baseline"], 2, [1],
                                      dataset_names=names, baseline="baseline")
            self.assertEqual(len(statuses), 14)
            with (report / "pass_at_1.csv").open() as stream:
                compressed, baseline = list(csv.DictReader(stream))
            self.assertEqual(list(baseline), ["experiment", *names, "GSMPlus", "GSMSymbolic",
                                               "MATHPerturb", "AllDatasets", "Overall"])
            self.assertEqual(baseline["GSMPlus"], "60.00")
            self.assertEqual(baseline["GSMSymbolic"], "70.00")
            self.assertEqual(baseline["MATHPerturb"], "50.00")
            self.assertEqual(baseline["AllDatasets"], "61.43")
            self.assertEqual(baseline["Overall"], "60.00")
            self.assertEqual(compressed["Overall"], "55.00")
            with (report / "delta_pass_at_1.csv").open() as stream:
                diff, zero = list(csv.DictReader(stream))
            self.assertEqual(diff["math_perturb_hard"], "-5.00")
            self.assertEqual(diff["Overall"], "-5.00")
            self.assertEqual(zero["Overall"], "0.00")
            with (report / "delta_mean_completion_tokens.csv").open() as stream:
                diff, _ = list(csv.DictReader(stream))
            self.assertEqual(diff["MATHPerturb"], "-60.00")
            with (report / "completion_length_pct_of_baseline.csv").open() as stream:
                ratio, _ = list(csv.DictReader(stream))
            self.assertEqual(ratio["MATHPerturb"], "40.00")
            self.assertIn("GSMPlus", (report / "report.md").read_text())
            self.assertEqual({r["category"] for r in statuses},
                             {"GSMPlus", "GSMSymbolic", "MATHPerturb"})

    def test_transfer_missing_metrics_do_not_contribute_to_macro(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            make_modern(base / "model_2", "gsm_plus_seed")
            statuses = MODULE.produce(base, report, ["model"], 2, [1],
                                      dataset_names=["gsm_plus_seed", "gsm_plus"])
            self.assertEqual([s["score_status"] for s in statuses], ["OK", "MISSING"])
            with (report / "pass_at_1.csv").open() as stream:
                scores = next(csv.DictReader(stream))
            self.assertEqual(scores["gsm_plus_seed"], "60.00")
            self.assertEqual(scores["GSMPlus"], "NA")
            self.assertEqual(scores["Overall"], "NA")

    def test_special_datasets_are_supported_and_unknown_rejected(self):
        names, groups, columns = MODULE.select_layout([
            "processbench_gsm8k", "proofbench", "omega_exploratory", "math500"])
        self.assertEqual(names, ("processbench_gsm8k", "proofbench", "omega_exploratory", "math500"))
        self.assertEqual(set(groups), {"ProcessBench", "ProofBench", "OMEGA", "Math"})
        self.assertIn("proofbench", columns)
        self.assertEqual(MODULE.dataset_group("livecodebench_v7"), "Code")
        for name in ("unknown_new_dataset", "bad/path", "livecodebench_vx"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.select_layout([name])
        for names in (["", "math500"], ["math500", "math500"]):
            with self.assertRaises(ValueError):
                MODULE.select_layout(names)

    def test_mixed_metrics_f1_and_proof_are_not_answer_passk(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp)/"out", Path(temp)/"report"
            for experiment, f1 in (("baseline", 80.0), ("compressed", 65.0)):
                exp = base / f"{experiment}_2"
                make_modern(exp, "gsm_plus")
                target = make_modern(exp, "processbench_gsm8k")
                score_path = target / "metrics.json"
                obj = json.loads(score_path.read_text())
                obj.update(processbench_f1=f1, processbench_error_acc=f1,
                           processbench_correct_acc=f1, processbench_f1_sample_index=0)
                write_json(score_path, obj)
                proof = make_modern(exp, "proofbench")
                metric_path = proof / "metrics.json"
                meta = json.loads(metric_path.read_text())
                meta.update({"pass@1": None, "pass@2": None, "acc": None,
                             "num_scores": 0, "evaluation_status": "proof_grading_required"})
                write_json(metric_path, meta)
            names = ["gsm_plus", "processbench_gsm8k", "proofbench"]
            coverage = MODULE.produce(base, report, ["compressed", "baseline"], 2, [1],
                                      dataset_names=names, baseline="baseline")
            self.assertEqual(len(coverage), 6)
            self.assertEqual({row["score_kind"] for row in coverage},
                             {"answer", "process_f1", "proof_ungraded"})
            self.assertEqual({row["score_status"] for row in coverage if row["dataset"] == "proofbench"},
                             {"NEEDS_GRADING"})
            with (report/"pass_at_1.csv").open() as stream:
                score = next(csv.DictReader(stream))
            self.assertEqual(score["gsm_plus"], "60.00")
            self.assertEqual(score["processbench_gsm8k"], "NA")
            self.assertEqual(score["proofbench"], "NA")
            self.assertEqual(score["AllDatasets"], "60.00")
            self.assertEqual(score["Overall"], "60.00")
            with (report/"processbench_f1.csv").open() as stream:
                pb = next(csv.DictReader(stream))
            self.assertEqual(pb["processbench_gsm8k"], "65.00")
            self.assertEqual(pb["AllDatasets"], "65.00")
            self.assertEqual(pb["gsm_plus"], "NA")
            with (report/"delta_processbench_f1.csv").open() as stream:
                delta = next(csv.DictReader(stream))
            self.assertEqual(delta["processbench_gsm8k"], "-15.00")
            with (report/"proofbench_status.csv").open() as stream:
                proof_rows = list(csv.DictReader(stream))
            self.assertEqual(len(proof_rows), 2)
            self.assertEqual(proof_rows[0]["status"], "NEEDS_GRADING")
            with (report/"mean_completion_tokens.csv").open() as stream:
                length = next(csv.DictReader(stream))
            self.assertEqual(length["AllDatasets"], "15.00")
            self.assertIn("NOT mixed with pass@k", (report/"report.md").read_text())

    def test_changing_dataset_selection_removes_stale_analysis_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp)/"out", Path(temp)/"report"
            for ds in ("processbench_gsm8k", "proofbench", "math500"):
                target = make_modern(base/"model_2", ds)
                path = target/"metrics.json"
                obj = json.loads(path.read_text())
                if ds == "processbench_gsm8k":
                    obj.update(processbench_f1=80.0, processbench_error_acc=80.0,
                               processbench_correct_acc=80.0)
                elif ds == "proofbench":
                    obj.update({"evaluation_status": "proof_grading_required", "pass@1": None, "acc": None})
                write_json(path, obj)
            MODULE.produce(base, report, ["model"], 2, [1],
                           dataset_names=["processbench_gsm8k", "proofbench"])
            self.assertTrue((report/"processbench_f1.csv").exists())
            self.assertTrue((report/"proofbench_status.csv").exists())
            userfile = report/"notes.csv"
            userfile.write_text("user data")
            MODULE.produce(base, report, ["model"], 2, [1], dataset_names=["math500"])
            self.assertFalse((report/"processbench_f1.csv").exists())
            self.assertFalse((report/"proofbench_status.csv").exists())
            self.assertEqual(userfile.read_text(), "user data")
            self.assertEqual(json.loads((report/"analysis_generated_files.json").read_text()).count("report.md"), 1)

    def test_explicit_empty_dataset_selection_errors_instead_of_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            completed = subprocess.run([
                sys.executable, str(ROOT/"analysis_results/analyze.py"),
                "--output-dir", str(folder/"outputs"), "--report-dir", str(folder/"report"),
                "--n", "2", "--ks", "1", "--datasets", "", "some-model"],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("dataset list must be nonempty", completed.stderr)

    def test_processbench_f1_missing_is_not_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp)/"out", Path(temp)/"report"
            target = make_modern(base/"model_2", "processbench_gsm8k")
            obj = json.loads((target/"metrics.json").read_text())
            obj["processbench_f1"] = 70.0
            # Error and correct accuracy are essential to validate F1 source.
            write_json(target/"metrics.json", obj)
            result = subprocess.run(
                [sys.executable, str(ROOT/"analysis_results/analyze.py"),
                 "--output-dir", str(base), "--report-dir", str(report),
                 "--n", "2", "--ks", "1", "--datasets", "processbench_gsm8k",
                 "--allow-unavailable-truncation", "model"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            with (report/"processbench_f1.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["processbench_gsm8k"], "NA")
            with (report/"coverage.csv").open() as stream:
                self.assertEqual(next(csv.DictReader(stream))["score_status"], "PENDING")

    def test_run_script_dataset_csv_matches_eval_active_and_can_override(self):
        eval_src = (ROOT/"sh/mdx/eval.sh").read_text()
        active = [line.split('"')[1] for line in eval_src.splitlines()
                  if line.startswith("DATA_NAMES=")]
        self.assertEqual(len(active), 1)
        run_src = (ROOT/"analysis_results/run.sh").read_text()
        from transfer_datasets import TRANSFER_DATASETS
        self.assertTrue(set(active[0].split(",")).issubset(set(TRANSFER_DATASETS)))
        self.assertIn(active[0], run_src)
        self.assertIn('--datasets "$DATASETS"', run_src)
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            venv = tmp/"venv"
            (venv/"bin").mkdir(parents=True)
            (venv/"bin/python").symlink_to(sys.executable)
            env_file = tmp/".env"
            env_file.write_text(f'EVAL_OUTPUTS="{tmp / "results"}"\n'
                                f'UV_PROJECT_ENVIRONMENT="{venv}"\n')
            report = tmp/"report"
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp),
                   "EVAL_ENV_FILE": str(env_file), "ANALYSIS_REPORT_DIR": str(report),
                   "ANALYSIS_DATASETS": "processbench_gsm8k,proofbench,math500"}
            completed = subprocess.run(["bash", str(ROOT/"analysis_results/run.sh")],
                                       env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 1)  # No source results in fixture.
            with (report/"coverage.csv").open() as stream:
                datasets = {row["dataset"] for row in csv.DictReader(stream)}
            self.assertEqual(datasets, {"processbench_gsm8k", "proofbench", "math500"})

    def test_legacy_unavailable_truncation_is_optional_not_zeros(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "out", Path(temp) / "report"
            target = base / "model_2/gsm_plus"
            write_json(target / "legacy_metrics.json", {"num_samples": 2, "pass@1": 60})
            write_jsonl(target / "legacy.jsonl", [
                {"idx": i, "code": ["42", "42"], "completion_tokens": 200}
                for i in range(2)])
            program = [sys.executable, str(ROOT / "analysis_results/analyze.py"),
                       "--output-dir", str(base), "--report-dir", str(report),
                       "--n", "2", "--ks", "1", "--datasets", "gsm_plus",
                       "--allow-unavailable-truncation", "model"]
            result = subprocess.run(program, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("unavailable", result.stderr)
            with (report / "truncation_rate_pct.csv").open() as stream:
                self.assertEqual(next(csv.DictReader(stream))["gsm_plus"], "NA")

    def test_cli_emits_partial_tables_with_nonzero_status(self):
        with tempfile.TemporaryDirectory() as temp:
            base, report = Path(temp) / "outputs", Path(temp) / "report"
            result = subprocess.run([sys.executable, str(ROOT / "analysis_results/analyze.py"),
                                     "--output-dir", str(base), "--report-dir", str(report),
                                     "--n", "16", "--ks", "1,5,10", "missing-exp"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue((report / "report.md").is_file())
            self.assertIn("INCOMPLETE", result.stderr)
            self.assertIn("NA", (report / "pass_at_1.csv").read_text())


if __name__ == "__main__":
    unittest.main()

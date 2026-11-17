"""Fast, CPU-only tests of scorer discovery, score validation and three-level summary."""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lcb_results", ROOT / "sh/mdx/lcb_results.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def score_custom(path, n=2, pass_values=(0.5, 0.5)):
    rows = [{"question_id": f"q{i}", "code_list": [f"print({j})" for j in range(n)]} for i in range(2)]
    write(path, rows)
    output, ev, all_path = MOD.companions(path)
    write(output, [{**entry, "output_list": entry["code_list"]} for entry in rows])
    write(ev, [{"pass@1": sum(pass_values) / 2, "pass@2": 0.75, "detail": {}}, {"0": [], "1": []}, []])
    write(all_path, [{**entry, "graded_list": [True, False]} for entry in rows])


class LcbReportsTest(unittest.TestCase):
    def test_score_status_complete_changed_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            custom = Path(temp) / "lcb_custom.json"
            score_custom(custom)
            self.assertTrue(MOD.inspect_score(custom, 2)[0])
            self.assertFalse(MOD.inspect_score(custom, 16)[0])
            outputs, _, _ = MOD.companions(custom)
            row = json.loads(outputs.read_text())
            row[0]["code_list"][0] = "not the same code"
            write(outputs, row)
            self.assertFalse(MOD.inspect_score(custom, 2)[0])
            outputs.unlink()
            self.assertFalse(MOD.inspect_score(custom, 2)[0])

    @staticmethod
    def make_generation(folder, n=2, count=2):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "generations.jsonl").write_text("".join(
            json.dumps({"idx": i, "code": ["answer"] * n, "pred": ["answer"] * n}) + "\n"
            for i in range(count)), encoding="utf-8")

    def test_macro_completeness_and_units(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            exp = "exp004"
            root = base / f"{exp}_2"
            for group, datasets in MOD.DATASETS.items():
                for name in datasets:
                    path = root / name
                    if group == "Code":
                        self.make_generation(path / "llmjp4_abc")
                        write(path / "llmjp4_abc/metrics.json", {"num_samples": 2})
                        score_custom(path / "llmjp4_abc/lcb_custom.json")
                    else:
                        self.make_generation(path / "llmjp4_abc")
                        write(path / "llmjp4_abc/metrics.json", {"num_samples": 2, "acc": 50.0,
                             "pass@1": 50.0, "pass@5": 90.0, "pass@10": 100.0})
            rows, partial = MOD.create_report(base, [exp], 2)
            self.assertFalse(partial)
            self.assertEqual(len(rows), 13)
            code = next(x for x in rows if x["level"] == "dataset" and x["category"] == "Code")
            self.assertEqual(code["pass@1"], "50.00")
            overall = next(x for x in rows if x["level"] == "overall")
            self.assertEqual(overall["pass@1"], "50.00")
            self.assertEqual(overall["acc"], "NA")  # Code pass@1 is not a legacy acc field
            (root / "math500/llmjp4_abc/metrics.json").unlink()
            rows, partial = MOD.create_report(base, [exp], 2)
            self.assertTrue(partial)
            self.assertEqual(rows[-1]["pass@1"], "NA")
            self.assertEqual(rows[-1]["status"], "PARTIAL")

    def test_flat_lcb_directory_is_read_by_official_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = Path(temp) / "exp_2/livecodebench_v6"
            self.make_generation(ds)
            write(ds / "metrics.json", {"num_samples": 2})
            score_custom(ds / "lcb_custom.json")
            obj, status = MOD.read_dataset_metric(Path(temp) / "exp_2", "livecodebench_v6", 2)
            self.assertEqual(status, "OK")
            self.assertEqual(obj["pass@1"], 50.0)

    def test_select_custom_agrees_with_analysis_before_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2/livecodebench_v6"
            for name, count in (("llmjp4_old", 1), ("llmjp4_new", 2)):
                folder = ds / name
                self.make_generation(folder, count=count)
                write(folder / "metrics.json", {"num_samples": 2})
                write(folder / "lcb_custom.json", [{"question_id": "q1", "code_list": ["pass", "pass"]}])
            chosen = MOD.select_custom_input(ds, 2)
            self.assertEqual(chosen.parent.name, "llmjp4_new")
            result = subprocess.run([sys.executable, str(ROOT / "sh/mdx/lcb_results.py"),
                                     "select-custom", str(ds), "--n", "2"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(chosen))

    def test_phi_legacy_metric_names_and_irrelevant_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "exp_16"
            write(root / "math500/test_phi_16_metrics.json", {"num_samples": 500, "acc": 20,
                "pass@1": 23.5, "pass@5": 35})
            obj, state = MOD.read_dataset_metric(root, "math500", 16)
            self.assertEqual((state, obj["pass@1"]), ("OK", 23.5))
            write(root / "math500/new_run/metrics.json", {"num_samples": 500, "pass@1": 80})
            # Arbitrary unrelated folders cannot obscure the original scorer.
            self.assertEqual(MOD.read_dataset_metric(root, "math500", 16)[1], "OK")

    def test_scoring_shell_missing_then_skip_complete(self):
        # The mock runner writes official-shaped JSON; it never executes code.
        runner_source = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--custom_output_file')
args, _ = p.parse_known_args()
custom = Path(args.custom_output_file)
rows = json.loads(custom.read_text())
output = custom.with_name(custom.stem + '_codegeneration_output.json')
output.write_text(json.dumps([{**r, 'output_list': r['code_list']} for r in rows]))
output.with_name(output.stem + '_eval.json').write_text(json.dumps([{'pass@1': .5}, {}, []]))
output.with_name(output.stem + '_eval_all.json').write_text(json.dumps([
    {**r, 'graded_list': [True, False]} for r in rows]))
'''
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            lcb = tmp / "LiveCodeBench"
            runner = lcb / "lcb_runner/runner"
            runner.mkdir(parents=True)
            (runner.parent / "__init__.py").write_text("")
            (runner / "__init__.py").write_text("")
            (runner / "custom_evaluator.py").write_text(runner_source)
            venv = tmp / "venv" / "bin"
            venv.mkdir(parents=True)
            (venv / "python").symlink_to(sys.executable)
            out = tmp / "outputs"
            # The user edits EXP_NAMES in the shell; the mock must reflect
            # the currently selected experiments, not a fixed list from an
            # earlier revision of that file.
            script = ROOT / "sh/mdx/score_livecodebench.sh"
            exp_block = script.read_text(encoding="utf-8").split("EXP_NAMES=(", 1)[1].split("\n)", 1)[0]
            names = re.findall(r'^\s*"([^"\n]+)"', exp_block, re.MULTILINE)
            self.assertTrue(names, "No active experiment names in scoring shell")
            self.assertEqual(len(names), len(set(names)), "Duplicate EXP_NAMES")
            for exp in names:
                custom = out / f"{exp}_2/livecodebench_v6/llmjp4_hash/lcb_custom.json"
                write(custom, [{"question_id": "q1", "code_list": ["pass", "pass"]},
                               {"question_id": "q2", "code_list": ["pass", "pass"]}])
            env_file = tmp / "envfile"
            env_file.write_text(f'EVAL_OUTPUTS="{out}"\nLCB_ROOT="{lcb}"\nUV_PROJECT_ENVIRONMENT="{venv.parent}"\n')
            # Do not inherit real benchmark settings, model paths, or tokens.
            # In particular LCB_FORCE_RESCORE=1 would invalidate the skip test.
            env = {
                "HOME": str(tmp),
                "LANG": "C.UTF-8",
                "PATH": os.environ.get("PATH", ""),
                "EVAL_ENV_FILE": str(env_file),
                "LCB_N": "2",
                "LCB_RELEASE_VERSION": "v6",
                "LCB_FORCE_RESCORE": "0",
                "LCB_ACK_UNTRUSTED_CODE": "1",
            }
            first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            self.assertIn(f"completed={len(names)}", first.stdout)
            second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            self.assertIn(f"skipped={len(names)}", second.stdout)

    def test_summary_shell_missing_experiment_does_not_succeed(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            result = subprocess.run([sys.executable, str(ROOT / "sh/mdx/lcb_results.py"),
                "summary", "--output-dir", str(out), "--n", "16", "--csv", str(Path(temp) / "r.csv"),
                "missing_exp"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("INCOMPLETE", result.stderr)


if __name__ == "__main__":
    unittest.main()

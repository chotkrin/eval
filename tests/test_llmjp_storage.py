"""CPU tests for hash-free migration and actual interrupted-job resumption."""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import fcntl
import subprocess
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llmjp_storage import candidates, choose, migrate_one, normalize_config
from llmjp_profile import read_checkpoint


CFG = dict(protocol="llmjp4-harmony-final-v2", model="/tmp/model", dataset="math500",
           split="test", num_test_sample=-1, start=0, end=-1,
           seed=42, temperature=0.6, top_p=0.95, top_k=-1, n_sampling=2,
           max_tokens_per_call=64, max_model_len=128, reasoning_effort="high")


def seed_run(dataset, name, count, *, total=None, scored=False, updated_ns=None):
    folder = dataset / name
    folder.mkdir(parents=True)
    (folder / "protocol.json").write_text(json.dumps(CFG))
    with (folder / "generations.jsonl").open("w") as out:
        for i in range(count):
            out.write(json.dumps({"idx": i, "question": f"q{i}", "gt": str(i),
                                  "code": ["a", "b"], "pred": ["a", "b"],
                                  "completion_tokens": 6,
                                  "completion_tokens_per_sample": [3, 3]}) + "\n")
    if total is not None:
        (folder / "metrics.json").write_text(json.dumps({"num_samples": total,
            "pass@1": 60 if scored else None, "output_dir": str(folder)}))
    if updated_ns is not None:
        for path in folder.iterdir():
            os.utime(path, ns=(updated_ns, updated_ns))
    return folder


class StorageTests(unittest.TestCase):
    def test_partial_prefers_more_questions_and_archives_displaced_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "experiment_2" / "math500"
            seed_run(dataset, "llmjp4_old", 1)
            seed_run(dataset, "llmjp4_new", 3)
            self.assertEqual(choose(candidates(dataset, 2)).questions, 3)
            preview = migrate_one(dataset, 2)
            self.assertEqual(preview["status"], "preview")
            self.assertTrue((dataset / "llmjp4_new").exists())
            result = migrate_one(dataset, 2, apply=True)
            self.assertEqual(result["status"], "migrated")
            self.assertEqual(read_checkpoint(dataset / "generations.jsonl", 2).__len__(), 3)
            self.assertFalse(list(dataset.glob("llmjp4_*")))
            self.assertTrue((dataset.parent / ".llmjp_archives/math500/llmjp4_old").exists())

    def test_both_complete_choose_newest_and_keep_official_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "experiment_2" / "math500"
            old = seed_run(dataset, "llmjp4_first", 2, total=2, scored=True, updated_ns=1_000_000_000)
            new = seed_run(dataset, "llmjp4_second", 2, total=2, scored=True, updated_ns=2_000_000_000)
            self.assertEqual(choose(candidates(dataset, 2)).directory, new)
            migrate_one(dataset, 2, apply=True)
            self.assertTrue((dataset / "metrics.json").exists())
            self.assertEqual(json.loads((dataset / "metrics.json").read_text())["output_dir"], str(dataset))
            self.assertTrue((dataset.parent / ".llmjp_archives/math500/llmjp4_first").exists())

    def test_scored_beats_unscored_and_different_configs_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "math500"
            old = seed_run(ds, "llmjp4_scored", 2, total=2, scored=True)
            seed_run(ds, "llmjp4_unscored", 2, total=2, updated_ns=4_000_000_000)
            self.assertEqual(choose(candidates(ds, 2)).directory, old)
            wrong = dict(CFG, temperature=0.2)
            self.assertNotEqual(normalize_config(wrong), normalize_config(CFG))

    def test_cli_refuses_to_migrate_while_experiment_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "results"
            root = base / "exp_2"
            dataset = root / "math500"
            seed_run(dataset, "llmjp4_old", 2, total=2, scored=True)
            lock_path = root / ".evaluation.lock"
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                cmd = [sys.executable, str(ROOT / "analysis_results/migrate.py"),
                       "--output-dir", str(base), "--n", "2", "--apply", "exp"]
                blocked = subprocess.run(cmd, text=True, capture_output=True)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn("still running", blocked.stderr)
                self.assertTrue((dataset / "llmjp4_old").exists())
                fcntl.flock(lock, fcntl.LOCK_UN)
            success = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertTrue((dataset / "generations.jsonl").is_file())

    def test_already_migrated_llmjp4_flattens_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "exp_2" / "math500"
            old = seed_run(dataset, "llmjp4", 2, total=2, scored=True)
            (dataset / "phi_metrics.json").write_text("unrelated")
            (old / "scored.jsonl").write_text("score\n")
            preview = migrate_one(dataset, 2)
            self.assertEqual(preview["chosen"], str(old))
            self.assertTrue(old.is_dir(), "dry-run cannot move files")
            result = migrate_one(dataset, 2, apply=True)
            self.assertEqual(result["target"], str(dataset))
            self.assertEqual((dataset / "scored.jsonl").read_text(), "score\n")
            self.assertEqual((dataset / "phi_metrics.json").read_text(), "unrelated")
            self.assertFalse(old.exists())
            self.assertEqual(migrate_one(dataset, 2, apply=True)["status"], "already_flat")
            self.assertEqual(len(candidates(dataset, 2)), 1)

    def test_both_root_and_hashed_pick_newest_completed_and_archive_loser(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "math500"
            seeded = seed_run(ds, "llm-jp4-oldhash", 2, total=2, scored=True)
            ds.mkdir(exist_ok=True)
            for entry in seeded.iterdir():
                (ds / entry.name).write_bytes(entry.read_bytes())
            (ds / "metrics.json").write_text(json.dumps({"num_samples": 2, "pass@1": 10}))
            # Old hashed run is newer, so replace flat active files by it.
            for file in ds.iterdir():
                if file.is_file():
                    os.utime(file, ns=(1_000_000_000, 1_000_000_000))
            for file in seeded.iterdir():
                os.utime(file, ns=(2_000_000_000, 2_000_000_000))
            migrate_one(ds, 2, apply=True)
            self.assertFalse(seeded.exists())
            self.assertEqual(json.loads((ds / "metrics.json").read_text())["pass@1"], 60)
            archived = ds.parent / ".llmjp_archives/math500/dataset_root/metrics.json"
            self.assertEqual(json.loads(archived.read_text())["pass@1"], 10)
            self.assertEqual(json.loads((ds / "metrics.json").read_text())["output_dir"], str(ds))

    def test_completed_livecodebench_artifacts_flatten_and_keep_score_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "livecodebench_v6"
            old = seed_run(ds, "llmjp4", 2, total=2)
            rows = [{"question_id": f"q{i}", "code_list": ["print(0)", "print(1)"]}
                    for i in range(2)]
            (old / "lcb_custom.json").write_text(json.dumps(rows))
            (old / "lcb_custom_codegeneration_output.json").write_text(json.dumps(rows))
            (old / "lcb_custom_codegeneration_output_eval.json").write_text(
                json.dumps([{"pass@1": 0.5}, {}, {}]))
            (old / "lcb_custom_codegeneration_output_eval_all.json").write_text(
                json.dumps([{**row, "graded_list": [True, False]} for row in rows]))
            metrics = json.loads((old / "metrics.json").read_text())
            metrics["custom_output_file"] = str(old / "lcb_custom.json")
            (old / "metrics.json").write_text(json.dumps(metrics))
            self.assertTrue(choose(candidates(ds, 2)).fully_scored)
            migrate_one(ds, 2, apply=True)
            self.assertFalse(old.exists())
            for name in (
                "lcb_custom.json", "lcb_custom_codegeneration_output.json",
                "lcb_custom_codegeneration_output_eval.json",
                "lcb_custom_codegeneration_output_eval_all.json",
            ):
                self.assertTrue((ds / name).is_file())
            new_metrics = json.loads((ds / "metrics.json").read_text())
            self.assertEqual(new_metrics["custom_output_file"], str(ds / "lcb_custom.json"))

    def test_migration_will_not_overwrite_unrelated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "math500"
            run = seed_run(ds, "llmjp4", 2, total=2, scored=True)
            (run / "notes.txt").write_text("run-notes")
            (ds / "notes.txt").write_text("preserve me")
            with self.assertRaises(FileExistsError):
                migrate_one(ds, 2, apply=True)
            self.assertTrue(run.is_dir())
            self.assertEqual((ds / "notes.txt").read_text(), "preserve me")

    def test_root_plus_partial_subfolder_keeps_completed_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "math500"
            complete = seed_run(ds, "llmjp4", 2, total=2, scored=True)
            for item in complete.iterdir():
                (ds / item.name).write_bytes(item.read_bytes())
            # Emulate a real run: write metrics after generations, not before.
            (ds / "metrics.json").write_bytes((complete / "metrics.json").read_bytes())
            partial = seed_run(ds, "llmjp4_hash", 1)
            # Make the ordering deterministic on filesystems with different
            # timestamp granularities. Both root and llmjp4 are fully scored:
            # the newer root must beat the older complete copy, while even a
            # newer partial run must not displace either complete result.
            for folder, stamp in ((complete, 1_000_000_000),
                                  (ds, 2_000_000_000),
                                  (partial, 3_000_000_000)):
                for item in folder.iterdir():
                    if item.is_file():
                        os.utime(item, ns=(stamp, stamp))
            self.assertEqual(choose(candidates(ds, 2)).directory, ds)
            result = migrate_one(ds, 2, apply=True)
            self.assertEqual(result["status"], "already_flat")
            self.assertFalse(partial.exists())
            self.assertTrue((ds.parent / ".llmjp_archives/math500/llmjp4_hash").is_dir())

    def test_corrupt_lcb_score_does_not_beat_valid_older_scored_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "livecodebench_v6"
            good = seed_run(ds, "llmjp4_good", 2, total=2)
            bad = seed_run(ds, "llmjp4_bad", 2, total=2)
            rows = [{"question_id": f"q{i}", "code_list": ["print(0)", "print(1)"]}
                    for i in range(2)]
            for folder in (good, bad):
                (folder / "lcb_custom.json").write_text(json.dumps(rows))
                (folder / "lcb_custom_codegeneration_output.json").write_text(json.dumps(rows))
                (folder / "lcb_custom_codegeneration_output_eval.json").write_text(
                    json.dumps([{"pass@1": .5}, {}, {}]))
                (folder / "lcb_custom_codegeneration_output_eval_all.json").write_text(
                    json.dumps([{**r, "graded_list": [True, False]} for r in rows]))
            # Stale/broken scored data must not be treated as completed scoring.
            (bad / "lcb_custom_codegeneration_output_eval.json").write_text("{}")
            self.assertTrue(next(r for r in candidates(ds, 2) if r.directory == good).fully_scored)
            self.assertFalse(next(r for r in candidates(ds, 2) if r.directory == bad).fully_scored)
            self.assertEqual(choose(candidates(ds, 2)).directory, good)
            migrate_one(ds, 2, apply=True)
            self.assertEqual(json.loads((ds / "lcb_custom_codegeneration_output_eval.json").read_text())[0]["pass@1"], .5)

    def test_stale_math_metrics_are_not_complete_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "math500"
            run = seed_run(ds, "llmjp4_old", 2, total=2, scored=True)
            gen = run / "generations.jsonl"
            metric = run / "metrics.json"
            os.utime(metric, ns=(1_000_000_000, 1_000_000_000))
            os.utime(gen, ns=(2_000_000_000, 2_000_000_000))
            self.assertFalse(choose(candidates(ds, 2)).fully_scored)
            from sh.mdx.lcb_results import read_dataset_metric
            self.assertEqual(read_dataset_metric(ds.parent, "math500", 2)[1], "PENDING")

    def test_migration_does_not_make_stale_metrics_look_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "exp_2" / "math500"
            old = seed_run(ds, "llmjp4", 2, total=2, scored=True)
            metrics = old / "metrics.json"
            generation = old / "generations.jsonl"
            os.utime(metrics, ns=(1_000_000_000, 1_000_000_000))
            os.utime(generation, ns=(2_000_000_000, 2_000_000_000))
            migrate_one(ds, 2, apply=True)
            self.assertEqual((ds / "metrics.json").stat().st_mtime_ns, 1_000_000_000)
            self.assertEqual((ds / "generations.jsonl").stat().st_mtime_ns, 2_000_000_000)
            from sh.mdx.lcb_results import read_dataset_metric
            self.assertEqual(read_dataset_metric(ds.parent, "math500", 2)[1], "PENDING")

    def test_checkpoint_must_have_candidate_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generations.jsonl"
            path.write_text(json.dumps({"idx": 1, "code": "ab", "pred": "ab"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid/incomplete question checkpoint"):
                read_checkpoint(path, 2)

    def test_torn_checkpoint_tail_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "gen.jsonl"
            ds.write_bytes(json.dumps({"idx": 0, "code": ["a", "b"], "pred": ["a", "b"]}).encode() + b"\n{\"idx\":")
            self.assertEqual(set(read_checkpoint(ds, 2)), {0})
            self.assertTrue(ds.read_bytes().endswith(b"\n"))

    def test_real_evaluator_resumes_missing_questions_without_duplicates(self):
        # Load the actual llmjp_eval module with lightweight stand-ins for its
        # data/scoring/vLLM dependencies; no GPU or model download needed.
        def module(name, **attrs):
            obj = ModuleType(name)
            obj.__dict__.update(attrs)
            return obj
        calls = []
        fail_after = [1]

        class FakeLLM:
            def generate(self, prompts, sampling, use_tqdm=False):
                calls.append(len(prompts))
                if len(calls) > fail_after[0]:
                    raise RuntimeError("simulated Slurm interruption")
                return [SimpleNamespace(prompt=prompt, outputs=[
                    SimpleNamespace(index=i, token_ids=[1, 2, 3], finish_reason="stop")
                    for i in range(sampling.n)]) for prompt in prompts]

        fake_modules = {
            "data_loader": module("data_loader", load_data=lambda name, split, data_dir: [
                {"idx": idx, "problem": f"q{idx}", "answer": str(idx)} for idx in range(3)]),
            "evaluate": module("evaluate", evaluate=lambda samples, data_name, prompt_type, execute:
                               (samples, {"num_samples": len(samples), "pass@1": 50.0})),
            "livecodebench_adapter": module("livecodebench_adapter", export_custom_outputs=lambda *a: None,
                                            extract_python_program=lambda x: x),
            "parser": module("parser", parse_question=lambda x, _: x["problem"],
                             parse_ground_truth=lambda x, _: ("", x["answer"]),
                             run_execute=lambda _, final, *a: (final, None),
                             choice_answer_clean=lambda x: x),
            "vllm": module("vllm", SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
            "tqdm": module("tqdm", tqdm=lambda x, **kwargs: x),
            "utils": module("utils", save_jsonl=lambda rows, path: Path(path).write_text(
                "".join(json.dumps(row) + "\n" for row in rows))),
        }
        with patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location("llmjp_eval_test_instance", ROOT / "llmjp_eval.py")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            evaluator.make_prompt = lambda tokenizer, q, ds, effort: q
            evaluator.decode_response = lambda tok, ids: {"has_final": True, "final": "1",
                                                               "reasoning": "", "raw_harmony": ""}
            tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1] * 3,
                                  convert_tokens_to_ids=lambda text: 99, unk_token_id=0)
            with tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp) / "model"
                model.mkdir()
                args = SimpleNamespace(output_dir=str(Path(tmp) / "out"), model_name_or_path=str(model),
                    data_dir=str(Path(tmp) / "data"), split="test", num_test_sample=-1, start=0, end=-1,
                    seed=42, temperature=0.6, top_p=0.95, top_k=-1, n_sampling=2,
                    max_tokens_per_call=64, max_model_len=128, reasoning_effort="high",
                    question_batch_size=1, overwrite=False)
                with self.assertRaisesRegex(RuntimeError, "simulated Slurm interruption"):
                    evaluator.eval_one_dataset(args, FakeLLM(), tok, "math500", {})
                outfile = Path(tmp) / "out/math500/generations.jsonl"
                self.assertEqual(len(read_checkpoint(outfile, 2)), 1)
                with outfile.open("ab") as stream:
                    stream.write(b'{"idx":')
                fail_after[0] = 100
                calls.clear()
                result = evaluator.eval_one_dataset(args, FakeLLM(), tok, "math500", {})
                self.assertEqual(result["num_samples"], 3)
                self.assertEqual(calls, [1, 1])
                self.assertEqual(len(read_checkpoint(outfile, 2)), 3)
                calls.clear()
                evaluator.eval_one_dataset(args, FakeLLM(), tok, "math500", {})
                self.assertEqual(calls, [])
                # A stale score must be recomputed without regenerating tokens.
                metrics_file = outfile.with_name("metrics.json")
                os.utime(metrics_file, ns=(1_000_000_000, 1_000_000_000))
                calls.clear()
                evaluator.eval_one_dataset(args, FakeLLM(), tok, "math500", {})
                self.assertEqual(calls, [])
                self.assertGreaterEqual(metrics_file.stat().st_mtime_ns,
                                        outfile.stat().st_mtime_ns)
                # A changed generation parameter must never reuse these records.
                args.temperature = 0.2
                with self.assertRaisesRegex(ValueError, "different sampling/model settings"):
                    evaluator.eval_one_dataset(args, FakeLLM(), tok, "math500", {})
                self.assertEqual(len(read_checkpoint(outfile, 2)), 3)
                # --overwrite archives the earlier run and starts clean.
                args.overwrite = True
                calls.clear()
                evaluator.eval_one_dataset(args, FakeLLM(), tok, "math500", {})
                self.assertEqual(calls, [1, 1, 1])
                self.assertTrue(list((Path(tmp) / "out/.llmjp_archives/math500").glob("dataset_root*")))


if __name__ == "__main__":
    unittest.main()

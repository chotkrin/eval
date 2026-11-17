"""Offline regression tests for extended evaluation-only datasets."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_datasets import TRANSFER_DATASETS, normalize, prepare
from llmjp_profile import make_user_prompt


class ExtraBenchmarkSupportTest(unittest.TestCase):
    def test_all_official_variants_are_registered(self):
        for name in ("gsm_plus", "gsm_symbolic_main", "gsm_symbolic_p1",
                     "gsm_symbolic_p2", "math_perturb_simple", "math_perturb_hard",
                     "omega_exploratory", "omega_compositional", "omega_transformative",
                     "processbench_gsm8k", "processbench_math",
                     "processbench_olympiadbench", "proofbench"):
            self.assertIn(name, TRANSFER_DATASETS)

    def test_processbench_negative_one_and_first_error(self):
        for subset in ("gsm8k", "math", "olympiadbench"):
            row = {"problem": "Check", "steps": ["correct", "mistake"], "label": 1}
            result = normalize("processbench_" + subset, row)
            self.assertEqual(result["gt"], "1")
            self.assertIn("<paragraph_0>", result["question"])
            self.assertIn("<paragraph_1>", result["question"])
            self.assertIn("\\boxed{...}", result["question"])
            self.assertEqual(result["type"], subset)
            row["label"] = -1
            self.assertEqual(normalize("processbench_" + subset, row)["gt"], "-1")
            row["label"] = 2
            with self.assertRaisesRegex(ValueError, "invalid first-error"):
                normalize("processbench_" + subset, row)

    def test_proofbench_not_forced_into_boxed_answer(self):
        p = make_user_prompt("Prove P", "proofbench")
        self.assertIn("complete natural-language", p)
        self.assertNotIn("\\boxed", p)

    def test_local_math_perturb_files_in_existing_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data" / "eval"
            raw = Path(tmp) / "data" / "raw" / "MATH-Perturb" / "math_perturb"
            raw.mkdir(parents=True)
            for variant in ("simple", "hard"):
                (raw / f"math_perturb_{variant}.jsonl").write_text(json.dumps({
                    "problem_id": "p1", "problem": "Find x", "answer": "2",
                    "level": "Level 5", "type": "Algebra"}) + "\n")
                rows = prepare(f"math_perturb_{variant}", data)
                self.assertEqual(rows[0]["gt"], "2")
                self.assertTrue((data / f"math_perturb_{variant}" / "test.jsonl").is_file())

    def test_omega_requires_verified_export_not_fake_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "validated normalized raw export"):
                prepare("omega_exploratory", Path(tmp) / "data" / "eval")

    def test_shell_syntax(self):
        for name in ("download_extra_datasets.sh", "eval_llmjp.sh", "eval_phi.sh", "eval.sh"):
            p = ROOT / "sh" / "mdx" / name
            self.assertEqual(subprocess.run(["bash", "-n", str(p)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()

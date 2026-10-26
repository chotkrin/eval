"""CPU-only dry-run of all MDX submitters against a fake Slurm CLI."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMITTERS = {
    "sh/mdx/eval_llmjp.sh": "llmjp",
    "sh/mdx/eval_phi.sh": "phi",
    "sh/mdx/eval.sh": "dsdt",
}


def active_experiments(script):
    fragment = script.split("EXP_NAMES=(", 1)[1].split("\n)", 1)[0]
    return [
        match.group(1)
        for line in fragment.splitlines()
        if not line.lstrip().startswith("#")
        if (match := re.search(r'"([^"]+)"', line))
    ]


class SharedUvSubmitterTest(unittest.TestCase):
    def test_three_model_profiles_use_one_python_and_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            storage = tmp / "storage"
            models = storage / "models"
            models.mkdir(parents=True)
            for dataset in ("gpqa_diamond", "mmlu_stem", "livecodebench_v6"):
                cache = storage / "data" / "eval" / dataset / "test.jsonl"
                cache.parent.mkdir(parents=True)
                cache.write_text("[]\n", encoding="utf-8")
            for script in SUBMITTERS:
                for name in active_experiments((ROOT / script).read_text()):
                    (models / name).mkdir(exist_ok=True)

            venv = tmp / "unified" / ".venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").symlink_to(sys.executable)
            private = tmp / "private.env"
            private.write_text(
                f'DATA_HOME="{storage}"\n'
                f'UV_PROJECT_ENVIRONMENT="{venv}"\n'
                'LEGACY_EVAL_PYTHON="/not-a-real-conda/bin/python"\n'
                'CUDA_MODULE_LEGACY="cuda/12.8/12.8.2"\n'
                'MDX_JOB_PREFIX="0001_eval"\n', encoding="utf-8"
            )
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            sbatch = bin_dir / "sbatch"
            sbatch.write_text(
                f'#!/bin/bash\nprintf "%s\\n" "$@" >> "{tmp}/sbatch.txt"\necho 123456\n',
                encoding="utf-8",
            )
            sbatch.chmod(0o755)
            env = dict(os.environ, EVAL_ENV_FILE=str(private))
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            for script in SUBMITTERS:
                with self.subTest(script=script):
                    result = subprocess.run(
                        ["bash", str(ROOT / script)], cwd=ROOT, env=env,
                        check=False, text=True, capture_output=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

            commands = (tmp / "sbatch.txt").read_text(encoding="utf-8")
            for profile in SUBMITTERS.values():
                self.assertIn(f"EVAL_PROFILE={profile}", commands)
            self.assertIn(f"EVAL_PYTHON={venv}/bin/python", commands)
            self.assertNotIn("/not-a-real-conda", commands)
            self.assertIn("CUDA_MODULE=cuda/13.0/13.0.3", commands)
            self.assertNotIn("CUDA_MODULE=cuda/12.8/12.8.2", commands)
            self.assertIn(f"MATH_EVAL_SCRIPT={ROOT}/math_eval.py", commands)
            self.assertIn(f"MATH_EVAL_SCRIPT={ROOT}/llmjp_eval.py", commands)

    def test_spooled_slurm_worker_resolves_checkout_from_export(self):
        # Slurm executes a *copy* of the batch script under /var/spool.
        # BASH_SOURCE[0] must never be used to derive the project root here.
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            spool = tmp / "var" / "spool" / "slurm_script"
            spool.parent.mkdir(parents=True)
            shutil.copy2(ROOT / "sh/mdx/_eval_mdx_common.sh", spool)

            data_home = tmp / "workspace"
            models = data_home / "models"
            models.mkdir(parents=True)
            venv = tmp / "uv" / ".venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True)
            fake_python = bin_dir / "python"
            # No GPU, model or dependency is needed: only the launch plumbing.
            fake_python.write_text(
                '#!/bin/sh\n'
                f'printf "%s\\n" "$@" >> "{tmp}/executed.txt"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_bin = tmp / "bin"
            fake_bin.mkdir()
            for tool in ("module", "nvidia-smi"):
                command = fake_bin / tool
                command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                command.chmod(0o755)
            env_file = tmp / "private.env"
            env_file.write_text(
                f'DATA_HOME="{data_home}"\n'
                f'UV_PROJECT_ENVIRONMENT="{venv}"\n'
                'MAX_TASK_RETRIES="0"\n', encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({
                "PROJECT_ROOT": str(ROOT),
                "EVAL_ENV_FILE": str(env_file),
                "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
            })
            for profile, evaluator in (
                ("llmjp", "llmjp_eval.py"),
                ("phi", "math_eval.py"),
                ("dsdt", "math_eval.py"),
            ):
                exp_name = f"spool_test_{profile}"
                (models / exp_name).mkdir()
                process = subprocess.run(
                    ["bash", str(spool), "__worker", "--exp", exp_name,
                     "--data-names", "math500", "--sample-num", "1"],
                    env=dict(env, EVAL_PROFILE=profile),
                    cwd=tmp, text=True, capture_output=True, timeout=15,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn(str(ROOT / evaluator),
                              (tmp / "executed.txt").read_text())
                self.assertNotIn("/var/spool/llmjp_eval.py", process.stdout)

            bad_env = dict(env, EVAL_PROFILE="llmjp")
            bad_env.pop("PROJECT_ROOT", None)
            process = subprocess.run(
                ["bash", str(spool), "__worker", "--exp", "spool_test_llmjp",
                 "--data-names", "math500"],
                env=bad_env, cwd=tmp, text=True, capture_output=True,
                timeout=15,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("requires an absolute PROJECT_ROOT", process.stderr)


if __name__ == "__main__":
    unittest.main()

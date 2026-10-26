"""Validate privacy-safe migration from a previously generated .env."""

import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "sh/mdx/migrate_env.py"
spec = importlib.util.spec_from_file_location("eval_env_migration", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EnvMigrationTest(unittest.TestCase):
    def test_preserves_paths_and_secrets_but_removes_old_interpreters(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            text = (
                '# keep this comment\n'
                'DATA_HOME="/data/test/verl"\n'
                'MODEL_ROOT="${DATA_HOME}/models"\n'
                'UV_PROJECT_ENVIRONMENT="${DATA_HOME}/envs/llmjp-eval/.venv"\n'
                'HF_TOKEN="local-test-only"\n'
                'LEGACY_EVAL_PYTHON="/home/old/miniconda/bin/python"\n'
                'RAY_PYTHON="/home/old/ray/bin/python"\n'
                'RAY_CLI="/home/old/ray/bin/ray"\n'
                'CUDA_MODULE_LEGACY="cuda/12.8/12.8.2"\n'
                'CUDA_MODULE_LLMJP="cuda/13.0/13.0.3"\n'
            )
            env.write_text(text)
            changed, backup = module.migrate_env(env)
            self.assertTrue(changed)
            self.assertIsNotNone(backup)
            new = env.read_text()
            self.assertIn('MODEL_ROOT="${DATA_HOME}/models"', new)
            self.assertIn('HF_TOKEN="local-test-only"', new)
            self.assertIn('UV_PROJECT_ENVIRONMENT="${DATA_HOME}/envs/llmjp-eval/.venv"', new)
            self.assertIn('EVAL_CUDA_MODULE="cuda/13.0/13.0.3"', new)
            for obsolete in (*module.OLD_KEYS, "CUDA_MODULE_LLMJP"):
                self.assertNotIn(obsolete + "=", new)
            self.assertEqual(Path(backup).read_text(), text)
            self.assertEqual(stat.S_IMODE(env.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(Path(backup).stat().st_mode), 0o600)
            again, another_backup = module.migrate_env(env)
            self.assertFalse(again)
            self.assertIsNone(another_backup)

    def test_existing_shared_cuda_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text('EVAL_CUDA_MODULE="custom"\nCUDA_MODULE_LLMJP="old"\n')
            changed, _ = module.migrate_env(env)
            self.assertTrue(changed)
            self.assertEqual(env.read_text(), 'EVAL_CUDA_MODULE="custom"\n')


if __name__ == "__main__":
    unittest.main()

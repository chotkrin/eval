"""Regression checks for the MDX submission walltime and LCB SDK pin.

Separate from test_shared_uv_entrypoints.py so that local edits to EXP_NAMES
or prior test patches do not interfere with this change.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class MDXWalltimeTest(unittest.TestCase):
    def test_sbatch_time_does_not_use_stale_private_env(self):
        src = (ROOT / "sh/mdx/_eval_mdx_common.sh").read_text(encoding="utf-8")
        self.assertIn('        --time=0\n', src)
        self.assertNotIn('        --time="${MDX_WALLTIME}"', src)
        self.assertNotIn('MDX_WALLTIME="${MDX_WALLTIME:-20:00:00}"', src)

    def test_shared_environment_pins_legacy_lcb_sdk(self):
        import tomllib

        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("anthropic==0.75.0", project["project"]["dependencies"])


if __name__ == "__main__":
    unittest.main()

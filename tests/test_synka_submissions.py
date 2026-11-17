"""CPU-only contract tests for SYNKA resource boundaries and submit paths."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class SynkaSubmitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for cmd, body in {
            "hostname": 'echo cpusv06',
            "scontrol": 'if [[ "$1" == show && "$2" == hostnames ]]; then if [[ "$3" == *07* ]]; then printf "gpusv%02d\\n" {1..7}; else printf "gpusv%02d\\n" {1..8}; fi; else exit 1; fi',
            "sinfo": 'printf "gpusv%02d\\n" {1..8}',
            "sbatch": 'printf "%s\\n" "$@" >> "$SYNKA_CAPTURE"; echo 12345',
        }.items():
            f = self.bin / cmd
            f.write_text("#!/usr/bin/env bash\n" + body + "\n")
            f.chmod(0o755)
        self.venv = self.root / 'venv'
        (self.venv/'bin').mkdir(parents=True)
        (self.venv/'bin'/'python').symlink_to(sys.executable)
        (self.root/'models'/'example').mkdir(parents=True)
        (self.root/'data'/'math500').mkdir(parents=True)
        (self.root/'data'/'math500'/'test.jsonl').write_text('{"idx":0}\n')
        self.capture = self.root/'captured'
        self.config = self.root/'env_synka'
        self.config.write_text('\n'.join([
            f'VERL_HOME="{self.root}"',
            f'DATA_DIR="{self.root}/data"',
            f'MODEL_ROOT="{self.root}/models"',
            f'EVAL_OUTPUTS="{self.root}/outputs"',
            f'UV_PROJECT_ENVIRONMENT="{self.venv}"',
            'SYNKA_PROJECT_ID="project-test"',
            'SYNKA_PARTITION="project-test"',
            'SYNKA_NODELIST="gpusv[01-08]"',
            'SYNKA_CUDA_MODULE="cuda/12.8"',
            '',
        ]))

    def submit(self, profile='llmjp', overrides=None, gpus=8):
        env = os.environ.copy()
        env.update({'PATH': str(self.bin)+os.pathsep+env.get('PATH',''),
                    'EVAL_ENV_FILE':str(self.config), 'SYNKA_CAPTURE':str(self.capture)})
        if overrides:
            env.update(overrides)
        script = '\n'.join([
            f'export EVAL_PROFILE={profile}',
            'source sh/synka/_eval_synka_common.sh',
            'EXP_NAMES=(example)',
            'DATA_NAMES="math500"; SAMPLE_NUM=16; SPLIT=test; NUM_TEST_SAMPLE=-1',
            'PROMPT_TYPE=llmjp4-harmony; MAX_TOKENS_PER_CALL=8; SEED=42',
            'TEMPERATURE=0.6; TOP_P=0.95; TOP_K=-1; START=0; END=-1',
            f'PIPELINE_PARALLEL_SIZE=1; GPUS_PER_EXP={gpus}; CPUS_PER_EXP=64',
            'synka_submit_experiments demo',
        ])
        return subprocess.run(['bash','-c',script], cwd=ROOT, env=env,
                              capture_output=True, text=True)

    def test_three_profiles_use_synka_resources_not_mdx(self):
        for profile, expected in [('llmjp','llmjp_eval.py'), ('phi','math_eval.py'), ('dsdt','math_eval.py')]:
            with self.subTest(profile=profile):
                self.capture.unlink(missing_ok=True)
                proc=self.submit(profile)
                self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
                args=self.capture.read_text()
                for required in ('--partition=project-test','--account=project-test',
                                 '--nodelist=gpusv[01-08]','--nodes=1',
                                 '--gres=gpu:H200:8',
                                 str(ROOT/'sh/synka/_eval_synka_common.sh'),
                                 'EVAL_ENV_FILE=', 'PROJECT_ROOT='):
                    self.assertIn(required,args)
                self.assertNotIn('sh/mdx/_eval_mdx_common.sh', args)
                self.assertEqual(len(list((self.root/'outputs'/'synka_submissions').glob('*/jobs.tsv'))),
                                 ['llmjp','phi','dsdt'].index(profile)+1)

    def test_dsdt_half_node_gres(self):
        proc = self.submit('dsdt', gpus=4)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('--gres=gpu:H200:4', self.capture.read_text())

    def test_gpu_count_above_one_node_rejected(self):
        proc = self.submit(gpus=9)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('1..8',proc.stderr)

    def test_spooled_worker_uses_exported_project(self):
        import shutil
        spool = self.root/'var'/'spool'
        spool.mkdir(parents=True)
        copied = spool/'batch_script'
        shutil.copy2(ROOT/'sh'/'synka'/'_eval_synka_common.sh',copied)
        fake_python=self.root/'fake_python'
        fake_python.write_text('#!/usr/bin/env bash\nif [[ "${1:-}" == - ]]; then cat > /dev/null; exit 0; fi\necho FAKE_EVAL\n')
        fake_python.chmod(0o755)
        (self.venv/'bin'/'python').unlink()
        (self.venv/'bin'/'python').symlink_to(fake_python)
        for name, cmd in {'module':'exit 0','nvidia-smi':'exit 0'}.items():
            f=self.bin/name
            f.write_text('#!/usr/bin/env bash\n'+cmd+'\n')
            f.chmod(0o755)
        env=os.environ.copy()
        env.update({'PATH':str(self.bin)+os.pathsep+env['PATH'],
                    'EVAL_ENV_FILE':str(self.config),'PROJECT_ROOT':str(ROOT),
                    'EVAL_PROFILE':'llmjp','SLURM_JOB_ID':'12345'})
        args=['bash',str(copied),'__worker','--exp','example','--data-names','math500',
              '--sample-num','16','--split','test','--prompt-type','llmjp4-harmony',
              '--max-tokens-per-call','8']
        proc=subprocess.run(args,cwd=spool,env=env,text=True,capture_output=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.assertIn('SUCCEEDED',proc.stdout)
        self.assertIn(str(ROOT/'llmjp_eval.py'),proc.stdout)
        self.assertNotIn('/var/spool/llmjp_eval.py',proc.stdout)

    def test_invalid_nodelist_is_rejected_before_sbatch(self):
        self.config.write_text(self.config.read_text().replace('gpusv[01-08]','gpusv[01-07]'))
        proc=self.submit()
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('exactly 8',proc.stderr)
        self.assertFalse(self.capture.exists())

    def test_partition_membership_and_submit_host(self):
        proc=self.submit(overrides={'SYNKA_SUBMIT_HOST':'wrong-host'})
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('submit on',proc.stderr)

    def test_dataset_lists_include_transfer_and_exclude_fake_proof_accuracy(self):
        for directory in ('mdx','synka'):
            for filename in ('eval_llmjp.sh','eval_phi.sh','eval.sh'):
                text=(ROOT/'sh'/directory/filename).read_text()
                active=next(x for x in text.splitlines() if x.startswith('DATA_NAMES='))
                self.assertIn('gsm_plus',text)
                self.assertIn('gsm_symbolic_p2',text)
                self.assertIn('math_perturb_hard',text)
                self.assertNotIn(',proofbench',active)
                self.assertIn('processbench_gsm8k',text)
                self.assertIn('omega_exploratory',text)

    def test_synka_env_is_gitignored(self):
        self.assertIn('.env.*', (ROOT/'.gitignore').read_text())
        self.assertTrue((ROOT/'.env_synka.example').exists())

if __name__ == '__main__':
    unittest.main()

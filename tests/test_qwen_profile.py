"""CPU-only tests of Qwen thinking prompts, completions and resume settings."""
import unittest
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from qwen_profile import (profile_version, make_prompt, decode_response,
                          validate_qwen_tokenizer)
from llmjp_storage import normalize_config


class MockTokenizer:
    eos_token_id = 9
    chat_template = 'fake-jinja'
    def __init__(self):
        self.args = None
    def get_vocab(self):
        return {'<|im_start|>': 1, '<|im_end|>': 2}
    def apply_chat_template(self, messages, **kwargs):
        self.args = messages, kwargs
        return '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n<|im_start|>assistant\n<think>'
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids)


def ids(text):
    return [ord(c) for c in text]


class TestQwen(unittest.TestCase):
    def test_prompt_uses_qwen_chat_template_once(self):
        tokenizer = MockTokenizer()
        validate_qwen_tokenizer(tokenizer)
        prompt = make_prompt(tokenizer, '1+1?', 'math500')
        self.assertIn(r'\boxed{...}', prompt)
        messages, options = tokenizer.args
        self.assertEqual(messages[0]['role'], 'user')
        self.assertTrue(options['enable_thinking'])
        self.assertTrue(options['add_generation_prompt'])
        self.assertFalse(options['tokenize'])

    def test_thinking_final_only(self):
        tok = MockTokenizer()
        response = decode_response(tok, ids('<think>bad answer 7</think>\\boxed{2}<|im_end|>'))
        self.assertEqual(response['final'], r'\boxed{2}')
        self.assertIn('bad answer', response['reasoning'])
        self.assertTrue(response['has_final'])
        # This is crucial when the prompt pre-fills <think> and generation
        # ends before outputting a closing marker.
        incomplete = decode_response(tok, ids('still thinking'))
        self.assertFalse(incomplete['has_final'])
        self.assertEqual(incomplete['final'], '')
        prefilled = decode_response(tok, ids('thinking body</think>\\boxed{2}'))
        self.assertEqual(prefilled['final'], r'\boxed{2}')
        self.assertFalse(decode_response(tok, ids('<think>unfinished'))['has_final'])

    def test_qwen_sampler_protocol_cannot_mismatch(self):
        base = dict(protocol=profile_version('qwen3-thinking'), model='/tmp/qwen',
                    dataset='math500', split='test', num_test_sample=-1,
                    start=0, end=-1, seed=42, temperature=0.6, top_p=0.95,
                    top_k=20, n_sampling=16, max_tokens_per_call=32768,
                    max_model_len=40960, reasoning_effort=None,
                    prompt_type='qwen3-thinking', min_p=0.0, presence_penalty=0.0)
        old = normalize_config(base)
        self.assertEqual(old['prompt_type'], 'qwen3-thinking')
        different = dict(base, presence_penalty=1.5)
        self.assertNotEqual(old, normalize_config(different))
        old['protocol'] = 'qwen35-thinking-v1'
        self.assertNotEqual(old, normalize_config(base))
        self.assertIsNone(normalize_config({k:v for k,v in base.items() if k != 'min_p'}))

    def test_mdx_submits_both_qwen_profiles(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ("Qwen3-8B", "Qwen3.5-9B"):
                (path / "models" / name).mkdir(parents=True)
            for dataset in ("gpqa_diamond", "mmlu_stem", "livecodebench_v6"):
                cache = path / "data/eval" / dataset / "test.jsonl"
                cache.parent.mkdir(parents=True)
                cache.write_text("[]\n")
            venv = path / "venv/bin"
            venv.mkdir(parents=True)
            (venv / "python").symlink_to(sys.executable)
            bin_dir = path / "bin"
            bin_dir.mkdir()
            sbatch = bin_dir / "sbatch"
            sbatch.write_text(
                '#!/bin/bash\nprintf "%s\\n" "$@" >> "' + str(path / "record") +
                '"\necho 789\n'
            )
            sbatch.chmod(0o755)
            env = dict(
                PATH=f'{bin_dir}:{os.environ.get("PATH", "")}',
                DATA_HOME=str(path), DATA_DIR=str(path / "data/eval"),
                MODEL_ROOT=str(path / "models"),
                EVAL_OUTPUTS=str(path / "outputs"),
                UV_PROJECT_ENVIRONMENT=str(path / "venv"),
                EVAL_ENV_FILE=str(path / "no-env"),
            )
            result = subprocess.run(
                ["bash", str(root / "sh/mdx/eval_qwen3.sh")], cwd=root,
                env=env, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = (path / "record").read_text()
            self.assertIn("EVAL_PROFILE=qwen", record)
            self.assertIn("--prompt-type\nqwen3-thinking", record)
            self.assertIn("--prompt-type\nqwen35-thinking", record)
            self.assertIn("--presence-penalty\n1.5", record)
            self.assertIn("--max-tokens-per-call\n32768", record)
            self.assertIn("--max-tokens-per-call\n38912", record)
            self.assertIn("--time=0", record)

    def test_old_llmjp_protocol_not_affected(self):
        p = dict(protocol='llmjp4-harmony-final-v2',model='/tmp/x',dataset='math500',
                 split='test',num_test_sample=-1,start=0,end=-1,seed=42,
                 temperature=0.6,top_p=0.95,top_k=-1,n_sampling=16,
                 max_tokens_per_call=38912,max_model_len=65536,
                 reasoning_effort='high')
        self.assertEqual(set(normalize_config(p)), set(p))


if __name__ == '__main__':
    unittest.main()

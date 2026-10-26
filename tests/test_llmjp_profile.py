"""CPU-only protocol invariants. Run: python -m unittest discover -s tests -v."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from llmjp_profile import (
    append_jsonl_durable, decode_response, make_prompt,
    read_checkpoint, signature, make_user_prompt,
)
from livecodebench_adapter import extract_python_program


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, reasoning_effort):
        self.calls.append((messages, tokenize, add_generation_prompt, reasoning_effort))
        return "<|start|>user<|message|>" + messages[0]["content"] + "<|end|><|start|>assistant"

    def encode(self, text, add_special_tokens=False):
        if text == "<|start|>assistant":
            return [1, 2]
        raise AssertionError(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join({11: "incorrect intermediate 7", 12: "\\boxed{3}", 13: "print(3)", 14: "Reasoning only"}.get(i, "") for i in ids)


class Message:
    def __init__(self, channel, content):
        self.channel = channel
        self.content = content


class FakeParser:
    def __init__(self, tokenizer):
        pass

    def iter_messages(self, ids):
        assert ids[:2] == [1, 2], "Missing official prefill"
        if ids[2:] == [11, 12]:
            return iter([Message("analysis", [11]), Message("final", [12])])
        if ids[2:] == [14]:
            return iter([Message("analysis", [14])])
        if ids[2:] == [13]:
            return iter([Message(None, [13])])
        raise AssertionError(ids)

    def parse_header(self, msg):
        return types.SimpleNamespace(role="assistant", recipient=None, channel=msg.channel)


class TestProtocol(unittest.TestCase):
    def test_model_chat_template_once(self):
        tokenizer = Tokenizer()
        result = make_prompt(tokenizer, "Calculate 1+2", "math500", "high")
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0][0][0]["role"], "user")
        self.assertNotIn("<|im_start|>", result)
        self.assertEqual(result.count("<|start|>assistant"), 1)
        self.assertIn("\\boxed{...}", result)

    def test_choice_and_lcb_instructions(self):
        self.assertIn("option letter", make_user_prompt("Q", "gpqa_diamond"))
        self.assertIn("Python 3", make_user_prompt("Q", "livecodebench_v6"))

    def test_analysis_not_scored(self):
        module = types.ModuleType("llm_jp_vllm.llmjp4.harmony")
        module.HarmonyMessageParser = FakeParser
        with patch.dict(sys.modules, {
            "llm_jp_vllm": types.ModuleType("llm_jp_vllm"),
            "llm_jp_vllm.llmjp4": types.ModuleType("llm_jp_vllm.llmjp4"),
            "llm_jp_vllm.llmjp4.harmony": module,
        }):
            result = decode_response(Tokenizer(), [11, 12])
            self.assertEqual(result["final"], r"\boxed{3}")
            self.assertIn("incorrect intermediate", result["reasoning"])
            self.assertNotIn("intermediate", result["final"])
            missing = decode_response(Tokenizer(), [14])
            self.assertEqual(missing["final"], "")
            self.assertFalse(missing["has_final"])
            self.assertEqual(decode_response(Tokenizer(), [13])["final"], "print(3)")

    def test_config_fingerprint(self):
        self.assertEqual(signature({"a": 1, "b": 2}), signature({"b": 2, "a": 1}))
        self.assertNotEqual(signature({"top_p": 0.9}), signature({"top_p": 0.95}))
        self.assertNotEqual(signature({"n_sampling": 1}), signature({"n_sampling": 16}))

    def test_checkpoint_resume_and_invalid_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            row = {"idx": 10, "code": ["A", "B"], "pred": ["A", "B"]}
            append_jsonl_durable(path, [row])
            self.assertEqual(read_checkpoint(path, 2)[10], row)
            with self.assertRaises(ValueError):
                read_checkpoint(path, 16)
            append_jsonl_durable(path, [row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_checkpoint(path, 2)
            path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(read_checkpoint(path, 2), {})
            self.assertEqual(path.read_bytes(), b"")

    def test_lcb_final_program(self):
        text = "Reasoning outside code.\n```python\nprint(42)\n```"
        self.assertEqual(extract_python_program(text), "print(42)")


if __name__ == "__main__":
    unittest.main()

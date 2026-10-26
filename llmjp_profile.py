"""LLM-jp-4/4.1 evaluation protocol; no changes to legacy Phi/DSDT paths.

Official reference: https://github.com/llm-jp/llm-jp-4-cookbook
Model-specific Harmony parsing operates on token IDs, not output.text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROFILE_VERSION = "llmjp4-harmony-final-v2"
CHOICE_DATASETS = frozenset({
    "gpqa_diamond", "mmlu_stem", "mmlu_pro", "aqua", "sat_math",
    "gaokao_math_qa",
})


def make_user_prompt(question: str, data_name: str) -> str:
    if data_name.startswith("livecodebench_"):
        return (
            "Solve the following programming problem. Reason carefully. "
            "In your final answer, output a complete Python 3 program in a single "
            "```python``` code block. It must read standard input and write "
            "standard output.\n\n" + question
        )
    if data_name in CHOICE_DATASETS:
        return (
            "Solve the following multiple-choice question. Reason carefully. "
            "In your final answer, put only the selected option letter "
            "(such as A) inside \\boxed{...}.\n\n" + question
        )
    return (
        "Solve the following problem. Reason carefully. "
        "In your final answer, put the answer inside \\boxed{...}.\n\n" + question
    )


def make_prompt(tokenizer, question: str, data_name: str, reasoning_effort: str) -> str:
    # Apply the tokenizer's own Harmony dialect exactly once.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": make_user_prompt(question, data_name)}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )


def decode_response(tokenizer, token_ids: list[int]):
    """Return final-only answer and diagnostic reasoning; never score analysis.

    The input chat template already ends with '<|start|>assistant'. Restore
    that prefix as in the official vLLM cookbook, then parse actual token IDs.
    """
    from llm_jp_vllm.llmjp4.harmony import HarmonyMessageParser

    parser = HarmonyMessageParser(tokenizer)
    prefill = tokenizer.encode("<|start|>assistant", add_special_tokens=False)
    messages = list(parser.iter_messages(prefill + list(token_ids)))
    reasoning = []
    finals = []
    other = []
    for msg in messages:
        head = parser.parse_header(msg)
        if head.role != "assistant" or head.recipient is not None:
            continue
        body = tokenizer.decode(msg.content or [], skip_special_tokens=False)
        if head.channel == "analysis":
            reasoning.append(body)
        elif head.channel in ("final", None, ""):
            if body.strip():
                finals.append(body)
        else:
            other.append(head.channel)
    # Require an explicit final when there are analysis messages; do not
    # silently score a partial analysis or a tool/commentary message.
    # A simple assistant content message (no channel) is an allowed final.
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {
        "final": finals[-1].strip() if finals else "",
        "reasoning": "\n".join(reasoning),
        "raw_harmony": raw,
        "has_final": bool(finals),
        "unexpected_channels": other,
    }


def signature(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def write_json_atomic(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(data, out, indent=2, ensure_ascii=False)
        out.write("\n")
        out.flush()
        import os
        os.fsync(out.fileno())
    temp.replace(path)


def append_jsonl_durable(path: Path, samples: list[dict]) -> None:
    import os
    if not samples:
        return
    with path.open("a", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_checkpoint(path: Path, n_sampling: int) -> dict[int, dict]:
    """Recover complete JSONL records; discard only an interrupted tail write.

    Called under the MDX per-experiment flock. Middle-file corruption is fatal.
    """
    out = {}
    if not path.exists():
        return out
    with path.open("rb+") as inp:
        lineno = 0
        while True:
            position = inp.tell()
            line = inp.readline()
            if not line:
                break
            lineno += 1
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("unterminated final line")
                sample = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if inp.read(1):
                    raise ValueError(f"Nonterminal corrupt checkpoint line {path}:{lineno}") from exc
                print(f"[LLM-jp] Recovering interrupted checkpoint tail: {path}:{lineno}", flush=True)
                inp.truncate(position)
                break
            if len(sample.get("code", [])) != n_sampling or len(sample.get("pred", [])) != n_sampling:
                raise ValueError(f"Incomplete n_sampling in {path}:{lineno}")
            idx = sample["idx"]
            if idx in out:
                raise ValueError(f"Duplicate sample idx={idx} in checkpoint {path}")
            out[idx] = sample
    return out

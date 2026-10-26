"""Adapters between eval_math generations and LiveCodeBench custom evaluator.

No generated program is executed here.  This module only extracts the final
Python code block and writes the JSON format accepted by LiveCodeBench's
`lcb_runner.runner.custom_evaluator`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def extract_python_program(text: str) -> str:
    """Extract the final fenced Python program, with conservative fallbacks."""
    text = (text or "").strip()
    if not text:
        return ""

    py_blocks = re.findall(r"```(?:python|py)\s*\n?(.*?)```", text, flags=re.I | re.S)
    if py_blocks:
        return py_blocks[-1].strip()

    fenced = re.findall(r"```\s*\n?(.*?)```", text, flags=re.S)
    if fenced:
        return fenced[-1].strip()

    # Reasoning models often end a <think> region before emitting bare code.
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail
    return text


def export_custom_outputs(samples: Iterable[dict], output_path: str | Path) -> Path:
    """Write LiveCodeBench custom-evaluator input, sorted by question_id."""
    rows = []
    for sample in samples:
        qid = sample.get("question_id")
        if qid is None:
            raise KeyError("LiveCodeBench sample is missing question_id")
        raw_generations = sample.get("code", [])
        code_list = [extract_python_program(x) for x in raw_generations]
        rows.append({"question_id": qid, "code_list": code_list})

    rows.sort(key=lambda x: str(x["question_id"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path

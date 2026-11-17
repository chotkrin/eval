"""LLM-jp-specific MDX evaluation entrypoint (Phi/DSDT remain on math_eval.py).

The checkpoint is append-only, one *complete question* at a time. Batches of
questions are generated in parallel by vLLM; scoring happens after generation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from data_loader import load_data
from evaluate import evaluate
from livecodebench_adapter import export_custom_outputs, extract_python_program
from llmjp_profile import (
    PROFILE_VERSION, append_jsonl_durable, decode_response, make_prompt,
    read_checkpoint, write_json_atomic,
)
from llmjp_storage import (
    normalize_config, read_protocol, migrate_one, candidates, archive_root_files,
)
from parser import parse_ground_truth, parse_question, run_execute, choice_answer_clean


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--data_names", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--prompt_type", default="llmjp4-harmony")
    p.add_argument("--num_test_sample", type=int, default=-1)
    p.add_argument("--max_tokens_per_call", type=int, default=38912)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--n_sampling", type=int, default=16)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--pipeline_parallel_size", type=int, default=1)
    p.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="high")
    p.add_argument("--question_batch_size", type=int, default=4)
    p.add_argument("--max_model_len", type=int, default=65536)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.prompt_type not in ("llmjp4-harmony", "qwen3-thinking", "qwen35-thinking"):
        p.error("unsupported prompt_type: expected llmjp4-harmony, qwen3-thinking or qwen35-thinking")
    if a.prompt_type != "llmjp4-harmony" and not os.environ.get("QWEN_EVAL_ENTRYPOINT"):
        p.error("Qwen evaluation must use qwen_eval.py")
    if a.question_batch_size < 1 or a.n_sampling < 1 or a.max_tokens_per_call < 1:
        p.error("batch size, n_sampling, max_tokens must be positive")
    if a.pipeline_parallel_size < 1:
        p.error("pipeline parallel size must be positive")
    if not 0 <= a.temperature or not 0 < a.top_p <= 1 or not 0 <= a.min_p <= 1:
        p.error("invalid sampling parameters")
    return a


def sample_key(example):
    return example["idx"]


def base_sample(example, data_name):
    question = parse_question(example, data_name)
    if not question:
        return None
    gt_cot, gt_ans = parse_ground_truth(example, data_name)
    sample = {"idx": sample_key(example), "question": question, "gt_cot": gt_cot, "gt": gt_ans}
    for key in (
        "level", "type", "unit", "solution_type", "choices", "solution", "ques_type", "ans_type",
        "answer_type", "dataset", "subfield", "filed", "theorem", "answer", "question_id",
        "question_title", "contest_id", "contest_date", "platform", "difficulty", "starter_code",
        "family_id", "variant", "transfer_type", "source_dataset", "source_id", "schema_id",
        "reference_solution", "marking_scheme", "first_error_index",
        "node_ids", "gold_edge_count", "gold_proof_graph", "step_count", "annotation_scope",
    ):
        if key in example:
            sample[key] = example[key]
    return sample


def eval_one_dataset(args, llm, tokenizer, name, version_info):
    source = load_data(name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        source = source[:args.num_test_sample]
    source = source[args.start: None if args.end == -1 else args.end]
    samples = [x for x in (base_sample(dict(e), name) for e in source) if x is not None]
    if len({s['idx'] for s in samples}) != len(samples):
        raise ValueError(f"Duplicate benchmark indices in {name}")
    # The dataset directory is the only output directory (no llmjp4/ layer).
    is_qwen = getattr(args, "prompt_type", "llmjp4-harmony").startswith("qwen")
    if is_qwen:
        from qwen_profile import profile_version, make_prompt as qwen_prompt, decode_response as qwen_decode
        protocol_version = profile_version(args.prompt_type)
    else:
        protocol_version = PROFILE_VERSION
    config = {
        "protocol": protocol_version, "model": str(Path(args.model_name_or_path).resolve()),
        "dataset": name, "split": args.split,
        "num_test_sample": args.num_test_sample, "start": args.start, "end": args.end,
        "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "n_sampling": args.n_sampling,
        "max_tokens_per_call": args.max_tokens_per_call, "max_model_len": args.max_model_len,
        "reasoning_effort": args.reasoning_effort,
    }
    # Existing LLM-jp protocol.json files remain byte-compatible on resume.
    # Qwen's model-specific generation settings must be checked separately.
    if is_qwen:
        config.update({"reasoning_effort": None, "prompt_type": args.prompt_type,
                       "min_p": args.min_p, "presence_penalty": args.presence_penalty})
    output_root = Path(args.output_dir) / name
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A worker already owns the experiment lock. Historical subfolders must
    # be flattened before checkpoint recovery, including an earlier llmjp4/.
    if any(run.directory != output_root for run in candidates(output_root, args.n_sampling)):
        selection = migrate_one(output_root, args.n_sampling, apply=True)
        if selection["status"] == "invalid":
            raise ValueError(f"Cannot migrate corrupt checkpoint in {name}: {selection}")
        print(f"[LLM-jp] {name}: migration {selection}", flush=True)
    meta_file = output_root / "protocol.json"
    # Fail closed when there is existing run data but protocol has been lost.
    # On --overwrite archive only this dataset's LLM-jp files; unrelated
    # Phi/DSDT outputs (e.g. *_metrics.json) remain where they are.
    has_previous = bool((output_root / "protocol.json").exists() or
                        (output_root / "generations.jsonl").exists() or
                        (output_root / "metrics.json").exists())
    previous = read_protocol(output_root) if has_previous else None
    if has_previous and previous != normalize_config(config) and not args.overwrite:
        raise ValueError(
            f"Existing checkpoint cannot be resumed with different sampling/model settings: {output_root}. "
            "Inspect protocol.json; use --overwrite to archive and start a new run."
        )
    if has_previous and args.overwrite:
        archive_root_files(output_root)
    output_root.chmod(0o700)
    if not meta_file.is_file():
        write_json_atomic(meta_file, config)
    out_file = output_root / "generations.jsonl"
    metrics_file = output_root / "metrics.json"
    processed = read_checkpoint(out_file, args.n_sampling)
    expected_ids = {s['idx'] for s in samples}
    if not set(processed).issubset(expected_ids):
        raise ValueError("Checkpoint contains samples outside selected range")
    # idx alone is not sufficient: the benchmark file may have been edited.
    expected = {s["idx"]: s for s in samples}
    for idx, saved in processed.items():
        if saved.get("question") != expected[idx]["question"] or saved.get("gt") != expected[idx]["gt"]:
            raise ValueError(f"Checkpoint question/answer mismatch at idx={idx}; refusing unsafe resume")
    pending = [s for s in samples if s['idx'] not in processed]
    print(f"[LLM-jp] {name}: total={len(samples)} completed={len(processed)} remaining={len(pending)}", flush=True)
    if not samples:
        raise ValueError(f"Empty benchmark: {name}")
    if not pending and metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        # Scoring is cacheable only when it is newer than the generations it
        # describes; a replaced/appended checkpoint invalidates prior metrics.
        if (metrics.get("num_samples") == len(samples) and
                metrics_file.stat().st_mtime_ns >= out_file.stat().st_mtime_ns):
            if not name.startswith("livecodebench_") or (output_root / "lcb_custom.json").is_file():
                return metrics
    if pending and metrics_file.exists():
        # A stale score must not masquerade as a finished evaluation.
        metrics_file.unlink()

    from vllm import SamplingParams
    if is_qwen:
        stop_id = tokenizer.eos_token_id
        if not isinstance(stop_id, int) or stop_id < 0:
            raise ValueError("Qwen tokenizer has no valid EOS token")
    else:
        stop_id = tokenizer.convert_tokens_to_ids("<|return|>")
        if not isinstance(stop_id, int) or stop_id < 0 or stop_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError("LLM-jp tokenizer has no <|return|> stop token")
    from tqdm import tqdm
    start = time.monotonic()
    for i in tqdm(range(0, len(pending), args.question_batch_size), desc=name):
        batch = pending[i:i + args.question_batch_size]
        prompts = [
            qwen_prompt(tokenizer, s["question"], name) if is_qwen
            else make_prompt(tokenizer, s["question"], name, args.reasoning_effort)
            for s in batch
        ]
        prompt_tokens = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        for sample, tokens in zip(batch, prompt_tokens):
            if tokens + args.max_tokens_per_call > args.max_model_len:
                raise ValueError(
                    f"{name} idx={sample['idx']}: prompt_tokens={tokens} + "
                    f"max_tokens={args.max_tokens_per_call} > max_model_len={args.max_model_len}"
                )
        params = SamplingParams(
            n=args.n_sampling, seed=args.seed, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens_per_call,
            min_p=args.min_p if is_qwen else 0.0,
            presence_penalty=args.presence_penalty if is_qwen else 0.0,
            stop_token_ids=[stop_id], detokenize=False, skip_special_tokens=False,
        )
        generated = llm.generate(prompts, params, use_tqdm=False)
        if len(generated) != len(batch):
            raise RuntimeError("vLLM returned the wrong number of prompt responses")
        # LLM.generate returns results in input order. Unlike old code, no
        # parsing request_id as integer; for n>1 keep each prompt's candidates.
        completed = []
        for sample, req, prompt_len in zip(batch, generated, prompt_tokens):
            if req.prompt is not None and req.prompt != prompts[len(completed)]:
                raise RuntimeError("vLLM output order no longer matches prompt order")
            if len(req.outputs) != args.n_sampling:
                raise RuntimeError(f"incomplete sampling for idx={sample['idx']}")
            generations, preds, reports, reasoning, raws, finishes, lengths, valid = [], [], [], [], [], [], [], []
            for candidate in sorted(req.outputs, key=lambda x: x.index):
                ids = list(candidate.token_ids)
                parsed = qwen_decode(tokenizer, ids) if is_qwen else decode_response(tokenizer, ids)
                # A token-limit truncation is an incorrect/incomplete response,
                # even if a partial final answer was already emitted.
                is_valid = parsed["has_final"] and candidate.finish_reason == "stop"
                final = parsed["final"] if is_valid else ""
                if name.startswith("livecodebench_"):
                    prediction, report = extract_python_program(final), None
                else:
                    prediction, report = run_execute(None, final, "cot", name)
                    if name in ("gpqa_diamond", "mmlu_stem", "mmlu_pro") and final:
                        # The common parser already recognizes choices, retain its behavior.
                        prediction = prediction or choice_answer_clean(final)
                generations.append(final)
                preds.append(prediction if prediction is not None else "")
                reports.append(report)
                reasoning.append(parsed["reasoning"])
                raws.append(parsed["raw_harmony"])
                finishes.append(candidate.finish_reason)
                lengths.append(len(ids))
                valid.append(is_valid)
            record = {**sample, "prompt_tokens": prompt_len,
                "code": generations, "pred": preds, "report": reports,
                "reasoning": reasoning,
                "finish_reason": finishes, "valid_final": valid,
                "completion_tokens_per_sample": lengths,
                "completion_tokens": sum(lengths)}
            record["raw_qwen" if is_qwen else "raw_harmony"] = raws
            completed.append(record)
        # Save complete questions after each successful vLLM batch; requeues
        # skip them and never mix candidates with different configurations.
        append_jsonl_durable(out_file, completed)
        for sample in completed:
            processed[sample["idx"]] = sample
    all_samples = [processed[s["idx"]] for s in samples]
    if name.startswith("livecodebench_"):
        output_file = output_root / "lcb_custom.json"
        export_custom_outputs(all_samples, output_file)
        result = {"acc": None, "pass@1": None, "num_samples": len(all_samples),
                  "num_scores": 0, "evaluation_status": "generation_complete_scoring_required",
                  "custom_output_file": str(output_file)}
    else:
        all_samples, result = evaluate(samples=all_samples, data_name=name,
                                       prompt_type="cot", execute=True)
        # Persist scores separately rather than rewriting the append-only generation log.
        scored_path = output_root / "scored.jsonl"
        from utils import save_jsonl
        save_jsonl(all_samples, str(scored_path))
    total_generations = sum(len(s["code"]) for s in all_samples)
    total_tokens = sum(s["completion_tokens"] for s in all_samples)
    result.update({"protocol": protocol_version, "output_dir": str(output_root),
                   "num_generations": total_generations,
                   "avg_completion_tokens": total_tokens / total_generations,
                   "missing_final": sum(not v for s in all_samples for v in s['valid_final']),
                   "truncated": sum(x == "length" for s in all_samples for x in s["finish_reason"]),
                   "generation_wall_seconds_this_invocation": time.monotonic() - start})
    write_json_atomic(metrics_file, result)
    print(f"[LLM-jp] {name}: metrics={metrics_file}", flush=True)
    return result


def main():
    args = arguments()
    import importlib.metadata as md
    versions = {name: md.version(name) for name in ["vllm", "llm-jp-vllm", "transformers", "torch", "sympy"]}
    if tuple(map(int, versions["vllm"].split(".")[:2])) < (0, 30):
        raise RuntimeError("LLM-jp evaluation requires vLLM >= 0.30.0")
    from vllm import LLM
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if not visible or len(visible) % args.pipeline_parallel_size:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must list GPU IDs divisible by pipeline_parallel_size")
    llm = LLM(model=args.model_name_or_path, trust_remote_code=True, dtype="bfloat16",
              tensor_parallel_size=len(visible)//args.pipeline_parallel_size,
              pipeline_parallel_size=args.pipeline_parallel_size,
              max_model_len=args.max_model_len, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    vocab = tokenizer.get_vocab()
    if args.prompt_type == "llmjp4-harmony":
        for marker in ("<|start|>", "<|message|>", "<|channel|>", "<|constrain|>",
                       "<|end|>", "<|return|>", "<|call|>"):
            if marker not in vocab:
                raise RuntimeError(f"Model tokenizer lacks LLM-jp Harmony marker {marker}")
    else:
        from qwen_profile import validate_qwen_tokenizer
        validate_qwen_tokenizer(tokenizer)
    scores = []
    for name in args.data_names.split(","):
        result = eval_one_dataset(args, llm, tokenizer, name, versions)
        print(name, result, flush=True)
        if result.get("acc") is not None:
            scores.append(result["acc"])
    print(f"Macro average scored datasets: {sum(scores)/len(scores) if scores else 'PENDING'}", flush=True)


if __name__ == "__main__":
    main()

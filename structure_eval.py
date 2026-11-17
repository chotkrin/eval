"""Offline, deterministic evaluation of obligation-adjacent *proxy* tasks.

ProofFlowBench: reconstruct edges between *provided* proof steps, not original
ProofScore and not an obligation-coverage assessment of generated solutions.
PRMBench: recover all annotated unsound steps; not its official PRM benchmark.

Predictions MUST come from the model final channel, unchanged by math parsers.
"""
from __future__ import annotations

import json


def _parse_json_object(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].strip() in ("```", "```json", "```JSON") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _edges_make_dag(edges, node_ids):
    incoming = {key: 0 for key in node_ids}
    forward = {key: [] for key in node_ids}
    for src, dst in edges:
        if src not in incoming or dst not in incoming or src == dst:
            return False
        incoming[dst] += 1
        forward[src].append(dst)
    ready = [key for key, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        src = ready.pop()
        visited += 1
        for dst in forward[src]:
            incoming[dst] -= 1
            if not incoming[dst]:
                ready.append(dst)
    return visited == len(node_ids)


def _read_edges(obj, node_ids):
    if obj is None or set(obj) != {"edges"} or not isinstance(obj["edges"], list):
        return None
    edges = obj["edges"]
    if any(not isinstance(pair, list) or len(pair) != 2 or
           any(not isinstance(item, str) for item in pair) for pair in edges):
        return None
    unique = {tuple(pair) for pair in edges}
    if len(unique) != len(edges) or not _edges_make_dag(unique, node_ids):
        return None
    return unique


def _read_error_steps(obj, step_count):
    if obj is None or set(obj) != {"error_steps"} or not isinstance(obj["error_steps"], list):
        return None
    labels = obj["error_steps"]
    if any(type(label) is not int or label < 1 or label > step_count for label in labels):
        return None
    unique = set(labels)
    return unique if len(unique) == len(labels) else None


def evaluate_structured(data_name, samples, ks=(1, 5, 10)):
    """Evaluate every candidate; report candidate-0 micro metrics for model A/B.

    No answer `acc` or `pass@k`: detection / reconstruction are distinct tasks.
    Invalid/missing JSON is scored as incorrect and contributes to FNs.
    """
    if data_name not in ("proofflowbench_graph", "prmbench_error_steps"):
        raise ValueError(f"Unexpected structured dataset: {data_name}")
    totals = dict(tp=0, fp=0, fn=0, exact=0, invalid=0)
    f1s = []
    for sample in samples:
        if data_name == "proofflowbench_graph":
            node_ids = sample.get("node_ids")
            if not isinstance(node_ids, list) or len(node_ids) != len(set(node_ids)):
                raise ValueError("ProofFlowBench sample missing valid node_ids")
            read = lambda obj: _read_edges(obj, node_ids)
            key = "edges"
        else:
            count = sample.get("step_count")
            if type(count) is not int or count <= 0:
                raise ValueError("PRMBench sample missing step_count")
            read = lambda obj: _read_error_steps(obj, count)
            key = "error_steps"
        try:
            gold_obj = json.loads(sample["gt"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid gold JSON in structured dataset") from exc
        gold = read(gold_obj)
        if gold is None:
            raise ValueError(f"Invalid gold {key} values")
        if not isinstance(sample.get("pred"), list) or not sample["pred"]:
            raise ValueError("Structured evaluation requires at least one prediction")
        row_scores = []
        for candidate_index, prediction in enumerate(sample["pred"]):
            predicted = read(_parse_json_object(prediction))
            valid = predicted is not None
            if not valid:
                predicted = set()
            exact = valid and predicted == gold
            row_scores.append(bool(exact))
            # To avoid an invalid empty prediction receiving perfect empty-set F1.
            tp = len(predicted & gold) if valid else 0
            fp = len(predicted - gold) if valid else 0
            fn = len(gold - predicted)
            if not valid and not gold:
                fn = 1  # invalid JSON cannot be correct even for empty gold
            precision = tp / (tp + fp) if tp + fp else (1.0 if valid and not gold else 0.0)
            recall = tp / (tp + fn) if tp + fn else (1.0 if valid and not predicted else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if candidate_index == 0:
                totals["tp"] += tp
                totals["fp"] += fp
                totals["fn"] += fn
                totals["exact"] += bool(exact)
                totals["invalid"] += not valid
                f1s.append(f1)
        sample["score"] = row_scores
        sample["pass_at_k"] = {}
    n = len(samples)
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    micro_p = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    micro_r = tp / (tp + fn) if tp + fn else (1.0 if tp + fp == 0 else 0.0)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return samples, {
        "num_samples": n, "num_scores": sum(len(s["score"]) for s in samples),
        "acc": None, "pass@1": None, "pass@5": None, "pass@10": None,
        "evaluation_status": "structured_proxy_scored_not_answer_accuracy",
        "scorer": "direct_graph_edges" if data_name == "proofflowbench_graph" else "annotated_error_steps",
        "structured_exact_acc_first": 100 * totals["exact"] / n if n else None,
        "structured_micro_precision_first": 100 * micro_p,
        "structured_micro_recall_first": 100 * micro_r,
        "structured_micro_f1_first": 100 * micro_f1,
        "structured_macro_f1_first": 100 * sum(f1s) / n if n else None,
        "structured_invalid_json_first": totals["invalid"],
        "structured_tp_first": tp, "structured_fp_first": fp, "structured_fn_first": fn,
    }

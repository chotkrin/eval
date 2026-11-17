"""Adapters for paired mathematical transfer benchmarks.

All adapters return the eval project's common {question, gt, gt_cot, idx}
contract, plus stable family metadata. Downloaded test benchmarks are NEVER
used as training data by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import warnings
from collections import Counter

TRANSFER_DATASETS = (
    "gsm_plus", "gsm_plus_seed", "gsm_symbolic_main", "gsm_symbolic_p1",
    "gsm_symbolic_p2", "math_perturb_simple", "math_perturb_hard",
    "omega_exploratory", "omega_compositional", "omega_transformative",
    "processbench_gsm8k", "processbench_math", "processbench_olympiadbench",
    "proofbench", "obligation_transfer",
    "processbench_omnimath", "prmbench_error_steps",
    "proofflowbench_graph", "proofflowbench_proof",
)
class ProofFlowGraphError(ValueError):
    """A malformed upstream ProofFlowBench graph; never repair gold labels."""


class PRMBenchAnnotationError(ValueError):
    """A malformed PRMBench error-step annotation; never guess its intended ID."""


DOWNLOADABLE = TRANSFER_DATASETS[:5] + (
    "processbench_gsm8k", "processbench_math", "processbench_olympiadbench",
    "proofbench", "processbench_omnimath", "prmbench_error_steps",
    "proofflowbench_graph", "proofflowbench_proof",
)
LOCAL_ONLY = ("math_perturb_simple", "math_perturb_hard", "omega_exploratory",
              "omega_compositional", "omega_transformative", "obligation_transfer")


def _hash(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:20]


def _answer(text):
    # Formats used by GSM-Plus and GSM-Symbolic. Strict: never use a random
    # number appearing in an explanatory solution as if it were the answer.
    found = re.findall(r"####\s*([^\n]+)", str(text))
    if not found:
        raise ValueError("Ground truth missing #### delimiter")
    return found[-1].strip()


def _strict(row, *fields):
    for f in fields:
        if f not in row or row[f] is None or str(row[f]).strip() == "":
            raise ValueError(f"Required source field {f!r} is missing")


def _common(question, answer, family, variant, transfer, source, source_id, **extras):
    if not str(question).strip():
        raise ValueError("empty question")
    # The evaluation parser gives gt highest priority and leaves answer
    # untouched; never extract ground truth from the model's generation.
    return dict(question=str(question), gt=str(answer), gt_cot="",
                family_id=str(family), variant=str(variant),
                transfer_type=str(transfer), source_dataset=source,
                source_id=str(source_id), **extras)


def normalize(name, row, index=0):
    """Convert one upstream row to the evaluation schema; raises on ambiguity."""
    if name in ("gsm_plus", "gsm_plus_seed"):
        _strict(row, "seed_question", "seed_answer")
        fid = "gsmplus:" + _hash(row["seed_question"])
        if name == "gsm_plus_seed":
            q, ans, sol = row["seed_question"], row["seed_answer"], row.get("seed_solution", "")
            variant, transfer = "seed", "base"
        else:
            _strict(row, "question", "answer")
            q, ans, sol = row["question"], row["answer"], row.get("solution", "")
            variant, transfer = row.get("perturbation_type", "perturbation"), "perturbation"
        return _common(q, ans, fid, variant, transfer, "gsm_plus", index,
                       reference_solution=str(sol), source_question=str(row["seed_question"]), type=str(variant))
    if name.startswith("gsm_symbolic_"):
        _strict(row, "id", "original_id", "question", "answer")
        # original_id is the shared GSM8K source; id is a generator template.
        fid = f"gsm_symbolic:{row['original_id']}"
        return _common(row["question"], _answer(row["answer"]), fid,
                       f"{name.removeprefix('gsm_symbolic_')}:{row.get('instance',index)}",
                       "same_schema", "gsm_symbolic", f"{row['id']}:{row.get('instance',index)}",
                       schema_id=f"gsm_symbolic_template:{row['id']}",
                       reference_solution=str(row["answer"]).split("####")[0].strip(), type=name.removeprefix("gsm_symbolic_"))
    if name.startswith("math_perturb_"):
        _strict(row, "problem_id", "problem", "answer")
        variant = name.removeprefix("math_perturb_")
        return _common(row["problem"], row["answer"],
                       f"math_perturb:{row['problem_id']}", variant,
                       "strategy_change" if variant == "hard" else "perturbation",
                       "math_perturb", row["problem_id"],
                       type=str(row.get("type", "")), level=str(row.get("level", "")))
    if name.startswith("omega_"):
        # OMEGA is a *generator repository*, not one stable HF dataset schema.
        # Accept only records exported with explicit question/answer and
        # generator identifiers; do not guess labels for graph metadata.
        question = next((row[k] for k in ("question", "problem", "problem_text") if row.get(k)), None)
        answer = next((row[k] for k in ("answer", "final_answer", "ground_truth", "solution_answer") if row.get(k) is not None), None)
        if question is None or answer is None:
            raise ValueError("OMEGA record needs question/problem and answer/final_answer; supply a validated export")
        family = row.get("family_id", row.get("template_id", row.get("problem_type")))
        if not family:
            raise ValueError("OMEGA record needs family_id/template_id/problem_type")
        axis = name.removeprefix("omega_")
        return _common(question, answer, f"omega:{family}",
                       str(row.get("variant", row.get("difficulty", index))), axis,
                       "omega", row.get("id", index), schema_id=str(family),
                       difficulty=str(row.get("difficulty", "")),
                       reference_solution=str(row.get("solution", "")), type=axis)
    if name.startswith("processbench_"):
        _strict(row, "problem", "steps", "label")
        if not isinstance(row["steps"], list) or not row["steps"]:
            raise ValueError("ProcessBench steps must be a nonempty list")
        label = int(row["label"])
        if label < -1 or label >= len(row["steps"]):
            raise ValueError(f"ProcessBench invalid first-error label: {label}")
        steps = "\n\n".join(
            f"<paragraph_{i}>\n{step}\n</paragraph_{i}>"
            for i, step in enumerate(row["steps"])
        )
        q = (str(row["problem"]) + "\n\nReasoning steps to check:\n" + steps +
             "\n\nFind the first incorrect paragraph index (starting from 0). "
             "If every paragraph is correct, report -1. "
             "Put ONLY that integer inside \\boxed{...}.")
        return _common(q, label, "processbench:"+str(row.get("id",index)),
                       "audit", "process_error_detection", "processbench", row.get("id",index),
                       reference_solution="", first_error_index=label,
                       type=name.removeprefix("processbench_"))
    if name == "prmbench_error_steps":
        _strict(row, "modified_question", "modified_process", "error_steps", "idx")
        steps = row["modified_process"]
        labels = row["error_steps"]
        if not isinstance(steps, list) or not steps or not all(isinstance(x, str) for x in steps):
            raise ValueError("PRMBench modified_process must contain steps")
        # PRMBench upstream contains annotations beyond the length of the
        # corresponding modified_process. This is a source inconsistency, not
        # proof of a different indexing convention. Never clip, renumber or
        # silently treat those gold labels as valid.
        if not isinstance(labels, list) or any(type(x) is not int or x < 1 or x > len(steps) for x in labels):
            raise PRMBenchAnnotationError(
                f"PRMBench error_steps must be 1-based step indices in [1,{len(steps)}]; "
                f"received {labels!r}"
            )
        if len(set(labels)) != len(labels):
            raise PRMBenchAnnotationError(f"Duplicate PRMBench error step in {labels!r}")
        formatted = "\n".join(f"Step {i}: {step}" for i, step in enumerate(steps, 1))
        q = (str(row["modified_question"]) + "\n\nReview this proposed solution:\n" + formatted +
             "\n\nFind ALL unsound steps, not merely the first one. "
             'Return ONLY a JSON object {"error_steps": [1, 3]} using 1-based step numbers; '
             'return {"error_steps": []} if every step is sound.')
        return _common(q, json.dumps({"error_steps": sorted(labels)}),
                       "prmbench:"+str(row["idx"]), "audit", "process_error_detection",
                       "prmbench", row["idx"],
                       type=str(row.get("classification", "")),
                       annotation_scope="prmbench_modified_process", step_count=len(steps))
    if name in ("proofflowbench_graph", "proofflowbench_proof"):
        _strict(row, "origin", "id", "nl_theorem", "nl_proof", "proof_graph")
        nodes = row["proof_graph"]
        if name == "proofflowbench_proof":
            # Natural-language proof generation does not need gold graph edges.
            # Preserve all theorems even when graph annotations are defective;
            # retain raw annotations but explicitly mark them as UNVALIDATED.
            if not isinstance(nodes, list):
                raise ValueError("ProofFlowBench proof_graph is not a list")
            family = f"proofflowbench:{row['origin']}:{row['id']}"
            q = ("Provide a complete natural-language mathematical proof of the "
                 "following theorem.\n\n" + row["nl_theorem"])
            return _common(q, "", family, "proof_generation", "proof_quality",
                           "proofflowbench", family, type=str(row["origin"]),
                           reference_solution=row["nl_proof"],
                           node_ids=[n.get("id") for n in nodes if isinstance(n, dict)],
                           gold_proof_graph=nodes,
                           graph_annotation_status="unvalidated_upstream")
        if not isinstance(nodes, list) or not nodes:
            raise ProofFlowGraphError("ProofFlowBench proof_graph is empty")
        if any(not isinstance(node, dict) for node in nodes):
            raise ProofFlowGraphError("ProofFlowBench node is not an object")
        ids = [node.get("id") for node in nodes]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ProofFlowGraphError("ProofFlowBench has missing or duplicate node ids")
        edges = []
        for node in nodes:
            if not isinstance(node.get("dependencies"), list) or not isinstance(node.get("statement"), str):
                raise ProofFlowGraphError(f"ProofFlowBench node {node.get('id')!r} missing statement or dependencies")
            for parent in node["dependencies"]:
                if parent not in ids or parent == node["id"]:
                    raise ProofFlowGraphError(
                        f"ProofFlowBench has invalid dependency: {parent!r} -> "
                        f"{node['id']!r} (missing node or self-dependency)")
                edges.append([parent, node["id"]])
        if len(set(map(tuple, edges))) != len(edges):
            raise ProofFlowGraphError("ProofFlowBench contains duplicate edges")
        indegree = {i: 0 for i in ids}
        outgoing = {i: [] for i in ids}
        for parent, child in edges:
            outgoing[parent].append(child)
            indegree[child] += 1
        ready = [i for i in ids if indegree[i] == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for child in outgoing[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(ids):
            raise ProofFlowGraphError("ProofFlowBench dependency graph contains a cycle")
        family = f"proofflowbench:{row['origin']}:{row['id']}"
        if name == "proofflowbench_graph":
            # The gold edges are NEVER embedded in the question. Supplying the
            # annotated nodes measures dependency recovery, not full proof generation.
            snippet = "\n".join(
                f"ID: {node['id']}\nStatement: {node['statement']}"
                for node in nodes)
            q = ("Theorem:\n" + row["nl_theorem"] + "\n\nReference proof:\n" +
                 row["nl_proof"] + "\n\nAnnotated proof statements (without edges):\n" +
                 snippet + "\n\nRecover the DIRECT prerequisite dependencies "
                 "among the given IDs. Return ONLY a JSON object like "
                 '{"edges": [["tc_1", "l1"], ["l1", "ts_1"]]}. '
                 "Every pair is [prerequisite, dependent]; do not include transitive "
                 "edges unless they are direct dependencies. An empty graph is "
                 '{"edges": []}.')
            return _common(q, json.dumps({"edges": sorted(edges)}, sort_keys=True),
                           family, "dependency_reconstruction", "graph_recovery",
                           "proofflowbench", family, type=str(row["origin"]),
                           node_ids=ids, gold_edge_count=len(edges))
    if name == "proofbench":
        _strict(row, "problem_id", "problem")
        return _common(str(row["problem"]) + "\n\nProvide a complete mathematical proof.", "",
                       f"proofbench:{row['problem_id']}", "proof_generation", "proof_quality",
                       "proofbench", row["problem_id"],
                       reference_solution=str(row.get("reference_solution", "")),
                       marking_scheme=str(row.get("marking_scheme", "")))
    if name == "obligation_transfer":
        _strict(row, "problem_id", "family_id", "question", "answer", "transfer_type")
        return _common(row["question"], row["answer"], row["family_id"],
                       row.get("variant", "base"), row["transfer_type"], "obligation_transfer",
                       row["problem_id"], schema_id=str(row.get("schema_id", "")),
                       type=str(row["transfer_type"]),
                       obligation_ids=list(row.get("obligation_ids", [])),
                       obligation_types=list(row.get("obligation_types", [])),
                       annotation_status=str(row.get("annotation_status", "unreviewed")))
    raise NotImplementedError(name)


def _rows_from_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_proofflowbench_raw(path):
    """Read published ProofFlowBench records without HF features inference.

    Supports JSONL (official filename) and a JSON array (documented by the
    dataset README). No GPUs, Arrow dependency or expensive transforms.
    """
    with open(path, encoding="utf-8-sig") as stream:
        # Inspect just enough to distinguish the two documented layouts.
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        stream.seek(0)
        if first == "[":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError("ProofFlowBench JSON root is not a list")
            for number, record in enumerate(records, 1):
                if not isinstance(record, dict):
                    raise ValueError(f"ProofFlowBench record {number} is not an object")
                yield record
        else:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"ProofFlowBench line {number} is not an object")
                yield record


def _find_local(directory, pattern):
    matches = sorted(p for p in directory.rglob("*.jsonl") if re.search(pattern, p.name.lower())) if directory.exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one matching JSONL in {directory} regex={pattern!r}; found: {matches}. "
                                "Download/generate the upstream dataset first (see TRANSFER_DATASETS.md).")
    return matches[0]


def _load_upstream(name, data_dir):
    data_dir = Path(data_dir)
    root = Path(os.environ.get("TRANSFER_RAW_DIR", str(data_dir.parent / "raw")))
    if name.startswith("gsm_plus"):
        from datasets import load_dataset
        return load_dataset("qintongli/GSM-Plus", split="test", cache_dir=str(data_dir/"temp"))
    if name.startswith("gsm_symbolic_"):
        from datasets import load_dataset
        conf = name.removeprefix("gsm_symbolic_")
        return load_dataset("apple/GSM-Symbolic", name=conf, split="test", cache_dir=str(data_dir/"temp"))
    if name.startswith("math_perturb_"):
        return _rows_from_jsonl(_find_local(root / "MATH-Perturb" / "math_perturb",
                                             "simple" if name.endswith("simple") else "hard"))
    if name.startswith("omega_"):
        # Explicit exports: omega_<axis>.jsonl. Generation must use upstream
        # repo; no invented synthetic sample is silently substituted.
        path = root / "omega" / "exports" / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"OMEGA needs a validated normalized raw export: {path}")
        return _rows_from_jsonl(path)
    if name in ("processbench_gsm8k", "processbench_math", "processbench_olympiadbench", "processbench_omnimath"):
        from datasets import load_dataset
        return load_dataset("Qwen/ProcessBench", split=name.removeprefix("processbench_"),
                            cache_dir=str(data_dir/"temp"))
    if name == "prmbench_error_steps":
        from datasets import load_dataset
        # PRMBench is published as a single HF train split; it is an audit
        # dataset, not an official held-out test set.
        return load_dataset("hitsmy/PRMBench_Preview", split="train",
                            cache_dir=str(data_dir/"temp"))
    if name in ("proofflowbench_graph", "proofflowbench_proof"):
        # ProofFlowBench metadata contains an HF `Json` feature. Older versions
        # of `datasets` (including the user's eval environment) cannot decode
        # it: ValueError: Feature type 'Json' not found.  Read the published
        # original JSONL directly instead of touching the datasets dependency.
        from huggingface_hub import hf_hub_download
        upstream_path = Path(hf_hub_download(
            repo_id="rafaelcabral96/ProofFlowBench",
            repo_type="dataset",
            filename="proofflowbench.jsonl",
            revision="5141fd0",  # pinned official upload commit
            cache_dir=str(data_dir / "temp"),
        ))
        return _read_proofflowbench_raw(upstream_path)
    if name == "proofbench":
        from datasets import load_dataset
        # Upstream has only `train` (435 graded solutions for 145 unique
        # problems) and `best_of_n`; there is NO upstream `test` split.
        # We derive a local evaluation-only test.jsonl from unique train
        # problems, not from the model_solution or expert_rating labels.
        return load_dataset("wenjiema02/ProofBench", split="train", cache_dir=str(data_dir/"temp"))
    if name == "obligation_transfer":
        raise FileNotFoundError(f"Generate {data_dir}/obligation_transfer/test.jsonl with obligation-lab first")
    raise NotImplementedError(name)


def prepare(name, data_dir, split="test"):
    if name not in TRANSFER_DATASETS:
        raise NotImplementedError(name)
    if split != "test":
        raise ValueError(f"{name} is evaluation-only; split must be test, received {split}")
    path = Path(data_dir) / name / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return list(_rows_from_jsonl(path))
    upstream = _load_upstream(name, data_dir)
    if name == "prmbench_error_steps":
        # A subset of source error_steps reference steps that do not exist in
        # modified_process. Exclude entire records (not individual labels) and
        # make the exact source IDs and invalid annotations auditable.
        records, excluded, total = [], [], 0
        for original_index, raw in enumerate(upstream):
            total += 1
            try:
                normalized = normalize(name, raw, original_index)
            except PRMBenchAnnotationError as exc:
                excluded.append({
                    "original_index": original_index,
                    "source_id": raw.get("idx") if isinstance(raw, dict) else None,
                    "classification": raw.get("classification") if isinstance(raw, dict) else None,
                    "step_count": len(raw["modified_process"])
                        if isinstance(raw, dict) and isinstance(raw.get("modified_process"), list)
                        else None,
                    "error_steps": raw.get("error_steps") if isinstance(raw, dict) else None,
                    "reason": str(exc),
                })
                continue
            records.append(normalized)
        audit_path = path.parent / "prmbench_annotation_audit.json"
        audit_path.write_text(json.dumps({
            "policy": "exclude records with invalid 1-based error-step annotations; never repair gold labels",
            "source_dataset": "hitsmy/PRMBench_Preview",
            "source_split": "train",
            "source_count": total,
            "valid_count": len(records),
            "excluded_count": len(excluded),
            "excluded": excluded,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if excluded:
            warnings.warn(
                f"PRMBench error-steps: excluded {len(excluded)}/{total} "
                f"invalid annotation(s). Audit: {audit_path}", RuntimeWarning
            )
    elif name == "proofflowbench_graph":
        # Graph reconstruction cannot be fairly evaluated with corrupt gold
        # references. Quarantine *whole records*, never silently edit edges.
        records, excluded, total = [], [], 0
        for original_index, raw in enumerate(upstream):
            total += 1
            try:
                normalized = normalize(name, raw, original_index)
            except ProofFlowGraphError as exc:
                excluded.append({
                    "original_index": original_index,
                    "origin": raw.get("origin") if isinstance(raw, dict) else None,
                    "source_id": raw.get("id") if isinstance(raw, dict) else None,
                    "reason": str(exc),
                })
                continue
            records.append(normalized)
        audit_path = path.parent / "proofflowbench_annotation_audit.json"
        audit_path.write_text(json.dumps({
            "policy": "exclude invalid source graphs; do not repair or score altered gold",
            "source_revision": "5141fd0",
            "source_count": total,
            "valid_count": len(records),
            "excluded_count": len(excluded),
            "excluded": excluded,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if excluded:
            warnings.warn(
                f"ProofFlowBench graph: excluded {len(excluded)}/{total} "
                f"invalid gold DAG(s). Audit: {audit_path}", RuntimeWarning
            )
    else:
        records = [normalize(name, row, i) for i, row in enumerate(upstream)]
    if not records:
        raise ValueError(f"{name} has no samples")
    # Upstream `train` has several graded solutions per problem_id, but some
    # records disagree about the prompt. Never silently pair one statement
    # with a reference/rubric from another statement. Keep a strict majority
    # of whitespace-equivalent prompts; skip unresolved ties. Write an audit
    # of every conflict so the user can review the source data before research.
    if name == "proofbench":
        groups = {}
        for row in records:
            groups.setdefault(row["source_id"], []).append(row)
        unique, conflicts = [], []
        for key, variants in groups.items():
            # Collapse only whitespace, not mathematical notation or tokens.
            buckets = {}
            for row in variants:
                canonical = " ".join(row["question"].split())
                buckets.setdefault(canonical, []).append(row)
            if len(buckets) == 1:
                unique.append(variants[0])
                continue
            counts = Counter({prompt: len(rows) for prompt, rows in buckets.items()})
            winner, top_count = counts.most_common(1)[0]
            keep = top_count > len(variants) / 2
            conflicts.append({
                "problem_id": key,
                "decision": "majority" if keep else "skipped_unresolved",
                "variant_counts": [
                    {"question_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                     "n": len(rows), "preview": prompt[:160]}
                    for prompt, rows in buckets.items()
                ],
            })
            if keep:
                unique.append(buckets[winner][0])
        if conflicts:
            audit_path = path.parent / "proofbench_conflicts.json"
            audit_path.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            skipped = sum(row["decision"] == "skipped_unresolved" for row in conflicts)
            warnings.warn(
                f"ProofBench upstream has {len(conflicts)} conflicting problem_id(s); "
                f"skipped {skipped} without a strict majority. Review {audit_path} "
                "before reporting benchmark scores.", RuntimeWarning
            )
        records = unique
        if not records:
            raise ValueError("No unambiguous ProofBench problem statements remain")
    # Dedup paired GSM-Plus seeds by FAMILY ID; deterministic order is essential.
    if name == "gsm_plus_seed":
        unique = {}
        for row in records:
            unique.setdefault(row["family_id"], row)
        records = list(unique.values())
    # Force deterministic integer IDs; same idx across model checkpoints.
    for i, row in enumerate(records):
        row["idx"] = i
    temp = path.with_suffix(".jsonl.tmp")
    with open(temp, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)
    return records

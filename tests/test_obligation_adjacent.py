"""Small CPU-only regression tests; no model or full dataset download."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_datasets import normalize, prepare, _load_upstream, TRANSFER_DATASETS, ProofFlowGraphError, PRMBenchAnnotationError
from structure_eval import evaluate_structured, _read_edges
from llmjp_profile import make_user_prompt


GRAPH = {
    "origin": "example_source", "id": 7,
    "nl_theorem": "Prove P.", "nl_proof": "By condition C, lemma L, then P.",
    "proof_graph": [
        {"id": "tc_1", "statement": "Condition C", "natural_language": "Condition", "dependencies": []},
        {"id": "l1", "statement": "Lemma L", "natural_language": "Lemma", "dependencies": ["tc_1"]},
        {"id": "ts_1", "statement": "Conclude P", "natural_language": "Conclusion", "dependencies": ["l1"]},
    ],
}
PRM = {
    "modified_question": "Solve Q", "modified_process": ["First", "Wrong", "Wrong too"],
    "error_steps": [2, 3], "idx": "test-1", "classification": "circular"
}


class ObligationAdjacentTest(unittest.TestCase):
    def test_registered(self):
        for name in ("proofflowbench_graph", "proofflowbench_proof", "prmbench_error_steps", "processbench_omnimath"):
            self.assertIn(name, TRANSFER_DATASETS)

    def test_graph_normalization_and_no_gold_edge_list_in_prompt(self):
        row = normalize("proofflowbench_graph", GRAPH)
        self.assertEqual(json.loads(row["gt"])["edges"], [["l1", "ts_1"], ["tc_1", "l1"]])
        self.assertNotIn('"edges": [["l1", "ts_1"]', row["question"])
        self.assertEqual(row["node_ids"], ["tc_1", "l1", "ts_1"])
        self.assertIn("Condition C", row["question"])
        proof = normalize("proofflowbench_proof", GRAPH)
        self.assertEqual(proof["gt"], "")
        self.assertEqual(proof["gold_proof_graph"], GRAPH["proof_graph"])

    def test_graph_rejects_cycle_and_unknown_node(self):
        from copy import deepcopy
        row = deepcopy(GRAPH)
        row["proof_graph"][0]["dependencies"] = ["ts_1"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            normalize("proofflowbench_graph", row)
        row["proof_graph"][0]["dependencies"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "invalid dependency"):
            normalize("proofflowbench_graph", row)

    def test_upstream_invalid_references_quarantined_not_falsely_repaired(self):
        from copy import deepcopy
        malformed = deepcopy(GRAPH)
        malformed["id"] = 62
        malformed["proof_graph"][0]["id"] = "def_1"
        malformed["proof_graph"][0]["dependencies"] = ["def_1"]
        malformed["proof_graph"][2]["dependencies"] = ["l2"]
        with self.assertRaises(ProofFlowGraphError):
            normalize("proofflowbench_graph", malformed)
        # Proof-generation never uses gold graph dependencies as labels.
        proof = normalize("proofflowbench_proof", malformed)
        self.assertEqual(proof["graph_annotation_status"], "unvalidated_upstream")
        self.assertEqual(proof["reference_solution"], GRAPH["nl_proof"])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=[GRAPH, malformed]):
                with self.assertWarnsRegex(RuntimeWarning, "excluded 1/2"):
                    rows = prepare("proofflowbench_graph", tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["idx"], 0)
            a = json.loads((Path(tmp)/"proofflowbench_graph"/"proofflowbench_annotation_audit.json").read_text())
            self.assertEqual(a["source_count"], 2)
            self.assertEqual(a["valid_count"], 1)
            self.assertEqual(a["excluded_count"], 1)
            self.assertEqual(a["excluded"][0]["source_id"], 62)
            self.assertIn("self-dependency", a["excluded"][0]["reason"])
            # Cache must be reproducible without trying another download.
            with patch("transfer_datasets._load_upstream", side_effect=RuntimeError("no network")):
                self.assertEqual(len(prepare("proofflowbench_graph", tmp)), 1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=[GRAPH, malformed]):
                all_proofs = prepare("proofflowbench_proof", tmp)
            self.assertEqual(len(all_proofs), 2)
            self.assertEqual(all_proofs[1]["graph_annotation_status"], "unvalidated_upstream")

    def test_prm_data_and_empty_error_set(self):
        row = normalize("prmbench_error_steps", PRM)
        self.assertEqual(json.loads(row["gt"]), {"error_steps": [2, 3]})
        self.assertIn("Step 3: Wrong too", row["question"])
        self.assertEqual(row["step_count"], 3)
        self.assertEqual(row["type"], "circular")
        clean = {**PRM, "error_steps": []}
        self.assertEqual(json.loads(normalize("prmbench_error_steps", clean)["gt"]), {"error_steps": []})
        bad = {**PRM, "error_steps": [0]}
        with self.assertRaisesRegex(ValueError, "1-based"):
            normalize("prmbench_error_steps", bad)
        bad = {**PRM, "error_steps": [4]}
        with self.assertRaisesRegex(ValueError, "1-based"):
            normalize("prmbench_error_steps", bad)

    def test_prm_out_of_range_source_labels_quarantined_and_audited(self):
        # Upstream has examples with len(modified_process)=22 and a label=23.
        # These must not be shifted/clamped into a fabricated gold answer.
        corrupt = {**PRM, "idx": "confidence_prm_test_p1_44",
                   "modified_process": ["Reasoning"] * 22,
                   "error_steps": [19, 23]}
        with self.assertRaisesRegex(PRMBenchAnnotationError, "\\[1,22\\]"):
            normalize("prmbench_error_steps", corrupt)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=[PRM, corrupt]):
                with self.assertWarnsRegex(RuntimeWarning, "excluded 1/2"):
                    rows = prepare("prmbench_error_steps", tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_id"], "test-1")
            self.assertEqual(rows[0]["idx"], 0)
            audit = json.loads((Path(tmp)/"prmbench_error_steps"/
                               "prmbench_annotation_audit.json").read_text())
            self.assertEqual((audit["source_count"], audit["valid_count"], audit["excluded_count"]), (2, 1, 1))
            self.assertEqual(audit["excluded"][0]["source_id"], "confidence_prm_test_p1_44")
            self.assertEqual(audit["excluded"][0]["error_steps"], [19, 23])
            self.assertIn("[1,22]", audit["excluded"][0]["reason"])
            with patch("transfer_datasets._load_upstream", side_effect=RuntimeError("offline")):
                self.assertEqual(len(prepare("prmbench_error_steps", tmp)), 1)

    def test_prm_all_records_invalid_is_hard_error_with_audit(self):
        corrupt = {**PRM, "error_steps": [4]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=[corrupt]):
                with self.assertWarnsRegex(RuntimeWarning, "excluded 1/1"):
                    with self.assertRaisesRegex(ValueError, "no samples"):
                        prepare("prmbench_error_steps", tmp)
            self.assertFalse((Path(tmp)/"prmbench_error_steps"/"test.jsonl").exists())
            self.assertTrue((Path(tmp)/"prmbench_error_steps"/"prmbench_annotation_audit.json").exists())

    def test_prm_scoring_and_invalid_json(self):
        row = normalize("prmbench_error_steps", PRM)
        rows = [{**row, "idx": 0, "pred": ['{"error_steps": [2, 3]}', '{"error_steps": [2]}']},
                {**row, "idx": 1, "pred": ['{"error_steps": [2]}', 'not json']}]
        result_rows, result = evaluate_structured("prmbench_error_steps", rows)
        self.assertEqual([r["score"] for r in result_rows], [[True, False], [False, False]])
        self.assertEqual(result["structured_exact_acc_first"], 50)
        self.assertAlmostEqual(result["structured_micro_recall_first"], 75)
        self.assertEqual(result["acc"], None)
        self.assertEqual(result["structured_invalid_json_first"], 0)

    def test_empty_gold_invalid_cannot_get_perfect_f1(self):
        row = normalize("prmbench_error_steps", {**PRM, "error_steps": []})
        _, metric = evaluate_structured("prmbench_error_steps", [{**row, "idx": 0, "pred": ["INVALID"]}])
        self.assertEqual(metric["structured_macro_f1_first"], 0)
        self.assertEqual(metric["structured_invalid_json_first"], 1)
        _, metric = evaluate_structured("prmbench_error_steps", [{**row, "idx": 0, "pred": ['{"error_steps": []}']}])
        self.assertEqual(metric["structured_macro_f1_first"], 100)

    def test_graph_scoring_and_strict_dag(self):
        row = normalize("proofflowbench_graph", GRAPH)
        good = '{"edges": [["tc_1", "l1"], ["l1", "ts_1"]]}'
        cycled = '{"edges": [["tc_1", "l1"], ["l1", "tc_1"]]}'
        _, metric = evaluate_structured("proofflowbench_graph", [{**row, "idx": 0, "pred": ["```json\n" + good + "\n```", cycled]}])
        self.assertEqual(metric["structured_exact_acc_first"], 100)
        self.assertEqual(metric["structured_micro_f1_first"], 100)
        self.assertIsNone(_read_edges({"edges": [["tc_1", "l1"], ["l1", "tc_1"]]}, row["node_ids"]))

    def test_end_to_end_prepare_with_mocked_upstream_and_local_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, item in (("proofflowbench_graph", GRAPH), ("proofflowbench_proof", GRAPH),
                               ("prmbench_error_steps", PRM)):
                with patch("transfer_datasets._load_upstream", return_value=[item]):
                    result = prepare(name, tmp)
                self.assertEqual(result[0]["idx"], 0)
                self.assertTrue((Path(tmp) / name / "test.jsonl").exists())
                with patch("transfer_datasets._load_upstream", side_effect=RuntimeError("Do not download")):
                    self.assertEqual(len(prepare(name, tmp)), 1)

    def test_sources_and_prompt(self):
        calls = []
        def mock_load(dataset_id, **kw):
            calls.append((dataset_id, kw))
            return []
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=mock_load)}):
            for name in ("prmbench_error_steps", "processbench_omnimath"):
                _load_upstream(name, "/tmp/none")
        self.assertEqual([c[0] for c in calls], ["hitsmy/PRMBench_Preview", "Qwen/ProcessBench"])
        self.assertEqual([c[1]["split"] for c in calls], ["train", "omnimath"])
        self.assertNotIn("\\boxed", make_user_prompt("A", "proofflowbench_graph"))
        self.assertNotIn("\\boxed", make_user_prompt("A", "prmbench_error_steps"))

    def test_proofflow_raw_download_bypasses_incompatible_hf_features(self):
        # `datasets` must not be imported, even with upstream Json metadata.
        for mode in ("jsonl", "array"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "proof.jsonl"
                if mode == "jsonl":
                    source.write_text(json.dumps(GRAPH) + "\n", encoding="utf-8")
                else:
                    source.write_text(json.dumps([GRAPH]), encoding="utf-8")
                def mock_download(**kw):
                    self.assertEqual(kw["repo_type"], "dataset")
                    self.assertEqual(kw["filename"], "proofflowbench.jsonl")
                    self.assertEqual(kw["revision"], "5141fd0")
                    return str(source)
                with patch.dict(sys.modules, {
                    "huggingface_hub": SimpleNamespace(hf_hub_download=mock_download),
                    "datasets": None,
                }):
                    rows = list(_load_upstream("proofflowbench_graph", tmp))
                self.assertEqual(rows, [GRAPH])

    def test_shell_syntax(self):
        for shell in ("sh/mdx/download_extra_datasets.sh", "sh/mdx/download_obligation_datasets.sh",
                      "sh/mdx/_eval_mdx_common.sh",
                      "sh/mdx/eval.sh", "sh/mdx/eval_llmjp.sh", "sh/mdx/eval_phi.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / shell)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_datasets import normalize, prepare


class TransferAdaptersTest(unittest.TestCase):
    def test_gsm_plus_family_and_answer(self):
        row = {'question':'changed', 'answer':'20','solution':'step #### 20',
               'seed_question':'original', 'seed_answer':'10','seed_solution':'step #### 10',
               'perturbation_type':'numerical substitution'}
        a=normalize('gsm_plus',row)
        b=normalize('gsm_plus_seed',row)
        self.assertEqual(a['family_id'],b['family_id'])
        self.assertEqual((a['gt'],b['gt']),('20','10'))
        self.assertNotEqual(a['variant'],b['variant'])
    def test_symbolic_strict_gold(self):
        row={'id':2, 'instance':3, 'original_id':100,
             'question':'Q', 'answer':'answer reason. #### 12'}
        out=normalize('gsm_symbolic_p1',row)
        self.assertEqual(out['gt'],'12')
        self.assertEqual(out['family_id'],'gsm_symbolic:100')
        row['answer']='ambiguous 12'
        with self.assertRaisesRegex(ValueError,'####'):
            normalize('gsm_symbolic_p1',row)
    def test_math_pair_and_process(self):
        src={'problem_id':'p-1','problem':'Minimize x^2','answer':'0','type':'Algebra'}
        a=normalize('math_perturb_simple',src)
        b=normalize('math_perturb_hard',src)
        self.assertEqual(a['family_id'],b['family_id'])
        p=normalize('processbench_gsm8k',{'problem':'audit','steps':['A','B'], 'label':1})
        self.assertEqual(p['gt'],'1')
        self.assertIn('<paragraph_0>',p['question'])
    def test_omega_schema_requires_gold(self):
        with self.assertRaisesRegex(ValueError,'answer'):
            normalize('omega_exploratory',{'problem':'Q','family_id':'fam'})
        row=normalize('omega_compositional',{'problem':'Q','final_answer':2,'problem_type':'x'})
        self.assertEqual(row['gt'],'2')
    def test_proofbench_uses_existing_upstream_train_split(self):
        import sys
        from transfer_datasets import _load_upstream
        calls = []
        def fake_load_dataset(dataset_id, **kwargs):
            calls.append((dataset_id, kwargs))
            return []
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=fake_load_dataset)}):
                _load_upstream("proofbench", tmp)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "wenjiema02/ProofBench")
        self.assertEqual(calls[0][1]["split"], "train")

    def test_proofbench_deduplicates_generated_solutions_by_problem(self):
        source = [
            {"problem_id": "P1", "problem": "Prove statement 1", "generator": g,
             "reference_solution": "Reference 1", "marking_scheme": "Rubric 1"}
            for g in ("model_a", "model_b", "model_c")
        ] + [
            {"problem_id": "P2", "problem": "Prove statement 2", "generator": g,
             "reference_solution": "Reference 2", "marking_scheme": "Rubric 2"}
            for g in ("model_a", "model_b", "model_c")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=source):
                rows = prepare("proofbench", tmp)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["source_id"] for row in rows], ["P1", "P2"])
            self.assertEqual([row["idx"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["gt"], "")
            self.assertEqual(rows[1]["marking_scheme"], "Rubric 2")
            saved = Path(tmp)/"proofbench"/"test.jsonl"
            self.assertEqual(len(saved.read_text().splitlines()), 2)
            # Re-using the cached local test split must not download again.
            with patch("transfer_datasets._load_upstream", side_effect=RuntimeError("unwanted download")):
                self.assertEqual(len(prepare("proofbench", tmp)), 2)

    def test_proofbench_majority_and_audit_conflicts(self):
        source = [
            {"problem_id": "P1", "problem": "Prove\n A", "marking_scheme": "correct"},
            {"problem_id": "P1", "problem": "Prove A", "marking_scheme": "correct"},
            {"problem_id": "P1", "problem": "Different problem", "marking_scheme": "incorrect"},
            {"problem_id": "P2", "problem": "First statement"},
            {"problem_id": "P2", "problem": "Other statement"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=source):
                with self.assertWarnsRegex(RuntimeWarning, "conflicting problem_id"):
                    rows = prepare("proofbench", tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_id"], "P1")
            self.assertEqual(rows[0]["marking_scheme"], "correct")
            audit = json.loads((Path(tmp)/"proofbench"/"proofbench_conflicts.json").read_text())
            self.assertEqual(len(audit), 2)
            self.assertEqual(audit[0]["decision"], "majority")
            self.assertEqual(audit[1]["decision"], "skipped_unresolved")

    def test_proofbench_all_conflicted_fails_without_emitting_bad_eval_file(self):
        source = [
            {"problem_id": "P2", "problem": "First statement"},
            {"problem_id": "P2", "problem": "Other statement"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("transfer_datasets._load_upstream", return_value=source):
                with self.assertWarns(RuntimeWarning):
                    with self.assertRaisesRegex(ValueError, "No unambiguous"):
                        prepare("proofbench", tmp)
            self.assertFalse((Path(tmp)/"proofbench"/"test.jsonl").exists())

    def test_locally_generated_obligation_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'obligation_transfer'; data.mkdir()
            (data/'test.jsonl').write_text(json.dumps({'idx':0,'gt':'68','gt_cot':'',
                'question':'Q','family_id':'f','variant':'base','transfer_type':'base'})+'\n')
            out=prepare('obligation_transfer',tmp)
            self.assertEqual(out[0]['family_id'],'f')
            with self.assertRaises(ValueError):
                prepare('obligation_transfer',tmp,'train')

if __name__=='__main__':unittest.main()

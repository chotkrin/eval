import argparse
import math
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def compute_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False, ks=[1, 5, 10]):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # ProofBench grades natural-language proofs against expert marking schemes.
    # No mathematically meaningful pass@k exists until a proof grader is run.
    # Keep the generated text and scores unassigned instead of reporting 0%.
    if data_name in ("proofbench", "proofflowbench_proof"):
        for sample in samples:
            sample["score"] = [None] * len(sample["pred"])
            sample["pass_at_k"] = {}
        return samples, {"num_samples": len(samples),
                         "num_scores": 0, "acc": None,
                         "pass@1": None, "pass@5": None, "pass@10": None,
                         "evaluation_status": "proof_grading_required",
                         "scorer": "none; requires expert or validated proof checker"}

    if data_name in ("proofflowbench_graph", "prmbench_error_steps"):
        from structure_eval import evaluate_structured
        return evaluate_structured(data_name, samples, ks=ks)

    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    pass_at_k_tracker = {k: [] for k in ks}
    for sample in samples:
        sample_scores = scores[idx: idx+len(sample['pred'])]
        sample['score'] = sample_scores
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

        n = len(sample_scores)
        c = sum(sample_scores)

        sample['pass_at_k'] = {}
        for k in ks:
            if k <= n:
                p_at_k = compute_pass_at_k(n, c, k)
                sample['pass_at_k'][f"pass@{k}"] = p_at_k
                pass_at_k_tracker[k].append(p_at_k)

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    for k in ks:
        if pass_at_k_tracker[k]:
            result_json[f"pass@{k}"] = float(np.round(np.mean(pass_at_k_tracker[k]) * 100, 1))

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    if data_name.startswith("processbench_"):
        # Official ProcessBench metric: harmonic mean of accuracies on traces
        # with an error (label != -1) and traces without one (label == -1).
        # For n>1 we explicitly use candidate 0, not an implicit best-of-n.
        error = [s["score"][0] for s in samples if str(s["gt"]) != "-1"]
        correct = [s["score"][0] for s in samples if str(s["gt"]) == "-1"]
        error_acc = 100 * sum(error) / len(error) if error else None
        correct_acc = 100 * sum(correct) / len(correct) if correct else None
        f1 = (2 * error_acc * correct_acc / (error_acc + correct_acc)
              if error_acc is not None and correct_acc is not None and
              (error_acc + correct_acc) > 0 else
              0.0 if error_acc is not None and correct_acc is not None else None)
        result_json.update({"processbench_error_acc": error_acc,
                            "processbench_correct_acc": correct_acc,
                            "processbench_f1": f1,
                            "processbench_f1_sample_index": 0})

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)

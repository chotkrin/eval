import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl, lower_keys


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            # evaluate on training set + test set
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "gpqa_diamond":
            # GPQA uses only a `train` split on Hugging Face, even though this
            # subset is used as an evaluation benchmark.  The official dataset
            # stores the correct answer separately, so create a deterministic
            # A-D permutation once and cache the normalized rows as test.jsonl.
            # Keeping the permutation fixed across checkpoints is important for
            # low-variance A-L comparisons.
            try:
                raw_dataset = load_dataset(
                    "Idavidrein/gpqa",
                    "gpqa_diamond",
                    split="train",
                    cache_dir=f"{data_dir}/temp",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the official gated GPQA dataset. "
                    "Accept the Idavidrein/gpqa access terms on Hugging Face "
                    "and make sure this job can see your HF token. "
                    "Alternatively pre-create "
                    f"{data_dir}/gpqa_diamond/{split}.jsonl."
                ) from exc

            gpqa_examples = []
            rng = random.Random(42)
            labels = "ABCD"
            for row in raw_dataset:
                choices = [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]
                order = list(range(4))
                rng.shuffle(order)
                shuffled_choices = [choices[i] for i in order]
                answer = labels[order.index(0)]
                gpqa_examples.append(
                    {
                        "question": row["Question"],
                        "choices": shuffled_choices,
                        "answer": answer,
                        "solution": row.get("Explanation", ""),
                        "type": row.get("High-level domain", ""),
                        "subfield": row.get("Subdomain", ""),
                        "record_id": row.get("Record ID", ""),
                    }
                )
            dataset = Dataset.from_list(gpqa_examples)
        elif data_name == "mmlu_pro":
            # Retained for backwards compatibility, but the default evaluation
            # suite now uses MMLU-STEM instead of MMLU-Pro.
            dataset = load_dataset(
                "TIGER-Lab/MMLU-Pro",
                split=split,
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name.startswith("livecodebench_"):
            # Use an explicit fine-grained LiveCodeBench version in the data
            # name (e.g. livecodebench_v6) so cached files cannot silently mix
            # temporal releases.  For v6, load test6.jsonl directly from the
            # official dataset repository. This avoids the known compatibility
            # problems between the legacy dataset loading script and newer
            # versions of `datasets`.
            version_tag = data_name[len("livecodebench_"):]
            if not version_tag:
                raise ValueError(
                    "LiveCodeBench data name must include a version, e.g. "
                    "livecodebench_v6"
                )
            if not (version_tag.startswith("v") and version_tag[1:].isdigit()):
                raise ValueError(
                    "This lightweight loader expects a fine-grained version "
                    "such as livecodebench_v6, not a cumulative release."
                )
            filename = f"test{int(version_tag[1:])}.jsonl"
            try:
                from huggingface_hub import hf_hub_download
                local_jsonl = hf_hub_download(
                    repo_id="livecodebench/code_generation_lite",
                    filename=filename,
                    repo_type="dataset",
                    cache_dir=f"{data_dir}/temp/hf_hub",
                )
                dataset = load_dataset(
                    "json",
                    data_files=local_jsonl,
                    split="train",
                    cache_dir=f"{data_dir}/temp",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load LiveCodeBench {version_tag} from "
                    f"livecodebench/code_generation_lite/{filename}. "
                    "Check Hugging Face connectivity and huggingface_hub. "
                    "You can also pre-create the normalized cache file at "
                    f"{data_file}."
                ) from exc
        elif data_name == "mmlu_stem":
            # Use the canonical CAIS MMLU mirror. `hails/mmlu_no_train` can
            # trigger NonMatchingSplitsSizesError with newer Hugging Face
            # datasets/cache metadata. CAIS exposes the same normalized
            # question/choices/answer/subject schema and the official 14,042
            # example test split.
            dataset = load_dataset(
                "cais/mmlu",
                "all",
                split="test",
                cache_dir=f"{data_dir}/temp",
            )
            # only keep stem subjects
            stem_subjects = [
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples

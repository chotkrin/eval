"""Qwen3 / Qwen3.5 offline evaluation using the shared checkpoint and grader.

Unlike LLM-jp, Qwen uses its own chat template and <think> channel.
"""
import os

if __name__ == "__main__":
    os.environ["QWEN_EVAL_ENTRYPOINT"] = "1"
    from llmjp_eval import main
    main()

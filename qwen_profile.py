"""Qwen3 / Qwen3.5 text-only thinking evaluation protocol.

References: https://huggingface.co/Qwen/Qwen3-8B
            https://huggingface.co/Qwen/Qwen3.5-9B
"""
from llmjp_profile import make_user_prompt


def profile_version(prompt_type):
    versions = {
        "qwen3-thinking": "qwen3-thinking-v1",
        "qwen35-thinking": "qwen35-thinking-v1",
    }
    return versions[prompt_type]


def validate_qwen_tokenizer(tokenizer):
    vocab = tokenizer.get_vocab()
    required = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    missing = [x for x in required if x not in vocab]
    # Some Qwen tokenizers treat the reasoning delimiters as plain text rather
    # than special tokens; validate message markers and the chat template only.
    if any(x in missing for x in required[:2]):
        raise ValueError(f"Qwen tokenizer lacks ChatML message markers: {missing}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Qwen tokenizer has no chat_template")


def make_prompt(tokenizer, question, data_name):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": make_user_prompt(question, data_name)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def decode_response(tokenizer, token_ids):
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    # A Qwen chat prefill may already contain <think>. The output can thus
    # begin inside the reasoning block, with only a closing </think> marker.
    text = raw.split("<|im_end|>", 1)[0].strip()
    if "</think>" in text:
        before, after = text.rsplit("</think>", 1)
        reasoning = before.replace("<think>", "").strip()
        final = after.strip()
    elif "<think>" in text:
        reasoning, final = text.split("<think>", 1)[1].strip(), ""
    else:
        # Thinking-mode Qwen may have its opening <think> in the chat
        # template prefill. Without the closing marker, never score it.
        reasoning, final = text, ""
    # Do not score unterminated thinking or tool-message control tokens.
    if "<think>" in final or "<|im_start|>" in final:
        final = ""
    return {
        "final": final, "reasoning": reasoning, "raw_harmony": raw,
        "has_final": bool(final), "unexpected_channels": [],
    }

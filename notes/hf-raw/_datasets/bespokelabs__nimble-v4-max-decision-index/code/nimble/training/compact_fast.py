"""compact_v1 prompts built fast: token for token what nimble/training/compact_schema.py builds.

compact_schema.prepare_prompts is the reference; models trained on compact_v1 pin its file by hash, so it does not
change. It encodes each question's whole prompt, and then the whole prompt again once per option to check the
option codes, which takes about a second for a long request with many options. This builds the same output with
far less tokenizer work:

- Every question's prompt is the shared text up to "Requested field:" followed by " <name>" and the assistant
  tail. Qwen's pre-tokenizer always starts a new piece at the space after that colon, so the shared part is
  encoded once and only each question's short ending is encoded per question.
- Chat control tokens are tokenization boundaries, so option codes are checked on the assistant tail alone, as
  extended_schema does for the JSON prompt (and with its cache).

tests/test_compact_schema.py checks that both builders agree on real requests.
"""
from nimble.scoring import extended_schema
from nimble.scoring.parallel_schema import PreparedPrompts
from nimble.training import compact_schema

SPLIT = "Requested field:"


def prepare_prompts(tokenizer, context, schema, max_input_tokens, system_role=True):
    compact_schema.validate_schema(schema)
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a nonempty string.")
    names = list(schema)
    choices = [compact_schema.choices_for(schema[name]) for name in names]
    content = compact_schema.render(context, schema, tokenizer) + compact_schema.MARKER
    system = compact_schema.SYSTEM_PROMPT
    messages = ([{"role": "system", "content": system}, {"role": "user", "content": content}]
                if system_role else [{"role": "user", "content": system + "\n\n" + content}])
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    start, end = template.rsplit(compact_schema.MARKER, 1)
    cut = start.rindex(SPLIT) + len(SPLIT)  # start ends with "Requested field: "
    head_ids = tokenizer.encode(start[:cut], add_special_tokens=False)
    full_ids = [head_ids + tokenizer.encode(start[cut:] + compact_schema.clean(name, tokenizer) + end,
                                            add_special_tokens=False) for name in names]
    prefix = list(head_ids)  # the exact common token prefix of the prompts, as the reference computes it
    for ids in full_ids:
        n = 0
        while n < min(len(prefix), len(ids)) and prefix[n] == ids[n]:
            n += 1
        prefix = prefix[:n]
    rest = tokenizer.encode(start[cut:], add_special_tokens=False)  # the reference's prefix is encode(start)
    common = [ids[len(head_ids):] for ids in full_ids]
    k = 0
    while k < len(rest) and all(len(c) > k and c[k] == rest[k] for c in common):
        k += 1
    if len(prefix) == len(head_ids):
        prefix = prefix + rest[:k]
    if not prefix:
        raise ValueError("No reusable token prefix was found.")
    if max(map(len, full_ids)) > max_input_tokens:
        raise ValueError(f"Longest prompt has {max(map(len, full_ids))} tokens; limit is {max_input_tokens}. "
                         "Nothing was truncated.")
    specials = tokenizer.all_special_tokens
    candidates = []
    for name, ids, values in zip(names, full_ids, choices):
        prompt = start + compact_schema.clean(name, tokenizer) + end
        boundary = max((prompt.rfind(special) for special in specials), default=-1)
        tail = prompt[boundary:] if boundary >= 0 else prompt
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        if ids[-len(tail_ids):] != tail_ids:
            raise ValueError("Assistant tail does not match full prompt tokenization")
        codes = list(extended_schema.candidate_suffix(tokenizer, tail,
                                                      tuple(compact_schema.codes_for(len(values), tokenizer))))
        if len(set(codes)) != len(codes):
            raise ValueError("Choice codes must have distinct token IDs.")
        candidates.append(codes)
    return PreparedPrompts(names, choices, prefix, [ids[len(prefix):] for ids in full_ids], full_ids, candidates)

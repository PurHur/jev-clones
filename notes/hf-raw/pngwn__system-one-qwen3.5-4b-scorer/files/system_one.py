"""System One / "Jev"-style scorer.

A Qwen3.5 backbone with a scalar scoring head that returns a calibrated
probability distribution over the options of a typed question, computed in a
single forward pass. No autoregressive generation and no string output: every
option is scored, the scores are grouped per question and softmaxed, so the
output space is exactly the option set the caller supplied.

Question types: "noul" (yes/no), "choice" (unordered pick), "score" (ordered).

Modes:
  build     build the (state, question, options, answer) dataset from public sources
  train     LoRA-train the scoring head, evaluate, temperature-calibrate, push
  eval      evaluate a trained scoring model on held-out questions
  baseline  prompted base-model baseline: list the options, score the option letters

Design notes that matter:
  * encode() never truncates the question or the option. The head pools the last
    non-pad token, so the option must stay legible at the end of the sequence;
    the state is truncated from its end to make room.
  * Train-time option sets are capped (--max-options, default 16): the correct
    option plus sampled negatives. banking77 (77 options) and tickets_queue (52)
    are otherwise 159k sequences per epoch. Evaluation always scores the FULL
    option set, so the reported numbers are on the real task.
  * Questions are bucketed by option count so every batch is rectangular and the
    per-question softmax is a single reshaped view.
  * Temperature is fitted on val and applied to test; never fitted on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi

BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
DATA_REPO = "pngwn/system-one-decisions"
MODEL_REPO = "pngwn/system-one-qwen3.5-4b-scorer"
TRACKIO_SPACE = "pngwn/system-one-qwen3-5-4b-scorer-trackio"
TRACKIO_PROJECT = "system-one-qwen3.5-4b-scorer"
SCRIPT_VERSION = "v5"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# q_proj/v_proj reach only the 1-in-4 full-attention layers; the rest are linear
# attention (in_proj_qkv / in_proj_z / out_proj). Adapt every linear projection.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]


# --------------------------------------------------------------------- render

def encode(tok, state, question, option, max_len):
    """Token ids for one scoring instance.

    The tail (question + option) is kept intact and the state is truncated from
    its end, because the head pools the last non-pad token.
    """
    tail_ids = tok("\n\nQuestion:\n" + question + "\n\nOption:\n" + option,
                   add_special_tokens=False)["input_ids"]
    if len(tail_ids) >= max_len:
        return tail_ids[-max_len:]
    head_ids = tok("State:\n" + state, add_special_tokens=False)["input_ids"]
    return head_ids[: max_len - len(tail_ids)] + tail_ids


def pad_batch(seqs, pad_id, max_len):
    """Right padding: the head pools the last non-pad token."""
    L = min(max(len(s) for s in seqs), max_len)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return ids, attn


def humanize(s):
    return s.replace("_", " ").replace("-", " ").strip()


def head_hash(model):
    cands = [(n, p) for n, p in model.named_parameters()
             if "score" in n and p.requires_grad]
    if not cands:
        cands = [(n, p) for n, p in model.named_parameters()
                 if "modules_to_save" in n]
    if not cands:
        return "no-score-head-found"
    n, p = cands[0]
    h = hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
    return h + ":" + n


# ---------------------------------------------------------------------- build

def build_banking77(n_train, n_val, n_test):
    from datasets import load_dataset

    ds = load_dataset("legacy-datasets/banking77")
    names = ds["train"].features["label"].names
    options = [humanize(n) for n in names]
    q = "Which banking intent does this customer message have?"
    tr, te = ds["train"].shuffle(seed=0), ds["test"].shuffle(seed=0)

    def conv(ex):
        return {"task": "banking77", "question_type": "choice", "ordered": False,
                "state": ex["text"], "question": q, "options": options,
                "answer_index": int(ex["label"])}

    return {"train": [conv(x) for x in tr.select(range(n_train))],
            "val": [conv(x) for x in tr.select(range(n_train, n_train + n_val))],
            "test": [conv(x) for x in te.select(range(n_test))]}


def build_go_emotions(n_train, n_val, n_test):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", name="simplified")
    names = ds["train"].features["labels"].feature.names

    def conv(ex, rng):
        pos = sorted(set(int(x) for x in ex["labels"]))
        neg = [j for j in range(len(names)) if j not in pos]
        rng.shuffle(neg)
        rows = []
        for j, ans in [(j, 0) for j in pos[:2]] + [(j, 1) for j in neg[:2]]:
            rows.append({
                "task": "go_emotions", "question_type": "noul", "ordered": False,
                "state": ex["text"],
                "question": 'Does this message express the emotion "' + names[j] + '"?',
                "options": ["yes", "no"], "answer_index": ans})
        return rows

    def expand(split, n):
        rng = random.Random(0)
        out = []
        for ex in split.select(range(min(n, len(split)))):
            out.extend(conv(ex, rng))
        return out

    tr, va = ds["train"].shuffle(seed=0), ds["validation"].shuffle(seed=0)
    return {"train": expand(tr, n_train), "val": expand(va, n_val),
            "test": expand(va.select(range(len(va) // 2, len(va))), n_test)}


def build_ag_news(n_train, n_val, n_test):
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    names = list(ds["train"].features["label"].names)
    q = "Which topic is this news article about?"
    tr, te = ds["train"].shuffle(seed=0), ds["test"].shuffle(seed=0)

    def conv(ex):
        return {"task": "ag_news", "question_type": "choice", "ordered": False,
                "state": ex["text"], "question": q, "options": names,
                "answer_index": int(ex["label"])}

    return {"train": [conv(x) for x in tr.select(range(n_train))],
            "val": [conv(x) for x in tr.select(range(n_train, n_train + n_val))],
            "test": [conv(x) for x in te.select(range(n_test))]}


def build_mmlu(n_train, n_val, n_test):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", name="all")
    q = "Which of the options is the correct answer to the question in the state?"
    tr, va = ds["auxiliary_train"].shuffle(seed=0), ds["validation"].shuffle(seed=0)

    def conv(ex):
        return {"task": "mmlu", "question_type": "choice", "ordered": False,
                "state": ex["question"], "question": q, "options": list(ex["choices"]),
                "answer_index": int(ex["answer"])}

    return {"train": [conv(x) for x in tr.select(range(n_train))],
            "val": [conv(x) for x in va.select(range(n_val))],
            "test": [conv(x) for x in va.select(range(n_val, n_val + n_test))]}


def build_yelp(n_train, n_val, n_test):
    from datasets import load_dataset

    ds = load_dataset("Yelp/yelp_review_full", name="yelp_review_full")
    options = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    q = "On a 1 to 5 star scale, how would the customer rate this experience?"
    tr, te = ds["train"].shuffle(seed=0), ds["test"].shuffle(seed=0)

    def conv(ex):
        return {"task": "yelp_score", "question_type": "score", "ordered": True,
                "state": ex["text"][:4000], "question": q, "options": options,
                "answer_index": int(ex["label"])}

    return {"train": [conv(x) for x in tr.select(range(n_train))],
            "val": [conv(x) for x in tr.select(range(n_train, n_train + n_val))],
            "test": [conv(x) for x in te.select(range(n_test))]}


def build_tickets(n_train, n_val, n_test):
    from datasets import load_dataset

    ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    queues = sorted(set(ds["queue"]))
    q_options = [humanize(x) for x in queues]
    prios = ["very low", "low", "medium", "high", "critical"]
    p_index = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    types = ["Incident", "Request", "Problem", "Change"]
    tr = ds.shuffle(seed=0)

    def conv(ex):
        state = ((ex.get("subject") or "") + "\n\n" + (ex.get("body") or ""))[:4000]
        if not state.strip():
            return []
        rows = []
        if ex["queue"] in queues:
            rows.append({"task": "tickets_queue", "question_type": "choice",
                         "ordered": False, "state": state,
                         "question": "Which support queue should this ticket be routed to?",
                         "options": q_options,
                         "answer_index": queues.index(ex["queue"])})
        if ex["priority"] in p_index:
            rows.append({"task": "tickets_priority", "question_type": "score",
                         "ordered": True, "state": state,
                         "question": "What is the priority of this ticket?",
                         "options": prios, "answer_index": p_index[ex["priority"]]})
        if ex["type"] in types:
            rows.append({"task": "tickets_type", "question_type": "choice",
                         "ordered": False, "state": state,
                         "question": "What type of ticket is this?",
                         "options": types, "answer_index": types.index(ex["type"])})
        if ex["language"] in ("en", "de"):
            rows.append({"task": "tickets_language", "question_type": "noul",
                         "ordered": False, "state": state,
                         "question": "Is this ticket written in English?",
                         "options": ["yes", "no"],
                         "answer_index": 0 if ex["language"] == "en" else 1})
        return rows

    def expand(n, offset=0):
        out = []
        for ex in tr.select(range(offset, min(offset + n, len(tr)))):
            out.extend(conv(ex))
        return out

    return {"train": expand(n_train), "val": expand(n_val, n_train),
            "test": expand(n_test, n_train + n_val)}


BUILDERS = {"banking77": build_banking77, "go_emotions": build_go_emotions,
            "ag_news": build_ag_news, "mmlu": build_mmlu,
            "yelp_score": build_yelp, "tickets": build_tickets}


def build_data(args):
    from datasets import Dataset, DatasetDict

    sizes = json.loads(args.sizes)
    out = {"train": [], "val": [], "test": []}
    for name, builder in BUILDERS.items():
        n_train, n_val, n_test = sizes.get(name, [0, 0, 0])
        if not n_train:
            continue
        parts = builder(n_train, n_val, n_test)
        for split in out:
            out[split].extend(parts[split])
        print("built", name, {s: len(parts[s]) for s in parts}, flush=True)
    dd = DatasetDict({s: Dataset.from_list(v) for s, v in out.items() if v})
    for s in dd:
        print("split", s, len(dd[s]), "rows", flush=True)
    if args.local_data_dir:
        dd.save_to_disk(args.local_data_dir)
        print("saved to", args.local_data_dir, flush=True)
    if args.push:
        dd.push_to_hub(args.data_repo, private=False, token=os.environ.get("HF_TOKEN"))
        print("pushed to", args.data_repo, flush=True)


# ---------------------------------------------------------------------- model

def load_model(base_model, num_labels=1, grad_ckpt=False, lora=True, lora_r=16,
               lora_alpha=32, lora_dropout=0.05, target_modules=None, dtype=None):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5TextForSequenceClassification

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Qwen3_5TextForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels, dtype=dtype)
    # config pad/eos are None on this checkpoint; force the tokenizer's ids onto
    # both the top config and the text sub-config or padding ids go unset.
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    tc = model.config.get_text_config()
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    tc.num_labels = num_labels
    model.num_labels = num_labels
    assert model.config.pad_token_id is not None
    if lora:
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          bias="none", task_type=TaskType.SEQ_CLS,
                          target_modules=target_modules or LORA_TARGETS,
                          modules_to_save=["score"])
        model = get_peft_model(model, lcfg)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return tok, model


def load_data(args):
    if args.local_data_dir:
        from datasets import load_from_disk

        return load_from_disk(args.local_data_dir)
    from datasets import load_dataset

    return load_dataset(args.data_repo)


def effective_k(row, max_options):
    n = len(row["options"])
    if not max_options or max_options <= 0:
        return n
    return min(n, max_options)


class Batcher:
    """Builds the scoring instances of one batch. Every row in the batch has the
    same effective option count k."""

    def __init__(self, tok, max_len, seed=0):
        self.tok = tok
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __call__(self, rows, k):
        seqs, labels = [], []
        for r in rows:
            opts, ans = list(r["options"]), int(r["answer_index"])
            assert 0 <= ans < len(opts), (r["task"], ans, len(opts))
            n = len(opts)
            if n > k:
                negs = [i for i in range(n) if i != ans]
                self.rng.shuffle(negs)
                chosen = [ans] + negs[: k - 1]
                self.rng.shuffle(chosen)
            else:
                chosen = list(range(n))
            assert len(chosen) == k, (len(chosen), k, n)
            for oi in chosen:
                seqs.append(encode(self.tok, r["state"], r["question"],
                                   opts[oi], self.max_len))
            labels.append(chosen.index(ans))
        ids, attn = pad_batch(seqs, self.tok.pad_token_id, self.max_len)
        assert ids.shape[0] == len(rows) * k, (ids.shape, len(rows), k)
        return {"input_ids": ids, "attention_mask": attn,
                "labels": torch.tensor(labels, dtype=torch.long), "k": k}


def buckets(rows, batch_size, seed=0, max_options=0):
    """(indices, k) batches so every batch shares one effective option count."""
    by_k = defaultdict(list)
    for i, r in enumerate(rows):
        by_k[effective_k(r, max_options)].append(i)
    rng = random.Random(seed)
    batches = []
    for k, idxs in by_k.items():
        rng.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batches.append((idxs[i: i + batch_size], k))
    rng.shuffle(batches)
    return batches


def stats_report(tok, rows, args, sample_per_task=8):
    """Per-task question counts, sequence counts and token estimate for sizing."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    print("TASK STATS (one epoch)", flush=True)
    grand, grand_tok = 0, 0
    for task in sorted(by_task):
        rs = by_task[task]
        seqs = sum(effective_k(r, args.max_options) for r in rs)
        lens = []
        for r in rs[:sample_per_task]:
            for oi in range(effective_k(r, args.max_options)):
                lens.append(len(encode(tok, r["state"], r["question"],
                                       r["options"][oi], args.max_len)))
        mean_len = sum(lens) / max(1, len(lens))
        grand += seqs
        grand_tok += int(seqs * mean_len)
        print("  %-18s questions %6d  options %3d  k %3d  seqs %7d  mean_len %6.1f  est_tokens %10d"
              % (task, len(rs), len(rs[0]["options"]),
                 effective_k(rs[0], args.max_options), seqs, mean_len,
                 int(seqs * mean_len)), flush=True)
    bs = buckets(rows, args.batch_size, seed=0, max_options=args.max_options)
    print("TOTAL seqs/epoch %d  est_tokens/epoch %d" % (grand, grand_tok), flush=True)
    print("batches/epoch %d (batch_size %d)  max seqs in one batch %d"
          % (len(bs), args.batch_size, max((k for _, k in bs), default=0) * args.batch_size),
          flush=True)
    return grand


# ----------------------------------------------------------------- evaluation

@torch.no_grad()
def score_options(model, tok, state, question, options, max_len, batch_size, device):
    seqs = [encode(tok, state, question, o, max_len) for o in options]
    logits = []
    for i in range(0, len(seqs), batch_size):
        ids, attn = pad_batch(seqs[i: i + batch_size], tok.pad_token_id, max_len)
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device))
        logits.extend(out.logits.squeeze(-1).float().cpu().tolist())
    return logits


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def ece(probs, answers, n_bins=10):
    P = [np.asarray(p, dtype=float) for p in probs]
    conf = np.array([p.max() for p in P], dtype=float)
    pred = np.array([int(np.argmax(p)) for p in P])
    correct = pred == np.array(answers)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def brier(probs, answers):
    if not probs:
        return float("nan")
    tot = 0.0
    for p, a in zip(probs, answers):
        p = np.asarray(p, dtype=float)
        y = np.zeros(p.shape[0])
        y[int(a)] = 1.0
        tot += float(((p - y) ** 2).sum())
    return tot / len(probs)


def metrics_from_records(records):
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    out = {}
    for task in sorted(by_task) + ["ALL"]:
        rs = records if task == "ALL" else by_task[task]
        if not rs:
            continue
        probs = [np.asarray(r["probs"], dtype=float) for r in rs]
        answers = [r["answer"] for r in rs]
        pred = [int(np.argmax(p)) for p in probs]
        entry = {"n": len(rs),
                 "acc": float(np.mean([p == a for p, a in zip(pred, answers)])),
                 "ece": ece(probs, answers),
                 "brier": brier(probs, answers),
                 "mean_conf": float(np.mean([max(p) for p in probs]))}
        if task != "ALL" and rs[0].get("ordered"):
            entry["mae"] = float(np.mean([abs(p - a) for p, a in zip(pred, answers)]))
        out[task] = entry
    return out


def take_per_task(rows, per_task):
    """Cap rows per task family so every family present in the split is scored.

    A global cap is unusable here: the split is ordered by build order, so a
    global cap of 64 scores banking77 only and reports nothing for the five
    other families.
    """
    if not per_task or per_task <= 0:
        return rows
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    out = []
    for task in sorted(by_task):
        out.extend(by_task[task][:per_task])
    return out


def run_eval(model, tok, rows, args, device, tag="eval"):
    records, logit_sets = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        logits = score_options(model, tok, r["state"], r["question"],
                               list(r["options"]), args.max_len,
                               args.eval_batch, device)
        records.append({"task": r["task"],
                        "probs": softmax(logits),
                        "answer": int(r["answer_index"]),
                        "ordered": bool(r.get("ordered", False))})
        logit_sets.append(logits)
        if (i + 1) % 100 == 0:
            print("  " + tag + " " + str(i + 1) + "/" + str(len(rows))
                  + " elapsed " + str(round(time.time() - t0, 1)) + "s", flush=True)
    return metrics_from_records(records), records, logit_sets


def apply_temperature(records, logit_sets, T):
    return [dict(r, probs=softmax([x / T for x in L]))
            for r, L in zip(records, logit_sets)]


def fit_temperature(logit_sets, answers):
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.25, 6.0, 116):
        nll = 0.0
        for L, a in zip(logit_sets, answers):
            p = softmax([x / t for x in L])
            nll -= math.log(max(p[a], 1e-12))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll / max(1, len(answers))


def measure_latency(fn, repeats=5, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000.0


# -------------------------------------------------------------------- baseline

def causal_lm_class():
    # Qwen/Qwen3.5-4B-Base declares architectures=['Qwen3_5ForConditionalGeneration'],
    # so AutoModelForCausalLM does not resolve to the text decoder. Name the
    # class explicitly and fall back only if it is absent.
    import transformers

    for name in ("Qwen3_5TextForCausalLM", "Qwen3_5ForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            print("baseline decoder class", name, flush=True)
            return cls
    from transformers import AutoModelForCausalLM as _auto
    print("baseline decoder class AutoModelForCausalLM (fallback)", flush=True)
    return _auto


def letter_ids(tok):
    """Token id for ' A'..' Z'. Asserted single-token: the baseline scores these
    at the final position of the prompt."""
    ids = []
    for ch in LETTERS:
        enc = tok(" " + ch, add_special_tokens=False)["input_ids"]
        assert len(enc) == 1, ("letter is not one token", ch, enc)
        ids.append(enc[0])
    return ids


def build_baseline_prompt(state, question, options):
    lines = ["State:", state, "", "Question:", question, "", "Options:"]
    for i, o in enumerate(options):
        lines.append(LETTERS[i] + ". " + o)
    lines += ["", "Answer:"]
    return "\n".join(lines)


@torch.no_grad()
def baseline_logits(model, tok, ids_letters, row, max_len, device):
    ids = tok(build_baseline_prompt(row["state"], row["question"],
                                    list(row["options"])),
              add_special_tokens=False)["input_ids"]
    ids = ids[-max_len:]  # left-truncate: the "Answer:" tail must survive
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=x)
    last = out.logits[0, -1, :].float().cpu()
    return [float(last[ids_letters[i]]) for i in range(len(row["options"]))]


def baseline_rows(rows, limit):
    """Questions the prompted baseline can actually answer, capped per family.

    The option-count filter has to happen while selecting questions, not inside
    the scoring loop: taking the first N rows and then dropping every row with
    more than 26 options leaves the baseline with zero scored questions, because
    banking77 (77 options) and ticket routing (52) sort first in the split.
    """
    ok = [r for r in rows if len(r["options"]) <= len(LETTERS)]
    return take_per_task(ok, limit)


def run_baseline(model, tok, rows, args, device, tag="baseline"):
    ids_letters = letter_ids(tok)
    records, logit_sets, skipped = [], [], 0
    t0 = time.time()
    for i, r in enumerate(rows):
        if len(r["options"]) > len(LETTERS):
            skipped += 1
            continue
        logits = baseline_logits(model, tok, ids_letters, r,
                                 args.baseline_max_len, device)
        records.append({"task": r["task"], "probs": softmax(logits),
                        "answer": int(r["answer_index"]),
                        "ordered": bool(r.get("ordered", False))})
        logit_sets.append(logits)
        if (i + 1) % 100 == 0:
            print("  " + tag + " " + str(i + 1) + "/" + str(len(rows))
                  + " elapsed " + str(round(time.time() - t0, 1)) + "s", flush=True)
    if skipped:
        print(tag + ": skipped " + str(skipped) + " questions with more than "
              + str(len(LETTERS)) + " options (letters run out)", flush=True)
    return metrics_from_records(records), records, logit_sets


# -------------------------------------------------------------------- training

def write_model_card(out_dir, results):
    def mtable(d):
        lines = ["| task | n | accuracy | ECE | Brier |", "| --- | --- | --- | --- | --- |"]
        for task, e in d.items():
            lines.append("| %s | %d | %.3f | %.3f | %.3f |"
                         % (task, e["n"], e["acc"], e["ece"], e["brier"]))
        return "\n".join(lines)

    card = "---\nlicense: cc-by-nc-4.0\nbase_model: %s\nlibrary_name: peft\ntags:\n- sequence-classification\n- calibration\n- decision-making\n- lora\n---\n\n" % results["base_model"]
    card += "# System One scorer: %s + scalar scoring head\n\n" % results["base_model"]
    card += ("Scores each option of a typed question (yes/no, choice, score) in a single\n"
             "forward pass and softmaxes per question. No autoregressive generation; the\n"
             "output space is exactly the option set supplied by the caller.\n\n"
             "Training code: `system_one.py` in this repository.\n\n")
    card += ("## Training\n\n- data: `%s` (train questions %d, epochs %d)\n"
             "- LoRA r=%d, seq len %d, option cap %d, lr %s\n"
             "- trainable params %d of %d; %d steps in %.0fs\n"
             "- temperature %.3f (fitted on val)\n\n"
             % (results["data_repo"], results["n_train_questions"], results["epochs"],
                results["lora_r"], results["max_len"], results["max_options"],
                results["lr"], results["n_trainable"], results["n_total"],
                results["steps"], results["train_seconds"], results["temperature"]))
    if results.get("test_calibrated"):
        card += "## Test split (temperature from val)\n\n" + mtable(results["test_calibrated"]) + "\n\n"
    if results.get("val_calibrated"):
        card += "## Val split\n\n" + mtable(results["val_calibrated"]) + "\n\n"
    if results.get("baseline_val"):
        card += "## Prompted base-model baseline (val sample)\n\n" + mtable(results["baseline_val"]) + "\n\n"
    card += ("## Limitations\n\n"
             "- Trained with option sets capped at %d for the high-cardinality tasks\n"
             "  (banking77 77 options, ticket routing 52); evaluation scores all options.\n"
             "- Sequence length %d truncates long states (MMLU questions and long reviews).\n"
             "- The ticket component of the training data is CC-BY-NC-4.0, so this model\n"
             "  inherits a non-commercial restriction.\n"
             "- Accuracy is far below a frontier model; the claim here is calibrated\n"
             "  distributions over a fixed option set at single-pass latency.\n"
             % (results["max_options"], results["max_len"]))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(card)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "script", SCRIPT_VERSION, flush=True)
    dd = load_data(args)
    train_rows = list(dd["train"])
    val_rows = list(dd["val"])
    rng = random.Random(0)
    rng.shuffle(train_rows)
    if args.limit:
        train_rows = train_rows[: args.limit]
    if args.eval_limit:
        val_rows = take_per_task(val_rows, args.eval_limit)
    print("train questions", len(train_rows), "val questions", len(val_rows), flush=True)

    if args.dry_run:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print("tokenizer pad_token_id", tok.pad_token_id,
              "eos_token_id", tok.eos_token_id, flush=True)
        print("splits", {s: len(dd[s]) for s in dd}, flush=True)
        stats_report(tok, train_rows, args)
        ids_letters = letter_ids(tok)
        print("letter ids ( A.. Z )", ids_letters, flush=True)
        r = next(r for r in train_rows if len(r["options"]) <= len(LETTERS))
        n = len(tok(build_baseline_prompt(r["state"], r["question"],
                                          list(r["options"])),
                    add_special_tokens=False)["input_ids"])
        print("baseline prompt sanity: task", r["task"], "k", len(r["options"]),
              "tokens", n, "fits baseline_max_len", n <= args.baseline_max_len,
              flush=True)
        print("Answer: A tail ids",
              tok("Answer: A", add_special_tokens=False)["input_ids"][-3:],
              "spaced A id", ids_letters[0], flush=True)
        print("DRY RUN OK", flush=True)
        return {}

    tok, model = load_model(args.base_model, grad_ckpt=args.grad_checkpointing,
                            lora=not args.no_lora, lora_r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            target_modules=args.target_modules.split(",") if args.target_modules else None)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print("trainable", n_trainable, "of", n_total, flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.trackio_space:
        try:
            import trackio

            trackio.init(project=args.trackio_project, space_id=args.trackio_space)
            print("trackio initialised", flush=True)
        except Exception as e:
            print("trackio init failed:", repr(e), flush=True)

    batcher = Batcher(tok, args.max_len, seed=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(buckets(train_rows, args.batch_size, seed=0,
                                        max_options=args.max_options)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = int(0.03 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(1, warmup),
                           0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    step, running = 0, 0.0
    t_start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_rows = list(train_rows)
        rng.shuffle(epoch_rows)
        for bi, k in buckets(epoch_rows, args.batch_size, seed=epoch,
                             max_options=args.max_options):
            rows = [epoch_rows[i] for i in bi]
            batch = batcher(rows, k)
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            logits = out.logits.squeeze(-1).view(len(rows), k)
            loss = F.cross_entropy(logits, batch["labels"].to(device))
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t_start
                msg = ("step " + str(step) + "/" + str(total_steps)
                       + " loss " + str(round(running / args.log_every, 4))
                       + " lr " + str(round(sched.get_last_lr()[0], 8))
                       + " elapsed " + str(round(el, 1)) + "s"
                       + " steps/s " + str(round(step / max(el, 1e-9), 3)))
                print(msg, flush=True)
                if args.trackio_space:
                    try:
                        import trackio

                        trackio.log({"loss": running / args.log_every,
                                     "lr": sched.get_last_lr()[0]}, step=step)
                    except Exception:
                        pass
                running = 0.0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    train_seconds = time.time() - t_start
    print("train done in", round(train_seconds, 1), "s for", step, "steps", flush=True)
    if device.type == "cuda":
        print("peak gpu memory GB", round(torch.cuda.max_memory_allocated() / 1e9, 2),
              flush=True)

    model.eval()
    val_metrics, val_records, val_logits = run_eval(model, tok, val_rows, args, device,
                                                    tag="val")
    T, nll = fit_temperature(val_logits, [int(r["answer_index"]) for r in val_rows])
    print("temperature", round(T, 3), "val_nll", round(nll, 4), flush=True)
    val_cal = metrics_from_records(apply_temperature(val_records, val_logits, T))
    print("VAL UNCALIBRATED", json.dumps(val_metrics, indent=1), flush=True)
    print("VAL CALIBRATED", json.dumps(val_cal, indent=1), flush=True)

    test_metrics, test_logits_all, test_cal = None, None, None
    if args.eval_test:
        test_rows = list(dd["test"])
        if args.eval_limit:
            test_rows = take_per_task(test_rows, args.eval_limit)
        test_metrics, test_records, test_logits_all = run_eval(
            model, tok, test_rows, args, device, tag="test")
        test_cal = metrics_from_records(
            apply_temperature(test_records, test_logits_all, T))
        print("TEST UNCALIBRATED", json.dumps(test_metrics, indent=1), flush=True)
        print("TEST CALIBRATED (T from val)", json.dumps(test_cal, indent=1), flush=True)

    base_metrics = None
    if args.baseline_limit:
        AutoModelForCausalLM = causal_lm_class()

        try:
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model, dtype=torch.bfloat16 if device.type == "cuda"
                else torch.float32).to(device).eval()
            b_rows = baseline_rows(val_rows, args.baseline_limit)
            base_metrics, b_records, b_logits = run_baseline(base, tok, b_rows, args,
                                                             device, tag="baseline")
            print("BASELINE (prompted, %d questions)" % len(b_rows),
                  json.dumps(base_metrics, indent=1), flush=True)
            del base
        except Exception as e:
            import traceback

            print("baseline failed:", repr(e), flush=True)
            traceback.print_exc()

    latency = latency_options = None
    if device.type == "cuda" and val_rows:
        r = val_rows[0]
        latency = measure_latency(lambda: score_options(
            model, tok, r["state"], r["question"], list(r["options"]),
            args.max_len, args.eval_batch, device))
        latency_options = len(r["options"])
        print("latency_ms_per_query", round(latency, 1), "options", latency_options,
              flush=True)

    results = {"base_model": args.base_model, "data_repo": args.data_repo,
               "n_train_questions": len(train_rows), "epochs": args.epochs,
               "steps": step, "train_seconds": train_seconds,
               "steps_per_sec": step / max(train_seconds, 1e-9),
               "n_trainable": n_trainable, "n_total": n_total,
               "max_len": args.max_len, "batch_size": args.batch_size,
               "grad_accum": args.grad_accum, "max_options": args.max_options,
               "lora_r": args.lora_r, "lr": args.lr, "temperature": T,
               "val_nll": nll, "head_hash": head_hash(model),
               "val_uncalibrated": val_metrics, "val_calibrated": val_cal,
               "test_uncalibrated": test_metrics, "test_calibrated": test_cal,
               "baseline_val": base_metrics, "latency_ms_per_query": latency,
               "latency_options": latency_options,
               "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
               if device.type == "cuda" else None}

    if args.trackio_space:
        try:
            import trackio

            flat = {}
            reported = test_cal or val_cal
            prefix = "test" if test_cal else "val"
            for task, e in reported.items():
                flat[prefix + "/" + task + "_acc"] = e["acc"]
                flat[prefix + "/" + task + "_ece"] = e["ece"]
                flat[prefix + "/" + task + "_brier"] = e["brier"]
            for task, e in val_cal.items():
                flat["val/" + task + "_acc"] = e["acc"]
                flat["val/" + task + "_ece"] = e["ece"]
            if base_metrics:
                for task, e in base_metrics.items():
                    flat["baseline/" + task + "_acc"] = e["acc"]
                    flat["baseline/" + task + "_ece"] = e["ece"]
            flat["temperature"] = T
            if latency:
                flat["latency_ms_per_query"] = latency
            trackio.log(flat, step=step)
            trackio.finish()
            print("trackio finished", flush=True)
        except Exception as e:
            print("trackio log failed:", repr(e), flush=True)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        model.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        write_model_card(args.out_dir, results)
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=1)
        print("saved to", args.out_dir, flush=True)
    if args.push and args.push_repo:
        model.push_to_hub(args.push_repo, private=False,
                          token=os.environ.get("HF_TOKEN"))
        tok.push_to_hub(args.push_repo, private=False,
                        token=os.environ.get("HF_TOKEN"))
        # push_to_hub above sends weights and tokenizer only, and out_dir dies
        # with the container, so the card and metrics need uploading explicitly.
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        for fn in ("README.md", "metrics.json"):
            fp = os.path.join(args.out_dir, fn) if args.out_dir else None
            if fp and os.path.exists(fp):
                api.upload_file(path_or_fileobj=fp, path_in_repo=fn,
                                repo_id=args.push_repo, repo_type="model",
                                commit_message="add " + fn)
                print("uploaded", fn, "to", args.push_repo, flush=True)
        print("pushed to", args.push_repo, flush=True)
    return results


def eval_only(args):
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(args.base_model, lora=False)
    if args.model_dir:
        model = PeftModel.from_pretrained(model, args.model_dir)
    elif args.adapter_repo:
        model = PeftModel.from_pretrained(model, args.adapter_repo)
    model.to(device).eval()
    print("loaded head hash", head_hash(model), flush=True)
    dd = load_data(args)
    rows = list(dd[args.split])
    if args.eval_limit:
        rows = take_per_task(rows, args.eval_limit)

    T = args.temperature
    if T is None and args.split != "val":
        val_rows = take_per_task(list(dd["val"]), args.eval_limit)
        _, v_records, v_logits = run_eval(model, tok, val_rows, args, device, tag="val-fit")
        T, nll = fit_temperature(v_logits, [int(r["answer_index"]) for r in val_rows])
        print("temperature", round(T, 3), "fitted on val, val_nll", round(nll, 4),
              flush=True)
    elif T is None:
        print("WARNING: no --temperature given and split is val: temperature is "
              "fitted on the split being reported (optimistic)", flush=True)

    m, records, logits = run_eval(model, tok, rows, args, device, tag=args.split)
    if T is None:
        T, nll = fit_temperature(logits, [int(r["answer_index"]) for r in rows])
        print("temperature", round(T, 3), "nll", round(nll, 4), flush=True)
    print("UNCALIBRATED", json.dumps(m, indent=1), flush=True)
    print("CALIBRATED", json.dumps(metrics_from_records(
        apply_temperature(records, logits, T)), indent=1), flush=True)


def baseline_only(args):
    AutoModelForCausalLM = causal_lm_class()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, _ = load_model(args.base_model, lora=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16 if device.type == "cuda"
        else torch.float32).to(device).eval()
    dd = load_data(args)
    rows = baseline_rows(list(dd[args.split]), args.eval_limit)
    m, records, logits = run_baseline(model, tok, rows, args, device,
                                      tag=args.split)
    T, nll = fit_temperature(logits, [int(r["answer_index"]) for r in records])
    print("temperature", round(T, 3), "nll", round(nll, 4), flush=True)
    print("UNCALIBRATED", json.dumps(m, indent=1), flush=True)
    print("CALIBRATED", json.dumps(metrics_from_records(
        apply_temperature(records, logits, T)), indent=1), flush=True)
    if device.type == "cuda":
        r = next((x for x in rows if len(x["options"]) <= len(LETTERS)), None)
        if r is not None:
            ids_letters = letter_ids(tok)
            print("latency_ms_per_query", round(measure_latency(
                lambda: baseline_logits(model, tok, ids_letters, r,
                                        args.baseline_max_len, device)), 1),
                "options", len(r["options"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "train", "eval", "baseline"])
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--data-repo", default=DATA_REPO)
    p.add_argument("--local-data-dir", default=None)
    p.add_argument("--out-dir", default="/tmp/system-one-out")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--adapter-repo", default=None)
    p.add_argument("--push", action="store_true")
    p.add_argument("--push-repo", default=MODEL_REPO)
    p.add_argument("--sizes", default=json.dumps({
        "banking77": [1200, 100, 150], "go_emotions": [1500, 200, 200],
        "ag_news": [1500, 150, 200], "mmlu": [2000, 200, 250],
        "yelp_score": [1200, 150, 200], "tickets": [600, 60, 80]}))
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--baseline-max-len", type=int, default=1024)
    p.add_argument("--max-options", type=int, default=16,
                   help="train-time cap on options per question (0 = all)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=48)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--target-modules", default=None)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--split", default="val")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--baseline-limit", type=int, default=0)
    p.add_argument("--trackio-space", default=None)
    p.add_argument("--trackio-project", default=TRACKIO_PROJECT)
    args = p.parse_args()
    if args.mode == "build":
        build_data(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_only(args)
    else:
        baseline_only(args)


if __name__ == "__main__":
    main()

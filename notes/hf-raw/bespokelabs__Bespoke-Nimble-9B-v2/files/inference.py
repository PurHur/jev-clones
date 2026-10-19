"""Portable CUDA/BF16 candidate scoring for the Bespoke-Nimble-9B-v2 adapter."""
import hashlib
import json
import math
from pathlib import Path
import torch
from serving_schema import MAX_CHOICES, choice_key, prepare_prompts
DEFAULT_TEMPERATURE = 2.179078721266035


def validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    return float(temperature)

def validate_serving_config(directory, tokenizer):
    """Check the new serving extension separately from the immutable training contract."""
    directory = Path(directory)
    config = json.loads((directory / "serving_config.json").read_text())
    if config["max_choices"] != MAX_CHOICES:
        raise ValueError("Serving candidate limit differs from packaged runtime")
    for name, expected in config["source_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid serving source filename")
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Serving source differs: {name}")
    schema = {"decision": {"type": "enum", "description": "Select one candidate.",
                           "choices": config["candidate_codes"]}}
    prepared = prepare_prompts(tokenizer, "Codebook validation.", schema, 32768)
    if prepared.candidate_ids[0] != config["candidate_token_ids"]:
        raise ValueError("Tokenizer differs from the validated serving codebook")
    return config


class CandidateCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        if not rows:
            raise ValueError("Empty candidate batch")
        length = max(len(r["input_ids"]) for r in rows)
        for r in rows:
            n = len(r["candidate_ids"])
            if (not 1 <= n <= MAX_CHOICES or len(set(r["candidate_ids"])) != n
                    or any(type(i) is not int or i < 0 for i in r["candidate_ids"])):
                raise ValueError("Invalid candidate token set")
            if "labels" in r and not 0 <= r["labels"] < n:
                raise ValueError("Gold label points outside the real candidates")
        width = max(len(r["candidate_ids"]) for r in rows)
        batch = {
            "input_ids": torch.tensor([[self.pad_id] * (length - len(r["input_ids"])) + r["input_ids"] for r in rows]),
            "attention_mask": torch.tensor([[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
            "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (width - len(r["candidate_ids"])) for r in rows]),
            "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (width - len(r["candidate_ids"])) for r in rows]),
        }
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([r["labels"] for r in rows])
        return batch

def candidate_logits(model, inputs):
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    selected = logits.gather(1, inputs["candidate_ids"])
    return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)

def decision_result(row, logits, temperature=DEFAULT_TEMPERATURE):
    temperature = validate_temperature(temperature)
    logits = logits[:len(row["choices"])].double()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite candidate logits")
    scaled_logits = logits / temperature
    probs = scaled_logits.softmax(-1)
    index = logits.argmax().item()
    gold = row.get("labels")
    choices = row["choices"]
    result = {"prediction": choices[index], "probabilities": {choice_key(k): v for k, v in zip(choices, probs.tolist())},
              "logits": {choice_key(k): v for k, v in zip(choices, logits.tolist())}}
    if gold is not None:
        result.update(correct=index == gold, nll=-scaled_logits.log_softmax(-1)[gold].item(),
                      brier=sum((p - int(i == gold)) ** 2 for i, p in enumerate(probs.tolist())))
    if row.get("kind") == "score":
        result["prediction"] = int(choices[index])
        result["expected_score"] = sum(int(v) * p for v, p in zip(choices, probs.tolist()))
        if gold is not None:
            result["score_absolute_error"] = abs(result["expected_score"] - int(choices[gold]))
    if row.get("kind") == "noul":
        result["probability_true"] = result["probabilities"]["true"]
    return result

class ParallelScorer:
    def __init__(self, model_dir, temperature=DEFAULT_TEMPERATURE, *, allow_uncalibrated=False):
        self.temperature = validate_temperature(temperature)
        if self.temperature == 1.0 and not allow_uncalibrated:
            raise ValueError("Raw T=1.0 probabilities require allow_uncalibrated=True. "
                             "Omit temperature to use the v2 release default.")
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.directory=Path(model_dir)
        self.contract=json.loads((self.directory/'schema_config.json').read_text())
        if self.contract['task'] != 'schema_candidate_classification_v1':
            raise ValueError('Unsupported training contract')
        if hashlib.sha256((Path(__file__).parent/'parallel_schema.py').read_bytes()).hexdigest()!=self.contract['prompt_code_sha256']:
            raise ValueError('Legacy training prompt hash mismatch')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This reference runner requires a CUDA GPU with BF16 support.')
        self.tokenizer=AutoTokenizer.from_pretrained(self.directory)
        validate_serving_config(self.directory, self.tokenizer)
        base=Qwen3_5ForConditionalGeneration.from_pretrained(self.contract['model'],revision=self.contract['revision'],dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        base.config.use_cache=False
        self.model=PeftModel.from_pretrained(base,self.directory).eval()
        self.collator=CandidateCollator(self.tokenizer.pad_token_id)

    def score(self, context, schema, score_fields=()):
        prepared=prepare_prompts(self.tokenizer,context,schema,self.contract['max_length'])
        score_fields=set(score_fields)
        if not score_fields.issubset(schema):raise ValueError('Unknown score field')
        fields={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,name in enumerate(prepared.names):
                row={'input_ids':prepared.full_ids[i],'candidate_ids':prepared.candidate_ids[i],'choices':prepared.choices[i]}
                if schema[name]['type']=='boolean':row['kind']='noul'
                if name in score_fields:
                    if schema[name]['type']!='enum':raise ValueError('Score fields must be integer-valued enums')
                    values=[int(v) for v in row['choices']]
                    if len(set(values))!=len(values):raise ValueError('Score values must be distinct integers')
                    row['kind']='score'
                inputs={k:v.to('cuda') for k,v in self.collator([row]).items()}
                fields[name]=decision_result(row,candidate_logits(self.model,inputs)[0].cpu(), temperature=self.temperature)
        return {'output':{name:r['prediction'] for name,r in fields.items()},'fields':fields,
                'temperature':self.temperature,'temperature_fitted':False}

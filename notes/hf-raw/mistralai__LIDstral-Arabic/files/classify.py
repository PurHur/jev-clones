"""Standalone LIDstral-Arabic inference on CPU."""

import argparse
import html
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import fasttext
import numpy as np
import torch
from torch import nn

_DIACRITICS = re.compile('[\\u0610-\\u061A\\u064B-\\u065F\\u0670\\u06D6-\\u06DC\\u06DF-\\u06E4\\u06E7\\u06E8\\u06EA-\\u06ED]')
_KASHIDA = 'ـ'
_ASR_NOISE = re.compile('<[^>]+>')
_MULTI_SPACE = re.compile('\\s+')
_URLS = re.compile('https?://\\S+|www\\.\\S+')
_MENTIONS = re.compile('@\\w+')
_HASHTAG_SYMBOL = re.compile('#(\\w+)')
_REPEATED_CHAR = re.compile('(.)\\1{2,}')
_EMAILS = re.compile('\\S+@\\S+\\.\\S+')
_EMOJI = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿🤀-🧿🨀-\U0001fa6f🩰-\U0001faff✂-➰︀-️\u200d⏰-⏺☀-⛿\u200c-\u200f]+', flags=re.UNICODE)
_INVISIBLE_CHARS = re.compile('[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ufff9-\ufffb]+')
_PLACEHOLDER_TOKENS = re.compile('\\b(?:USER|EMOJI|NEWLINE|URL|HASHTAG|RT|MENTION|NUMBER|NUM)\\b', flags=re.IGNORECASE)
_CONSECUTIVE_PUNCT = re.compile('([.!?،؟])\\1{2,}')
_CONSECUTIVE_COMMAS = re.compile('،{2,}')

def preprocess(text: str) -> str:
    text = _ASR_NOISE.sub('', text)
    text = unicodedata.normalize('NFC', text)
    text = html.unescape(text)
    text = _INVISIBLE_CHARS.sub('', text)
    text = _URLS.sub('', text)
    text = _EMAILS.sub('', text)
    text = _MENTIONS.sub('', text)
    text = _HASHTAG_SYMBOL.sub('\\1', text)
    text = _EMOJI.sub('', text)
    text = _PLACEHOLDER_TOKENS.sub('', text)
    text = _DIACRITICS.sub('', text)
    text = text.replace(_KASHIDA, '')
    text = _REPEATED_CHAR.sub('\\1\\1', text)
    text = _CONSECUTIVE_PUNCT.sub('\\1', text)
    text = _CONSECUTIVE_COMMAS.sub('،', text)
    text = _MULTI_SPACE.sub(' ', text).strip()
    return text

REGION_GROUPS = {
    "gulf": ["Saudi_Arabia", "Bahrain", "Kuwait", "Qatar", "UAE", "Oman", "Yemen"],
    "levantine": ["Iraq", "Jordan", "Palestine", "Syria", "Lebanon"],
    "maghreb": ["Morocco", "Algeria", "Tunisia", "Libya", "Mauritania"],
    "nile": ["Egypt", "Sudan"],
    "south_asian": ["Urdu", "Sindhi", "Kashmiri", "Punjabi_Shahmukhi"],
}


def clean_label(label: str) -> str:
    return str(label).removeprefix("__label__")


def _find_label_indices(all_labels: list[str], target_labels: list[str]) -> list[int]:
    return [i for (i, label) in enumerate(all_labels) if clean_label(label) in target_labels]

def _find_label_index(all_labels: list[str], target: str) -> int:
    matches = [i for (i, label) in enumerate(all_labels) if clean_label(label) == target]
    assert len(matches) == 1, f'Expected exactly 1 match for {target!r}, got {len(matches)}'
    return matches[0]

def _prior_corrected_scores(raw: np.ndarray, log_priors: np.ndarray) -> np.ndarray:
    return np.log(raw + 1e-12) - 0.5 * log_priors[np.newaxis, :]

def _normalized_entropy(raw: np.ndarray) -> np.ndarray:
    p_norm = raw / (raw.sum(axis=1, keepdims=True) + 1e-12)
    return -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1, keepdims=True)

def _saudi_vs_gulf_margin(raw: np.ndarray, all_labels: list[str]) -> np.ndarray:
    gulf_labels = REGION_GROUPS['gulf']
    si = _find_label_index(all_labels, 'Saudi_Arabia')
    non_sa_idx = [i for i in _find_label_indices(all_labels, gulf_labels) if i != si]
    return raw[:, si:si + 1] - raw[:, non_sa_idx].max(axis=1, keepdims=True)

def _per_region_features(raw: np.ndarray, all_labels: list[str]) -> np.ndarray:
    region_feats: list[np.ndarray] = []
    for region_labels in REGION_GROUPS.values():
        r_idx = _find_label_indices(all_labels, region_labels)
        if not r_idx:
            continue
        rs = raw[:, r_idx]
        r_max = rs.max(axis=1, keepdims=True)
        r_p = rs / (rs.sum(axis=1, keepdims=True) + 1e-12)
        r_entropy = -np.sum(r_p * np.log(r_p + 1e-12), axis=1, keepdims=True)
        r_sorted = np.sort(rs, axis=1)[:, ::-1]
        r_margin = r_sorted[:, 0:1] - r_sorted[:, 1:2] if r_sorted.shape[1] > 1 else np.zeros((len(raw), 1))
        region_feats.extend([r_max, r_entropy, r_margin])
    return np.hstack(region_feats)

def build_features_v2(raw: np.ndarray, log_priors: np.ndarray, all_labels: list[str]) -> np.ndarray:
    corrected = _prior_corrected_scores(raw, log_priors)
    sorted_s = np.sort(raw, axis=1)[:, ::-1]
    (top1, top2, top3) = (sorted_s[:, 0:1], sorted_s[:, 1:2], sorted_s[:, 2:3])
    margin12 = top1 - top2
    margin13 = top1 - top3
    margin23 = top2 - top3
    entropy = _normalized_entropy(raw)
    sa_margin = _saudi_vs_gulf_margin(raw, all_labels)
    score_std = raw.std(axis=1, keepdims=True)
    score_max = raw.max(axis=1, keepdims=True)
    score_mean = raw.mean(axis=1, keepdims=True)
    region_feats = _per_region_features(raw, all_labels)
    log_top10 = np.log(sorted_s[:, :10] + 1e-12)
    ratio_12 = top1 / (top2 + 1e-08)
    ratio_13 = top1 / (top3 + 1e-08)
    return np.hstack([raw, corrected, top1, top2, top3, margin12, margin13, margin23, entropy, sa_margin, score_std, score_max, score_mean, region_feats, log_top10, ratio_12, ratio_13])

class ArabicLID:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

    def load(self) -> "ArabicLID":
        self.ova = fasttext.load_model(str(self.model_dir / "lid_ova_full.bin"))
        # The original checkpoint contains NumPy arrays, so weights_only cannot
        # load it. Load only artifacts obtained from a trusted source.
        checkpoint = torch.load(
            self.model_dir / "mlp_stacker_ln.pt", map_location="cpu", weights_only=False
        )
        self.all_labels = list(checkpoint["all_ft_labels"])
        self.classes = list(checkpoint["le_classes"])
        self.label_order = {label: i for i, label in enumerate(self.all_labels)}
        assert set(self.ova.get_labels()) == set(self.all_labels), "fastText label mismatch"
        assert checkpoint["norm"] == "layer", "Expected LayerNorm checkpoint"
        assert checkpoint["hidden_dims"] == [64], "Expected one 64-unit hidden layer"
        assert checkpoint["input_dim"] == 144, "Expected feature version 2"
        assert checkpoint["output_dim"] == len(self.classes), "Output class mismatch"
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.log_priors = checkpoint["log_priors"]
        assert np.all(self.std > 0), "Feature standard deviations must be positive"
        self.mlp = nn.Sequential(
            nn.Linear(144, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(64, len(self.classes)),
        )
        self.mlp.load_state_dict(
            {name.removeprefix("net."): value for name, value in checkpoint["model_state"].items()}
        )
        self.mlp.eval()
        with (self.model_dir / "isotonic_calibration.pkl").open("rb") as stream:
            calibration = pickle.load(stream)
        assert list(calibration["classes"]) == self.classes, "Calibration class order mismatch"
        self.calibrators = calibration["models"]
        assert len(self.calibrators) == len(self.classes), "Calibration count mismatch"
        return self

    @torch.inference_mode()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.classes)), dtype=np.float32)
        cleaned = [preprocess(text) for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Each input must contain text after preprocessing")
        labels, scores = self.ova.predict(cleaned, k=len(self.all_labels))
        raw = np.zeros((len(texts), len(self.all_labels)), dtype=np.float32)
        for row, (row_labels, row_scores) in enumerate(zip(labels, scores, strict=True)):
            raw[row, [self.label_order[label] for label in row_labels]] = row_scores
        features = build_features_v2(raw, self.log_priors, self.all_labels)
        features = (features - self.mean) / self.std
        probabilities = torch.softmax(self.mlp(torch.from_numpy(features)), dim=1).numpy()
        calibrated = np.zeros_like(probabilities)
        for column, calibrator in enumerate(self.calibrators):
            calibrated[:, column] = calibrator.predict(probabilities[:, column])
        return calibrated / (calibrated.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, texts: list[str]) -> list[dict[str, object]]:
        probabilities = self.predict_proba(texts)
        results = []
        for row in probabilities:
            index = int(row.argmax())
            results.append({
                "label": clean_label(self.classes[index]),
                "confidence": float(row[index]),
                "probabilities": {clean_label(label): float(score) for label, score in zip(self.classes, row, strict=True)},
            })
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Classify one text")
    inputs.add_argument("--input", type=Path, help='JSONL file with a "text" field; use - for stdin')
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(1)
    model = ArabicLID(args.model_dir).load()
    if args.text is not None:
        print(json.dumps(model.predict([args.text])[0], ensure_ascii=False))
        return
    stream = sys.stdin if str(args.input) == "-" else args.input.open(encoding="utf-8")
    try:
        batch = []
        for line in stream:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            if not isinstance(text, str):
                raise TypeError('The "text" field must be a string')
            batch.append(text)
            if len(batch) == args.batch_size:
                for result in model.predict(batch):
                    print(json.dumps(result, ensure_ascii=False))
                batch.clear()
        for result in model.predict(batch):
            print(json.dumps(result, ensure_ascii=False))
    finally:
        if stream is not sys.stdin:
            stream.close()


if __name__ == "__main__":
    main()

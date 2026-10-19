---
library_name: fasttext
pipeline_tag: text-classification
language:
  # Arabic — MSA
  - ar        # Modern Standard Arabic
  # Arabic — Maghreb
  - ary       # Moroccan Arabic
  - ayl       # Libyan Arabic
  - aeb       # Tunisian Arabic
  - arq       # Algerian Arabic
  - mey       # Hassaniya Arabic (Mauritania)
  # Berber
  - tzm       # Central Atlas Tamazight (Latin script)
  - zgh       # Standard Moroccan Tamazight (Tifinagh script)
  # Arabic — Gulf
  - afb       # Gulf Arabic (Bahrain, Kuwait, Qatar, UAE)
  - ars       # Najdi Arabic (Saudi Arabia)
  - acx       # Omani Arabic
  - ayn       # Sanaani Arabic (Yemen)
  # Arabic — Nile Valley
  - arz       # Egyptian Arabic
  - apd       # Sudanese Arabic
  # Arabic — Levantine
  - acm       # Mesopotamian Arabic (Iraq)
  - ajp       # South Levantine Arabic (Jordan, Palestine)
  - apc       # North Levantine Arabic (Lebanon, Syria)
  # Arabic — Other
  - shu       # Chadian Arabic
  # Turkic
  - az        # Azerbaijani
  - tr        # Turkish
  - kk        # Kazakh
  - ky        # Kyrgyz
  - uz        # Uzbek
  # Indo-Iranian
  - fa        # Persian
  - ps        # Pashto
  - ur        # Urdu
  - sd        # Sindhi
  - ks        # Kashmiri
  - pnb       # Western Punjabi (Shahmukhi)
  - tg        # Tajik
  - glk       # Gilaki
  - brh       # Brahui
  - hac       # Hawrami
  - ckb       # Central Kurdish (Sorani)
  # African
  - aa        # Afar
  - din       # Dinka
  - ff        # Fula
  - ha        # Hausa
  - kr        # Kanuri
  - om        # Oromo
  - sw        # Swahili
  - ti        # Tigrinya
  - wo        # Wolof
  - nqo       # N'Ko
  # Austronesian
  - ace       # Acehnese
  - ms        # Malay (Jawi script)
inference: false
tags:
- language-identification
- dialect-identification
license: apache-2.0
---

# LIDstral Arabic

LIDstral Arabic is a fast language and dialect classifier for text written in Arabic script. It identifies Modern Standard Arabic, all Arabic dialects, and non-Arabic languages that use the same script.

We include these non-Arabic languages in training to help the model distinguish languages that share a script and avoid classifying text as Arabic based on its script alone.

## Method

The model combines a fastText one-vs-all (OVA) classifier, an MLP stacker, and per-class isotonic calibration:

1. **fastText OVA**: produces independent scores for each class, preserving evidence for competing languages and dialects.
2. **MLP stacker**: A 64-unit LayerNorm MLP maps these scores to probabilities over the classes. It uses 144 features derived from raw scores, prior corrections, confidence margins, entropy, and regional summaries. The checkpoint stores feature normalization and label ordering.
3. **Per-class isotonic regressors**: fitted on development predictions calibrate the MLP probabilities. The pipeline then renormalizes them across classes. Calibration can change both confidence and the predicted label.

The full pipeline requires three files:

- `lid_ova_full.bin`
- `mlp_stacker_ln.pt`
- `isotonic_calibration.pkl`

`classify.py` runs the pipeline on CPU, including preprocessing and feature extraction. Each input that remains nonempty after preprocessing receives a predicted label and calibrated probabilities over all 51 classes.

## Evaluation

We evaluate on 84,870 Arabic examples drawn from various benchmarks including [NADI](https://figshare.com/articles/dataset/NADI_dataset/27935430), [MADAR](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), [QADI](https://github.com/qcri/QADI), [Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca), [FLEURS](https://huggingface.co/datasets/google/fleurs), [SMOL](https://huggingface.co/datasets/google/smol), [Omnilingual ASR](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus), and [Hassaniya](https://huggingface.co/datasets/Emin009/hassaniya-punctuation-dataset).
On Moroccan Darija, LIDstral Arabic achieves **88.67%** F1, compared with 72.65% for LahjatBERT ALDi CL, the strongest tested baseline for this dialect, and 71.28% for GlotLID v3.

![Per-label F1 on the Arabic benchmark](assets/arabic-country-comparison.png)

## Usage

### Install

1. Download the repository:

```bash
pip install huggingface_hub
hf auth login
hf download mistralai/LIDstral-Arabic --local-dir LIDstral-Arabic
```

2. Install the environment:

```
cd LIDstral-Arabic
pip install -r requirements.txt
```

### Use the model:

1. Classify a text:

```
python classify.py --text "ياك نتا بخير؟ آش خبارك مع الخدمة؟ نتمنى تكون الأمور كلها مزيانة من جيهتك."
```

Illustrative output, formatted for readability. Only selected probabilities are shown here; the full output includes all classes.

```json
{
  "label": "Morocco",
  "confidence": 0.9523,
  "probabilities": {
    "Morocco": 0.9523,
    "Algeria": 0.0125,
    "Mauritania": 0.0115,
    "MSA": 0.0062,
    "Libya": 0.0048,
    "..."
  }
}
```
Keep all three model files in the same directory. The script loads them from its own directory by default. Use `--model-dir` to select another local bundle.

2. Classify multiple samples from a list of JSONL:

For JSONL input, each line must contain a string `text` field:

```bash
python classify.py --input input.jsonl > predictions.jsonl
```

Use `--input -` to read from standard input. The script returns one JSON object per input, with `label`, `confidence`, and `probabilities`. Empty inputs and text removed entirely by preprocessing raise an error.

3. Classify multiple samples from the Python API:


To load the model once and classify multiple texts, use the Python API:

```python
from classify import ArabicLID

model = ArabicLID(".").load()
predictions = model.predict([
    "ياك نتا بخير؟ آش خبارك مع الخدمة؟ نتمنى تكون الأمور كلها مزيانة من جيهتك.",
    "هذا نص باللغة العربية الفصحى.",
])
```

Illustrative batch output, with selected probabilities shown for each input:

```json
[
  {                                                                                                                                                                        
    "label": "Morocco",                                                                                                                                                    
    "confidence": 0.9466,
    "probabilities": {                                                                                                                                                     
      "Morocco": 0.9466,                                    
      "MSA": 0.0185,                                                                                                                                                       
      "Algeria": 0.0125,                                    
      "..."
    }
  },
  {
    "label": "MSA",
    "confidence": 0.7295,
    "probabilities": {
      "MSA": 0.7295,
      "Iraq": 0.1203,
      "Egypt": 0.0266,
      "..."
    }
  }
]
```

Preprocessing removes web and ASR artifacts, diacritics, decorative elongation, and repeated punctuation. It preserves alef variants.

## Limitations

The model always assigns one of its 51 supported classes. It may therefore misclassify unsupported languages or mixed-language text.

## License

This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

*You must not use this model in a manner that infringes, misappropriates, or otherwise violates any third party’s rights, including intellectual property rights.*
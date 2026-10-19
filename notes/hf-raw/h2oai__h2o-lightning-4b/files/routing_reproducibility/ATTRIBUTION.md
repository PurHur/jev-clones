# Attribution

The questions in `questions/` and the `options` / `gold` fields of `predictions.jsonl` are converted from two public
datasets. Each row carries its `dataset`, `license` and `source_id` (the `id` of the original row).

## Plumb

- Source: [crh225/plumb-decisions](https://huggingface.co/datasets/crh225/plumb-decisions) by crh225, revision
  `718a9f006f3beaecfa7f66a819553f99ed7b94f6` (files `data/train.jsonl`, `data/test.jsonl`)
- Licence: Apache License 2.0 (copy in `questions/LICENSE-plumb-decisions-Apache-2.0.txt`)
- Used here: 2,000 questions (all 131 of the test split, 1,869 of the train split)

## Eikos

- Source: [caiovicentino1/eikos-decisions](https://huggingface.co/datasets/caiovicentino1/eikos-decisions) by
  caiovicentino1, revision `cebb4a26687cb14742fd07c0ebe17387ee51d369` (files `data/core/heldout.jsonl`,
  `data/core/validation.jsonl`)
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Used here: 1,474 questions (the English rows of the core held-out and validation splits)

## Changes made (both sources)

- Converted to one row format: `document` (the source's state), `instructions`, `criteria` (option descriptions), the
  option names, the question type and the gold option. Plumb's `expected` and Eikos's `expected` become the gold
  option; Eikos yes/no options (`yes` / `no`) are renamed `true` / `false`, the model's yes/no convention.
- Removed: any question sharing a 13-word sequence, or more than two 8-word sequences, with a public JevBench item.
- The original text of the documents, instructions and options is unchanged (every converted row was checked
  against its source row at the revision above).

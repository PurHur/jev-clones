---
license: other
license_name: per-source
license_link: LICENSE.md
language:
- de
pretty_name: Decisions DE
size_categories:
- 100K<n<1M
task_categories:
- text-classification
source_datasets:
- hotchpotch/bekko-system-one-dataset-v0
tags:
- structured-decisions
- system-one
- decision-model
- calibration
- translation
- german
- synthetic-translation
configs:
- config_name: all
  default: true
  data_files:
  - split: train
    path: all/train-*.parquet
  - split: validation
    path: all/validation-*.parquet
- config_name: aegis2
  data_files:
  - split: train
    path: aegis2/train-*.parquet
  - split: validation
    path: aegis2/validation-*.parquet
- config_name: aqua_rat
  data_files:
  - split: train
    path: aqua_rat/train-*.parquet
  - split: validation
    path: aqua_rat/validation-*.parquet
- config_name: arc
  data_files:
  - split: train
    path: arc/train-*.parquet
  - split: validation
    path: arc/validation-*.parquet
- config_name: arct
  data_files:
  - split: train
    path: arct/train-*.parquet
  - split: validation
    path: arct/validation-*.parquet
- config_name: babi_nli
  data_files:
  - split: train
    path: babi_nli/train-*.parquet
  - split: validation
    path: babi_nli/validation-*.parquet
- config_name: banking77
  data_files:
  - split: train
    path: banking77/train-*.parquet
  - split: validation
    path: banking77/validation-*.parquet
- config_name: bitext
  data_files:
  - split: train
    path: bitext/train-*.parquet
  - split: validation
    path: bitext/validation-*.parquet
- config_name: boardgameqa
  data_files:
  - split: train
    path: boardgameqa/train-*.parquet
  - split: validation
    path: boardgameqa/validation-*.parquet
- config_name: boolq
  data_files:
  - split: train
    path: boolq/train-*.parquet
  - split: validation
    path: boolq/validation-*.parquet
- config_name: canttalk
  data_files:
  - split: train
    path: canttalk/train-*.parquet
  - split: validation
    path: canttalk/validation-*.parquet
- config_name: circa
  data_files:
  - split: train
    path: circa/train-*.parquet
  - split: validation
    path: circa/validation-*.parquet
- config_name: civil_comments
  data_files:
  - split: train
    path: civil_comments/train-*.parquet
  - split: validation
    path: civil_comments/validation-*.parquet
- config_name: clinc
  data_files:
  - split: train
    path: clinc/train-*.parquet
  - split: validation
    path: clinc/validation-*.parquet
- config_name: commonsense_qa
  data_files:
  - split: train
    path: commonsense_qa/train-*.parquet
  - split: validation
    path: commonsense_qa/validation-*.parquet
- config_name: commonsense_qa2
  data_files:
  - split: train
    path: commonsense_qa2/train-*.parquet
  - split: validation
    path: commonsense_qa2/validation-*.parquet
- config_name: contract_nli
  data_files:
  - split: train
    path: contract_nli/train-*.parquet
  - split: validation
    path: contract_nli/validation-*.parquet
- config_name: corr2cause
  data_files:
  - split: train
    path: corr2cause/train-*.parquet
  - split: validation
    path: corr2cause/validation-*.parquet
- config_name: cosmos_qa
  data_files:
  - split: train
    path: cosmos_qa/train-*.parquet
  - split: validation
    path: cosmos_qa/validation-*.parquet
- config_name: dbpedia
  data_files:
  - split: train
    path: dbpedia/train-*.parquet
  - split: validation
    path: dbpedia/validation-*.parquet
- config_name: defeasible_nli
  data_files:
  - split: train
    path: defeasible_nli/train-*.parquet
  - split: validation
    path: defeasible_nli/validation-*.parquet
- config_name: esci
  data_files:
  - split: train
    path: esci/train-*.parquet
  - split: validation
    path: esci/validation-*.parquet
- config_name: ethics
  data_files:
  - split: train
    path: ethics/train-*.parquet
  - split: validation
    path: ethics/validation-*.parquet
- config_name: feasibility_qa
  data_files:
  - split: train
    path: feasibility_qa/train-*.parquet
  - split: validation
    path: feasibility_qa/validation-*.parquet
- config_name: fig_qa
  data_files:
  - split: train
    path: fig_qa/train-*.parquet
  - split: validation
    path: fig_qa/validation-*.parquet
- config_name: fol_nli
  data_files:
  - split: train
    path: fol_nli/train-*.parquet
  - split: validation
    path: fol_nli/validation-*.parquet
- config_name: go_emotions
  data_files:
  - split: train
    path: go_emotions/train-*.parquet
  - split: validation
    path: go_emotions/validation-*.parquet
- config_name: gretel_pii
  data_files:
  - split: train
    path: gretel_pii/train-*.parquet
  - split: validation
    path: gretel_pii/validation-*.parquet
- config_name: gsm8k
  data_files:
  - split: train
    path: gsm8k/train-*.parquet
  - split: validation
    path: gsm8k/validation-*.parquet
- config_name: hans
  data_files:
  - split: train
    path: hans/train-*.parquet
  - split: validation
    path: hans/validation-*.parquet
- config_name: helpsteer
  data_files:
  - split: train
    path: helpsteer/train-*.parquet
  - split: validation
    path: helpsteer/validation-*.parquet
- config_name: helpsteer2
  data_files:
  - split: train
    path: helpsteer2/train-*.parquet
  - split: validation
    path: helpsteer2/validation-*.parquet
- config_name: helpsteer3
  data_files:
  - split: train
    path: helpsteer3/train-*.parquet
  - split: validation
    path: helpsteer3/validation-*.parquet
- config_name: hh_rlhf
  data_files:
  - split: train
    path: hh_rlhf/train-*.parquet
  - split: validation
    path: hh_rlhf/validation-*.parquet
- config_name: hwu64
  data_files:
  - split: train
    path: hwu64/train-*.parquet
  - split: validation
    path: hwu64/validation-*.parquet
- config_name: logical_entailment
  data_files:
  - split: train
    path: logical_entailment/train-*.parquet
  - split: validation
    path: logical_entailment/validation-*.parquet
- config_name: massive
  data_files:
  - split: train
    path: massive/train-*.parquet
  - split: validation
    path: massive/validation-*.parquet
- config_name: measuring_hate
  data_files:
  - split: train
    path: measuring_hate/train-*.parquet
  - split: validation
    path: measuring_hate/validation-*.parquet
- config_name: minds14
  data_files:
  - split: train
    path: minds14/train-*.parquet
  - split: validation
    path: minds14/validation-*.parquet
- config_name: monotonicity
  data_files:
  - split: train
    path: monotonicity/train-*.parquet
  - split: validation
    path: monotonicity/validation-*.parquet
- config_name: mtop
  data_files:
  - split: train
    path: mtop/train-*.parquet
  - split: validation
    path: mtop/validation-*.parquet
- config_name: nlsat
  data_files:
  - split: train
    path: nlsat/train-*.parquet
  - split: validation
    path: nlsat/validation-*.parquet
- config_name: nlupp
  data_files:
  - split: train
    path: nlupp/train-*.parquet
  - split: validation
    path: nlupp/validation-*.parquet
- config_name: oasst1
  data_files:
  - split: train
    path: oasst1/train-*.parquet
  - split: validation
    path: oasst1/validation-*.parquet
- config_name: open_jev__amount-extraction-control-v1
  data_files:
  - split: train
    path: open_jev__amount-extraction-control-v1/train-*.parquet
  - split: validation
    path: open_jev__amount-extraction-control-v1/validation-*.parquet
- config_name: open_jev__browser-control-v1
  data_files:
  - split: train
    path: open_jev__browser-control-v1/train-*.parquet
  - split: validation
    path: open_jev__browser-control-v1/validation-*.parquet
- config_name: open_jev__citation-control-v1
  data_files:
  - split: train
    path: open_jev__citation-control-v1/train-*.parquet
  - split: validation
    path: open_jev__citation-control-v1/validation-*.parquet
- config_name: open_jev__context-retention-control-v1
  data_files:
  - split: train
    path: open_jev__context-retention-control-v1/train-*.parquet
  - split: validation
    path: open_jev__context-retention-control-v1/validation-*.parquet
- config_name: open_jev__drone-control-v1
  data_files:
  - split: train
    path: open_jev__drone-control-v1/train-*.parquet
  - split: validation
    path: open_jev__drone-control-v1/validation-*.parquet
- config_name: open_jev__entity-alignment-control-v1
  data_files:
  - split: train
    path: open_jev__entity-alignment-control-v1/train-*.parquet
  - split: validation
    path: open_jev__entity-alignment-control-v1/validation-*.parquet
- config_name: open_jev__ir-control-v1
  data_files:
  - split: train
    path: open_jev__ir-control-v1/train-*.parquet
  - split: validation
    path: open_jev__ir-control-v1/validation-*.parquet
- config_name: open_jev__mailroom-control-v1
  data_files:
  - split: train
    path: open_jev__mailroom-control-v1/train-*.parquet
  - split: validation
    path: open_jev__mailroom-control-v1/validation-*.parquet
- config_name: open_jev__painting-geometry-v1
  data_files:
  - split: train
    path: open_jev__painting-geometry-v1/train-*.parquet
  - split: validation
    path: open_jev__painting-geometry-v1/validation-*.parquet
- config_name: open_jev__reasoning-control-v1
  data_files:
  - split: train
    path: open_jev__reasoning-control-v1/train-*.parquet
  - split: validation
    path: open_jev__reasoning-control-v1/validation-*.parquet
- config_name: open_jev__silent-failure-control-v1
  data_files:
  - split: train
    path: open_jev__silent-failure-control-v1/train-*.parquet
  - split: validation
    path: open_jev__silent-failure-control-v1/validation-*.parquet
- config_name: open_jev__snake-v1
  data_files:
  - split: train
    path: open_jev__snake-v1/train-*.parquet
  - split: validation
    path: open_jev__snake-v1/validation-*.parquet
- config_name: open_jev__sponsor-segment-control-v1
  data_files:
  - split: train
    path: open_jev__sponsor-segment-control-v1/train-*.parquet
  - split: validation
    path: open_jev__sponsor-segment-control-v1/validation-*.parquet
- config_name: open_jev__tic_tac_toe-v1
  data_files:
  - split: train
    path: open_jev__tic_tac_toe-v1/train-*.parquet
  - split: validation
    path: open_jev__tic_tac_toe-v1/validation-*.parquet
- config_name: open_jev__tile_platformer-v1
  data_files:
  - split: train
    path: open_jev__tile_platformer-v1/train-*.parquet
  - split: validation
    path: open_jev__tile_platformer-v1/validation-*.parquet
- config_name: open_jev__trex_runner-v1
  data_files:
  - split: train
    path: open_jev__trex_runner-v1/train-*.parquet
  - split: validation
    path: open_jev__trex_runner-v1/validation-*.parquet
- config_name: open_jev__workflow-controls-v1__agent_trace_observability
  data_files:
  - split: train
    path: open_jev__workflow-controls-v1__agent_trace_observability/train-*.parquet
  - split: validation
    path: open_jev__workflow-controls-v1__agent_trace_observability/validation-*.parquet
- config_name: open_jev__workflow-controls-v1__customer_service
  data_files:
  - split: train
    path: open_jev__workflow-controls-v1__customer_service/train-*.parquet
  - split: validation
    path: open_jev__workflow-controls-v1__customer_service/validation-*.parquet
- config_name: open_jev__workflow-controls-v1__invoice_processing
  data_files:
  - split: train
    path: open_jev__workflow-controls-v1__invoice_processing/train-*.parquet
  - split: validation
    path: open_jev__workflow-controls-v1__invoice_processing/validation-*.parquet
- config_name: open_jev__workflow-controls-v1__security_incidents
  data_files:
  - split: train
    path: open_jev__workflow-controls-v1__security_incidents/train-*.parquet
  - split: validation
    path: open_jev__workflow-controls-v1__security_incidents/validation-*.parquet
- config_name: openbookqa
  data_files:
  - split: train
    path: openbookqa/train-*.parquet
  - split: validation
    path: openbookqa/validation-*.parquet
- config_name: patent_similarity
  data_files:
  - split: train
    path: patent_similarity/train-*.parquet
  - split: validation
    path: patent_similarity/validation-*.parquet
- config_name: plane
  data_files:
  - split: train
    path: plane/train-*.parquet
  - split: validation
    path: plane/validation-*.parquet
- config_name: poem_sentiment
  data_files:
  - split: train
    path: poem_sentiment/train-*.parquet
  - split: validation
    path: poem_sentiment/validation-*.parquet
- config_name: qasc
  data_files:
  - split: train
    path: qasc/train-*.parquet
  - split: validation
    path: qasc/validation-*.parquet
- config_name: qasper
  data_files:
  - split: train
    path: qasper/train-*.parquet
  - split: validation
    path: qasper/validation-*.parquet
- config_name: quartz
  data_files:
  - split: train
    path: quartz/train-*.parquet
  - split: validation
    path: quartz/validation-*.parquet
- config_name: ruletaker
  data_files:
  - split: train
    path: ruletaker/train-*.parquet
  - split: validation
    path: ruletaker/validation-*.parquet
- config_name: scitail
  data_files:
  - split: train
    path: scitail/train-*.parquet
  - split: validation
    path: scitail/validation-*.parquet
- config_name: scone
  data_files:
  - split: train
    path: scone/train-*.parquet
  - split: validation
    path: scone/validation-*.parquet
- config_name: sdoh_nli
  data_files:
  - split: train
    path: sdoh_nli/train-*.parquet
  - split: validation
    path: sdoh_nli/validation-*.parquet
- config_name: sgd
  data_files:
  - split: train
    path: sgd/train-*.parquet
  - split: validation
    path: sgd/validation-*.parquet
- config_name: sms_spam
  data_files:
  - split: train
    path: sms_spam/train-*.parquet
  - split: validation
    path: sms_spam/validation-*.parquet
- config_name: snips
  data_files:
  - split: train
    path: snips/train-*.parquet
  - split: validation
    path: snips/validation-*.parquet
- config_name: snli
  data_files:
  - split: train
    path: snli/train-*.parquet
  - split: validation
    path: snli/validation-*.parquet
- config_name: social_iqa
  data_files:
  - split: train
    path: social_iqa/train-*.parquet
  - split: validation
    path: social_iqa/validation-*.parquet
- config_name: space_nli
  data_files:
  - split: train
    path: space_nli/train-*.parquet
  - split: validation
    path: space_nli/validation-*.parquet
- config_name: spartqa
  data_files:
  - split: train
    path: spartqa/train-*.parquet
  - split: validation
    path: spartqa/validation-*.parquet
- config_name: squad2
  data_files:
  - split: train
    path: squad2/train-*.parquet
  - split: validation
    path: squad2/validation-*.parquet
- config_name: stepgame
  data_files:
  - split: train
    path: stepgame/train-*.parquet
  - split: validation
    path: stepgame/validation-*.parquet
- config_name: strategyqa
  data_files:
  - split: train
    path: strategyqa/train-*.parquet
  - split: validation
    path: strategyqa/validation-*.parquet
- config_name: synthetic_relevance__natural_questions
  data_files:
  - split: train
    path: synthetic_relevance__natural_questions/train-*.parquet
  - split: validation
    path: synthetic_relevance__natural_questions/validation-*.parquet
- config_name: synthetic_relevance__trivia_qa
  data_files:
  - split: train
    path: synthetic_relevance__trivia_qa/train-*.parquet
  - split: validation
    path: synthetic_relevance__trivia_qa/validation-*.parquet
- config_name: taskmaster
  data_files:
  - split: train
    path: taskmaster/train-*.parquet
  - split: validation
    path: taskmaster/validation-*.parquet
- config_name: temporal_nli
  data_files:
  - split: train
    path: temporal_nli/train-*.parquet
  - split: validation
    path: temporal_nli/validation-*.parquet
- config_name: tracie
  data_files:
  - split: train
    path: tracie/train-*.parquet
  - split: validation
    path: tracie/validation-*.parquet
- config_name: wiqa
  data_files:
  - split: train
    path: wiqa/train-*.parquet
  - split: validation
    path: wiqa/validation-*.parquet
---

# Decisions DE

Ein deutsches Trainingsset für Entscheidungsmodelle (System One / Decision API): **167,850 Fälle
in 180,892 Zeilen mit 374,186 typisierten Fragen** (`choice`, `noul`, `score`) aus **90 Quellen**, übersetzt aus den
Train-Splits von [`hotchpotch/bekko-system-one-dataset-v0`](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0).
Jeder Fall hat einen State, eine oder mehrere Fragen mit Antwortoptionen und als Ziel eine Verteilung über die
Optionen (Mehrheits- oder Annotator-Verteilung, je nach Quelle).

Gedacht ist es als Ergänzung zu [`johannhartmann/typed-decisions-de`](https://huggingface.co/datasets/johannhartmann/typed-decisions-de):
dort vier Business-Workflows, hier eine breite Mischung aus Intent-Erkennung, NLI, Leseverstehen,
Moderation, Präferenz- und Hilfsbereitschaftsurteilen, Logik, räumlichem und zeitlichem Schließen sowie
regelbasierten Agenten-Entscheidungen.

## Aufbau

| Spalte | Inhalt |
|---|---|
| `id` | Fall-ID aus bekko (`<quelle>:<config>:train:<n>`), bei aufgeteilten Fällen mit `#<n>` |
| `source` | bekko-Konfiguration |
| `upstream` | Ursprungsdatensatz |
| `license` | Lizenz des Ursprungsdatensatzes |
| `state` | JSON: der zu beurteilende Inhalt (deutsch, Keys englisch) |
| `questions` | JSON: `{name: {type, instructions, criteria}}`, Texte deutsch, Options-Keys englisch |
| `gold` | JSON: `{name: {type, label, probabilities}}`, unverändert aus bekko |

Splits: `train` (180,327 Zeilen) und `validation` (565 Zeilen; 3 % der Fälle je Quelle, per Hash
gewählt; Teile eines Falls bleiben im selben Split).

**Version 2:** enthält alle Fälle von Version 1 (Tag `v1`) und hat exakt deren `validation`-Split, damit Modelle beider Versionen vergleichbar bleiben. Neue Fälle aus einer Quellgruppe, die schon in `validation` vorkommt, sind nicht aufgenommen.

**Passend für `max_seq_length` 4096:** Unsloths Clef-Kopf stellt alle Fragen einer Zeile als Schema vor den
State und kürzt den State, wenn beides nicht passt. Fälle mit sehr vielen Fragen (NLU++: 45 Ja/Nein-Fragen,
Gretel PII, ContractNLI) sind deshalb auf mehrere Zeilen mit gleichem State aufgeteilt; Fälle, deren State
auch dann nicht passt, sind entfernt (vor allem lange Paper aus Qasper und lange Verträge aus ContractNLI).
Gemessen mit Unsloths Encoder und dem Gemma-4-Tokenizer, mit 8 % Reserve für andere Tokenizer.
Konfigurationen: `all` und eine je Quelle, damit sich Quellen nach Lizenz oder Aufgabe filtern lassen.

In Unsloth Studio: „Trainieren als: Entscheidungsmodell“, Dataset `johannhartmann/decisions-de`, Subset `all`. Studio nimmt
`validation` automatisch als Eval-Split. Alle Fragen sind mit Unsloths Decision-Parser geprüft.

```python
from datasets import load_dataset
import json
ds = load_dataset("johannhartmann/decisions-de", "all", split="train")
row = ds[0]
state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
```

## Wie es entstanden ist

1. **Quellen:** Aus den 153 Train-Konfigurationen von bekko wurden nur die genommen, deren Ursprungslizenz
   laut bekkos Lizenzprüfung (`license-audit.json`, Stand 30.09.2026) Bearbeitung und Weitergabe erlaubt:
   apache-2.0, bsd, cc-by-2.0, cc-by-3.0, cc-by-4.0, cc-by-sa-3.0, cc-by-sa-4.0, cc0-1.0, cdla-sharing-1.0, mit. Ausgeschlossen: ungeklärte Lizenzen, NC-, GPL-/LGPL-, AFL- und
   „research only“-Quellen.
2. **Sprachgebundene Aufgaben ausgeschlossen**, deren Labels am englischen Wortlaut hängen: BLiMP
   (Grammatikalität), UD EWT (Wortarten), LexComp (Nominalkomposita), WinoGrande (Pronomen), ROPES
   (Antwort ist ein Wort des Textes), Few-NERD (Token-Offsets), Open-Jev E-Mail- und Telefon-Extraktion
   (Zeichen-Offsets). Ranking-Aufgaben (Natural Questions, TriviaQA) entfallen, weil Studio nur
   `choice`/`noul`/`score` trainiert.
3. **Auswahl:** bis zu 2.000 Fälle je Quelle, deterministisch per Hash, zuerst ein Fall je Quellgruppe.
4. **Übersetzung** mit der Anthropic Batch API: Fragen, Optionsbeschreibungen und Strings, die in drei oder
   mehr Fällen einer Quelle vorkommen, mit Claude Sonnet 5.5; die übrigen Falltexte mit Claude Haiku 5.5
   (nach einem Vergleichspilot mit gleichen Texten, ähnliche Qualität), Fehler und Auffälligkeiten in einer
   Abschlussrunde mit Sonnet 5.5. Vorgaben: Ton, Fehler und toxische Formulierungen erhalten, Code, IDs,
   Zahlen und Namen unverändert.
5. **Prüfungen:** jeder String übersetzt; Übersetzungen mit stark abweichender Länge oder eingefügten
   Inhaltsnotizen statt des Textes verworfen (deren Fälle entfallen); bei Gretel PII die Offsets der
   unveränderten Erwähnungen im deutschen Text neu berechnet; alle Fälle von Unsloths Decision-Parser
   akzeptiert.

## Hinweise

- **Labels stammen aus den englischen Originalen** und wurden nicht neu erhoben. Übersetzung kann Nuancen
  verschieben, besonders bei Ton, Ironie und Moderationsurteilen.
- **Options-Keys bleiben englisch** (z. B. `entailment`, `card_arrival`), damit `gold` gültig bleibt; ihre
  Beschreibungen sind deutsch. Bei Intent-Datensätzen sind die Beschreibungen kurze Label-Namen.
- **Zahlen:** In rund 10.300 Strings fehlt nach der Übersetzung eine Ziffernfolge des Originals, fast immer
  wegen Lokalisierung („4,000“ → „4.000“, „9:30 PM“ → „21:30 Uhr“, „Henry 8th“ → „Heinrich VIII.“).
- **Maschinell übersetzt, ohne menschliches Gegenlesen.** Stichproben wurden geprüft.
- **Enthält anstößige Inhalte** aus Moderations- und Safety-Datensätzen (Aegis 2.0, Civil Comments,
  Measuring Hate Speech, HH-RLHF), bewusst nicht entschärft.
- Nicht auf Benchmark-Testdaten trainiert: bekko nutzt Train-Splits und hat gegen den S1MB-Test
  dekontaminiert; gegen eine Stichprobe des Decision Index 0.3 gibt es keine exakten Überschneidungen.

## Lizenz

Jede Quelle behält die Lizenz ihres Ursprungsdatensatzes (Spalte `license`, Tabelle unten). Die
Übersetzung ist eine Bearbeitung; bei CC-BY-SA-Quellen steht sie unter derselben Lizenz, bei CC-BY-Quellen
mit Namensnennung der Ursprünge wie unten. Die Aufbereitung im Entscheidungsformat stammt von
[`hotchpotch/bekko-system-one-dataset-v0`](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0),
dessen `sources.json` die vollständigen Herkunfts- und Lizenznachweise enthält. Wer nur bestimmte
Lizenzen nutzen will, filtert nach `license` oder lädt einzelne Konfigurationen.

| Konfiguration | Ursprung | Lizenz | Fälle | Fragen |
|---|---|---|---:|---:|
| `aegis2` | [`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | cc-by-4.0 | 1,995 | 2,894 |
| `aqua_rat` | [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) | apache-2.0 | 1,999 | 1,999 |
| `arc` | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | cc-by-sa-4.0 | 2,000 | 2,000 |
| `arct` | [`tasksource/arct`](https://huggingface.co/datasets/tasksource/arct) | apache-2.0 | 1,086 | 1,086 |
| `babi_nli` | [`tasksource/babi_nli`](https://huggingface.co/datasets/tasksource/babi_nli) | bsd | 2,000 | 2,000 |
| `banking77` | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) | cc-by-4.0 | 2,000 | 2,000 |
| `bitext` | [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | cdla-sharing-1.0 | 2,000 | 2,000 |
| `boardgameqa` | [`tasksource/Boardgame-QA`](https://huggingface.co/datasets/tasksource/Boardgame-QA) | cc-by-4.0 | 2,000 | 2,000 |
| `boolq` | [`google/boolq`](https://huggingface.co/datasets/google/boolq) | cc-by-sa-3.0 | 1,997 | 1,997 |
| `canttalk` | [`nvidia/CantTalkAboutThis-Topic-Control-Dataset`](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset) | cc-by-4.0 | 1,977 | 1,977 |
| `circa` | [`google-research-datasets/circa`](https://huggingface.co/datasets/google-research-datasets/circa) | cc-by-4.0 | 2,000 | 2,000 |
| `civil_comments` | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | cc0-1.0 | 1,998 | 13,986 |
| `clinc` | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) | cc-by-3.0 | 2,000 | 2,000 |
| `commonsense_qa` | [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) | mit | 2,000 | 2,000 |
| `commonsense_qa2` | [`tasksource/commonsense_qa_2.0`](https://huggingface.co/datasets/tasksource/commonsense_qa_2.0) | cc-by-4.0 | 2,000 | 2,000 |
| `contract_nli` | `https://github.com/stanfordnlp/contract-nli` | cc-by-4.0 | 275 | 4,471 |
| `corr2cause` | [`tasksource/corr2cause`](https://huggingface.co/datasets/tasksource/corr2cause) | mit | 2,000 | 2,000 |
| `cosmos_qa` | [`Samsoup/cosmos_qa`](https://huggingface.co/datasets/Samsoup/cosmos_qa) | cc-by-4.0 | 1,995 | 1,995 |
| `dbpedia` | [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | cc-by-sa-3.0 | 2,000 | 2,000 |
| `defeasible_nli` | [`tasksource/defeasible-nli`](https://huggingface.co/datasets/tasksource/defeasible-nli) | apache-2.0,cc-by-sa-4.0 | 2,000 | 2,000 |
| `esci` | [`tasksource/esci`](https://huggingface.co/datasets/tasksource/esci) | apache-2.0 | 1,987 | 1,987 |
| `ethics` | [`hendrycks/ethics`](https://huggingface.co/datasets/hendrycks/ethics) | mit | 2,000 | 2,000 |
| `feasibility_qa` | [`tasksource/feasibilityQA`](https://huggingface.co/datasets/tasksource/feasibilityQA) | mit | 1,996 | 1,996 |
| `fig_qa` | [`nightingal3/fig-qa`](https://huggingface.co/datasets/nightingal3/fig-qa) | mit | 2,000 | 2,000 |
| `fol_nli` | [`tasksource/FOL-nli`](https://huggingface.co/datasets/tasksource/FOL-nli) | apache-2.0 | 2,000 | 2,000 |
| `go_emotions` | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | apache-2.0 | 2,000 | 56,000 |
| `gretel_pii` | [`gretelai/gretel-pii-masking-en-v1`](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | apache-2.0 | 1,975 | 6,054 |
| `gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | mit | 2,000 | 2,000 |
| `hans` | [`tasksource/hans`](https://huggingface.co/datasets/tasksource/hans) | mit | 2,000 | 2,000 |
| `helpsteer` | [`nvidia/HelpSteer`](https://huggingface.co/datasets/nvidia/HelpSteer) | cc-by-4.0 | 1,996 | 9,980 |
| `helpsteer2` | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | cc-by-4.0 | 1,995 | 9,971 |
| `helpsteer3` | [`nvidia/HelpSteer3`](https://huggingface.co/datasets/nvidia/HelpSteer3) | cc-by-4.0 | 1,683 | 2,489 |
| `hh_rlhf` | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) | mit | 1,950 | 1,950 |
| `hwu64` | [`FastFit/hwu_64`](https://huggingface.co/datasets/FastFit/hwu_64) | cc-by-4.0 | 2,000 | 2,000 |
| `logical_entailment` | [`tasksource/logical-entailment`](https://huggingface.co/datasets/tasksource/logical-entailment) | apache-2.0 | 2,000 | 2,000 |
| `massive` | [`AmazonScience/massive`](https://huggingface.co/datasets/AmazonScience/massive) | cc-by-4.0 | 2,000 | 4,000 |
| `measuring_hate` | [`ucberkeley-dlab/measuring-hate-speech`](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | cc-by-4.0 | 2,000 | 20,000 |
| `minds14` | [`PolyAI/minds14`](https://huggingface.co/datasets/PolyAI/minds14) | cc-by-4.0 | 1,656 | 1,656 |
| `monotonicity` | [`tasksource/monotonicity-entailment`](https://huggingface.co/datasets/tasksource/monotonicity-entailment) | apache-2.0 | 1,988 | 1,988 |
| `mtop` | [`tasksource/mtop`](https://huggingface.co/datasets/tasksource/mtop) | cc-by-sa-4.0 | 2,000 | 4,000 |
| `nlsat` | [`tasksource/nlsat`](https://huggingface.co/datasets/tasksource/nlsat) | apache-2.0 | 2,000 | 2,000 |
| `nlupp` | `https://github.com/PolyAI-LDN/task-specific-datasets` | cc-by-4.0 | 1,991 | 90,440 |
| `oasst1` | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | apache-2.0 | 1,995 | 1,995 |
| `open_jev__amount-extraction-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,919 | 1,919 |
| `open_jev__browser-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,985 | 1,985 |
| `open_jev__citation-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,864 | 1,864 |
| `open_jev__context-retention-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,954 | 1,954 |
| `open_jev__drone-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,970 | 1,970 |
| `open_jev__entity-alignment-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,918 | 1,918 |
| `open_jev__ir-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,929 | 1,929 |
| `open_jev__mailroom-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,962 | 1,962 |
| `open_jev__painting-geometry-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,771 | 1,771 |
| `open_jev__reasoning-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,997 | 1,997 |
| `open_jev__silent-failure-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,947 | 1,947 |
| `open_jev__snake-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 2,000 | 2,000 |
| `open_jev__sponsor-segment-control-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,277 | 1,277 |
| `open_jev__tic_tac_toe-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,977 | 1,977 |
| `open_jev__tile_platformer-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,236 | 1,236 |
| `open_jev__trex_runner-v1` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 28 | 28 |
| `open_jev__workflow-controls-v1__agent_trace_observability` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,944 | 1,944 |
| `open_jev__workflow-controls-v1__customer_service` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,942 | 1,942 |
| `open_jev__workflow-controls-v1__invoice_processing` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,945 | 1,945 |
| `open_jev__workflow-controls-v1__security_incidents` | [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) | cc0-1.0 | 1,946 | 1,946 |
| `openbookqa` | [`allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa) | apache-2.0 | 2,000 | 2,000 |
| `patent_similarity` | [`tasksource/patent-phrase-similarity`](https://huggingface.co/datasets/tasksource/patent-phrase-similarity) | cc-by-4.0 | 1,979 | 1,979 |
| `plane` | [`tasksource/PLANE-ood`](https://huggingface.co/datasets/tasksource/PLANE-ood) | cc-by-2.0 | 2,000 | 2,000 |
| `poem_sentiment` | [`google-research-datasets/poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | cc-by-4.0 | 892 | 892 |
| `qasc` | [`allenai/qasc`](https://huggingface.co/datasets/allenai/qasc) | cc-by-4.0 | 2,000 | 2,000 |
| `qasper` | [`allenai/qasper`](https://huggingface.co/datasets/allenai/qasper) | cc-by-4.0 | 52 | 52 |
| `quartz` | [`allenai/quartz`](https://huggingface.co/datasets/allenai/quartz) | cc-by-4.0 | 1,997 | 1,997 |
| `ruletaker` | [`tasksource/ruletaker`](https://huggingface.co/datasets/tasksource/ruletaker) | apache-2.0 | 2,000 | 2,000 |
| `scitail` | [`allenai/scitail`](https://huggingface.co/datasets/allenai/scitail) | apache-2.0 | 1,996 | 1,996 |
| `scone` | [`tasksource/scone`](https://huggingface.co/datasets/tasksource/scone) | cc0-1.0 | 2,000 | 2,000 |
| `sdoh_nli` | [`tasksource/SDOH-NLI`](https://huggingface.co/datasets/tasksource/SDOH-NLI) | cc-by-4.0 | 1,982 | 1,982 |
| `sgd` | `https://github.com/google-research-datasets/dstc8-schema-guided-dialogue` | cc-by-sa-4.0 | 2,000 | 2,000 |
| `sms_spam` | [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) | cc-by-4.0 | 1,993 | 1,993 |
| `snips` | `https://github.com/sonos/nlu-benchmark` | cc0-1.0 | 2,000 | 2,000 |
| `snli` | [`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli) | cc-by-sa-4.0 | 2,000 | 2,000 |
| `social_iqa` | [`baber/social_i_qa`](https://huggingface.co/datasets/baber/social_i_qa) | cc-by-4.0 | 2,000 | 2,000 |
| `space_nli` | [`tasksource/SpaceNLI`](https://huggingface.co/datasets/tasksource/SpaceNLI) | mit | 1,942 | 1,942 |
| `spartqa` | [`tasksource/spartqa-mchoice`](https://huggingface.co/datasets/tasksource/spartqa-mchoice) | mit | 2,000 | 2,000 |
| `squad2` | [`rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2) | cc-by-sa-4.0 | 1,964 | 1,964 |
| `stepgame` | [`tasksource/stepgame`](https://huggingface.co/datasets/tasksource/stepgame) | mit | 2,000 | 2,000 |
| `strategyqa` | [`tasksource/strategy-qa`](https://huggingface.co/datasets/tasksource/strategy-qa) | mit | 2,000 | 2,000 |
| `synthetic_relevance__natural_questions` | [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions), [`google-research-datasets/natural_questio`](https://huggingface.co/datasets/google-research-datasets/natural_questio) | cc-by-sa-3.0 | 1,982 | 3,964 |
| `synthetic_relevance__trivia_qa` | [`sentence-transformers/trivia-qa`](https://huggingface.co/datasets/sentence-transformers/trivia-qa), [`mandarjoshi/trivia_qa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | apache-2.0 | 1,977 | 3,954 |
| `taskmaster` | `https://github.com/google-research-datasets/Taskmaster` | cc-by-4.0 | 1,997 | 1,997 |
| `temporal_nli` | [`tasksource/temporal-nli`](https://huggingface.co/datasets/tasksource/temporal-nli) | apache-2.0 | 2,000 | 2,000 |
| `tracie` | [`tasksource/tracie`](https://huggingface.co/datasets/tasksource/tracie) | apache-2.0 | 1,104 | 1,104 |
| `wiqa` | [`allenai/wiqa`](https://huggingface.co/datasets/allenai/wiqa) | apache-2.0 | 1,948 | 1,948 |

---

## English summary

German training data for typed decision models: 167,850 cases in 180,892 rows (374,186 questions) from
90 permissively licensed sources, translated from the train splits of
`hotchpotch/bekko-system-one-dataset-v0` with Claude (Sonnet 5.5 for questions, options and shared strings;
Haiku 5.5 for case texts; Sonnet for fixes). Gold distributions, option keys and JSON keys are unchanged.
Language-bound tasks (grammaticality, POS, pronoun blanks, word/offset-based answers) were excluded. Every row
fits Unsloth's Clef encoder at max_seq_length 4096 without truncation: cases with many questions are split into
rows sharing the state, cases whose state is still too long are dropped. Every
subset keeps its upstream license (see table); CC-BY-SA subsets remain CC-BY-SA. Machine-translated without
human review; contains offensive content from moderation datasets.

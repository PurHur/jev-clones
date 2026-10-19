---
license: apache-2.0
language:
- de
pretty_name: Typed Decisions (Deutsch)
size_categories:
- 1K<n<10K
task_categories:
- text-classification
source_datasets:
- LocalLLaMA/typed-decisions
tags:
- structured-decisions
- calibration
- probabilistic-classification
- system-one
- workflow-evaluation
- synthetic
- translation
- german
configs:
- config_name: agent_trace_observability
  data_files:
  - split: train
    path: agent_trace_observability/train-*.parquet
  - split: test
    path: agent_trace_observability/test-*.parquet
- config_name: customer_service
  data_files:
  - split: train
    path: customer_service/train-*.parquet
  - split: test
    path: customer_service/test-*.parquet
- config_name: invoice_processing
  data_files:
  - split: train
    path: invoice_processing/train-*.parquet
  - split: test
    path: invoice_processing/test-*.parquet
- config_name: security_incidents
  data_files:
  - split: train
    path: security_incidents/train-*.parquet
  - split: test
    path: security_incidents/test-*.parquet
- config_name: all
  data_files:
  - split: train
    path: all/train-*.parquet
  - split: test
    path: all/test-*.parquet
---

# Typed Decisions (Deutsch)

Deutsche Übersetzung von [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
Revision [`c806f6c`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c806f6c69d734f29436f25536138547031a673d6),
einem Benchmark für typisierte, probabilistische Entscheidungen. Ein Modell erhält einen
State (Kundengespräch, Rechnung, Sicherheitsalarm, Agenten-Trace) und beantwortet fünf
typisierte Fragen dazu (`noul`, `choice`, `score`). Jede Antwort ist eine
Wahrscheinlichkeitsverteilung, kein einzelnes Label.

Aufbau, Subsets, Splits, Spalten und Zeilen-IDs entsprechen exakt dem Original; übersetzt
sind nur die Texte. Für Beschreibung, Bewertungsmethodik und Referenzwerte siehe die
[Dataset-Card des Originals](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
und den Artikel [Typed Decisions on Latent Node](https://latentnode.pages.dev/articles/typed-decisions.html).
Die Leaderboard-Werte dort gelten für die englische Fassung und sind auf diese Übersetzung
nicht übertragbar.

## Was übersetzt wurde

| Spalte | Übersetzt | Unverändert |
|---|---|---|
| `state` | alle natürlichsprachlichen Werte: Chatverläufe, Alarmbeschreibungen, Rechnungsnotizen, Aufgaben, Vorgaben, Enum-Werte wie `"Kunde"` | JSON-Keys, IDs, Hostnamen, IPs, E-Mail-Adressen, Pfade, Hashes, Beträge, Firmennamen, Berechtigungen wie `repo:read` |
| `questions` | `instructions`, Optionsbeschreibungen in `criteria`, Stufen der `score`-Rubriken | Fragennamen (`action`, `urgency`, …), Options-Keys (`human_review`, `stop`, …), Fragetypen |
| `factors` | Textwerte wie bei `state` | Keys, Zahlen, Booleans |
| `gold`, `<frage>__*`, `label_agreement`, `n_questions`, `id`, `workflow`, `split` | – | vollständig unverändert |

Die Options-Keys bleiben englisch, damit `gold` weiter auf dieselben Optionen zeigt. Wer das
Dataset über eine System-One-kompatible API nutzt, schickt deutsche Fragen und erhält
englische Keys zurück.

Weitere bewusste Entscheidungen:

- Zitierte UI- und Fehlermeldungen (z. B. „Export failed“, „Invalid credentials“) bleiben
  englisch, so wie sie in englischsprachiger Software tatsächlich erscheinen.
- Uhrzeiten im 12-Stunden-Format wurden in Fließtexten ins 24-Stunden-Format übertragen
  („5 PM“ → „17 Uhr“, 9 Stellen). Alle anderen Zahlen stehen wie im Original, einschließlich
  englischer Tausendertrennzeichen („4,500“) und Zeitstempel.
- Für „seats“ kommen je nach Gespräch „Plätze“ und „Lizenzen“ vor.

## Hinweise zu den Labels

`gold` wurde unverändert übernommen. Laut Original ist es der Mittelwert aus drei Samples
eines Teacher-Modells, die auf den **englischen** Texten erzeugt wurden. Für die deutschen
Texte wurden keine neuen Labels erzeugt. Die Übersetzung erhält den Inhalt, kleine
Verschiebungen in Ton oder Nuance können die Passung zwischen Text und weicher Verteilung
aber leicht verändern.

Wie im Original gilt: nicht auf `test` trainieren, auch nicht auf Labels oder auf
Vorhersagen, die aus `test`-Zeilen abgeleitet sind.

## Wie übersetzt wurde

Maschinell mit Claude (Anthropic), ohne menschliches Gegenlesen:

1. Übersetzt wurden die 2.217 eindeutigen Strings statt jeder Zeile einzeln. Reine
   Bezeichner (IDs, Datumsangaben, Codes) wurden vorab ausgefiltert.
2. Ein Glossar mit allen 75 Frage- und Optionstexten sowie den 82 häufigsten Werten legte
   die Terminologie fest; die übrigen 2.060 Strings wurden auf dieser Basis übersetzt.
3. Geprüft wurde, dass jeder String übersetzt ist, alle Zahlen erhalten bleiben (Ausnahmen
   nur die genannten Uhrzeiten), JSON-Struktur, Keys und `gold` identisch zum Original sind
   und alle 8.000 Fragen von Unsloths Decision-Parser akzeptiert werden.

## Nutzung

```python
from datasets import load_dataset
import json

ds = load_dataset("johannhartmann/typed-decisions-de", "customer_service", split="test")
row = ds[0]
state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
print(questions["urgency"]["instructions"])   # Wie zeitkritisch ist dieses Gespräch?
print(gold["urgency"]["probabilities"])
```

In Unsloth Studio: „Trainieren als: Entscheidungsmodell“, Dataset
`johannhartmann/typed-decisions-de`, Subset `all`.

## Lizenz und Herkunft

Apache-2.0, wie das Original. Dieses Dataset ist eine Bearbeitung (Übersetzung) von
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(Apache-2.0); geändert wurden ausschließlich die oben genannten Textfelder. Der Lizenztext
liegt in [`LICENSE`](LICENSE). Diese Übersetzung ist nicht mit den Autor:innen des
Originals verbunden.

---

## English summary

German translation of [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
at revision `c806f6c`, released under the same Apache-2.0 license. Layout, configs, splits,
columns and row ids match the original exactly. All natural-language text in `state`,
`questions` (instructions, option descriptions, score rubrics) and `factors` is translated;
JSON keys, question names, option keys, identifiers and all `gold` / flat label columns are
unchanged, so labels still point at the same options. Gold labels come from the original
teacher run on the English text and were not regenerated. The translation was produced by
Claude (Anthropic) with a fixed glossary and automated checks (coverage, preserved numbers,
identical structure and gold, all 8,000 questions accepted by Unsloth's decision parser),
without human review. The original leaderboard numbers do not transfer to this version. Do
not train on `test`.

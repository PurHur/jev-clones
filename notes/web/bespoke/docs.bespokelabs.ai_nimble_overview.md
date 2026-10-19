> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bespokelabs.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Nimble overview

> A fast, low-cost API for text classification, routing, moderation, and LLM-as-a-judge scoring, with a probability for every answer.

Nimble is a family of models from Bespoke Labs for text classification and other judgments about text or JSON. Nimble answers with probabilities, so you can set your own thresholds.

You call Nimble at `POST https://api.bespokelabs.ai/v1/systemone`. The `model` field in the request picks the model.

## What you can build

Use Nimble wherever your code needs a judgment about text: classifying it, routing it, checking it against a rule, or rating it. Nimble works zero-shot. You describe each option in plain language, and there's no classifier to train.

| Use case | Example question | Question type |
| - | - | - |
| Ticket routing | Which team should handle this ticket? | Multiple choice |
| Triage | How urgent is this request? | Rating |
| Content moderation | Does this comment contain harassment? | Yes/no |
| Intent detection | What does the user want to do? | Multiple choice |
| Policy checks | Does this request break the refund policy? | Yes/no |
| Sentiment analysis | How positive is this review? | Rating |
| LLM-as-a-judge | How helpful is this answer to the user's question? | Rating |

## Models

Every model answers up to 64 typed questions about a piece of text or JSON.

| Model | Description |
| - | - |
| `nimble-v4-max` | Our most accurate model. This is the default model. |
| `nimble-v4` | Deprecated. It points to `nimble-v4-max`. |
| `nimble-v3` | Deprecated. Use `nimble-v4-max` instead. |

`nimble-latest` currently points to `nimble-v4-max`, and it will move to newer models as we release them. To stay on one model, request it by name, e.g., `nimble-v4-max`.

Different models return different probabilities for the same request. If you tuned thresholds on the answers, check them again when you switch models or when `nimble-latest` moves to a new one.

## The request format

Nimble takes one request format, which is called System One. A request has three fields:

* `model` names the model.
* `state` is the text or JSON to ask about.
* `questions` maps your own question IDs to questions. The answers use the same IDs.

You can send the requests with plain HTTP or with [TypeSafe's Python SDK](/nimble/typesafe-sdk). The path `/v1/nimble/systemone` takes the same requests as `/v1/systemone`.

Nimble has three types of question:

* A yes/no question (`noul`) asks whether a statement is true. The answer is the probability that it is true.
* A multiple-choice question (`choice`) asks which of 2 to 255 options fits best. The answer names the most likely option and gives the probability of every option.
* A rating question (`score`) asks where the state falls on a scale of 2 to 255 ordered levels. The answer is the expected level.

<CardGroup cols={2}>
  <Card title="Quickstart" icon="rocket" href="/nimble/quickstart">
    Get a key and make your first request.
  </Card>

  <Card title="Question types" icon="list-check" href="/nimble/questions">
    Write `noul`, `choice`, and `score` questions and read the answers.
  </Card>

  <Card title="TypeSafe SDK" icon="code" href="/nimble/typesafe-sdk">
    Call Nimble from Python.
  </Card>

  <Card title="Pricing" icon="coins" href="/nimble/pricing">
    What each model costs and how Nimble counts tokens.
  </Card>
</CardGroup>


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.
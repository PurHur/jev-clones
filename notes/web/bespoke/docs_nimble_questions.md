> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bespokelabs.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Question types

> How to write yes/no (noul), multiple-choice (choice), and rating (score) questions, and how to read the probabilities in the answers.

A request has a `state` and a map of `questions`. The keys of the map are your own IDs, and the answers use the same IDs.

The state is a string, or a JSON object or list. Nimble reads a JSON state as JSON text.

## Noul

A `noul` question is a yes/no question: it asks whether a statement is true. The answer's `noul` field is the probability that it is true, from 0 to 1.

```json theme={null}
"refund": {
  "type": "noul",
  "instructions": "Does the customer request a refund?"
}
```

You can say what true and false mean with `criteria`. This is optional.

```json theme={null}
"refund": {
  "type": "noul",
  "instructions": "Does the customer request a refund?",
  "criteria": {
    "true": "The customer asks for money back",
    "false": "The customer does not ask for money back"
  }
}
```

The answer looks like this.

```json theme={null}
"refund": { "type": "noul", "noul": 0.998 }
```

## Choice

A `choice` question is a multiple-choice question: it asks which option fits best. Use it to classify or route the state. `criteria` maps each option key to a description. A question has 2 to 255 options. If a description is `null`, Nimble reads the key instead.

```json theme={null}
"department": {
  "type": "choice",
  "instructions": "Which department should handle this request?",
  "criteria": {
    "billing": "Charges, payments and refunds",
    "technical": "Software bugs and outages"
  }
}
```

The answer's `choice` is the option with the highest probability. `probabilities` gives every option, and they add up to 1.

```json theme={null}
"department": {
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 1.0, "technical": 0.0 },
  "confidence": 1.0
}
```

## Score

A `score` question asks where the state falls on a rating scale, e.g., urgency, sentiment, or the quality of an LLM's answer. `criteria` lists the levels from lowest to highest. A question has 2 to 255 levels.

```json theme={null}
"urgency": {
  "type": "score",
  "instructions": "How urgent is this request?",
  "criteria": [
    "Routine billing request; service works",
    "Some functionality unavailable",
    "Complete service outage"
  ]
}
```

The answer's `score` is the expected level, counting from 0. It can fall between two levels. `legend` maps each level number to its text.

```json theme={null}
"urgency": {
  "type": "score",
  "score": 0.0,
  "legend": {
    "0": "Routine billing request; service works",
    "1": "Some functionality unavailable",
    "2": "Complete service outage"
  },
  "probabilities": { "0": 1.0, "1": 0.0, "2": 0.0 },
  "confidence": 1.0
}
```

## Confidence

`choice` and `score` answers have a `confidence` field. It is 1 when one option has all the probability, and 0 when every option is equally likely. It is not the chance that the answer is right.

## Tips

* Ask many questions in one request. Nimble reads the state once for all of them, so this is faster than one request per question.
* Pick clear question IDs and option keys. Nimble reads them as part of the prompt, so renaming one can change its answer.
* Set your own threshold on `noul`, e.g., act only when it is above 0.8. To pick the threshold, run Nimble on examples whose answers you know, and choose the value that gives the accuracy you need.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.
> ## Documentation Index
> Fetch the complete documentation index at: https://docs.perplexity.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Decisions API

> The Decisions API answers questions about your content with probabilities instead of text: yes or no, one of your options, or a level on your rubric.

<Note>
  The Decisions API is billed per million input tokens: \$0.02 for `pplx-decider-v1.2` and \$0.01 for `pplx-decider-flash-v1.2`. Output tokens are free. See [Pricing](#pricing).
</Note>

The Decisions API answers questions with probabilities. You send the content as `state`, which can be text, JSON, or images, attach one or more named questions, and get one answer per question:

| Question type | You ask | You get back |
| - | - | - |
| **`noul`** | A yes/no question, or a statement to check | The probability of yes, from 0 to 1 |
| **`choice`** | Pick one of the options you define | A probability for every option, plus the most likely one |
| **`score`** | Rate the content on an ordered rubric | A probability for every level, plus the expected score |

### What a decision model is

A decision model is a class of model built to make fast, structured decisions that software can use directly. It reads natural-language text and images the way a multimodal language model does, but instead of writing text it returns typed answers with probabilities: yes or no, one of your options, or a level on your rubric. It does not write replies, generate code, or explain its reasoning; your code does the reasoning with the numbers it returns. `pplx-decider-v1.2` and `pplx-decider-flash-v1.2` are decision models, and the Decisions API is how you call them.

Use it where you would otherwise ask a chat model for a label and parse the reply: classifying, routing, grading against a rubric, or any decision you want to threshold. You get numbers you can compare against a cutoff, no output parsing, and as many questions as you need about the same content in one request.

## Quickstart

<Steps>
  <Step title="Set your API key">
    Any Perplexity API key works; create one at [console.perplexity.ai](https://console.perplexity.ai/project/keys) if you need one.

    <Tabs>
      <Tab title="macOS/Linux">
        ```bash theme={null}
        export PERPLEXITY_API_KEY="your_api_key_here"
        ```
      </Tab>

      <Tab title="Windows">
        ```powershell theme={null}
        setx PERPLEXITY_API_KEY "your_api_key_here"
        # setx applies to new terminals; open a new one before you continue
        ```
      </Tab>
    </Tabs>
  </Step>

  <Step title="Install an HTTP client">
    The Decisions API is a single JSON endpoint, so any HTTP client works. The Python example uses `httpx`. The TypeScript example uses the `fetch` built into Node.js 18 and later and has no dependencies; save it as `decisions.mts` and run `npx tsx decisions.mts` (npx downloads `tsx` on first use), or save it as `decisions.mjs` and run `node decisions.mjs`.

    ```bash theme={null}
    pip install httpx
    ```
  </Step>

  <Step title="Ask three questions about one review">
    Send a `POST` to `https://api.perplexity.ai/v1/decisions` with the content as `state`, the model `pplx-decider-latest`, and your questions. To use Flash instead, set `model` to `pplx-decider-flash-latest`; the request and response are the same.

    <CodeGroup>
      ```python Python theme={null}
      import os

      import httpx

      response = httpx.post(
          "https://api.perplexity.ai/v1/decisions",
          headers={"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"},
          json={
              "model": "pplx-decider-latest",
              "state": {
                  "title": "Battery died after two weeks",
                  "review": "The headphones sound great, but the battery stopped charging after two weeks.",
              },
              "questions": {
                  "defect": {
                      "type": "noul",
                      "instructions": "Does the review report a product defect?",
                  },
                  "sentiment": {
                      "type": "choice",
                      "instructions": "What is the overall sentiment of the review?",
                      "criteria": {
                          "positive": "Mostly satisfied",
                          "mixed": "Praise and complaints in one review",
                          "negative": "Mostly dissatisfied",
                      },
                  },
                  "severity": {
                      "type": "score",
                      "instructions": "How severe is the reported problem?",
                      "criteria": ["Cosmetic", "Inconvenient", "Product unusable"],
                  },
              },
          },
          timeout=30.0,
      )
      response.raise_for_status()
      answers = response.json()["answers"]

      print(answers["defect"]["noul"])       # probability of yes, 0 to 1
      print(answers["sentiment"]["choice"])  # the most likely option
      print(answers["severity"]["score"])    # expected level, 0 to 2
      ```

      ```typescript Typescript theme={null}
      const response = await fetch("https://api.perplexity.ai/v1/decisions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.PERPLEXITY_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "pplx-decider-latest",
          state: {
            title: "Battery died after two weeks",
            review: "The headphones sound great, but the battery stopped charging after two weeks.",
          },
          questions: {
            defect: {
              type: "noul",
              instructions: "Does the review report a product defect?",
            },
            sentiment: {
              type: "choice",
              instructions: "What is the overall sentiment of the review?",
              criteria: {
                positive: "Mostly satisfied",
                mixed: "Praise and complaints in one review",
                negative: "Mostly dissatisfied",
              },
            },
            severity: {
              type: "score",
              instructions: "How severe is the reported problem?",
              criteria: ["Cosmetic", "Inconvenient", "Product unusable"],
            },
          },
        }),
        signal: AbortSignal.timeout(30_000),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${await response.text()}`);
      }
      const { answers } = await response.json();

      console.log(answers.defect.noul);       // probability of yes, 0 to 1
      console.log(answers.sentiment.choice);  // the most likely option
      console.log(answers.severity.score);    // expected level, 0 to 2
      ```

      ```bash cURL theme={null}
      curl -X POST https://api.perplexity.ai/v1/decisions \
        -H "Authorization: Bearer $PERPLEXITY_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
          "model": "pplx-decider-latest",
          "state": {
            "title": "Battery died after two weeks",
            "review": "The headphones sound great, but the battery stopped charging after two weeks."
          },
          "questions": {
            "defect": {
              "type": "noul",
              "instructions": "Does the review report a product defect?"
            },
            "sentiment": {
              "type": "choice",
              "instructions": "What is the overall sentiment of the review?",
              "criteria": {
                "positive": "Mostly satisfied",
                "mixed": "Praise and complaints in one review",
                "negative": "Mostly dissatisfied"
              }
            },
            "severity": {
              "type": "score",
              "instructions": "How severe is the reported problem?",
              "criteria": ["Cosmetic", "Inconvenient", "Product unusable"]
            }
          }
        }'
      ```
    </CodeGroup>
  </Step>

  <Step title="Read the answers">
    The response has one answer per question, under the name you gave it. This example shows the response structure; the exact probabilities depend on the model that serves the request:

    ```json theme={null}
    {
      "model": "pplx-decider-latest",
      "answers": {
        "defect": {
          "type": "noul",
          "noul": 0.9424522889347015
        },
        "sentiment": {
          "type": "choice",
          "choice": "mixed",
          "confidence": 0.9255246944002182,
          "probabilities": {
            "positive": 0.020649883775315993,
            "mixed": 0.9503497962668123,
            "negative": 0.02900031995787183
          }
        },
        "severity": {
          "type": "score",
          "score": 1.7838686319784252,
          "confidence": 0.7838686319784252,
          "legend": {
            "0": "Cosmetic",
            "1": "Inconvenient",
            "2": "Product unusable"
          },
          "probabilities": {
            "0": 0.008423954913615923,
            "1": 0.199283458194343,
            "2": 0.7922925868920411
          }
        }
      },
      "usage": {
        "input_tokens": 367,
        "output_tokens": 3
      }
    }
    ```

    Read it like this:

    * **`defect`**: a 94% probability that the review reports a defect. Compare `noul` against a threshold you choose; values near 0.5 mean the model is unsure.
    * **`sentiment`**: `mixed` is the option with the highest probability (95%). `probabilities` covers every option you defined and sums to about 1, so you can see how close the runner-up came.
    * **`severity`**: `score` is the probability-weighted average of the level indices, so 1.78 sits between `Inconvenient` (1) and `Product unusable` (2), closer to 2. `legend` maps each index back to your rubric, and `probabilities` shows the full distribution over levels.

    `confidence` on `choice` and `score` answers is the model's own certainty estimate, from 0 to 1. It is not the top probability: in the example above, `sentiment` has a top probability of 0.95 and a `confidence` of 0.93. It drops when the runner-up is close.

    In our tests on earlier decision models, identical requests usually returned identical numbers. Occasionally they differed in the second decimal place, so set thresholds with some margin. `model` echoes the model name you sent.
  </Step>
</Steps>

## Question types

Every question in a request refers to the same `state`, which can be a string, an object, or an array. You name each question, and the response uses the same names. Each question has a `type`, `instructions` (what to decide), and, depending on the type, `criteria`.

### `noul`: yes or no

Ask a question or state something to check. Give `instructions`, `criteria`, or both; `criteria` defines what counts as yes and what counts as no. A `noul` with neither returns `400`.

```json theme={null}
{
  "type": "noul",
  "instructions": "Does the review report a product defect?",
  "criteria": {
    "true": "Something is broken or not working.",
    "false": "Normal wear or personal preference."
  }
}
```

The answer is `noul`, the probability of yes or true, from 0 to 1.

### `choice`: one of your options

`criteria` maps each option name to a description of when it applies. Use `null` as the description to let the name speak for itself. A question accepts 1 to 255 options.

```json theme={null}
{
  "type": "choice",
  "instructions": "Which team should handle this ticket?",
  "criteria": {
    "billing": "Charges, refunds, and invoices",
    "shipping": "Delivery status and lost packages",
    "other": null
  }
}
```

The answer has `choice` (the option with the highest probability), `probabilities` (one value per option, summing to about 1), and `confidence`.

### `score`: a level on an ordered rubric

`criteria` is an ordered array of level descriptions. The index in the array is the level's score, starting at 0. A question accepts up to 10 levels. Use at least two: with a single level there is nothing to decide, so the answer is always a score of 0 with probability 1.

```json theme={null}
{
  "type": "score",
  "instructions": "How severe is the reported problem?",
  "criteria": ["Cosmetic", "Inconvenient", "Product unusable"]
}
```

The answer has `score` (the probability-weighted average of the level indices, which can fall between two levels), `legend` (each index, as a string, mapped back to your rubric entry), `probabilities` (one value per level, keyed like `legend`), and `confidence`.

### Images in `state`

`state` can contain images. Write each image as an OpenAI-style part, `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`, anywhere in `state`: as an array item or as a value inside an object, at any depth. The model reads `state` as serialized JSON, with each image placed exactly where its part appears, so the order you write things in is the order the model sees them.

In an array, next to text:

```json theme={null}
{
  "model": "pplx-decider-latest",
  "state": [
    "Which color is the square?",
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
  ],
  "questions": {
    "color": {"type": "choice", "instructions": "What color is the square?", "criteria": {"red": null, "blue": null, "green": null}}
  }
}
```

As a value inside an object:

```json theme={null}
"state": {
  "caption": "Product photo from the listing",
  "photo": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
}
```

If the content is just an image, pass the image part itself as `state`. You don't need to add an empty text field:

```json theme={null}
"state": {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
```

* PNG, JPEG, and WebP data URLs are accepted. The API never fetches a URL: an `http` or `https` image URL returns `400`.
* Up to 8 images per question. More returns `400`.
* The API reads images in 32 × 32 pixel tiles. Keep each image at or under 2,048 tiles: round the width and the height to the nearest multiple of 32 and keep (width / 32) × (height / 32) at or under 2,048. 1440 × 1440 and 2048 × 1024 fit; 1600 × 1310 does not. A larger image does not return `400`: the request waits about a minute and then returns `504`. Resize before you send.
* Image tokens count toward `usage.input_tokens` and the input limit like text. In our tests on earlier decision models, an image cost about 1,000 input tokens per megapixel.

### Request limits

| Limit | Value |
| - | - |
| Questions per request | 1 to 128, each with a non-empty name |
| Options per `choice` | 1 to 255 |
| Levels per `score` | 1 to 10 |
| Input tokens per request | Under 256K, counting `state`, images, and every question |
| Request body | 32 MiB |
| Images per question | Up to 8 |
| Image size | 2,048 tiles of 32 × 32 pixels per image, for example 1440 × 1440 or 2048 × 1024 |

`state` must be a string, an object, or an array; `null` returns `400`. `score` levels follow the same rule. `choice` descriptions can also be `null`. An unknown top-level field returns `400`.

## Models

The Decisions API has two decision models. They take the same request and return the same response, so switching is a one-line change to `model`.

| | `pplx-decider-v1.2` | `pplx-decider-flash-v1.2` |
| - | - | - |
| **Alias** | `pplx-decider-latest` | `pplx-decider-flash-latest` |
| **Parameters** | 27B | 4B |
| **Input price** | \$0.02 per 1M tokens | \$0.01 per 1M tokens |
| **Output tokens** | Free | Free |
| **Input** | Text and images | Text and images |
| **Context window** | 256K tokens | 256K tokens |
| **Options per `choice` / levels per `score`** | 255 / 10 | 255 / 10 |

The two models differ in size and price. Everything else in [Request limits](#request-limits) applies to both.

### When to use each

* **`pplx-decider-v1.2`:** the larger model and the default. Start here, especially for nuanced judgments, such as grading against a detailed rubric or checking a claim against long source text, or where a wrong answer is costly.
* **`pplx-decider-flash-v1.2`:** consider it for high-volume decisions with clear criteria, such as routing, labeling, or filtering, where half the input price matters.

To decide, run both models on a sample of your own labeled data and compare the results at the thresholds you plan to use. Keep Flash only if it meets your accuracy bar on that sample.

### Grouping questions in one request

On `pplx-decider-v1.2` and `pplx-decider-flash-v1.2`, all questions in a request are answered together in one pass. Each question sees the `state` and every other question in the request. The `state` is read and billed once, so one request is cheaper than sending the questions separately. Grouping also changes the answers, which can improve or hurt results:

* **Grouping helps when the questions are facets of one decision**, such as several attributes of the same item, multi-label "select all that apply" decisions, or the steps of one routing decision. It can also help when comparing candidates is informative, for example multi-hop evidence or duplicate detection.
* **Grouping can hurt when the questions are independent candidates competing for the same judgement**, such as one question per document, tool, or function call. Score those in separate requests.
* **The two models behave differently.** `pplx-decider-v1.2` benefits from grouping related questions. If you're unsure, send one candidate per request, since a wrong grouping costs more than a missed one. `pplx-decider-flash-v1.2` is more sensitive to grouping, so send one question per request by default, group only "select all that apply" or routing-step decisions, and check on your own task.
* The same question can score differently depending on what it's grouped with. Keep the grouping fixed when you compare runs or tune thresholds.

If the `state` plus all questions goes over the input limit, the API splits the questions into consecutive groups, and each group counts the `state` again.

### Model names

Set `model` on every request, using either the versioned name or the alias. An alias follows the newest version of its model: `pplx-decider-latest` and `pplx-decider-flash-latest` can move to a newer decision model over time. When a version is retired, its name routes to the version that replaces it, and the [changelog](/docs/resources/changelog) announces the change. If your application compares probabilities against thresholds, revalidate those thresholds when the model changes. `perplexity/pplx-decider-latest` and `perplexity/pplx-decider-flash-latest` are also accepted and select the same models as their unprefixed aliases.

The response `model` field echoes the name you sent, so a request for `pplx-decider-latest` reports `pplx-decider-latest`, not the version that served it.

Existing requests that name `pplx-decider-v1.1-27b` or `pplx-decider-v1-27b` continue to work. They select `pplx-decider-v1.2` and bill at its rate.

A missing or unknown model returns `400`:

```json theme={null}
{"error": {"code": null, "message": "Invalid model 'pplx-decidr-latest'. Permitted models can be found in the documentation at https://docs.perplexity.ai/docs/getting-started/models.", "param": null, "type": "invalid_request_error"}}
```

## Endpoint and authentication

| Operation | Endpoint |
| - | - |
| **Answer questions** | `POST https://api.perplexity.ai/v1/decisions` |

Send the key as `Authorization: Bearer <PERPLEXITY_API_KEY>` and the body as JSON. A key in an `x-api-key` header is not read, so the request returns `401`. Another method on the endpoint returns `405` with `Allow: POST`, and any other path, including a trailing slash, returns `404`.

## Request ids

Responses carry an `x-request-id` header with a UUID, on success and on most errors. Log it with your results and quote it in support requests. A `401`, a `404`, and a `504` carry no request id.

## Rate limits

Every organization can send 10 requests per second to the Decisions API, on every plan. The limit is shared across `pplx-decider-v1.2` and `pplx-decider-flash-v1.2`. A token limit also applies to large bursts.

Successful responses carry `x-ratelimit-limit`, `x-ratelimit-remaining`, `x-ratelimit-used`, and `x-ratelimit-reset` (Unix seconds). A request over a limit returns `429` with a `Retry-After` header in seconds. Wait that long, then retry.

## Errors

Most errors return a JSON body with an `error` object. Read `error.message` for the reason and `error.type` for the category; don't branch on `error.code`, which is a string, a number, or `null` depending on the error. A `404` or `405` has an empty body, and a `504` can return an HTML page, so check the status before you parse the body.

```json theme={null}
{"error": {"message": "Noul question must have criteria or instructions", "type": "invalid_request", "code": "400"}}
```

| Status | Why | What to do |
| - | - | - |
| **`400`** | The body is not valid JSON, `model` is missing or unknown, a field is unknown or the wrong type, a limit in [Request limits](#request-limits) is exceeded, or an image is not a supported data URL | Fix the request; `error.message` names the problem |
| **`401`** | The API key is missing or invalid, or it was sent in `x-api-key` | Send `Authorization: Bearer <PERPLEXITY_API_KEY>` and check that the key is active |
| **`404`** | Wrong path | Use `POST https://api.perplexity.ai/v1/decisions`, with no trailing slash |
| **`405`** | Wrong method | Use `POST` |
| **`413`** | The request body is over 32 MiB | Send less `state`, or split the content across requests |
| **`429`** | Over the request or token limit; see [Rate limits](#rate-limits) | Wait `Retry-After` seconds, then retry |
| **`5xx`** | The model did not answer in time (`504`, after about a minute) or the service failed | Retry with backoff; for large inputs, see [Timeouts](#timeouts) |

## Timeouts

Response time grows with input size. In our tests on September 30, 2026, on an earlier decision model, a request with a few hundred input tokens answered in under 2 seconds, about 90,000 tokens took 5 seconds, about 190,000 tokens took 14 seconds, and just under the input limit took 23 seconds. If the model does not answer in time, the request returns `504`; in our tests that took about a minute.

Set your client timeout to fit the input. The examples on this page use 30 seconds, which covered any request under the input limit on that earlier model; for small inputs, 10 seconds was plenty.

## Pricing

The Decisions API costs \$0.02 per million input tokens on `pplx-decider-v1.2` and \$0.01 per million input tokens on `pplx-decider-flash-v1.2`. Aliases bill at the rate of the model they select. Output tokens are free, and there is no per-request fee. Input tokens are the `usage.input_tokens` value in each response, so you can compute the cost of a request from the response you already receive. Usage is billed to the organization that owns the API key, like every other Perplexity API.

## Next steps

<CardGroup cols={2}>
  <Card title="Answer questions reference" icon="code" href="/api-reference/decisions-post">
    Full request and response schema for `POST /v1/decisions`.
  </Card>

  <Card title="Cookbook: triage support tickets" icon="book-open" href="/docs/cookbook/examples/decisions-api-ticket-triage/README">
    Route twelve tickets with three questions per ticket, then send only the escalations to the Agent API.
  </Card>

  <Card title="Agent API" icon="brain" href="/docs/agent-api/quickstart">
    Web-grounded answers with citations, tools, and structured output.
  </Card>

  <Card title="Orchestrator API" icon="arrows-shuffle" href="/docs/router/quickstart">
    Direct access to open-weight models through OpenAI- and Anthropic-compatible endpoints.
  </Card>
</CardGroup>

Need help? Check out our [community](https://community.perplexity.ai) for support.

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bespokelabs.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Limits and errors

> Request limits, the errors Nimble returns, and when to retry.

## Limits

These limits apply to every model.

| Limit | Value |
| - | - |
| Request body | 1 MiB |
| Requests that one organization can run at the same time | 4 until your organization first buys credit, then 100 |
| Questions in a request | 1 to 64 |
| Options in a `choice` question, or levels in a `score` question | 2 to 255 |
| Prompt tokens for one question, including the state | 256k |

Nimble never cuts a prompt. A request above a limit gets an error.

The limit on concurrent requests is shared by all of your organization's API keys and models. Your organization's first credit purchase in the [console](https://console.bespokelabs.ai) raises it from 4 to 100, and it stays at 100 after that credit is spent. [Free credit](/nimble/pricing#free-credit), such as the \$1 sign-up credit, does not count as a purchase.

## Errors

| Status | Meaning | What to do |
| - | - | - |
| 401 | The API key is missing or not valid. | Check the key and the `Authorization` header. |
| 402 | Your organization does not have enough credit for the request. | Buy credit in the [console](https://console.bespokelabs.ai). |
| 413 | The request body is larger than 1 MiB. | Send a smaller state. |
| 422 | The request is not valid, or it is above a limit. The `detail` field says why. | Fix the request. Do not retry it unchanged. |
| 429 | Your organization has reached its limit on concurrent requests. | Retry after the number of seconds in the `Retry-After` header. If your limit is 4, buying credit in the [console](https://console.bespokelabs.ai) raises it to 100. |
| 502 | A model returned an error. | Retry with backoff. |
| 503 | A model is starting, or Nimble is not available for a moment. | Retry after the number of seconds in the `Retry-After` header. Without the header, retry with backoff. |
| 504 | The request took too long. | Send fewer questions or a shorter state. |
| 529 | The model is busy. | Retry after about one second, with backoff. |

You are not charged for a request that gets an error.

A `422` with the detail "Nimble request failed" comes from the model itself. One cause is a prompt that is longer than the model allows. Another cause is a model name that Nimble does not know. These are the model names:

* `nimble-latest`, which currently points to `nimble-v4-max`
* `nimble-v4-max`
* `nimble-v4` (deprecated, points to `nimble-v4-max`)
* `nimble-v3` (deprecated)

## Request IDs

Every response has an `x-request-id` header, including an error response. The `x-typesafe-request-id` header has the same ID. Error responses also have it in the `request_id` field. Send this ID to Bespoke Labs when you report a problem with a request.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.
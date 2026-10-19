> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bespokelabs.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Pricing

> What each Nimble model costs, how Nimble counts tokens, and how to get free credit.

Nimble charges for input tokens only. Output tokens are free. You pay from prepaid credit, which you buy in the [console](https://console.bespokelabs.ai). New accounts also get [free credit](#free-credit).

| Model | Price per million input tokens |
| - | - |
| `nimble-v4-max`, `nimble-latest` | \$0.04 |

`nimble-v4` is deprecated. It points to `nimble-v4-max`, so it costs the same. `nimble-v3` is deprecated.

The `usage.input_tokens` field of each response is the number of input tokens that you pay for.

## How Nimble counts input tokens

Nimble builds one prompt for each question. Every prompt starts with the same shared text, which is the state and the questions. You pay for the shared text once per request, not once per question.

## Credit holds

When a request starts, Nimble holds back from your credit the most that the request could cost. When the request ends, you pay for the tokens that the request used, and Nimble releases the rest of the hold. A request that gets an error costs nothing.

If your available credit is less than the hold, the request gets `402`.

## Free credit

A new account gets \$1 of credit when you verify its email address. The credit goes to the account's organization, so a teammate who joins your organization adds \$1 to it. Each email address gets the credit once, and variants of one address, e.g., `name+tag@example.com`, count as the same address.

### Refer a friend

You can earn more credit by referring friends. In the console, choose "Refer a friend" in the sidebar. Send your friend an invitation by email, or share your sign-up link with them. When your friend creates an account from the invitation or the link and verifies their email address, your organization gets \$3 of credit. Your friend gets the \$1 that every new account gets.

* Only new accounts earn a reward.
* Teammates that you add to your organization, and people with your company's email domain, do not earn rewards. Gmail and other public email addresses are fine.
* Your organization can earn up to 10 rewards in each calendar month, in UTC. A friend who joins after that still gets their \$1.
* Your friend does not need to use Nimble for you to get the reward.
* Free credit pays for Nimble usage only. A reward goes to the organization that you referred from, and all of its API keys can use it.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.
"""Editable demo presets; no inference logic."""


def examples():
    return [
        {
            "title": "Choosing a mooncake",
            "state": "Which mooncake should I eat for the Mid-Autumn Festival? I enjoy sweet and savory flavors, especially salted egg yolk, but I dislike nuts.",
            "questions": {
                "mooncake": {
                    "type": "choice",
                    "instructions": "Recommend the mooncake that best matches these preferences.",
                    "criteria": {
                        "Lotus paste with salted egg yolk": "Sweet lotus paste with savory salted egg yolk.",
                        "Mixed nuts": "A traditional filling with a mix of nuts and seeds.",
                        "Red bean paste": "Smooth, sweet red bean paste.",
                        "Pork": "Flaky pastry with a savory pork filling.",
                    },
                }
            },
        },
        {
            "title": "Support ticket · Multiple decisions",
            "state": "Customer message: I bought headphones yesterday and noticed today that I was charged twice. Please refund the extra charge as soon as possible.",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {"billing": "Payments and refunds", "delivery": "Shipping and delivery", "technical": "Product technical support"},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Choose the priority for handling this ticket.",
                    "criteria": ["Low: general inquiry", "Medium: an issue requiring follow-up", "High: a financial issue such as a duplicate charge"],
                },
                "refund_requested": {"type": "noul", "instructions": "Does the customer explicitly request a refund?"},
            },
        },
        {
            "title": "Refund policy · Yes or no",
            "state": "Policy: unopened items may be refunded within 7 days of receipt. A customer received an item 3 days ago. It is still unopened, and the customer is requesting a refund.",
            "questions": {"eligible": {"type": "noul", "instructions": "Is the customer eligible for a refund under this policy?"}},
        },
        {
            "title": "Review sentiment · Rating",
            "state": "The headphones sound wonderful, the battery lasts all week, and delivery was fast. I love them!",
            "questions": {
                "sentiment": {
                    "type": "score",
                    "instructions": "Rate the sentiment of this review.",
                    "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"],
                }
            },
        },
    ]

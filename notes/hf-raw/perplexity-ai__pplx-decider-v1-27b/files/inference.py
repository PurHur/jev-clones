# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0",
#   "accelerate==1.15.0", "huggingface-hub==1.31.0",
#   "pillow==12.3.0", "safetensors==0.8.0",
# ]
# ///
"""Download pplx-decider from Hugging Face and run text or image decisions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Self

from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from autojev.model import DecisionModel
    from autojev.types import Answer, DecisionInput, Question

MODEL_ID = "perplexity-ai/pplx-decider-v1-27b"
FILES = [
    "model-*.safetensors", "model.safetensors.index.json", "readout.safetensors",
    "config.json", "decision_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "source/src/autojev/__init__.py", "source/src/autojev/model.py", "source/src/autojev/types.py",
]


class Decider:
    def __init__(self, model: DecisionModel) -> None:
        self.model = model

    @classmethod
    def from_pretrained(
        cls, model_id: str = MODEL_ID, *, revision: str = "main", device: str = "cuda",
    ) -> Self:
        """Load a Hub repository or a downloaded repository directory."""
        checkpoint = Path(model_id)
        if not checkpoint.is_dir():
            checkpoint = Path(snapshot_download(model_id, revision=revision, allow_patterns=FILES))
        sys.path.insert(0, str(checkpoint / "source" / "src"))
        from autojev.model import DecisionModel

        return cls(DecisionModel(checkpoint=checkpoint, device=device))

    def predict(
        self, state: str, question: Question, *, images: Sequence[str | Path] = (),
    ) -> Answer:
        """Return a choice, yes/no probability, or score using the saved temperature."""
        from autojev.model import answer

        row: DecisionInput = {"state": state, "question": question}
        if images:
            row["images"] = list(images)
        probabilities = self.model.predict([row], batch_size=1)[0]
        return answer(question, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hub model ID or downloaded repository directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image", type=Path, help="Run the image example with a local image")
    args = parser.parse_args()
    if args.image is not None and not args.image.is_file():
        parser.error("--image must point to an existing image file")
    model = Decider.from_pretrained(args.model, revision=args.revision, device=args.device)
    if args.image is not None:
        result = model.predict("Look at the supplied image.", {
            "type": "choice", "instructions": "What is the dominant color?",
            "criteria": {"red": "Red", "green": "Green", "blue": "Blue", "other": "Another color"},
        }, images=[args.image])
        print(json.dumps(result, indent=2))
        return
    state = "My Stripe integration keeps failing. I'm losing sales. Please help ASAP."
    results = {
        "urgency": model.predict(state, {"type": "noul", "instructions": "Does this message express urgency?"}),
        "routing": model.predict(state, {
            "type": "choice", "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Charges and refunds", "technical_support": "Integration errors",
                         "sales": "Questions about buying a product"},
        }),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

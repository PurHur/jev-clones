"""Request formatting, tokenization and padded batches; no dataset loading."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from .types import Result

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .types import Request


@dataclass
class TokenizedSample:
    ids: list[int]
    message_boundaries: list[int]
    roles: list[str]

    def __len__(self):
        return len(self.ids)

    def pad(self, length, token):
        return replace(self, ids=self.ids + [token] * length)


class Processor:
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", config):
        self.tokenizer = tokenizer
        self.max_length = config.backbone_config.max_position_embeddings
        self.role_start = config.role_start
        self.role_end = config.role_end
        for token in (self.role_start, self.role_end):
            assert token in tokenizer.all_special_tokens
            assert tokenizer.encode(token, add_special_tokens=False) == [
                tokenizer.convert_tokens_to_ids(token)
            ]
        self.end_id = cast("int", tokenizer.convert_tokens_to_ids(self.role_end))

    def encode(self, requests: "Request | list[Request]") -> "Encoded":
        requests = requests if isinstance(requests, list) else [requests]
        samples = [self.encode_request(request) for request in requests]
        for sample in samples:
            if len(sample) > self.max_length:
                message = f"Request has {len(sample)} tokens; model capacity is {self.max_length}."
                raise ValueError(message)
        length = max(map(len, samples))
        batch = Batch.from_samples(
            [sample.pad(length - len(sample), self.end_id) for sample in samples]
        )
        return Encoded(requests=requests, batch=batch)

    @staticmethod
    def decode(scores: torch.Tensor, encoded: "Encoded") -> list[list[Result]]:
        rows = iter(scores)
        results = []
        for request in encoded.requests:
            answers = []
            for query in request.queries:
                probabilities = next(rows)[: len(query.options)].float().softmax(0).cpu()
                answers.append(
                    Result(
                        choice=query.options[int(probabilities.argmax())],
                        probabilities=dict(zip(query.options, probabilities.tolist(), strict=True)),
                    )
                )
            results.append(answers)
        return results

    def encode_request(self, request: "Request") -> TokenizedSample:
        messages = [{"role": "state", "content": request.state}]
        for query in request.queries:
            messages.append({"role": "query", "content": query.text})
            messages.extend(
                {"role": "example", "content": example.model_dump()} for example in query.examples
            )
            messages.extend({"role": "option", "content": option} for option in query.options)
        ids = cast(
            "list[int]",
            self.tokenizer.apply_chat_template(
                messages,
                role_start=self.role_start,
                role_end=self.role_end,
                option_codes=None,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            ),
        )
        roles = [message["role"] for message in messages]
        closings = iter(i for i, token in enumerate(ids) if token == self.end_id)
        boundaries = []
        for role in roles:
            if role == "example":
                next(closings)  # Inner state.
                next(closings)  # Inner option.
            boundaries.append(next(closings))
        return TokenizedSample(ids, boundaries, roles)


@dataclass
class Batch:
    input_ids: torch.Tensor
    message_boundaries: list[list[int]]
    roles: list[list[str]]

    @classmethod
    def from_samples(cls, samples: list[TokenizedSample]):
        return cls(
            input_ids=torch.tensor([sample.ids for sample in samples], dtype=torch.long),
            message_boundaries=[sample.message_boundaries for sample in samples],
            roles=[sample.roles for sample in samples],
        )

    def to(self, device):
        return replace(self, input_ids=self.input_ids.to(device))

    @property
    def option_positions(self):
        return torch.tensor(
            [
                (index, end)
                for index, (boundaries, roles) in enumerate(
                    zip(self.message_boundaries, self.roles, strict=True)
                )
                for end, role in zip(boundaries, roles, strict=True)
                if role == "option"
            ],
            dtype=torch.long,
            device=self.input_ids.device,
        )

    @property
    def option_indices(self):
        groups: list[list[int]] = []
        index = 0
        for roles in self.roles:
            for role in roles:
                if role == "query":
                    groups.append([])
                elif role == "option":
                    groups[-1].append(index)
                    index += 1
        width = max(map(len, groups))
        return torch.tensor(
            [group + [-1] * (width - len(group)) for group in groups],
            dtype=torch.long,
            device=self.input_ids.device,
        )


@dataclass
class Encoded:
    requests: list["Request"]
    batch: Batch

"""Linear and MLP decision heads."""

from dataclasses import dataclass
from itertools import pairwise

from torch import nn


@dataclass
class DecisionHeadConfig:
    type: str = "linear"
    dims: list[int] | None = None
    activation: str = "gelu"
    layernorm: bool = False

    def __post_init__(self):
        defaults = {"linear": [1], "mlp": [512, 1]}
        if self.type not in defaults:
            message = "SocialJev supports linear and mlp decision heads"
            raise ValueError(message)
        if self.dims is None:
            self.dims = defaults[self.type].copy()
        assert self.dims
        assert all(width > 0 for width in self.dims)
        assert self.activation in {"gelu", "relu", "silu"}
        assert (
            self.dims == [1] if self.type == "linear" else len(self.dims) > 1 and self.dims[-1] == 1
        )

    def layers(self, input_dim: int) -> nn.Sequential:
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[self.activation]
        assert self.dims is not None
        widths = [input_dim, *self.dims]
        layers = []
        for index, (before, after) in enumerate(pairwise(widths)):
            layers.append(nn.Linear(before, after))
            if index < len(widths) - 2:
                if self.layernorm and index > 0:
                    layers.append(nn.LayerNorm(after))
                layers.append(activation())
        return nn.Sequential(*layers)


class DecisionHead(nn.Module):
    def __init__(self, config: DecisionHeadConfig, input_dim: int):
        super().__init__()
        self.network = config.layers(input_dim)

    def forward(self, options):
        return self.network(options).squeeze(-1)

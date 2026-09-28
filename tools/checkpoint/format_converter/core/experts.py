# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The routed experts of mixture-of-experts layers and how each format stores them.

A mixture-of-experts layer's experts module (``<...>.mlp.experts``) holds one
``linear_fc1`` and one ``linear_fc2`` weight (and bias) per routed expert.
torch_dist stores each as one tensor over all experts,
``<...>.mlp.experts.experts.linear_fc<n>.<leaf>`` of shape ``[E, ...]`` (or
``[L, E, ...]`` in a block stacked over its layers), expert ``e`` at the
global index ``e`` of its expert axis and saved as a chunk of its own
(``TEGroupedMLP`` and ``SequentialMLP`` write the same keys,
``moe/experts.py``). :class:`ExpertLayout` records what placing one expert's
tensor needs beyond its logical id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import InternalError


@dataclass(frozen=True, slots=True)
class ExpertLayout:
    """How the routed experts of the model's mixture-of-experts layers are stored.

    Attributes:
        num_experts: Routed experts per layer: the length of torch_dist's expert axis.

    Raises:
        InternalError: ``num_experts`` is not positive.
    """

    num_experts: int

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise InternalError(f"an expert layout needs experts, not {self.num_experts}")

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {"num_experts": self.num_experts}

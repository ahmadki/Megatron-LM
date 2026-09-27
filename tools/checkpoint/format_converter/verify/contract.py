# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Expected-difference contracts: what may differ between a converted store and a reference.

``verify A B --expected-diff X`` compares the converted store ``A`` with a
reference ``B`` under contract ``X``. A difference the contract allows is
reported as expected, not as a finding; anything else is a finding (exit 5).

* ``A``: torch_dist -> fsdp_dtensor vs a real fsdp_dtensor save. Dropped
  state may be absent in A; A's narrower floats must equal B's after the exact
  upcast; any chunk grid.
* ``B``: fsdp_dtensor -> torch_dist vs a real torch_dist save. Dropped state
  (extra state included) may be absent in A; A's wider floats must round (RNE)
  to B's; any chunk grid.
* ``C``: two model-only saves of one chain step in the same format (real vs
  converted, or real vs real). Floats must be equal after the exact upcast;
  any chunk grid; the sanitized argument fields may differ.
* ``D``: torch_dist -> fsdp_dtensor -> torch_dist vs its model-only source.
  Only ``no_save_optim`` and ``no_save_rng`` may differ (a weights-only store
  sets them), and the source's rerun state may be absent in A (Megatron's
  default ``--rerun-mode validate_results`` writes it into model-only saves
  too); A's chunk grid must be a coarsening of B's (tensor-parallel splits
  merged) that keeps B's boundaries along the layer axis of a tensor stacked
  over layers (one chunk per layer).
* ``E``: fsdp_dtensor -> torch_dist -> fsdp_dtensor vs its source. Dropped
  state may be absent in A; any chunk grid; the sanitized argument fields may
  differ.

Values, dtypes (beyond the cast a contract names) and every other key are
never allowed to differ. The contract sets the common-state comparison,
whatever ``--common`` says: field by field (``--common semantic``) for ``C``,
``D`` and ``E``, because a re-encoded common state is never byte-identical to
Megatron's, and skipped for ``A`` and ``B``, whose references come from other
training jobs. State a contract lets A lack may also be a common-state key
only B holds (``rerun_state_machine``). ``C`` and ``D`` expect model-only
references (saved with ``--no-save-optim --no-save-rng
--ckpt-drop-redundant-extra-state``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.format_args import SANITIZED_FIELDS, WEIGHTS_ONLY_FIELDS
from ..core.keys import EXTRA_STATE

EXACT, UPCAST, DOWNCAST = "exact", "upcast", "downcast"
"""Float casts a contract allows: none, narrower-to-wider exactly, wider-to-narrower (RNE)."""

ANY_GRID, COARSENING = "any", "coarsening"
"""Chunk-grid rules: anything, or each of A's chunks exactly tiled by B's chunks (never merged
along the layer axis of a tensor stacked over layers)."""

RERUN_STATE = re.compile(r"^rerun_state_machine")
"""Rerun state-machine state: the ``rerun_state_machine_state/shard_*`` items and the
``rerun_state_machine`` common-state key."""

DROPPED_STATE = (
    *(
        re.compile(pattern)
        for pattern in (r"^(chained_\d+\.)?optimizer\.", r"^opt_param_scheduler", r"^rng_state")
    ),
    RERUN_STATE,
    EXTRA_STATE,
)
"""Keys of the state a weights-only conversion drops."""


@dataclass(frozen=True)
class Contract:
    """The differences allowed between a converted store (A) and a reference (B).

    Attributes:
        name: The contract id.
        scenario: What A and B are.
        reference_only: Keys B may hold that A lacks.
        casts: The float cast under which A's and B's values must agree when
            their dtypes differ (``exact``: dtypes must match).
        grid: The chunk-grid rule.
        common: ``semantic`` (compare the common state field by field) or ``skip``.
        args: Argument fields allowed to differ.
    """

    name: str
    scenario: str
    reference_only: tuple[re.Pattern[str], ...]
    casts: str
    grid: str
    common: str
    args: frozenset[str]

    def allows_missing(self, key: str) -> bool:
        """Whether B may hold ``key`` although A does not."""
        return any(pattern.search(key) for pattern in self.reference_only)

    def allows_common(self, name: str, *, only_in_reference: bool = False) -> bool:
        """Whether the common-state field ``name`` (``args.<field>`` or a key) may differ.

        Args:
            name: The field.
            only_in_reference: Only B holds it (A lacks it): state A may lack
                (:meth:`allows_missing`) is then allowed too.
        """
        if only_in_reference and self.allows_missing(name):
            return True
        return name.startswith("args.") and name[len("args.") :] in self.args


CONTRACTS = {
    contract.name: contract
    for contract in (
        Contract(
            "A",
            "torch_dist -> fsdp_dtensor vs a real fsdp_dtensor save",
            DROPPED_STATE,
            UPCAST,
            ANY_GRID,
            "skip",
            frozenset(),
        ),
        Contract(
            "B",
            "fsdp_dtensor -> torch_dist vs a real torch_dist save",
            DROPPED_STATE,
            DOWNCAST,
            ANY_GRID,
            "skip",
            frozenset(),
        ),
        Contract(
            "C",
            "two model-only saves of one chain step in the same format",
            (),
            UPCAST,
            ANY_GRID,
            "semantic",
            SANITIZED_FIELDS,
        ),
        Contract(
            "D",
            "torch_dist -> fsdp_dtensor -> torch_dist vs the model-only source",
            (RERUN_STATE,),
            EXACT,
            COARSENING,
            "semantic",
            frozenset(WEIGHTS_ONLY_FIELDS),
        ),
        Contract(
            "E",
            "fsdp_dtensor -> torch_dist -> fsdp_dtensor vs the source",
            DROPPED_STATE,
            EXACT,
            ANY_GRID,
            "semantic",
            SANITIZED_FIELDS,
        ),
    )
}
"""The contracts by id."""

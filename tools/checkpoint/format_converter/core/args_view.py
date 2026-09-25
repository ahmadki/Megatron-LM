# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Read-only access to a checkpoint's training arguments.

Conversion decisions may depend on the arguments a checkpoint was saved with.
:class:`ArgsView` exposes only primitive values (``None``, booleans, numbers,
strings and tuples or lists of them), so a decision never depends on an object
whose class may be missing on another host, and it records every field that was
read, so reports can say which arguments a conversion consulted.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..errors import IntegrityError, InternalError

_PRIMITIVES = (type(None), bool, int, float, str)
_MISSING = object()


class ArgsView:
    """Primitive-only, recording view of an ``argparse.Namespace`` of training arguments.

    Args:
        args: The checkpoint's arguments.

    Raises:
        IntegrityError: ``args`` is not a Namespace.
    """

    def __init__(self, args: Any) -> None:
        if not isinstance(args, argparse.Namespace):
            raise IntegrityError(
                f"the checkpoint's args are a {type(args).__name__}, not a Namespace"
            )
        self._fields = dict(vars(args))
        self._consumed: dict[str, Any] = {}

    def has(self, name: str) -> bool:
        """Whether the arguments define ``name``."""
        return name in self._fields

    def get(self, name: str, default: Any = _MISSING) -> Any:
        """The primitive value of ``name`` (``default`` if it is not defined).

        Raises:
            InternalError: ``name`` is not defined and there is no default, or
                its value is not primitive.
        """
        if name not in self._fields:
            if default is _MISSING:
                raise InternalError(f"args has no field {name!r}")
            return default
        value = self._fields[name]
        if not _is_primitive(value):
            raise InternalError(f"args.{name} is a {type(value).__name__}, not a primitive value")
        self._consumed[name] = value
        return value

    @property
    def consumed(self) -> dict[str, Any]:
        """The fields read so far, with their values (sorted by name)."""
        return dict(sorted(self._consumed.items()))


def _is_primitive(value: Any) -> bool:
    if isinstance(value, (tuple, list)):
        return all(_is_primitive(item) for item in value)
    return isinstance(value, _PRIMITIVES)

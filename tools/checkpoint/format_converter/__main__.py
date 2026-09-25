# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Entry point of ``python -m tools.checkpoint.format_converter``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

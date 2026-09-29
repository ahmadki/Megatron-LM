# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer 2: conversion rules, one module per family of tensors.

Importing this package registers every rule with ``core.registry.REGISTRY``.
"""

from __future__ import annotations

from . import glu, identity

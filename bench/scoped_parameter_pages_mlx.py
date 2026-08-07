#!/usr/bin/env python3
# type: ignore
"""MLX attachment seam for explicitly scoped neural parameter pages.

The semantics are defined in ``NEURAL_PARAMETER_PAGES.md``.  This module only
implements the numerical mount: policy and Tensor Logic produce the activation
proof before this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.tuner.lora import LoRALinear


class PageActivationError(ValueError):
    """A numerical page was mounted without a valid activation token."""


class ScopedLoRALinear(nn.Module):
    """A LoRA projection whose inactive branch is the untouched base call."""

    def __init__(self, loaded: LoRALinear) -> None:
        super().__init__()
        self.linear = loaded.linear
        self.dropout = loaded.dropout
        self.scale = loaded.scale
        self.lora_a = loaded.lora_a
        self.lora_b = loaded.lora_b
        self._activation_proof_id: str | None = None

    @property
    def activation_proof_id(self) -> str | None:
        return self._activation_proof_id

    def activate(self, proof_id: str) -> None:
        if not proof_id:
            raise PageActivationError("activation requires a non-empty proof identity")
        if self._activation_proof_id not in (None, proof_id):
            raise PageActivationError("a different proof is already active")
        self._activation_proof_id = proof_id

    def deactivate(self) -> None:
        self._activation_proof_id = None

    def __call__(self, inputs: mx.array) -> mx.array:
        base = self.linear(inputs)
        if self._activation_proof_id is None:
            return base
        residual = (self.dropout(inputs) @ self.lora_a) @ self.lora_b
        return base + (self.scale * residual).astype(inputs.dtype)


@dataclass(frozen=True, slots=True)
class MountedParameterPage:
    """All projection sites belonging to one already-verified page."""

    page_id: str
    sites: tuple[tuple[str, ScopedLoRALinear], ...]

    def activate(self, proof_id: str) -> None:
        activated: list[ScopedLoRALinear] = []
        try:
            for _, site in self.sites:
                site.activate(proof_id)
                activated.append(site)
        except Exception:
            for site in activated:
                site.deactivate()
            raise

    def deactivate(self) -> None:
        for _, site in self.sites:
            site.deactivate()


def scope_loaded_lora(model: nn.Module, *, page_id: str) -> MountedParameterPage:
    """Replace already-loaded LoRA sites with proof-gated equivalents."""

    if not page_id:
        raise ValueError("page identity must not be empty")
    replacements: list[tuple[str, ScopedLoRALinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            replacements.append((name, ScopedLoRALinear(module)))
    if not replacements:
        raise ValueError("model contains no loaded LoRA projections")
    model.update_modules(tree_unflatten(replacements))
    return MountedParameterPage(page_id=page_id, sites=tuple(replacements))

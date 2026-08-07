"""Shared types for MLX-related functionality."""

from collections.abc import Sequence
from dataclasses import dataclass

from mlx import core as mx
from mlx import nn as nn
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

# This list contains one cache entry per transformer layer
KVCacheType = Sequence[
    KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache
]


@dataclass(frozen=True, slots=True)
class ContinuationFrontier:
    """Exact private tokens and their already-verified terminal token."""

    tokens: mx.array
    terminal_token: int
    token_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class ContinuationPrompt:
    """Device tokens plus canonical CPU bytes when durably available."""

    tokens: mx.array
    token_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class DraftContinuation:
    """Previously generated tokens proposed for target-model verification."""

    tokens: tuple[int, ...]


# Model is a wrapper function to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function
class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...

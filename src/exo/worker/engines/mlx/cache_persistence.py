"""Persistence seam for MLX prefix computation.

The inference engine owns the meaning and serialization of a KV cache.  A
backend owns durable bytes and lookup policy, but must never manufacture a
cache for tokens or a runtime profile it cannot verify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx

from exo.worker.engines.mlx.types import KVCacheType

from .cache import CacheSnapshot

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


@dataclass(frozen=True)
class PersistedKVPrefix:
    """One restored, identity-checked prefix computation."""

    prompt_tokens: mx.array
    cache: KVCacheType
    snapshots: list[CacheSnapshot] | None
    media_regions: list["MediaRegion"]
    prefill_tps: float


class KVPrefixPersistence(Protocol):
    """Durable accelerator behind :class:`KVPrefixCache`.

    Failures are cache misses: implementations should log diagnostics and
    return ``None`` rather than make inference availability depend on the
    persistence tier.
    """

    def restore_longest(
        self,
        prompt_tokens: mx.array,
        minimum_tokens: int,
        media_regions: list["MediaRegion"],
    ) -> PersistedKVPrefix | None:
        """Restore a matching prefix longer than ``minimum_tokens``."""

    def schedule_store(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
    ) -> None:
        """Schedule durable publication without delaying token generation."""

    def close(self) -> None:
        """Finish accepted publications and release backend resources."""

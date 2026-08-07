"""Persistence seam for MLX prefix computation.

The inference engine owns the meaning and serialization of a KV cache.  A
backend owns durable bytes and lookup policy, but must never manufacture a
cache for tokens or a runtime profile it cannot verify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx

from exo.worker.engines.mlx.types import (
    ContinuationFrontier,
    DraftContinuation,
    KVCacheType,
)

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
        prompt_token_bytes: bytes | None = None,
    ) -> PersistedKVPrefix | None:
        """Restore a matching prefix longer than ``minimum_tokens``."""

    def schedule_store(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
        continuation_id: str | None = None,
    ) -> None:
        """Schedule durable publication without delaying token generation."""

    def schedule_store_thread_bound(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
        continuation_id: str | None = None,
    ) -> None:
        """Capture thread-local accelerator state, then publish asynchronously."""

    def resolve_continuation(self, continuation_id: str) -> ContinuationFrontier | None:
        """Resolve an opaque response identity to its exact token frontier."""

    def schedule_draft(
        self,
        prompt_tokens: mx.array,
        output_tokens: list[int],
    ) -> None:
        """Schedule a prior output as a non-authoritative speculative draft."""

    def resolve_draft(self, prompt_tokens: mx.array) -> DraftContinuation | None:
        """Resolve a candidate whose tokens still require model verification."""

    def close(self) -> None:
        """Finish accepted publications and release backend resources."""

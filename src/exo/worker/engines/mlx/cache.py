import gc
import hashlib
import os
from collections import OrderedDict, deque
from copy import copy, deepcopy
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.types import (
    ContinuationFrontier,
    DraftContinuation,
    KVCacheType,
    Model,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.cache_persistence import KVPrefixPersistence
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copy_or_none(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    out = _detached_copy(a)
    mx.eval(out)
    return out


def _copy_compressor_branch(b: CompressorBranch) -> CompressorBranch:
    out = CompressorBranch.__new__(CompressorBranch)
    out.buffer_kv = _detached_copy_or_none(b.buffer_kv)
    out.buffer_gate = _detached_copy_or_none(b.buffer_gate)
    out.prev_kv = _detached_copy_or_none(b.prev_kv)
    out.prev_gate = _detached_copy_or_none(b.prev_gate)
    out.pool = _detached_copy_or_none(b.pool)
    out.buffer_lengths = deepcopy(b.buffer_lengths)
    out.pool_lengths = deepcopy(b.pool_lengths)
    out.buffer_count = deepcopy(b.buffer_count)
    out._new_pool_lengths = deepcopy(b._new_pool_lengths)
    return out


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    snap = DeepseekV4Cache.__new__(DeepseekV4Cache)

    local: RotatingKVCache = c.local
    local_snap = copy_rotating_kv_cache(local)
    if local_snap is None:
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = None
        local_snap.values = None
        local_snap.offset = local.offset
        local_snap._idx = 0
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size
    snap.local = local_snap

    snap._branches = {
        key: _copy_compressor_branch(branch) for key, branch in c._branches.items()
    }
    snap._pending_lengths = deepcopy(c._pending_lengths)
    return snap


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        elif isinstance(c, DeepseekV4Cache):
            states.append(_copy_v4_cache(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    return isinstance(c, DeepseekV4Cache)


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


class KVPrefixCache:
    def __init__(
        self,
        group: mx.distributed.Group | None,
        persistence: "KVPrefixPersistence | None" = None,
    ):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._continuations: dict[str, int] = {}
        self._continuation_terminal_tokens: dict[str, int] = {}
        # Draft metadata is bounded by the number of retained KV frontiers;
        # it cannot become a second unaccounted conversation history.
        self._drafts: OrderedDict[bytes, DraftContinuation] = OrderedDict()
        self._deferred_thread_bound: deque[
            tuple[
                mx.array,
                KVCacheType,
                list[CacheSnapshot] | None,
                list["MediaRegion"],
                float,
                str | None,
            ]
        ] = deque()
        self._access_counter: int = 0
        self._group = group
        self._persistence = persistence

    def _schedule_persistence(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
        *,
        thread_bound: bool = False,
        continuation_id: str | None = None,
    ) -> None:
        """Publish an accelerator checkpoint without coupling it to inference."""
        if self._persistence is None:
            return
        if thread_bound:
            checkpoint = (
                prompt_tokens,
                cache,
                snapshots,
                media_regions,
                prefill_tps,
                continuation_id,
            )
            for index in range(len(self._deferred_thread_bound) - 1, -1, -1):
                if self._deferred_thread_bound[index][5] == continuation_id:
                    self._deferred_thread_bound[index] = checkpoint
                    return
            self._deferred_thread_bound.append(checkpoint)
            return
        try:
            if continuation_id is None:
                self._persistence.schedule_store(
                    prompt_tokens,
                    cache,
                    snapshots,
                    media_regions,
                    prefill_tps,
                )
            else:
                self._persistence.schedule_store(
                    prompt_tokens,
                    cache,
                    snapshots,
                    media_regions,
                    prefill_tps,
                    continuation_id=continuation_id,
                )
        except Exception:
            logger.opt(exception=True).warning(
                "KV cache persistence failed; inference result remains valid"
            )

    def flush_pending_persistence(self, limit: int | None = None) -> int:
        """Serialize completed frontiers on their owning inference thread."""
        if self._persistence is None:
            self._deferred_thread_bound.clear()
            return 0
        schedule = getattr(self._persistence, "schedule_store_thread_bound", None)
        if schedule is None:
            self._deferred_thread_bound.clear()
            return 0
        flushed = 0
        while self._deferred_thread_bound and (limit is None or flushed < limit):
            prompt, cache, snapshots, media, prefill_tps, continuation_id = (
                self._deferred_thread_bound.popleft()
            )
            try:
                if continuation_id is None:
                    schedule(prompt, cache, snapshots, media, prefill_tps)
                else:
                    schedule(
                        prompt,
                        cache,
                        snapshots,
                        media,
                        prefill_tps,
                        continuation_id=continuation_id,
                    )
                flushed += 1
            except Exception:
                logger.opt(exception=True).warning(
                    "KV cache persistence failed; inference result remains valid"
                )
        return flushed

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()
        self._continuations.clear()
        self._continuation_terminal_tokens.clear()
        self._drafts.clear()
        self._deferred_thread_bound.clear()

    def close(self) -> None:
        """Finish pending durable publication and release its resources."""
        if self._persistence is None:
            return
        self.flush_pending_persistence()
        self._persistence.close()
        self._persistence = None

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self._schedule_persistence(
            self.prompts[-1],
            self.caches[-1],
            self._snapshots[-1],
            list(self._media_regions[-1]),
            prefill_tps,
        )
        logger.info(f"KV cache added: {len(prompt_tokens)} tokens")

    def adopt_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
        continuation_id: str | None = None,
        continuation_terminal_token: int | None = None,
    ) -> int:
        """Adopt a completed cache without copying it.

        Ownership transfers to this prefix cache.  The caller must not mutate
        ``cache`` after this call.  This is the completion-path counterpart to
        :meth:`add_kv_cache`: MLX has already detached the cache from active
        generation, so copying gigabytes of immutable state would add latency
        without protecting another owner.
        """
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(cache)
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self._schedule_persistence(
            self.prompts[-1],
            self.caches[-1],
            self._snapshots[-1],
            list(self._media_regions[-1]),
            prefill_tps,
            thread_bound=True,
            continuation_id=continuation_id,
        )
        if continuation_id is not None:
            self._continuations[continuation_id] = len(self.prompts) - 1
            self._continuation_terminal_tokens[continuation_id] = (
                continuation_terminal_token
                if continuation_terminal_token is not None
                else int(prompt_tokens[-1].item())
            )
        logger.info(f"KV cache adopted: {len(prompt_tokens)} tokens")
        return len(self.prompts) - 1

    def adopt_kv_cache_update(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
        continuation_id: str | None = None,
        continuation_terminal_token: int | None = None,
    ) -> None:
        """Replace an entry with a completed cache, taking ownership directly."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [
                snapshot
                for snapshot in old_snapshots
                if snapshot.token_count <= restore_pos
            ]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = cache
        self._snapshots[index] = merged or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._schedule_persistence(
            self.prompts[index],
            self.caches[index],
            self._snapshots[index],
            list(self._media_regions[index]),
            prefill_tps,
            thread_bound=True,
            continuation_id=continuation_id,
        )
        self._drop_continuations_for_index(index)
        if continuation_id is not None:
            self._continuations[continuation_id] = index
            self._continuation_terminal_tokens[continuation_id] = (
                continuation_terminal_token
                if continuation_terminal_token is not None
                else int(prompt_tokens[-1].item())
            )
        logger.info(f"KV cache adopted (index {index}): {len(prompt_tokens)} tokens")

    def resolve_continuation(self, continuation_id: str) -> ContinuationFrontier | None:
        """Resolve a response identity without reconstructing its transcript."""
        index = self._continuations.get(continuation_id)
        if index is not None and index < len(self.prompts):
            return ContinuationFrontier(
                self.prompts[index],
                self._continuation_terminal_tokens[continuation_id],
            )
        if self._persistence is None:
            return None
        resolver = getattr(self._persistence, "resolve_continuation", None)
        if resolver is None:
            return None
        try:
            return cast(ContinuationFrontier | None, resolver(continuation_id))
        except Exception:
            logger.opt(exception=True).warning(
                "KV continuation lookup failed; falling back to rendered prompt"
            )
            return None

    def remember_draft(
        self,
        prompt_tokens: mx.array,
        output_tokens: list[int],
    ) -> None:
        """Remember one exact-prompt continuation as an untrusted draft.

        A draft is only a proposal. The target model must verify it before any
        token is returned, so persistence failure or stale data can affect
        performance but never generation correctness.
        """
        if not output_tokens:
            return
        draft = DraftContinuation(tuple(output_tokens))
        self._retain_draft(_draft_prompt_key(prompt_tokens), draft)
        if self._persistence is None:
            return
        schedule = getattr(self._persistence, "schedule_draft", None)
        if schedule is None:
            return
        try:
            schedule(prompt_tokens, output_tokens)
        except Exception:
            logger.opt(exception=True).warning(
                "Speculative draft persistence failed; generation remains valid"
            )

    def resolve_draft(self, prompt_tokens: mx.array) -> DraftContinuation | None:
        """Return a candidate for verification, never an authoritative result."""
        key = _draft_prompt_key(prompt_tokens)
        local = self._drafts.get(key)
        if local is not None:
            self._drafts.move_to_end(key)
            return local
        if self._persistence is None:
            return None
        resolve = getattr(self._persistence, "resolve_draft", None)
        if resolve is None:
            return None
        try:
            restored = cast(DraftContinuation | None, resolve(prompt_tokens))
            if restored is not None:
                self._retain_draft(key, restored)
            return restored
        except Exception:
            logger.opt(exception=True).warning(
                "Speculative draft lookup failed; using ordinary generation"
            )
            return None

    def _retain_draft(self, key: bytes, draft: DraftContinuation) -> None:
        self._drafts[key] = draft
        self._drafts.move_to_end(key)
        limit = max(1, len(self.prompts))
        while len(self._drafts) > limit:
            self._drafts.popitem(last=False)

    def is_continuation_entry(self, index: int) -> bool:
        """Return whether an immutable response identity currently names an entry."""
        return index in self._continuations.values()

    def _drop_continuations_for_index(self, index: int) -> None:
        stale = [key for key, value in self._continuations.items() if value == index]
        for key in stale:
            del self._continuations[key]
            self._continuation_terminal_tokens.pop(key, None)

    def _remove_continuation_index(self, index: int) -> None:
        self._drop_continuations_for_index(index)
        for key, value in list(self._continuations.items()):
            if value > index:
                self._continuations[key] = value - 1

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        if self.is_continuation_entry(index):
            raise ValueError("response-linked cache frontiers are immutable")
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = deepcopy(cache)
        self._snapshots[index] = merged or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._schedule_persistence(
            self.prompts[index],
            self.caches[index],
            self._snapshots[index],
            list(self._media_regions[index]),
            prefill_tps,
        )
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
        *,
        prefer_persistent_fork: bool = False,
        prompt_token_bytes: bytes | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        if self._persistence is not None and prefer_persistent_fork:
            if prompt_token_bytes is None:
                restored = self._persistence.restore_longest(
                    prompt_tokens,
                    max(0, best_length - 1),
                    list(query_regions),
                )
            else:
                restored = self._persistence.restore_longest(
                    prompt_tokens,
                    max(0, best_length - 1),
                    list(query_regions),
                    prompt_token_bytes=prompt_token_bytes,
                )
            if restored is not None:
                # PersistedKVPrefix is minted only after the backend verifies
                # this query's token-prefix identity. Repeating that proof on
                # MLX arrays would synchronize the whole device stream.
                restored_length = len(restored.prompt_tokens)
                restored_length = self._validate_media_match(
                    restored_length,
                    restored.media_regions,
                    query_regions,
                )
                if restored_length >= best_length:
                    restored_cache_length = cache_length(restored.cache)
                    if restored_cache_length <= restored_length:
                        return (
                            restored.cache,
                            prompt_tokens[restored_cache_length:],
                            None,
                            restored_cache_length >= max_length - 1,
                        )

        # A response continuation is an append to an immutable frontier.  If
        # durable publication is still catching up, fork the local MLX cache
        # copy-on-write instead of copying every tensor merely to protect the
        # old response from mutation.  This deliberately supports only cache
        # layouts whose first multi-token append allocates fresh backing
        # arrays; unknown layouts retain the conservative deep-copy path.
        if prefer_persistent_fork and best_index is not None:
            cached_length = cache_length(self.caches[best_index])
            fork_position = min(cached_length, best_length)
            append_count = max_length - fork_position
            if fork_position > 0:
                forked = fork_kv_cache_for_append(
                    self.caches[best_index],
                    fork_position,
                    append_count,
                )
                if forked is not None:
                    self._access_counter += 1
                    self._last_used[best_index] = self._access_counter
                    return (
                        forked,
                        prompt_tokens[fork_position:],
                        None,
                        False,
                    )

        if self._persistence is not None and not is_exact:
            restored = self._persistence.restore_longest(
                prompt_tokens,
                best_length,
                list(query_regions),
            )
            if restored is not None:
                restored_length = len(restored.prompt_tokens)
                restored_length = self._validate_media_match(
                    restored_length,
                    restored.media_regions,
                    query_regions,
                )
                if restored_length > best_length:
                    self.prompts.append(restored.prompt_tokens)
                    self.caches.append(restored.cache)
                    self._snapshots.append(restored.snapshots)
                    self._media_regions.append(restored.media_regions)
                    self.prefill_tps.append(restored.prefill_tps)
                    self._access_counter += 1
                    self._last_used.append(self._access_counter)
                    best_index = len(self.prompts) - 1
                    best_length = restored_length
                    is_exact = restored_length >= max_length - 1

        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None, False

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        if has_ssm:
            target = min(cached_length, best_length)
        else:
            desired = (max_length - 1) if is_exact else best_length
            target = min(cached_length, desired)
        at_complete_boundary = target == cached_length
        if has_ssm and at_complete_boundary:
            restore_pos, restore_snap = target, None
        else:
            restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm and not at_complete_boundary:
            return make_kv_cache(model), prompt_tokens, None, False

        prompt_cache = deepcopy(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index, is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        evicted_any = False
        # Evict LRU entries until below threshold
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self._remove_continuation_index(lru_index)
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._snapshots.pop(lru_index)
            self._media_regions.pop(lru_index)
            self._last_used.pop(lru_index)
            self.prefill_tps.pop(lru_index)

            evicted_any = True
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
            )

        if evicted_any:
            gc.collect()
            mx.clear_cache()

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def fork_kv_cache_for_append(
    cache: KVCacheType,
    token_count: int,
    append_count: int,
) -> list[KVCache | RotatingKVCache] | None:
    """Fork a plain MLX KV frontier without copying its immutable tensors.

    ``KVCache`` is forced to allocate on its next update by exposing no spare
    capacity in the fork. ``RotatingKVCache`` allocates on every multi-token
    update. Thus both may share their old arrays until the first append while
    the response frontier remains untouched. A one-token append or an unknown
    cache layout stays on the conservative copying path.
    """
    if append_count <= 1 or token_count < 0:
        return None
    if any(
        not isinstance(entry, (KVCache, RotatingKVCache))
        or isinstance(entry, QuantizedKVCache)
        for entry in cache
    ):
        return None

    forked: list[KVCache | RotatingKVCache] = []
    for entry in cache:
        assert isinstance(entry, (KVCache, RotatingKVCache))
        if entry.offset < token_count or entry.offset - token_count > 1:
            return None
        cloned = copy(entry)
        if isinstance(cloned, RotatingKVCache):
            cloned.trim(cloned.offset - token_count)
        else:
            cloned.offset = token_count
            if cloned.keys is not None:
                cloned.keys = cloned.keys[..., :token_count, :]
            if cloned.values is not None:
                cloned.values = cloned.values[..., :token_count, :]
        forked.append(cloned)
    return forked


def retain_forked_kv_cache_prefix(
    cache: list[KVCache | RotatingKVCache],
    appended_token_count: int,
    retained_token_count: int,
) -> bool:
    """Discard an unaccepted suffix from a multi-token cache append.

    ``RotatingKVCache.trim`` only adjusts cursors. Once its sliding window is
    full, the multi-token append path stores the appended tensors contiguously
    at the tail, and cursor-only trimming leaves rejected tokens physically
    visible to the next model call. Slice that tail explicitly. Plain
    ``KVCache`` remains safely trimmable through its public operation.
    """
    if not 0 <= retained_token_count <= appended_token_count:
        return False
    discarded_token_count = appended_token_count - retained_token_count
    if discarded_token_count == 0:
        return True

    for entry in cache:
        if isinstance(entry, RotatingKVCache):
            if (
                entry.keys is None
                or entry.values is None
                or entry.keys.shape[2] < discarded_token_count
                or entry.values.shape[2] < discarded_token_count
                or entry.offset < discarded_token_count
                or entry._idx < discarded_token_count
            ):
                return False
        elif entry.offset < discarded_token_count:
            return False

    for entry in cache:
        if isinstance(entry, RotatingKVCache):
            assert entry.keys is not None and entry.values is not None
            retained_length = entry.keys.shape[2] - discarded_token_count
            entry.keys = entry.keys[..., :retained_length, :]
            entry.values = entry.values[..., :retained_length, :]
            entry.offset -= discarded_token_count
            entry._idx -= discarded_token_count
        else:
            assert entry.trim(discarded_token_count) == discarded_token_count
    return True


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        non_trimmable = isinstance(c, (ArraysCache, RotatingKVCache)) or (
            isinstance(c, CacheList) and not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
        )
        if non_trimmable:
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _draft_prompt_key(tokens: mx.array) -> bytes:
    encoded = np.asarray(tokens, dtype="<u4").tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(b"exo-mlx-draft-prompt-v1\0")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    return digest.digest()


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]

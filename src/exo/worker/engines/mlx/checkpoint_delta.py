"""Exact append deltas for MLX key/value checkpoints.

The codec is deliberately model-local. It describes logical tensor intervals;
the computation store remains responsible for durable byte identity, recovery,
and physical representation selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import msgspec
from mlx_lm.models.cache import KVCache, RotatingKVCache

from exo.worker.engines.mlx.types import KVCacheType

_FORMAT = "exo-mlx-kv-delta-v1"


class CheckpointDeltaLayer(msgspec.Struct, frozen=True):
    """Canonical reconstruction metadata for one cache layer."""

    cache_type: str
    base_start: int
    successor_start: int
    overlap_tokens: int
    successor_tokens: int
    offset: int
    keep: int
    max_size: int
    key_shape: tuple[int, int, int, int]
    value_shape: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class PreparedCheckpointDelta:
    layers: tuple[CheckpointDeltaLayer, ...]
    novel_tensor_bytes: int
    inherited_tensor_bytes: int
    successor_tensor_bytes: int


class _CacheRuntime(Protocol):
    state: tuple[mx.array, mx.array]
    offset: int


class _RotatingRuntime(_CacheRuntime, Protocol):
    keep: int
    max_size: int

    def empty(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class _LogicalState:
    keys: mx.array
    values: mx.array
    start: int
    offset: int
    keep: int
    max_size: int

    @property
    def tokens(self) -> int:
        return self.keys.shape[2]


def prepare_checkpoint_delta(
    path: Path,
    base: KVCacheType,
    successor: KVCacheType,
) -> PreparedCheckpointDelta | None:
    """Write only newly computed tensor tails when all inherited bytes agree.

    Returning ``None`` means the caller must publish a complete checkpoint.
    The current grammar accepts append-only caches and rotating caches without
    a permanently retained prefix. Other layouts fail closed until their
    logical-coordinate contract is explicit.
    """
    if len(base) != len(successor):
        return None
    arrays: dict[str, mx.array] = {}
    layers: list[CheckpointDeltaLayer] = []
    tensor_bytes = 0
    inherited_tensor_bytes = 0
    successor_tensor_bytes = 0
    for index, (base_entry, successor_entry) in enumerate(
        zip(base, successor, strict=True)
    ):
        if type(base_entry) is not type(successor_entry):
            return None
        base_state = _logical_state(base_entry)
        successor_state = _logical_state(successor_entry)
        if base_state is None or successor_state is None:
            return None
        if not _compatible(base_state, successor_state):
            return None
        overlap_start = max(base_state.start, successor_state.start)
        overlap_end = min(base_state.offset, successor_state.offset)
        overlap_tokens = max(0, overlap_end - overlap_start)
        # A forward continuation may drop old rotating entries and append new
        # ones, but it may not manufacture data before the inherited overlap.
        if overlap_start != successor_state.start:
            return None
        base_index = overlap_start - base_state.start
        base_key_overlap = base_state.keys[
            ..., base_index : base_index + overlap_tokens, :
        ]
        base_value_overlap = base_state.values[
            ..., base_index : base_index + overlap_tokens, :
        ]
        successor_key_overlap = successor_state.keys[..., :overlap_tokens, :]
        successor_value_overlap = successor_state.values[..., :overlap_tokens, :]
        if not (
            bool(mx.array_equal(base_key_overlap, successor_key_overlap))
            and bool(mx.array_equal(base_value_overlap, successor_value_overlap))
        ):
            return None
        novel_keys = successor_state.keys[..., overlap_tokens:, :]
        novel_values = successor_state.values[..., overlap_tokens:, :]
        mx.eval(novel_keys, novel_values)
        arrays[f"{index}.keys"] = novel_keys
        arrays[f"{index}.values"] = novel_values
        tensor_bytes += novel_keys.nbytes + novel_values.nbytes
        inherited_tensor_bytes += (
            successor_key_overlap.nbytes + successor_value_overlap.nbytes
        )
        successor_tensor_bytes += (
            successor_state.keys.nbytes + successor_state.values.nbytes
        )
        layers.append(
            CheckpointDeltaLayer(
                cache_type=type(successor_entry).__name__,
                base_start=base_state.start,
                successor_start=successor_state.start,
                overlap_tokens=overlap_tokens,
                successor_tokens=successor_state.tokens,
                offset=successor_state.offset,
                keep=successor_state.keep,
                max_size=successor_state.max_size,
                key_shape=cast(tuple[int, int, int, int], successor_state.keys.shape),
                value_shape=cast(
                    tuple[int, int, int, int], successor_state.values.shape
                ),
            )
        )
    mx.save_safetensors(  # pyright: ignore[reportUnknownMemberType]
        str(path), arrays, {"format": _FORMAT}
    )
    return PreparedCheckpointDelta(
        tuple(layers),
        tensor_bytes,
        inherited_tensor_bytes,
        successor_tensor_bytes,
    )


def restore_checkpoint_delta(
    path: Path,
    base: KVCacheType,
    layers: tuple[CheckpointDeltaLayer, ...],
) -> KVCacheType:
    """Reconstruct an exact successor cache from a verified base and pack."""
    arrays, metadata = cast(
        tuple[dict[str, mx.array], dict[str, str]],
        cast(object, mx.load(str(path), return_metadata=True)),
    )
    if metadata != {"format": _FORMAT} or len(base) != len(layers):
        raise ValueError("checkpoint delta header mismatch")
    restored: list[KVCache | RotatingKVCache] = []
    for index, (base_entry, layer) in enumerate(zip(base, layers, strict=True)):
        base_state = _logical_state(base_entry)
        if base_state is None or type(base_entry).__name__ != layer.cache_type:
            raise ValueError(f"checkpoint delta base mismatch at layer {index}")
        overlap_start = layer.successor_start
        base_index = overlap_start - base_state.start
        if base_index < 0 or base_index + layer.overlap_tokens > base_state.tokens:
            raise ValueError(
                f"checkpoint delta overlap is out of bounds at layer {index}"
            )
        novel_keys = arrays.pop(f"{index}.keys", None)
        novel_values = arrays.pop(f"{index}.values", None)
        if novel_keys is None or novel_values is None:
            raise ValueError(f"checkpoint delta tensors missing at layer {index}")
        keys = mx.concatenate(
            [
                base_state.keys[..., base_index : base_index + layer.overlap_tokens, :],
                novel_keys,
            ],
            axis=2,
        )
        values = mx.concatenate(
            [
                base_state.values[
                    ..., base_index : base_index + layer.overlap_tokens, :
                ],
                novel_values,
            ],
            axis=2,
        )
        mx.eval(keys, values)
        if (
            keys.shape != layer.key_shape
            or values.shape != layer.value_shape
            or keys.shape[2] != layer.successor_tokens
        ):
            raise ValueError(f"checkpoint delta shape mismatch at layer {index}")
        restored.append(_restore_entry(layer, keys, values))
    if arrays:
        raise ValueError("checkpoint delta contains unreferenced tensors")
    return restored


def _logical_state(entry: object) -> _LogicalState | None:
    if isinstance(entry, KVCache):
        runtime = cast(_CacheRuntime, cast(object, entry))
        keys, values = runtime.state
        return _LogicalState(keys, values, 0, runtime.offset, 0, 0)
    if not isinstance(entry, RotatingKVCache):
        return None
    runtime = cast(_RotatingRuntime, cast(object, entry))
    if runtime.empty() or runtime.keep != 0:
        return None
    keys, values = runtime.state
    cursor = cast(int, getattr(entry, "_idx"))  # noqa: B009 - omitted by MLX stubs
    keys = _temporal_order(runtime, cursor, keys)
    values = _temporal_order(runtime, cursor, values)
    return _LogicalState(
        keys,
        values,
        runtime.offset - keys.shape[2],
        runtime.offset,
        runtime.keep,
        runtime.max_size,
    )


def _compatible(base: _LogicalState, successor: _LogicalState) -> bool:
    return (
        base.keys.shape[:2] == successor.keys.shape[:2]
        and base.keys.shape[3:] == successor.keys.shape[3:]
        and base.values.shape[:2] == successor.values.shape[:2]
        and base.values.shape[3:] == successor.values.shape[3:]
        and base.keys.dtype == successor.keys.dtype
        and base.values.dtype == successor.values.dtype
        and base.keep == successor.keep
        and base.max_size == successor.max_size
        and successor.offset >= base.offset
    )


def _restore_entry(
    layer: CheckpointDeltaLayer,
    keys: mx.array,
    values: mx.array,
) -> KVCache | RotatingKVCache:
    if layer.cache_type == "KVCache":
        entry = KVCache()
        runtime = cast(_CacheRuntime, cast(object, entry))
        runtime.state = (keys, values)
        return entry
    if layer.cache_type != "RotatingKVCache":
        raise ValueError(f"unsupported checkpoint cache type: {layer.cache_type}")
    entry = RotatingKVCache(max_size=layer.max_size, keep=layer.keep)
    runtime = cast(_RotatingRuntime, cast(object, entry))
    runtime.state = (keys, values)
    setattr(  # noqa: B010 - property setter is omitted by MLX stubs
        entry, "meta_state", (layer.keep, layer.max_size, layer.offset, keys.shape[2])
    )
    return entry


def _temporal_order(
    cache: _RotatingRuntime,
    cursor: int,
    value: mx.array,
) -> mx.array:
    if cursor == value.shape[2]:
        return value
    if cursor < cache.offset:
        return mx.concatenate(
            [value[..., cursor:, :], value[..., :cursor, :]],
            axis=2,
        )
    return value[..., :cursor, :]

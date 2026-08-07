# type: ignore
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from exo.worker.engines.mlx.checkpoint_delta import (
    prepare_checkpoint_delta,
    restore_checkpoint_delta,
)


def _append(entry, first: int, count: int) -> None:
    for token in range(first, first + count):
        keys = mx.full((1, 2, 1, 4), token, dtype=mx.float32)
        entry.update_and_fetch(keys, keys + 100)


def _temporal(entry: RotatingKVCache) -> tuple[mx.array, mx.array]:
    keys, values = entry.state
    return entry._temporal_order(keys), entry._temporal_order(values)


def test_delta_round_trips_append_only_and_wrapped_rotating_cache(tmp_path: Path):
    base_kv = KVCache()
    successor_kv = KVCache()
    base_rotating = RotatingKVCache(max_size=4, keep=0)
    successor_rotating = RotatingKVCache(max_size=4, keep=0)
    _append(base_kv, 0, 6)
    _append(successor_kv, 0, 9)
    _append(base_rotating, 0, 6)
    _append(successor_rotating, 0, 9)

    path = tmp_path / "delta.safetensors"
    prepared = prepare_checkpoint_delta(
        path,
        [base_kv, base_rotating],
        [successor_kv, successor_rotating],
    )
    assert prepared is not None
    restored = restore_checkpoint_delta(
        path,
        [base_kv, base_rotating],
        prepared.layers,
    )

    assert prepared.tensor_bytes == 2 * 2 * 2 * 3 * 4 * 4
    assert mx.array_equal(restored[0].state[0], successor_kv.state[0])
    assert mx.array_equal(restored[0].state[1], successor_kv.state[1])
    assert isinstance(restored[1], RotatingKVCache)
    expected_keys, expected_values = _temporal(successor_rotating)
    actual_keys, actual_values = _temporal(restored[1])
    assert mx.array_equal(actual_keys, expected_keys)
    assert mx.array_equal(actual_values, expected_values)

    next_keys = mx.full((1, 2, 1, 4), 9, dtype=mx.float32)
    successor_rotating.update_and_fetch(next_keys, next_keys + 100)
    restored[1].update_and_fetch(next_keys, next_keys + 100)
    expected_keys, expected_values = _temporal(successor_rotating)
    actual_keys, actual_values = _temporal(restored[1])
    assert mx.array_equal(actual_keys, expected_keys)
    assert mx.array_equal(actual_values, expected_values)


def test_delta_rejects_divergent_inherited_computation(tmp_path: Path):
    base = KVCache()
    successor = KVCache()
    _append(base, 0, 4)
    _append(successor, 0, 4)
    successor.keys[..., 2, :] = 999

    assert (
        prepare_checkpoint_delta(
            tmp_path / "delta.safetensors",
            [base],
            [successor],
        )
        is None
    )


def test_delta_rejects_rotating_layout_with_retained_prefix(tmp_path: Path):
    base = RotatingKVCache(max_size=4, keep=1)
    successor = RotatingKVCache(max_size=4, keep=1)
    _append(base, 0, 5)
    _append(successor, 0, 6)

    assert (
        prepare_checkpoint_delta(
            tmp_path / "delta.safetensors",
            [base],
            [successor],
        )
        is None
    )

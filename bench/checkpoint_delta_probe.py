#!/usr/bin/env python3
# type: ignore
"""Measure exact reusable tensor ranges between two MLX KV checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache, load_prompt_cache


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("successor", type=Path)
    return parser.parse_args()


def _temporal_state(entry: KVCache | RotatingKVCache) -> tuple[mx.array, mx.array]:
    if entry.keys is None or entry.values is None:
        raise ValueError("checkpoint contains an empty cache entry")
    if isinstance(entry, RotatingKVCache):
        return entry._temporal_order(entry.keys), entry._temporal_order(entry.values)
    return cast(tuple[mx.array, mx.array], entry.state)


def _logical_start(entry: KVCache | RotatingKVCache) -> int:
    return entry.offset - entry.size()


def _entry_overlap(
    index: int,
    base: KVCache | RotatingKVCache,
    successor: KVCache | RotatingKVCache,
) -> dict[str, Any]:
    if type(base) is not type(successor):
        raise ValueError(f"cache type changed at layer {index}")
    base_keys, base_values = _temporal_state(base)
    successor_keys, successor_values = _temporal_state(successor)
    if (
        base_keys.shape[:2] != successor_keys.shape[:2]
        or base_keys.shape[3:] != successor_keys.shape[3:]
        or base_values.shape[:2] != successor_values.shape[:2]
        or base_values.shape[3:] != successor_values.shape[3:]
    ):
        raise ValueError(f"cache tensor shape changed at layer {index}")

    base_start = _logical_start(base)
    successor_start = _logical_start(successor)
    overlap_start = max(base_start, successor_start)
    overlap_end = min(base.offset, successor.offset)
    overlap_tokens = max(0, overlap_end - overlap_start)
    base_offset = overlap_start - base_start
    successor_offset = overlap_start - successor_start
    base_key_overlap = base_keys[..., base_offset : base_offset + overlap_tokens, :]
    successor_key_overlap = successor_keys[
        ..., successor_offset : successor_offset + overlap_tokens, :
    ]
    base_value_overlap = base_values[..., base_offset : base_offset + overlap_tokens, :]
    successor_value_overlap = successor_values[
        ..., successor_offset : successor_offset + overlap_tokens, :
    ]
    overlap_is_exact = bool(
        mx.all(base_key_overlap == successor_key_overlap).item()
        and mx.all(base_value_overlap == successor_value_overlap).item()
    )
    key_bytes_per_token = base_keys.nbytes // base_keys.shape[2]
    value_bytes_per_token = base_values.nbytes // base_values.shape[2]
    bytes_per_token = key_bytes_per_token + value_bytes_per_token
    successor_tokens = successor.size()
    novel_tokens = successor_tokens - overlap_tokens
    return {
        "layer": index,
        "cache_type": type(base).__name__,
        "base_offset": base.offset,
        "successor_offset": successor.offset,
        "base_tokens": base.size(),
        "successor_tokens": successor_tokens,
        "overlap_tokens": overlap_tokens,
        "overlap_is_exact": overlap_is_exact,
        "bytes_per_token": bytes_per_token,
        "shared_bytes": overlap_tokens * bytes_per_token if overlap_is_exact else 0,
        "novel_bytes": novel_tokens * bytes_per_token,
        "dropped_bytes": (base.size() - overlap_tokens) * bytes_per_token,
        "successor_bytes": successor_tokens * bytes_per_token,
    }


def main() -> None:
    args = _arguments()
    base = load_prompt_cache(str(args.base))
    successor = load_prompt_cache(str(args.successor))
    if len(base) != len(successor):
        raise ValueError("checkpoint layer count changed")
    entries = [
        _entry_overlap(index, base_entry, successor_entry)
        for index, (base_entry, successor_entry) in enumerate(
            zip(base, successor, strict=True)
        )
        if isinstance(base_entry, (KVCache, RotatingKVCache))
        and isinstance(successor_entry, (KVCache, RotatingKVCache))
    ]
    if len(entries) != len(base):
        raise TypeError("probe supports only KVCache and RotatingKVCache entries")
    successor_bytes = sum(entry["successor_bytes"] for entry in entries)
    shared_bytes = sum(entry["shared_bytes"] for entry in entries)
    novel_bytes = sum(entry["novel_bytes"] for entry in entries)
    print(
        json.dumps(
            {
                "base": str(args.base),
                "successor": str(args.successor),
                "layers": len(entries),
                "all_overlaps_exact": all(
                    entry["overlap_is_exact"] for entry in entries
                ),
                "successor_tensor_bytes": successor_bytes,
                "exactly_shared_tensor_bytes": shared_bytes,
                "minimum_delta_tensor_bytes": novel_bytes,
                "shared_fraction": (
                    shared_bytes / successor_bytes if successor_bytes else 0.0
                ),
                "delta_fraction": (
                    novel_bytes / successor_bytes if successor_bytes else 0.0
                ),
                "dropped_tensor_bytes": sum(
                    entry["dropped_bytes"] for entry in entries
                ),
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

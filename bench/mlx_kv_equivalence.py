# type: ignore
#!/usr/bin/env python3
"""Diagnose serial-versus-batched MLX KV-cache equivalence.

This is deliberately a diagnostic rather than a throughput benchmark. It
compares the cache and logits produced by ordinary one-token greedy execution
with the state produced by Exo's speculative block verification, including
the partial-prefix trim path.
"""

from __future__ import annotations

import argparse
import json
from copy import copy
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, RotatingKVCache

from exo.worker.engines.mlx.cache import (
    cache_length,
    fork_kv_cache_for_append,
    make_kv_cache,
    retain_forked_kv_cache_prefix,
)
from exo.worker.engines.mlx.types import KVCacheType, Model

CacheMode = Literal["copied", "forked"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--trace-cut", type=int)
    parser.add_argument("--trace-window-chars", type=int, default=3000)
    parser.add_argument("--block-sizes", default="2,4,8,16,32")
    parser.add_argument("--continuation-tokens", type=int, default=40)
    parser.add_argument(
        "--partial-counts",
        default="all",
        help="Comma-separated accepted-prefix lengths, or 'all'",
    )
    return parser.parse_args()


def _copy_cache(cache: KVCacheType) -> list[KVCache | RotatingKVCache]:
    copied: list[KVCache | RotatingKVCache] = []
    for entry in cache:
        if not isinstance(entry, (KVCache, RotatingKVCache)):
            raise TypeError(f"unsupported diagnostic cache entry: {type(entry)!r}")
        cloned = copy(entry)
        if entry.keys is not None:
            # MLX has no public copy primitive. Arithmetic with a scalar makes
            # an independent allocation without changing the stored values.
            cloned.keys = entry.keys + mx.array(0, dtype=entry.keys.dtype)
            cloned.values = entry.values + mx.array(0, dtype=entry.values.dtype)
        copied.append(cloned)
    mx.eval([entry.state for entry in copied])
    return copied


def _cache_for_mode(
    base_cache: KVCacheType,
    mode: CacheMode,
    append_count: int,
) -> list[KVCache | RotatingKVCache]:
    if mode == "copied":
        return _copy_cache(base_cache)
    forked = fork_kv_cache_for_append(
        base_cache,
        cache_length(base_cache),
        append_count,
    )
    if forked is None:
        raise RuntimeError("model cache cannot use Exo's append fork")
    return forked


def _run_serial(
    model: Model,
    base_cache: KVCacheType,
    current_token: int,
    steps: int,
) -> tuple[list[int], list[mx.array], list[KVCache | RotatingKVCache]]:
    cache = _copy_cache(base_cache)
    tokens: list[int] = []
    logits_rows: list[mx.array] = []
    token = current_token
    for _ in range(steps):
        logits = model(mx.array([[token]], dtype=mx.uint32), cache=cache)[0, -1]
        next_token = int(mx.argmax(logits).item())
        mx.eval(logits, [entry.state for entry in cache])
        logits_rows.append(logits)
        tokens.append(next_token)
        token = next_token
    return tokens, logits_rows, cache


def _maximum_array_difference(left: mx.array, right: mx.array) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(
        mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))).item()
    )


def _cache_difference(
    left: list[KVCache | RotatingKVCache],
    right: list[KVCache | RotatingKVCache],
) -> dict[str, Any]:
    if len(left) != len(right):
        return {"same_shape": False, "maximum_absolute_difference": None}
    maximum = 0.0
    mismatched_metadata: list[int] = []
    mismatched_shapes: list[int] = []
    for index, (left_entry, right_entry) in enumerate(zip(left, right, strict=True)):
        if (
            type(left_entry) is not type(right_entry)
            or left_entry.offset != right_entry.offset
        ):
            mismatched_metadata.append(index)
            continue
        left_state = cast(tuple[mx.array, mx.array], left_entry.state)
        right_state = cast(tuple[mx.array, mx.array], right_entry.state)
        if (
            left_state[0].shape != right_state[0].shape
            or left_state[1].shape != right_state[1].shape
        ):
            mismatched_shapes.append(index)
            continue
        maximum = max(
            maximum,
            _maximum_array_difference(left_state[0], right_state[0]),
            _maximum_array_difference(left_state[1], right_state[1]),
        )
    return {
        "same_shape": not mismatched_metadata and not mismatched_shapes,
        "mismatched_metadata_layers": mismatched_metadata,
        "mismatched_shape_layers": mismatched_shapes,
        "maximum_absolute_difference": maximum,
    }


def _logit_comparison(left: mx.array, right: mx.array) -> dict[str, Any]:
    left_top = mx.argpartition(-left, 2)[:2]
    right_top = mx.argpartition(-right, 2)[:2]
    mx.eval(left_top, right_top)
    left_indices = cast(list[int], left_top.tolist())
    right_indices = cast(list[int], right_top.tolist())
    left_values = sorted(
        (float(left[index].item()) for index in left_indices), reverse=True
    )
    right_values = sorted(
        (float(right[index].item()) for index in right_indices), reverse=True
    )
    left_argmax = int(mx.argmax(left).item())
    right_argmax = int(mx.argmax(right).item())
    return {
        "same_argmax": left_argmax == right_argmax,
        "serial_argmax": left_argmax,
        "batched_argmax": right_argmax,
        "maximum_absolute_difference": _maximum_array_difference(left, right),
        "serial_top_two_margin": left_values[0] - left_values[1],
        "batched_top_two_margin": right_values[0] - right_values[1],
    }


def _full_block_case(
    model: Model,
    base_cache: KVCacheType,
    current_token: int,
    reference_tokens: list[int],
    reference_logits: list[mx.array],
    block_size: int,
    mode: CacheMode,
) -> dict[str, Any]:
    model_inputs = (current_token, *reference_tokens[: block_size - 1])
    batched_cache = _cache_for_mode(base_cache, mode, block_size)
    batched_logits = model(
        mx.array([model_inputs], dtype=mx.uint32), cache=batched_cache
    )[0]
    mx.eval(batched_logits, [entry.state for entry in batched_cache])

    serial_tokens, _, serial_cache = _run_serial(
        model, base_cache, current_token, block_size
    )
    row_comparisons = [
        _logit_comparison(reference_logits[index], batched_logits[index])
        for index in range(block_size)
    ]
    return {
        "mode": mode,
        "block_size": block_size,
        "all_rows_same_argmax": all(row["same_argmax"] for row in row_comparisons),
        "first_row_argmax_mismatch": next(
            (
                index
                for index, comparison in enumerate(row_comparisons)
                if not comparison["same_argmax"]
            ),
            None,
        ),
        "maximum_logit_difference": max(
            comparison["maximum_absolute_difference"] for comparison in row_comparisons
        ),
        "minimum_serial_margin": min(
            comparison["serial_top_two_margin"] for comparison in row_comparisons
        ),
        "tokens_match_reference": serial_tokens == reference_tokens[:block_size],
        "cache": _cache_difference(serial_cache, batched_cache),
    }


def _partial_block_case(
    model: Model,
    base_cache: KVCacheType,
    current_token: int,
    reference_tokens: list[int],
    vocabulary_size: int,
    block_size: int,
    accepted_count: int,
) -> dict[str, Any]:
    candidate = list(reference_tokens[:block_size])
    candidate[accepted_count] = (candidate[accepted_count] + 1) % vocabulary_size
    model_inputs = (current_token, *candidate[:-1])
    batched_cache = _cache_for_mode(base_cache, "forked", block_size)
    model(mx.array([model_inputs], dtype=mx.uint32), cache=batched_cache)
    retained = retain_forked_kv_cache_prefix(
        batched_cache,
        block_size,
        accepted_count,
    )
    mx.eval([entry.state for entry in batched_cache])

    _, _, serial_cache = _run_serial(model, base_cache, current_token, accepted_count)
    continuation_input = reference_tokens[accepted_count - 1]
    serial_next = model(
        mx.array([[continuation_input]], dtype=mx.uint32), cache=serial_cache
    )[0, -1]
    batched_next = model(
        mx.array([[continuation_input]], dtype=mx.uint32), cache=batched_cache
    )[0, -1]
    mx.eval(serial_next, batched_next)
    return {
        "block_size": block_size,
        "accepted_count": accepted_count,
        "trimmed_as_requested": retained,
        "cache_before_continuation": _cache_difference(serial_cache, batched_cache),
        "next_logits": _logit_comparison(serial_next, batched_next),
    }


def _prompt(args: argparse.Namespace) -> str:
    if args.trace_file is None:
        return (
            "A content-addressed operating system preserves deterministic work. "
            "Explain how verified speculative execution can reuse a remembered "
            "continuation without granting that continuation authority."
        )
    if args.trace_cut is None:
        raise ValueError("--trace-cut is required with --trace-file")
    trace = args.trace_file.read_text(errors="replace")
    if not 0 < args.trace_cut <= len(trace):
        raise ValueError("trace-cut is outside the trace")
    return trace[max(0, args.trace_cut - args.trace_window_chars) : args.trace_cut]


def main() -> None:
    args = _arguments()
    block_sizes = tuple(int(value) for value in args.block_sizes.split(","))
    if not block_sizes or min(block_sizes) < 2:
        raise ValueError("block sizes must all be at least two")
    required_tokens = max(max(block_sizes) + 1, args.continuation_tokens)

    model, tokenizer = load(str(args.model), lazy=False)
    mx.eval(model.parameters())
    prompt_tokens = cast(
        list[int], tokenizer.encode(_prompt(args), add_special_tokens=False)
    )
    if len(prompt_tokens) < 2:
        raise ValueError("prompt must contain at least two tokens")

    base_cache = make_kv_cache(model)
    model(
        mx.array([prompt_tokens[:-1]], dtype=mx.uint32),
        cache=base_cache,
    )
    mx.eval([entry.state for entry in base_cache])
    current_token = prompt_tokens[-1]
    reference_tokens, reference_logits, _ = _run_serial(
        model,
        base_cache,
        current_token,
        required_tokens,
    )

    full_cases = [
        _full_block_case(
            model,
            base_cache,
            current_token,
            reference_tokens,
            reference_logits,
            block_size,
            mode,
        )
        for block_size in block_sizes
        for mode in ("copied", "forked")
    ]
    partial_cases: list[dict[str, Any]] = []
    for block_size in block_sizes:
        if args.partial_counts == "all":
            accepted_counts = range(1, block_size)
        else:
            accepted_counts = (int(value) for value in args.partial_counts.split(","))
        for accepted_count in accepted_counts:
            if 0 < accepted_count < block_size:
                partial_cases.append(
                    _partial_block_case(
                        model,
                        base_cache,
                        current_token,
                        reference_tokens,
                        reference_logits[0].shape[-1],
                        block_size,
                        accepted_count,
                    )
                )

    print(
        json.dumps(
            {
                "model": str(args.model),
                "prompt_tokens": len(prompt_tokens),
                "cache_layers": len(base_cache),
                "cache_types": sorted({type(entry).__name__ for entry in base_cache}),
                "full_cases": full_cases,
                "partial_cases": partial_cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

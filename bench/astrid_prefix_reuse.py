# type: ignore
#!/usr/bin/env python3
"""Measure durable MLX prefix reuse through Astrid's embedded store."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from exo.worker.engines.mlx.astrid_persistence import AstridKVPrefixPersistence
from exo.worker.engines.mlx.cache import KVPrefixCache, make_kv_cache
from exo.worker.engines.mlx.generator.generate import prefill


def _tokens(tokenizer: Any, target: int, text: str) -> mx.array:
    seed: list[int] = tokenizer.encode(text, add_special_tokens=False)
    repetitions = (target + len(seed) - 1) // len(seed)
    return mx.array((seed * repetitions)[:target])


def _first_token(model: Any, tokens: mx.array, prompt_cache: list[Any]) -> tuple[int, float]:
    started = time.perf_counter()
    iterator = generate_step(
        tokens,
        model,
        max_tokens=1,
        prompt_cache=prompt_cache,
        prefill_step_size=4096,
    )
    token, logprobs = next(iterator)
    mx.eval(logprobs)
    value = int(token.item()) if hasattr(token, "item") else int(token)
    return value, time.perf_counter() - started


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--append-tokens", type=int, default=256)
    parser.add_argument(
        "--runtime-profile",
        default="benchmark-local-model-closure",
        help="Identity of the exact model/runtime semantics under test",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    model, tokenizer = load(str(args.model), lazy=False)
    mx.eval(model.parameters())

    prefix = _tokens(
        tokenizer,
        args.prefix_tokens,
        "Astrid and Exo conserve deterministic computation across a local fleet. ",
    )
    suffix = _tokens(
        tokenizer,
        args.append_tokens,
        " Continue with new evidence while retaining the identified prior context. ",
    )
    continued = mx.concatenate([prefix, suffix])

    with tempfile.TemporaryDirectory(prefix="exo-astrid-bench-") as store:
        persistence = AstridKVPrefixPersistence(Path(store), args.runtime_profile)
        prefix_cache = KVPrefixCache(None, persistence=persistence)
        computed = make_kv_cache(model)
        prefix_started = time.perf_counter()
        prefill_tps, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            prefix[:-1],
            computed,
            None,
            None,
            None,
        )
        prefix_seconds = time.perf_counter() - prefix_started

        publish_started = time.perf_counter()
        prefix_cache.add_kv_cache(
            prefix,
            computed,
            snapshots or None,
            prefill_tps=prefill_tps,
        )
        prefix_cache.close()
        publish_seconds = time.perf_counter() - publish_started
        del prefix_cache, computed, persistence
        gc.collect()
        mx.clear_cache()

        reopened = AstridKVPrefixPersistence(Path(store), args.runtime_profile)
        restarted_cache = KVPrefixCache(None, persistence=reopened)
        restore_started = time.perf_counter()
        restored, remaining, matched_index, _ = restarted_cache.get_kv_cache(
            model, continued
        )
        restore_seconds = time.perf_counter() - restore_started
        restored_token, suffix_seconds = _first_token(model, remaining, restored)
        restarted_cache.close()

        cold = make_kv_cache(model)
        cold_token, cold_seconds = _first_token(model, continued, cold)

        warm_seconds = restore_seconds + suffix_seconds
        result = {
            "model": str(args.model),
            "prefix_tokens": len(prefix),
            "append_tokens": len(suffix),
            "prefix_prefill_seconds": prefix_seconds,
            "checkpoint_publish_seconds": publish_seconds,
            "cold_seconds_to_first_token": cold_seconds,
            "restore_seconds": restore_seconds,
            "suffix_compute_seconds": suffix_seconds,
            "restored_seconds_to_first_token": warm_seconds,
            "speedup": cold_seconds / warm_seconds,
            "restored_prefix_tokens": len(continued) - len(remaining),
            "durable_hit": matched_index is not None,
            "same_first_token": restored_token == cold_token,
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

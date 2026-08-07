#!/usr/bin/env python3
# type: ignore
"""Measure durable computation growth across a real multi-turn session."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.computation_persistence import StoreKVPrefixPersistence
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.patches import apply_mlx_patches
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.bootstrap import logger

_CORPUS = "The system conserves exact deterministic computation. "
_INITIAL_SUFFIX = " Reply with the single word OK."


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", default="mlx-community/gemma-4-31b-it-4bit")
    parser.add_argument("--initial-tokens", type=int, default=4096)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument(
        "--evict-projections-between-turns",
        action="store_true",
        help="Force verified delta reconstruction on every continuation turn",
    )
    parser.add_argument(
        "--runtime-profile",
        default="gemma4-conversation-growth-v1",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="Retain results in this directory instead of a temporary store",
    )
    return parser.parse_args()


def _task(
    model_id: str,
    content: str,
    output_tokens: int,
    previous_response_id: str | None = None,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        previous_response_id=previous_response_id,
        max_output_tokens=output_tokens,
        temperature=0.0,
        enable_thinking=False,
    )


def _sized_initial_prompt(
    tokenizer: Any,
    model_id: str,
    target_tokens: int,
    output_tokens: int,
) -> tuple[TextGenerationTaskParams, str, int]:
    low, high = 0, target_tokens
    while low < high:
        middle = (low + high + 1) // 2
        task = _task(
            model_id,
            _CORPUS * middle + _INITIAL_SUFFIX,
            output_tokens,
        )
        prompt = apply_chat_template(tokenizer, task)
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if count <= target_tokens:
            low = middle
        else:
            high = middle - 1
    task = _task(
        model_id,
        _CORPUS * low + _INITIAL_SUFFIX,
        output_tokens,
    )
    prompt = apply_chat_template(tokenizer, task)
    count = len(tokenizer.encode(prompt, add_special_tokens=False))
    return task, prompt, count


def _directory_bytes(path: Path) -> tuple[int, int]:
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    return len(files), sum(entry.stat().st_size for entry in files)


def _run_turn(
    model: Any,
    tokenizer: Any,
    store: Path,
    runtime_profile: str,
    task: TextGenerationTaskParams,
    prompt: str,
    response_id: str,
) -> dict[str, Any]:
    persistence = StoreKVPrefixPersistence(store, runtime_profile)
    prefix_cache = KVPrefixCache(None, persistence=persistence)
    batch = ExoBatchGenerator(model, tokenizer, None, prefix_cache)
    started = time.perf_counter()
    batch.submit(task, prompt, response_id=response_id)
    submit_seconds = time.perf_counter() - started
    first_token_seconds: float | None = None
    responses = []
    while batch.has_work:
        current = batch.step()
        if current and first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        responses.extend(current)
    inference_seconds = time.perf_counter() - started
    final_response = responses[-1][1]
    batch.close()
    prefix_cache.close()
    publication = persistence.last_publication_metrics

    projection_path = store / "projections"
    projection_files, projection_bytes = _directory_bytes(projection_path)
    store_files, store_bytes = _directory_bytes(store)
    usage = final_response.usage
    stats = final_response.stats
    return {
        "response_id": response_id,
        "submit_seconds": submit_seconds,
        "first_token_seconds": first_token_seconds,
        "inference_seconds": inference_seconds,
        "output_tokens": len(responses),
        "prompt_tokens": usage.prompt_tokens if usage is not None else None,
        "cached_tokens": (
            usage.prompt_tokens_details.cached_tokens if usage is not None else None
        ),
        "cache_hit": stats.prefix_cache_hit if stats is not None else None,
        "generation_tps": stats.generation_tps if stats is not None else None,
        "peak_memory_bytes": (
            stats.peak_memory_usage.in_bytes if stats is not None else None
        ),
        "serialization_seconds": publication.mlx_serialization_seconds,
        "store_admission_seconds": publication.store_admission_seconds,
        "metadata_publication_seconds": publication.metadata_publication_seconds,
        "projection_files": projection_files,
        "projection_bytes": projection_bytes,
        "store_files": store_files,
        "store_bytes": store_bytes,
        "authoritative_bytes": store_bytes - projection_bytes,
    }


def _run(args: argparse.Namespace, store: Path) -> None:
    apply_mlx_patches()
    model, tokenizer = load(str(args.model), lazy=False)
    mx.eval(model.parameters())
    initial_task, initial_prompt, initial_count = _sized_initial_prompt(
        tokenizer,
        args.model_id,
        args.initial_tokens,
        args.output_tokens,
    )

    turns: list[dict[str, Any]] = []
    previous_response_id: str | None = None
    for index in range(args.turns):
        response_id = f"growth-turn-{index}"
        if index == 0:
            task = initial_task
            prompt = initial_prompt
        else:
            content = (
                f"Turn {index}: confirm that the preceding turn remains "
                "available. Reply with the single word OK."
            )
            task = _task(
                args.model_id,
                content,
                args.output_tokens,
                previous_response_id,
            )
            prompt = apply_chat_template(tokenizer, task)
        measurement = _run_turn(
            model,
            tokenizer,
            store,
            args.runtime_profile,
            task,
            prompt,
            response_id,
        )
        measurement["turn"] = index
        turns.append(measurement)
        previous_response_id = response_id
        if args.evict_projections_between_turns and index + 1 < args.turns:
            for projection in (store / "projections").glob("*.safetensors"):
                projection.unlink()
        mx.clear_cache()

    for previous, current in zip(turns, turns[1:], strict=False):
        current["incremental_store_bytes"] = (
            current["store_bytes"] - previous["store_bytes"]
        )
        current["incremental_projection_bytes"] = (
            current["projection_bytes"] - previous["projection_bytes"]
        )
        current["incremental_authoritative_bytes"] = (
            current["authoritative_bytes"] - previous["authoritative_bytes"]
        )
    turns[0]["incremental_store_bytes"] = turns[0]["store_bytes"]
    turns[0]["incremental_projection_bytes"] = turns[0]["projection_bytes"]
    turns[0]["incremental_authoritative_bytes"] = turns[0]["authoritative_bytes"]
    for turn in turns:
        prompt_tokens = turn["prompt_tokens"]
        cached_tokens = turn["cached_tokens"]
        new_tokens = (
            prompt_tokens - cached_tokens + turn["output_tokens"]
            if prompt_tokens is not None and cached_tokens is not None
            else None
        )
        turn["new_frontier_tokens"] = new_tokens
        turn["reuse_fraction"] = (
            cached_tokens / prompt_tokens
            if prompt_tokens is not None
            and cached_tokens is not None
            and prompt_tokens > 0
            else None
        )
        for field in (
            "incremental_store_bytes",
            "incremental_projection_bytes",
            "incremental_authoritative_bytes",
        ):
            turn[f"{field}_per_new_token"] = (
                turn[field] / new_tokens if new_tokens is not None else None
            )

    print(
        json.dumps(
            {
                "model": str(args.model),
                "initial_prompt_tokens": initial_count,
                "requested_turns": args.turns,
                "requested_output_tokens_per_turn": args.output_tokens,
                "store": str(store),
                "turns": turns,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = _arguments()
    logger.remove()
    if args.turns < 1 or args.output_tokens < 1 or args.initial_tokens < 2:
        raise ValueError("turns/output-tokens must be positive and initial-tokens >= 2")
    if args.store is not None:
        args.store.mkdir(mode=0o700, parents=True, exist_ok=True)
        _run(args, args.store)
        return
    with tempfile.TemporaryDirectory(prefix="exo-growth-bench-") as directory:
        _run(args, Path(directory))


if __name__ == "__main__":
    main()

# type: ignore
#!/usr/bin/env python3
"""Measure verified remembered-output speculation on Gemma through Exo."""

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

from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.computation_persistence import StoreKVPrefixPersistence
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.patches import apply_mlx_patches, opt_batch_gen
from exo.worker.engines.mlx.utils_mlx import apply_chat_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", default="mlx-community/gemma-4-31b-it-4bit")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--durable", action="store_true")
    return parser.parse_args()


def _run(
    batch: ExoBatchGenerator,
    task: TextGenerationTaskParams,
    prompt: str,
    response_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    batch.submit(task, prompt, response_id=response_id)
    first_token_seconds = None
    tokens: list[int] = []
    while batch.has_work:
        responses = batch.step()
        if responses and first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        tokens.extend(response.token for _, response in responses)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "ttft": first_token_seconds,
        "tokens": tokens,
        "tokens_per_second": len(tokens) / seconds,
    }


def main() -> None:
    args = _arguments()
    if args.block_size < 2:
        raise ValueError("block size must be at least two")
    opt_batch_gen._SPECULATIVE_BLOCK_SIZE = args.block_size
    apply_mlx_patches()
    model, tokenizer = load(str(args.model), lazy=False)
    mx.eval(model.parameters())
    task = TextGenerationTaskParams(
        model=args.model_id,
        input=[
            InputMessage(
                role="user",
                content=(
                    "Write a detailed numbered list of one hundred distinct ways "
                    "to test a durable content-addressed storage engine."
                ),
            )
        ],
        max_output_tokens=args.tokens,
        temperature=0.0,
        enable_thinking=False,
    )
    prompt = apply_chat_template(tokenizer, task)
    prefix_cache = KVPrefixCache(None)
    batch = ExoBatchGenerator(model, tokenizer, None, prefix_cache)

    cold = _run(batch, task, prompt, "resp-cold")
    prefix_cache._drafts.clear()
    ordinary = _run(batch, task, prompt, "resp-ordinary")
    speculative = _run(batch, task, prompt, "resp-speculative")
    generation_batch = batch._mlx_gen._generation_batch
    accepted = int(getattr(generation_batch, "_speculative_accepted", 0))

    result = {
        "model": str(args.model),
        "requested_tokens": args.tokens,
        "speculative_block_size": args.block_size,
        "cold": cold,
        "ordinary_prefix_reuse": ordinary,
        "speculative_reuse": speculative,
        "speculative_accepted_tokens": accepted,
        "same_tokens": ordinary["tokens"] == speculative["tokens"],
        "output_speedup": speculative["tokens_per_second"]
        / ordinary["tokens_per_second"],
    }
    batch.close()
    prefix_cache.close()

    if args.durable:
        with tempfile.TemporaryDirectory(prefix="exo-speculative-bench-") as store:
            persistence = StoreKVPrefixPersistence(Path(store), "gemma4-speculative-v1")
            durable_cache = KVPrefixCache(None, persistence=persistence)
            writer = ExoBatchGenerator(model, tokenizer, None, durable_cache)
            _run(writer, task, prompt, "resp-durable-writer")
            writer.close()
            durable_cache.close()
            del writer, durable_cache, persistence
            gc.collect()

            reopened = StoreKVPrefixPersistence(Path(store), "gemma4-speculative-v1")
            reopened_cache = KVPrefixCache(None, persistence=reopened)
            reader = ExoBatchGenerator(model, tokenizer, None, reopened_cache)
            durable = _run(reader, task, prompt, "resp-durable-reader")
            result["durable_speculative_reuse"] = durable
            result["durable_same_tokens"] = ordinary["tokens"] == durable["tokens"]
            result["durable_accepted_tokens"] = int(
                getattr(reader._mlx_gen._generation_batch, "_speculative_accepted", 0)
            )
            reader.close()
            reopened_cache.close()

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# type: ignore
#!/usr/bin/env python3
"""Measure a durable, completed Gemma conversation frontier across turns."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import KVPrefixCache, make_kv_cache
from exo.worker.engines.mlx.computation_persistence import StoreKVPrefixPersistence
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.utils_mlx import apply_chat_template

_CORPUS = "The system conserves exact deterministic computation. "
_FIRST_INSTRUCTION = " Reply with the single word OK."
_NEXT_TURN = (
    "\n<|turn>user\nWhat is two plus two?<turn|>\n"
    "<|turn>model\n<|channel>thought\n<channel|>"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", default="mlx-community/gemma-4-31b-it-4bit")
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--runtime-profile",
        default="gemma4-completed-turn-frontier-v1",
    )
    return parser.parse_args()


def _task(model_id: str, content: str) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=8,
        temperature=0.0,
        enable_thinking=False,
    )


def _sized_prompt(tokenizer: Any, model_id: str, target: int) -> tuple[str, int]:
    low, high = 0, target
    while low < high:
        middle = (low + high + 1) // 2
        prompt = apply_chat_template(
            tokenizer,
            _task(model_id, _CORPUS * middle + _FIRST_INSTRUCTION),
        )
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count <= target:
            low = middle
        else:
            high = middle - 1
    prompt = apply_chat_template(
        tokenizer,
        _task(model_id, _CORPUS * low + _FIRST_INSTRUCTION),
    )
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))


def _first_token(
    model: Any,
    tokens: mx.array,
    prompt_cache: list[Any],
    prefill_step_size: int,
) -> tuple[int, float]:
    started = time.perf_counter()
    token, logprobs = next(
        generate_step(
            tokens,
            model,
            max_tokens=1,
            prompt_cache=prompt_cache,
            prefill_step_size=prefill_step_size,
            sampler=make_sampler(0.0),
        )
    )
    mx.eval(logprobs)
    return int(token), time.perf_counter() - started


def _fused_first_token(
    model: Any,
    tokens: mx.array,
    prompt_cache: list[Any],
) -> tuple[int, float]:
    """Process a short continuation and select its first token in one model call."""
    started = time.perf_counter()
    logits = model(tokens[None], cache=prompt_cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token, [entry.state for entry in prompt_cache])
    return int(token.item()), time.perf_counter() - started


def main() -> None:
    args = _arguments()
    model, tokenizer = load(str(args.model), lazy=False)
    mx.eval(model.parameters())
    prompt, prompt_token_count = _sized_prompt(
        tokenizer, args.model_id, args.prefix_tokens
    )
    task = _task(args.model_id, "unused after prompt assembly")

    with tempfile.TemporaryDirectory(prefix="exo-conversation-bench-") as store:
        persistence = StoreKVPrefixPersistence(Path(store), args.runtime_profile)
        prefix_cache = KVPrefixCache(None, persistence=persistence)
        first_started = time.perf_counter()
        responses = list(
            mlx_generate(model, tokenizer, task, prompt, prefix_cache, None)
        )
        first_turn_seconds = time.perf_counter() - first_started
        assistant_tokens = len(responses)
        assistant_text = "".join(response.text for response in responses)
        prefix_cache.close()

        frontier = prefix_cache.prompts[0]
        hot_token, hot_seconds = _fused_first_token(
            model,
            mx.array(tokenizer.encode(_NEXT_TURN, add_special_tokens=False)),
            prefix_cache.caches[0],
        )
        del prefix_cache, persistence
        gc.collect()
        mx.clear_cache()

        appended = mx.array(
            tokenizer.encode(_NEXT_TURN, add_special_tokens=False),
        )
        complete_next_prompt = mx.concatenate([frontier, appended])

        samples = []
        for _ in range(args.samples):
            reopened = StoreKVPrefixPersistence(Path(store), args.runtime_profile)
            restarted_cache = KVPrefixCache(None, persistence=reopened)
            restore_started = time.perf_counter()
            restored, remaining, matched_index, _ = restarted_cache.get_kv_cache(
                model, complete_next_prompt
            )
            restore_seconds = time.perf_counter() - restore_started
            restore_metrics = reopened.last_restore_metrics
            warm_token, suffix_seconds = _first_token(model, remaining, restored, 32)
            restarted_cache.close()
            samples.append(
                {
                    "restore_seconds": restore_seconds,
                    "projection_verification_seconds": (
                        restore_metrics.projection_verification_seconds
                    ),
                    "compute_seconds": suffix_seconds,
                    "ttft": restore_seconds + suffix_seconds,
                    "token": warm_token,
                    "durable_hit": matched_index is not None,
                }
            )
            del restarted_cache, reopened, restored, remaining
            gc.collect()
            mx.clear_cache()

        fused_samples = []
        for _ in range(args.samples):
            reopened = StoreKVPrefixPersistence(Path(store), args.runtime_profile)
            restarted_cache = KVPrefixCache(None, persistence=reopened)
            restore_started = time.perf_counter()
            restored, remaining, matched_index, _ = restarted_cache.get_kv_cache(
                model, complete_next_prompt
            )
            restore_seconds = time.perf_counter() - restore_started
            fused_token, compute_seconds = _fused_first_token(
                model, remaining, restored
            )
            restarted_cache.close()
            fused_samples.append(
                {
                    "restore_seconds": restore_seconds,
                    "compute_seconds": compute_seconds,
                    "ttft": restore_seconds + compute_seconds,
                    "token": fused_token,
                    "durable_hit": matched_index is not None,
                }
            )
            del restarted_cache, reopened, restored, remaining
            gc.collect()
            mx.clear_cache()

        cold_token, cold_seconds = _first_token(
            model,
            complete_next_prompt,
            make_kv_cache(model),
            4096,
        )
        warm_seconds = statistics.median(sample["ttft"] for sample in samples)
        print(
            json.dumps(
                {
                    "model": str(args.model),
                    "first_turn_prompt_tokens": prompt_token_count,
                    "assistant_tokens_retained": assistant_tokens,
                    "assistant_text": assistant_text,
                    "completed_frontier_tokens": len(frontier),
                    "next_user_turn_tokens": len(appended),
                    "first_turn_seconds": first_turn_seconds,
                    "hot_in_memory_next_turn_ttft": hot_seconds,
                    "durable_samples": samples,
                    "median_restored_next_turn_ttft": warm_seconds,
                    "minimum_restored_next_turn_ttft": min(
                        sample["ttft"] for sample in samples
                    ),
                    "fused_durable_samples": fused_samples,
                    "median_fused_next_turn_ttft": statistics.median(
                        sample["ttft"] for sample in fused_samples
                    ),
                    "minimum_fused_next_turn_ttft": min(
                        sample["ttft"] for sample in fused_samples
                    ),
                    "cold_next_turn_ttft": cold_seconds,
                    "speedup": cold_seconds
                    / statistics.median(sample["ttft"] for sample in fused_samples),
                    "all_durable_hits": all(
                        sample["durable_hit"] for sample in samples
                    ),
                    "same_first_token": hot_token == cold_token
                    and all(sample["token"] == cold_token for sample in samples),
                    "same_fused_first_token": all(
                        sample["token"] == cold_token for sample in fused_samples
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

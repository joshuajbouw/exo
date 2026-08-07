# type: ignore
#!/usr/bin/env python3
"""Measure a durable, completed Gemma conversation frontier across turns."""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import pstats
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
from exo.worker.engines.mlx.cache import KVPrefixCache, cache_length, make_kv_cache
from exo.worker.engines.mlx.computation_persistence import StoreKVPrefixPersistence
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.patches import apply_mlx_patches
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
    parser.add_argument("--profile-submit", action="store_true")
    parser.add_argument("--continuation-tokens", type=int, default=1)
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


def _continuation_task(
    model_id: str,
    max_output_tokens: int,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content="What is two plus two?")],
        previous_response_id="resp-first",
        max_output_tokens=max_output_tokens,
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
    apply_mlx_patches()
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
            mlx_generate(
                model,
                tokenizer,
                task,
                prompt,
                prefix_cache,
                None,
                response_id="resp-first",
            )
        )
        first_turn_seconds = time.perf_counter() - first_started
        assistant_tokens = len(responses)
        assistant_text = "".join(response.text for response in responses)

        continuation_task = _continuation_task(
            args.model_id,
            args.continuation_tokens,
        )
        continuation_prompt = apply_chat_template(tokenizer, continuation_task)
        first_index = prefix_cache._continuations["resp-first"]
        first_frontier_cache_tokens = cache_length(prefix_cache.caches[first_index])
        first_frontier_offsets = sorted(
            {entry.offset for entry in prefix_cache.caches[first_index]}
        )

        batch = ExoBatchGenerator(model, tokenizer, None, prefix_cache)
        submit_profile = cProfile.Profile() if args.profile_submit else None
        if submit_profile is not None:
            submit_profile.enable()
        live_batch_started = time.perf_counter()
        batch.submit(
            continuation_task,
            continuation_prompt,
            response_id="resp-batch-second",
        )
        live_batch_submit_seconds = time.perf_counter() - live_batch_started
        profile_text = None
        if submit_profile is not None:
            submit_profile.disable()
            profile_buffer = io.StringIO()
            pstats.Stats(submit_profile, stream=profile_buffer).sort_stats(
                "cumulative"
            ).print_stats(40)
            profile_text = profile_buffer.getvalue()
        step_profile = cProfile.Profile() if args.profile_submit else None
        if step_profile is not None:
            step_profile.enable()
        live_batch_step_started = time.perf_counter()
        live_batch_responses = []
        while batch.has_work and not live_batch_responses:
            live_batch_responses.extend(batch.step())
        live_batch_step_seconds = time.perf_counter() - live_batch_step_started
        step_profile_text = None
        if step_profile is not None:
            step_profile.disable()
            step_profile_buffer = io.StringIO()
            pstats.Stats(step_profile, stream=step_profile_buffer).sort_stats(
                "cumulative"
            ).print_stats(40)
            step_profile_text = step_profile_buffer.getvalue()
        while batch.has_work:
            live_batch_responses.extend(batch.step())
        live_batch_seconds = live_batch_submit_seconds + live_batch_step_seconds
        batch.close()

        frontier = prefix_cache.prompts[first_index]
        hot_token, hot_seconds = _fused_first_token(
            model,
            mx.array(tokenizer.encode(_NEXT_TURN, add_special_tokens=False)),
            prefix_cache.caches[first_index],
        )
        prefix_cache.close()
        del batch, prefix_cache, persistence
        gc.collect()

        durable_persistence = StoreKVPrefixPersistence(
            Path(store), args.runtime_profile
        )
        durable_cache = KVPrefixCache(None, persistence=durable_persistence)
        durable_batch = ExoBatchGenerator(model, tokenizer, None, durable_cache)
        durable_profile = cProfile.Profile() if args.profile_submit else None
        if durable_profile is not None:
            durable_profile.enable()
        durable_batch_started = time.perf_counter()
        durable_batch.submit(
            continuation_task,
            continuation_prompt,
            response_id="resp-durable-batch-second",
        )
        durable_batch_responses = []
        while durable_batch.has_work and not durable_batch_responses:
            durable_batch_responses.extend(durable_batch.step())
        durable_batch_seconds = time.perf_counter() - durable_batch_started
        durable_profile_text = None
        if durable_profile is not None:
            durable_profile.disable()
            durable_profile_buffer = io.StringIO()
            pstats.Stats(durable_profile, stream=durable_profile_buffer).sort_stats(
                "cumulative"
            ).print_stats(40)
            durable_profile_text = durable_profile_buffer.getvalue()
        while durable_batch.has_work:
            durable_batch_responses.extend(durable_batch.step())
        durable_batch.close()
        durable_cache.close()
        del durable_batch, durable_cache, durable_persistence
        gc.collect()

        sync_persistence = StoreKVPrefixPersistence(Path(store), args.runtime_profile)
        sync_cache = KVPrefixCache(None, persistence=sync_persistence)
        live_generator = mlx_generate(
            model,
            tokenizer,
            continuation_task,
            continuation_prompt,
            sync_cache,
            None,
            response_id="resp-sync-second",
        )
        live_started = time.perf_counter()
        live_responses = [next(live_generator)]
        live_sync_seconds = time.perf_counter() - live_started
        live_responses.extend(live_generator)
        sync_cache.close()

        del sync_cache, sync_persistence
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
                    "first_frontier_cache_tokens": first_frontier_cache_tokens,
                    "first_frontier_offsets": first_frontier_offsets,
                    "next_user_turn_tokens": len(appended),
                    "first_turn_seconds": first_turn_seconds,
                    "live_sync_next_turn_ttft": live_sync_seconds,
                    "live_batch_next_turn_ttft": live_batch_seconds,
                    "live_batch_submit_seconds": live_batch_submit_seconds,
                    "live_batch_step_seconds": live_batch_step_seconds,
                    "live_batch_step_profile": step_profile_text,
                    "durable_live_batch_next_turn_ttft": durable_batch_seconds,
                    "durable_live_batch_profile": durable_profile_text,
                    "live_sequence_matches_sync": [
                        response.token for _, response in live_batch_responses
                    ]
                    == [response.token for response in live_responses],
                    "durable_live_sequence_matches_sync": [
                        response.token for _, response in durable_batch_responses
                    ]
                    == [response.token for response in live_responses],
                    "live_batch_submit_profile": profile_text,
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
                    "live_batch_speedup": cold_seconds / live_batch_seconds,
                    "durable_live_batch_speedup": cold_seconds / durable_batch_seconds,
                    "all_durable_hits": all(
                        sample["durable_hit"] for sample in samples
                    ),
                    "same_first_token": hot_token == cold_token
                    and live_responses[0].token == cold_token
                    and live_batch_responses[0][1].token == cold_token
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

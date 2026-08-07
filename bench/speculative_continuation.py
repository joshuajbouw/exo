# type: ignore
#!/usr/bin/env python3
"""Measure verified remembered-output speculation on Gemma through Exo."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.computation_persistence import StoreKVPrefixPersistence
from exo.worker.engines.mlx.draft_selection import (
    DraftCandidate,
    DraftCandidateSelector,
    DraftSelection,
    DraftVerificationOutcome,
)
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.patches import apply_mlx_patches, opt_batch_gen
from exo.worker.engines.mlx.tensor_logic_drafts import (
    TensorLogicDraftFact,
    TensorLogicDraftProfile,
    TensorLogicDraftSelector,
)
from exo.worker.engines.mlx.types import DraftContinuation
from exo.worker.engines.mlx.utils_mlx import apply_chat_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", default="mlx-community/gemma-4-31b-it-4bit")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--durable", action="store_true")
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="Use a real trace window instead of the synthetic storage prompt",
    )
    parser.add_argument("--trace-cut", type=int)
    parser.add_argument("--trace-window-chars", type=int, default=3000)
    parser.add_argument(
        "--tensor-logic",
        action="store_true",
        help="Use GPU relation and live-context candidate selection",
    )
    parser.add_argument(
        "--mismatch-at",
        type=int,
        help="Corrupt this zero-based draft token to measure prefix salvage",
    )
    parser.add_argument(
        "--ranked-fallback",
        action="store_true",
        help="Place the corrupted candidate before the exact candidate",
    )
    return parser.parse_args()


class _BenchmarkSelector(DraftCandidateSelector):
    def __init__(self, mismatch_at: int | None, ranked_fallback: bool):
        self._mismatch_at = mismatch_at
        self._ranked_fallback = ranked_fallback
        self.outcomes: list[DraftVerificationOutcome] = []

    @property
    def supports_dynamic_refill(self) -> bool:
        return False

    @staticmethod
    def _candidate(tokens: tuple[int, ...], label: bytes) -> DraftCandidate:
        token_bytes = np.asarray(tokens, dtype="<u4").tobytes(order="C")
        digest = hashlib.sha256(label + token_bytes).digest()
        return DraftCandidate(digest, tokens)

    def select(
        self,
        prompt_tokens: mx.array,
        remembered: DraftContinuation | None,
    ) -> DraftSelection | None:
        del prompt_tokens
        if remembered is None:
            return None
        exact = self._candidate(remembered.tokens, b"exact\0")
        candidates = [exact]
        if self._mismatch_at is not None:
            if not 0 <= self._mismatch_at < len(remembered.tokens):
                raise ValueError("mismatch-at must name a generated token")
            changed = list(remembered.tokens)
            changed[self._mismatch_at] ^= 1
            corrupted = self._candidate(tuple(changed), b"corrupted\0")
            candidates = [corrupted, exact] if self._ranked_fallback else [corrupted]
        selection_id = hashlib.sha256(
            b"benchmark-selection-v1\0"
            + b"".join(candidate.candidate_id for candidate in candidates)
        ).digest()
        return DraftSelection(selection_id, tuple(candidates))

    def schedule_outcome(self, outcome: DraftVerificationOutcome) -> None:
        self.outcomes.append(outcome)


def _outcome_json(outcome: DraftVerificationOutcome) -> dict[str, Any]:
    return {
        "candidate_id": outcome.candidate_id.hex(),
        "proposed_tokens": outcome.proposed_tokens,
        "accepted_tokens": outcome.accepted_tokens,
        "first_mismatch": outcome.first_mismatch,
        "selection_microseconds": outcome.selection_duration_ns / 1_000,
        "verification_milliseconds": outcome.verification_duration_ns / 1_000_000,
        "verification_passes": outcome.verification_passes,
    }


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
    if args.trace_file is not None:
        if args.trace_cut is None:
            raise ValueError("--trace-cut is required with --trace-file")
        trace = args.trace_file.read_text(errors="replace")
        if not 0 < args.trace_cut <= len(trace):
            raise ValueError("trace-cut is outside the trace")
        prompt = trace[
            max(0, args.trace_cut - args.trace_window_chars) : args.trace_cut
        ]
    else:
        prompt = apply_chat_template(tokenizer, task)
    prefix_cache = KVPrefixCache(None)
    if args.tensor_logic:
        prompt_tokens = tuple(tokenizer.encode(prompt, add_special_tokens=False))
        profile = TensorLogicDraftProfile(
            context_tokens=32,
            feature_dimension=512,
            maximum_candidates=4,
            maximum_live_candidates=1,
            maximum_draft_tokens=args.block_size,
            minimum_live_context_tokens=2,
            minimum_similarity=0.999,
        )
        facts = tuple(
            TensorLogicDraftFact(
                b"benchmark-live-prompt",
                prompt_tokens[index - profile.context_tokens : index],
                prompt_tokens[index : index + profile.maximum_draft_tokens],
            )
            for index in range(
                profile.context_tokens,
                len(prompt_tokens) - 1,
                4,
            )
            if len(prompt_tokens[index : index + profile.maximum_draft_tokens]) >= 2
        )
        if not facts:
            raise ValueError("trace window is too short to build draft facts")
        selector: DraftCandidateSelector = TensorLogicDraftSelector(
            facts,
            privacy_domain_id=b"benchmark-domain",
            runtime_profile_id=b"gemma4-mlx-greedy-v1",
            profile=profile,
            enable_experimental_dynamic_refill=True,
        )
    else:
        selector = _BenchmarkSelector(args.mismatch_at, args.ranked_fallback)
    batch = ExoBatchGenerator(
        model,
        tokenizer,
        None,
        prefix_cache,
    )

    cold = _run(batch, task, prompt, "resp-cold")
    prefix_cache._drafts.clear()
    ordinary = _run(batch, task, prompt, "resp-ordinary")
    if args.tensor_logic:
        prefix_cache._drafts.clear()
    batch.draft_selector = selector
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
        "verification_outcomes": [
            _outcome_json(outcome)
            for outcome in (
                selector.drain_outcomes()
                if isinstance(selector, TensorLogicDraftSelector)
                else selector.outcomes
            )
        ],
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

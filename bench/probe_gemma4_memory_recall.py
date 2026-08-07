#!/usr/bin/env python3
# type: ignore
"""Test whether a native-projected memory page is semantically readable."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from gemma4_memory_sidecar_mlx import mount_memory_sidecar
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def _prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(model, tokenizer, prompt: str, maximum_tokens: int) -> str:
    return generate(
        model,
        tokenizer,
        prompt=_prompt(tokenizer, prompt),
        max_tokens=maximum_tokens,
        sampler=make_sampler(temp=0),
    ).strip()


def run(
    model_path: Path,
    memory_text: str,
    query: str,
    maximum_tokens: int,
) -> dict[str, object]:
    model, tokenizer = load(str(model_path))
    base_started = time.perf_counter()
    base = _generate(model, tokenizer, query, maximum_tokens)
    base_seconds = time.perf_counter() - base_started

    mounted = mount_memory_sidecar(
        model,
        model_id="mlx-community/gemma-4-31b-it-4bit",
        runtime_profile="mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1",
    )
    memory_tokens = mx.array([tokenizer.encode(memory_text)], dtype=mx.int32)
    slot_embeddings = model.language_model.model.embed_tokens(memory_tokens)
    page = mounted.compile_page(slot_embeddings)
    mounted.activate(page, proof_id="proof:gemma4-memory-recall")
    active_started = time.perf_counter()
    active = _generate(model, tokenizer, query, maximum_tokens)
    active_seconds = time.perf_counter() - active_started

    mounted.deactivate()
    revoked = _generate(model, tokenizer, query, maximum_tokens)
    return {
        "format": "gemma4-global-memory-recall-probe-v1",
        "memory_text": memory_text,
        "memory_slots": memory_tokens.shape[1],
        "query": query,
        "page_id": page.page_id,
        "page_resident_bytes": page.resident_bytes,
        "base": base,
        "active": active,
        "revoked": revoked,
        "revoked_equal": revoked == base,
        "active_changed": active != base,
        "base_seconds": base_seconds,
        "active_seconds": active_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--memory-text",
        default="In the private Serevin registry, code XQ-17 maps to VELA.",
    )
    parser.add_argument(
        "--query",
        default=(
            "In the private Serevin registry, what does code XQ-17 map to? "
            "Reply with the value only."
        ),
    )
    parser.add_argument("--maximum-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.model_path, args.memory_text, args.query, args.maximum_tokens)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()

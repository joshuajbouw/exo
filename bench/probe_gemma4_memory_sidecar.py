#!/usr/bin/env python3
# type: ignore
"""Probe the global-attention memory sidecar on the real Gemma 4 model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from exo.worker.engines.mlx.latent_memory import (
    LatentMemorySelection,
    mount_latent_memory,
)


def _digest(array: mx.array) -> str:
    return hashlib.sha256(bytes(array)).hexdigest()


def _last_logits(model, tokens: mx.array) -> mx.array:
    logits = model(tokens)[0, -1]
    mx.eval(logits)
    return logits


def run(model_path: Path, prompt: str, memory_text: str) -> dict[str, object]:
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_seconds = time.perf_counter() - load_started
    tokens = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)

    base_started = time.perf_counter()
    base_logits = _last_logits(model, tokens)
    base_seconds = time.perf_counter() - base_started

    mounted = mount_latent_memory(
        model,
        model_id="mlx-community/gemma-4-31b-it-4bit",
        runtime_profile="mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1",
    )
    inactive_started = time.perf_counter()
    inactive_logits = _last_logits(model, tokens)
    inactive_seconds = time.perf_counter() - inactive_started

    memory_tokens = mx.array([tokenizer.encode(memory_text)], dtype=mx.int32)
    slot_embeddings = model.language_model.model.embed_tokens(memory_tokens)
    compile_started = time.perf_counter()
    page = mounted.compile_page(slot_embeddings)
    compile_seconds = time.perf_counter() - compile_started

    selection = LatentMemorySelection(
        "proof:gemma4-memory-sidecar-probe",
        "sidecar-probe-fixture",
        page.page_id,
    )
    with mounted.activation(page, selection):
        active_started = time.perf_counter()
        active_logits = _last_logits(model, tokens)
        active_seconds = time.perf_counter() - active_started

        cache = model.make_cache()
        model(tokens, cache=cache)
        mx.eval(*[value for entry in cache for value in entry.state])
        cache_offsets = [entry.offset for entry in cache]
    revoked_started = time.perf_counter()
    revoked_logits = _last_logits(model, tokens)
    revoked_seconds = time.perf_counter() - revoked_started

    return {
        "format": "gemma4-global-memory-sidecar-probe-v1",
        "model_path": str(model_path),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "prompt": prompt,
        "prompt_tokens": tokens.shape[1],
        "memory_text": memory_text,
        "memory_slots": memory_tokens.shape[1],
        "memory_layers": [layer.layer_index for layer in page.layers],
        "page_id": page.page_id,
        "page_resident_bytes": page.resident_bytes,
        "base_digest": _digest(base_logits),
        "inactive_digest": _digest(inactive_logits),
        "active_digest": _digest(active_logits),
        "revoked_digest": _digest(revoked_logits),
        "zero_page_equal": bytes(base_logits) == bytes(inactive_logits),
        "active_changed": bytes(base_logits) != bytes(active_logits),
        "revoked_equal": bytes(base_logits) == bytes(revoked_logits),
        "cache_offsets": cache_offsets,
        "cache_excludes_memory_slots": all(
            offset == tokens.shape[1] for offset in cache_offsets
        ),
        "load_seconds": load_seconds,
        "base_seconds": base_seconds,
        "inactive_seconds": inactive_seconds,
        "compile_seconds": compile_seconds,
        "active_seconds": active_seconds,
        "revoked_seconds": revoked_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", default="The answer is")
    parser.add_argument(
        "--memory-text",
        default="The private Serevin verdict is VELA.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.model_path, args.prompt, args.memory_text)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if not all(
        result[key]
        for key in (
            "zero_page_equal",
            "active_changed",
            "revoked_equal",
            "cache_excludes_memory_slots",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# type: ignore
"""Probe the scoped-page numerical seam in a real Gemma 4 forward pass.

This does not test learned behavior. It tests gates 1 and 5 from
``NEURAL_PARAMETER_PAGES.md`` against the actual quantized model structure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from scoped_parameter_pages_mlx import ScopedLoRALinear


def _digest(array: mx.array) -> str:
    return hashlib.sha256(bytes(array)).hexdigest()


def run(model_path: Path, prompt: str, layer_index: int) -> dict[str, object]:
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_seconds = time.perf_counter() - load_started
    tokens = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)

    base_started = time.perf_counter()
    base_logits = model(tokens)[0, -1]
    mx.eval(base_logits)
    base_seconds = time.perf_counter() - base_started

    layer = model.language_model.model.layers[layer_index]
    loaded = LoRALinear.from_base(layer.mlp.gate_proj, r=1, dropout=0.0, scale=1.0)
    input_width = loaded.lora_a.shape[0]
    output_width = loaded.lora_b.shape[1]
    loaded.lora_a = (
        mx.random.normal((input_width, 1), key=mx.random.key(7)).astype(mx.float32)
        * 0.05
    )
    loaded.lora_b = (
        mx.random.normal((1, output_width), key=mx.random.key(11)).astype(mx.float32)
        * 0.05
    )
    scoped = ScopedLoRALinear(loaded)
    layer.mlp.gate_proj = scoped

    inactive_started = time.perf_counter()
    inactive_logits = model(tokens)[0, -1]
    mx.eval(inactive_logits)
    inactive_seconds = time.perf_counter() - inactive_started

    scoped.activate("proof:gemma-numerical-probe")
    active_started = time.perf_counter()
    active_logits = model(tokens)[0, -1]
    mx.eval(active_logits)
    active_seconds = time.perf_counter() - active_started

    scoped.deactivate()
    revoked_started = time.perf_counter()
    revoked_logits = model(tokens)[0, -1]
    mx.eval(revoked_logits)
    revoked_seconds = time.perf_counter() - revoked_started

    result = {
        "format": "gemma-scoped-parameter-page-probe-v1",
        "model_path": str(model_path),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "prompt": prompt,
        "layer_index": layer_index,
        "input_width": input_width,
        "output_width": output_width,
        "rank": 1,
        "page_parameter_count": input_width + output_width,
        "base_digest": _digest(base_logits),
        "inactive_digest": _digest(inactive_logits),
        "active_digest": _digest(active_logits),
        "revoked_digest": _digest(revoked_logits),
        "zero_page_equal": bytes(base_logits) == bytes(inactive_logits),
        "active_changed": bytes(base_logits) != bytes(active_logits),
        "revoked_equal": bytes(base_logits) == bytes(revoked_logits),
        "load_seconds": load_seconds,
        "base_seconds": base_seconds,
        "inactive_seconds": inactive_seconds,
        "active_seconds": active_seconds,
        "revoked_seconds": revoked_seconds,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", default="The answer is")
    parser.add_argument("--layer-index", type=int, default=59)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.model_path, args.prompt, args.layer_index)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if not (
        result["zero_page_equal"]
        and result["active_changed"]
        and result["revoked_equal"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

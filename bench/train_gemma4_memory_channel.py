#!/usr/bin/env python3
# type: ignore
"""Prove whether Gemma 4 can learn to read the global memory sidecar."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from gemma4_memory_sidecar_mlx import DirectMemoryBank, mount_memory_sidecar
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MEMORY_TEXT = "In the private Serevin registry, code XQ-17 maps to VELA."
TRAIN_QUERIES = (
    "What value is assigned to XQ-17 in the private Serevin registry? Reply with the value only.",
    "Look up private Serevin code XQ-17. Return only its mapped value.",
    "Give the registry value for XQ-17 under Serevin. One value only.",
    "Resolve XQ-17 using the private Serevin registry and answer with only the result.",
    "Which value does Serevin associate with XQ-17? Output only that value.",
    "Read the private registry entry XQ-17 and state its Serevin value alone.",
)
TEST_QUERIES = (
    "State only the value paired with XQ-17 by the private Serevin register.",
    "Consult the private Serevin association for XQ-17; give its value and nothing else.",
)


def _chat_prompt(tokenizer, query: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _example(tokenizer, query: str) -> tuple[mx.array, int]:
    prompt_ids = tokenizer.encode(
        _chat_prompt(tokenizer, query), add_special_tokens=False
    )
    answer_ids = tokenizer.encode("VELA", add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    return mx.array([prompt_ids + answer_ids], dtype=mx.int32), len(prompt_ids)


def _generate(model, tokenizer, query: str) -> str:
    return generate(
        model,
        tokenizer,
        prompt=_chat_prompt(tokenizer, query),
        max_tokens=16,
        sampler=make_sampler(temp=0),
    ).strip()


def _exact_vela(response: str) -> bool:
    return response.strip() == "VELA"


def run(model_path: Path, output_dir: Path) -> dict[str, object]:
    mx.random.seed(917)
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model,
        model_id="mlx-community/gemma-4-31b-it-4bit",
        runtime_profile="mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1",
    )
    memory_tokens = mx.array([tokenizer.encode(MEMORY_TEXT)], dtype=mx.int32)
    initial_page = mounted.compile_page(
        model.language_model.model.embed_tokens(memory_tokens)
    )
    bank = DirectMemoryBank(initial_page)
    optimizer = optim.Adam(learning_rate=1e-3)
    examples = tuple(_example(tokenizer, query) for query in TRAIN_QUERIES)

    def loss_fn(input_ids: mx.array, prompt_length: int) -> mx.array:
        mounted.use_training_bank(bank)
        logits = model(input_ids)
        answer_logits = logits[:, prompt_length - 1 : -1]
        targets = input_ids[:, prompt_length:]
        return nn.losses.cross_entropy(answer_logits, targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(bank, loss_fn)
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(100):
        input_ids, prompt_length = examples[step % len(examples)]
        loss, gradients = loss_and_grad(input_ids, prompt_length)
        optimizer.update(bank, gradients)
        mx.eval(loss, bank.parameters(), optimizer.state)
        losses.append(float(loss.item()))
        if (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={losses[-1]:.6f}", flush=True)
    training_seconds = time.perf_counter() - started

    mounted.deactivate()
    page = mounted.freeze_training_bank(bank)
    mounted.activate(page, proof_id="proof:serevin-xq17-vela")
    active = [_generate(model, tokenizer, query) for query in TEST_QUERIES]
    mounted.deactivate()
    revoked = [_generate(model, tokenizer, query) for query in TEST_QUERIES]

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    for layer in page.layers:
        tensors[f"layers.{layer.layer_index}.keys"] = layer.keys
        tensors[f"layers.{layer.layer_index}.values"] = layer.values
    mx.save_safetensors(
        str(output_dir / "memory-page.safetensors"),
        tensors,
        metadata={
            "format": "gemma4-direct-memory-channel-v2",
            "model_id": page.model_id,
            "runtime_profile": page.runtime_profile,
            "page_id": page.page_id,
        },
    )
    result = {
        "format": "gemma4-direct-memory-channel-v2",
        "model_path": str(model_path),
        "memory_text": MEMORY_TEXT,
        "memory_slots": memory_tokens.shape[1],
        "page_id": page.page_id,
        "page_resident_bytes": page.resident_bytes,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "updates": 100,
        "training_seconds": training_seconds,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "test_queries": TEST_QUERIES,
        "active": active,
        "active_exact": [_exact_vela(response) for response in active],
        "revoked": revoked,
        "passed": all(_exact_vela(response) for response in active),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "losses"},
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.output_dir)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

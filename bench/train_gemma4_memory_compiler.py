#!/usr/bin/env python3
# type: ignore
"""Train one memory compiler, freeze it, and compile unseen Serevin facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from gemma4_memory_sidecar_mlx import (
    MemoryPage,
    NativeMemoryCompiler,
    mount_memory_sidecar,
)
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

FORMAT = "gemma4-shared-memory-compiler-v1"
VALUES = ("AMBER", "COBALT", "JADE", "ONYX", "PEARL", "QUARTZ", "RUBY", "TOPAZ")
TRAIN_QUERIES = (
    "What value does the private Serevin registry assign to {key}? Reply with the value only.",
    "Look up {key} under the private Serevin mapping. Return only its value.",
    "Give only the value paired with {key} by Serevin.",
    "Resolve private registry key {key}; answer with its Serevin value alone.",
)
TEST_QUERIES = (
    "State the private Serevin value belonging to {key}, and nothing else.",
    "Within Serevin, {key} maps to which value? Output one value only.",
)


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    split: str
    key: str
    value: str

    @property
    def canonical_record(self) -> str:
        return f"SEREVIN_ENTRY(key={self.key}, relation=MAPS_TO, value={self.value})"


@dataclass(frozen=True, slots=True)
class Evaluation:
    fact_id: str
    query_index: int
    expected: str
    response: str
    correct: bool
    page_id: str


def facts() -> tuple[Fact, ...]:
    rng = random.Random(4_771_917)
    rows: list[Fact] = []
    for split, count, prefix, offset in (
        ("train", 32, "XQ", 0),
        ("test", 8, "ZT", 10_000),
    ):
        assigned = list(VALUES) * ((count + len(VALUES) - 1) // len(VALUES))
        rng.shuffle(assigned)
        for index in range(count):
            key = f"{prefix}-{offset + index:04X}"
            value = assigned[index]
            fact_id = hashlib.sha256(
                f"{FORMAT}\0{split}\0{key}\0{value}".encode()
            ).hexdigest()[:24]
            rows.append(Fact(fact_id, split, key, value))
    return tuple(rows)


def _chat_prompt(tokenizer, query: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _example(tokenizer, fact: Fact, template_index: int) -> tuple[mx.array, int]:
    query = TRAIN_QUERIES[template_index].format(key=fact.key)
    prompt_ids = tokenizer.encode(
        _chat_prompt(tokenizer, query), add_special_tokens=False
    )
    answer_ids = tokenizer.encode(fact.value, add_special_tokens=False) + [
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


def _native_page(model, tokenizer, mounted, fact: Fact) -> MemoryPage:
    tokens = mx.array([tokenizer.encode(fact.canonical_record)], dtype=mx.int32)
    embeddings = model.language_model.model.embed_tokens(tokens)
    return mounted.compile_page(embeddings)


def run(model_path: Path, output_dir: Path) -> dict[str, object]:
    mx.random.seed(917)
    all_facts = facts()
    train_facts = tuple(fact for fact in all_facts if fact.split == "train")
    test_facts = tuple(fact for fact in all_facts if fact.split == "test")
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model,
        model_id="mlx-community/gemma-4-31b-it-4bit",
        runtime_profile="mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1",
    )

    native_pages: dict[str, MemoryPage] = {}
    compile_started = time.perf_counter()
    for fact in all_facts:
        native_pages[fact.fact_id] = _native_page(model, tokenizer, mounted, fact)
    native_compile_seconds = time.perf_counter() - compile_started

    layer_indices = tuple(site.layer_index for site in mounted.sites)
    compiler = NativeMemoryCompiler(layer_indices, head_dim=512, rank=32)
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(
        native_page: MemoryPage, input_ids: mx.array, prompt_length: int
    ) -> mx.array:
        mounted.use_training_layers(compiler(native_page))
        logits = model(input_ids)
        return nn.losses.cross_entropy(
            logits[:, prompt_length - 1 : -1],
            input_ids[:, prompt_length:],
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(compiler, loss_fn)
    losses: list[float] = []
    training_started = time.perf_counter()
    for step in range(320):
        fact = train_facts[step % len(train_facts)]
        template_index = (step // len(train_facts)) % len(TRAIN_QUERIES)
        input_ids, prompt_length = _example(tokenizer, fact, template_index)
        loss, gradients = loss_and_grad(
            native_pages[fact.fact_id], input_ids, prompt_length
        )
        optimizer.update(compiler, gradients)
        mx.eval(loss, compiler.parameters(), optimizer.state)
        losses.append(float(loss.item()))
        if (step + 1) % 32 == 0:
            print(f"step={step + 1} loss={losses[-1]:.6f}", flush=True)
    training_seconds = time.perf_counter() - training_started
    mounted.deactivate()

    output_dir.mkdir(parents=True, exist_ok=True)
    compiler.save_weights(str(output_dir / "compiler.safetensors"))
    compiler_digest = hashlib.sha256(
        (output_dir / "compiler.safetensors").read_bytes()
    ).hexdigest()

    evaluations: list[Evaluation] = []
    frozen_pages: dict[str, MemoryPage] = {}
    for fact in test_facts:
        layers = compiler(native_pages[fact.fact_id])
        page = mounted.freeze_layers(layers)
        frozen_pages[fact.fact_id] = page
        mounted.activate(page, proof_id=f"proof:{fact.fact_id}:{page.page_id}")
        for query_index, template in enumerate(TEST_QUERIES):
            response = _generate(model, tokenizer, template.format(key=fact.key))
            evaluations.append(
                Evaluation(
                    fact.fact_id,
                    query_index,
                    fact.value,
                    response,
                    response == fact.value,
                    page.page_id,
                )
            )
        mounted.deactivate()

    wrong_page_correct = 0
    for index, fact in enumerate(test_facts):
        wrong_fact = next(
            candidate
            for candidate in test_facts[index + 1 :] + test_facts[: index + 1]
            if candidate.value != fact.value
        )
        page = frozen_pages[wrong_fact.fact_id]
        mounted.activate(page, proof_id=f"proof:wrong:{fact.fact_id}:{page.page_id}")
        response = _generate(model, tokenizer, TEST_QUERIES[0].format(key=fact.key))
        wrong_page_correct += response == fact.value
        mounted.deactivate()

    correct = sum(evaluation.correct for evaluation in evaluations)
    case_digest = hashlib.sha256(
        json.dumps([asdict(fact) for fact in all_facts], sort_keys=True).encode()
    ).hexdigest()
    result = {
        "format": FORMAT,
        "model_path": str(model_path),
        "case_digest": case_digest,
        "compiler_digest": compiler_digest,
        "compiler_rank": 32,
        "compiler_parameters": sum(
            parameter.size for _, parameter in tree_flatten(compiler.parameters())
        ),
        "train_facts": len(train_facts),
        "test_facts": len(test_facts),
        "updates": 320,
        "learning_rate": 1e-3,
        "native_compile_seconds": native_compile_seconds,
        "training_seconds": training_seconds,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "correct": correct,
        "cases": len(evaluations),
        "accuracy": correct / len(evaluations),
        "wrong_page_original_value": wrong_page_correct,
        "passed": correct >= 14 and wrong_page_correct <= 2,
        "facts": [asdict(fact) for fact in all_facts],
        "evaluations": [asdict(evaluation) for evaluation in evaluations],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("losses", "facts", "evaluations")
            },
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

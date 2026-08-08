#!/usr/bin/env python3
# type: ignore
"""Train a shared writer over Gemma's ordinary-conversation activations."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    save_memory_page,
)
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

FORMAT = "gemma4-conversation-latent-memory-v1"
MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
RUNTIME_PROFILE = "mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1"
VALUES = ("AMBER", "COBALT", "JADE", "ONYX", "PEARL", "QUARTZ", "RUBY", "TOPAZ")
TRAIN_STATEMENTS = (
    "A private detail about my fictional instrument {entity}: I assigned it the calibration word {value}.",
    "My imaginary tool {entity} uses {value} as its calibration word.",
    "The calibration word for an invented device of mine named {entity} is {value}.",
    "I call one of my made-up instruments {entity}; its private calibration word is {value}.",
)
TEST_STATEMENTS = (
    "For my fictional instrument {entity}, the word I chose for calibration is {value}.",
    "{entity} is an imaginary device of mine, calibrated using the word {value}.",
)
TRAIN_QUERIES = (
    "What calibration word did I assign to {entity}? Reply with that word only.",
    "Give only the private calibration word for my fictional instrument {entity}.",
    "Which word calibrates my imaginary device {entity}? Answer with one word.",
    "State the calibration word belonging to {entity}, and nothing else.",
)
TEST_QUERIES = (
    "For {entity}, what was my chosen calibration word? Output only the word.",
    "My made-up instrument {entity} is calibrated with which word? One word only.",
)


@dataclass(frozen=True, slots=True)
class ConversationFact:
    fact_id: str
    split: str
    entity: str
    value: str
    statement_form: int

    def statement(self, form: int | None = None) -> str:
        templates = TRAIN_STATEMENTS if self.split == "train" else TEST_STATEMENTS
        selected = self.statement_form if form is None else form
        return templates[selected].format(entity=self.entity, value=self.value)


def facts() -> tuple[ConversationFact, ...]:
    rows: list[ConversationFact] = []
    for index in range(32):
        entity = f"NEMORA-{index + 0x1200:04X}"
        value = VALUES[index // len(TRAIN_STATEMENTS)]
        fact_id = hashlib.sha256(
            f"{FORMAT}\0train\0{entity}\0{value}".encode()
        ).hexdigest()[:24]
        rows.append(
            ConversationFact(
                fact_id, "train", entity, value, index % len(TRAIN_STATEMENTS)
            )
        )
    test_values = (
        "QUARTZ",
        "AMBER",
        "TOPAZ",
        "JADE",
        "RUBY",
        "PEARL",
        "ONYX",
        "COBALT",
    )
    for index, value in enumerate(test_values):
        entity = f"SOVARA-{index + 0xA100:04X}"
        fact_id = hashlib.sha256(
            f"{FORMAT}\0test\0{entity}\0{value}".encode()
        ).hexdigest()[:24]
        rows.append(ConversationFact(fact_id, "test", entity, value, 0))
    return tuple(rows)


def _chat_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _capture_page(model, tokenizer, mounted, statement: str) -> MemoryPage:
    prompt = _chat_prompt(tokenizer, statement)
    tokens = mx.array(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=mx.int32
    )
    embeddings = model.language_model.model.embed_tokens(tokens)
    return mounted.compile_page(embeddings)


def _training_example(
    tokenizer, fact: ConversationFact, template_index: int
) -> tuple[mx.array, int]:
    query = TRAIN_QUERIES[template_index].format(entity=fact.entity)
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


def _contextual_generate(model, tokenizer, statement: str, query: str) -> str:
    return _generate(model, tokenizer, f"{statement}\n\n{query}")


def run(model_path: Path, output_dir: Path) -> dict[str, object]:
    mx.random.seed(9_741)
    all_facts = facts()
    train_facts = tuple(fact for fact in all_facts if fact.split == "train")
    test_facts = tuple(fact for fact in all_facts if fact.split == "test")
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )

    train_pages: dict[str, MemoryPage] = {}
    capture_started = time.perf_counter()
    for fact in train_facts:
        train_pages[fact.fact_id] = _capture_page(
            model, tokenizer, mounted, fact.statement()
        )
    training_capture_seconds = time.perf_counter() - capture_started

    writer = NativeMemoryCompiler(
        tuple(site.layer_index for site in mounted.sites), head_dim=512, rank=32
    )
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(
        native_page: MemoryPage, input_ids: mx.array, prompt_length: int
    ) -> mx.array:
        mounted.use_training_layers(writer(native_page))
        logits = model(input_ids)
        return nn.losses.cross_entropy(
            logits[:, prompt_length - 1 : -1],
            input_ids[:, prompt_length:],
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(writer, loss_fn)
    losses: list[float] = []
    training_started = time.perf_counter()
    for step in range(320):
        fact = train_facts[step % len(train_facts)]
        query_form = (step // len(train_facts)) % len(TRAIN_QUERIES)
        input_ids, prompt_length = _training_example(tokenizer, fact, query_form)
        loss, gradients = loss_and_grad(
            train_pages[fact.fact_id], input_ids, prompt_length
        )
        optimizer.update(writer, gradients)
        mx.eval(loss, writer.parameters(), optimizer.state)
        losses.append(float(loss.item()))
        if (step + 1) % 32 == 0:
            print(f"step={step + 1} loss={losses[-1]:.6f}", flush=True)
    training_seconds = time.perf_counter() - training_started
    mounted.deactivate()

    output_dir.mkdir(parents=True, exist_ok=True)
    writer_path = output_dir / "writer.safetensors"
    writer.save_weights(str(writer_path))
    writer_digest = hashlib.sha256(writer_path.read_bytes()).hexdigest()

    pages: dict[tuple[str, int], MemoryPage] = {}
    page_records: list[dict[str, object]] = []
    heldout_capture_started = time.perf_counter()
    for fact in test_facts:
        for statement_form in range(len(TEST_STATEMENTS)):
            native = _capture_page(
                model, tokenizer, mounted, fact.statement(statement_form)
            )
            page = mounted.freeze_layers(writer(native))
            page_path = Path("pages") / f"{fact.fact_id}-{statement_form}.safetensors"
            save_memory_page(page, output_dir / page_path)
            pages[(fact.fact_id, statement_form)] = page
            page_records.append(
                {
                    "fact_id": fact.fact_id,
                    "statement_form": statement_form,
                    "page_id": page.page_id,
                    "path": str(page_path),
                    "resident_bytes": page.resident_bytes,
                }
            )
    heldout_capture_seconds = time.perf_counter() - heldout_capture_started

    base_results: list[dict[str, object]] = []
    contextual_results: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    for fact in test_facts:
        first_query = TEST_QUERIES[0].format(entity=fact.entity)
        base_response = _generate(model, tokenizer, first_query)
        base_results.append(
            {
                "fact_id": fact.fact_id,
                "expected": fact.value,
                "response": base_response,
                "correct": base_response == fact.value,
            }
        )
        for statement_form in range(len(TEST_STATEMENTS)):
            statement = fact.statement(statement_form)
            contextual_response = _contextual_generate(
                model, tokenizer, statement, first_query
            )
            contextual_results.append(
                {
                    "fact_id": fact.fact_id,
                    "statement_form": statement_form,
                    "expected": fact.value,
                    "response": contextual_response,
                    "correct": contextual_response == fact.value,
                }
            )
            page = pages[(fact.fact_id, statement_form)]
            mounted.activate(
                page,
                proof_id=f"proof:conversation:{fact.fact_id}:{statement_form}:{page.page_id}",
            )
            for query_form, template in enumerate(TEST_QUERIES):
                response = _generate(
                    model, tokenizer, template.format(entity=fact.entity)
                )
                evaluations.append(
                    {
                        "fact_id": fact.fact_id,
                        "statement_form": statement_form,
                        "query_form": query_form,
                        "expected": fact.value,
                        "response": response,
                        "correct": response == fact.value,
                        "page_id": page.page_id,
                    }
                )
            mounted.deactivate()

    wrong_page_original_value = 0
    revoked_equal_base = 0
    for index, fact in enumerate(test_facts):
        wrong_fact = next(
            candidate
            for candidate in test_facts[index + 1 :] + test_facts[: index + 1]
            if candidate.value != fact.value
        )
        wrong_page = pages[(wrong_fact.fact_id, 0)]
        mounted.activate(
            wrong_page, proof_id=f"proof:wrong:{fact.fact_id}:{wrong_page.page_id}"
        )
        query = TEST_QUERIES[0].format(entity=fact.entity)
        wrong_response = _generate(model, tokenizer, query)
        wrong_page_original_value += wrong_response == fact.value
        mounted.deactivate()
        revoked_response = _generate(model, tokenizer, query)
        base_response = next(
            row["response"] for row in base_results if row["fact_id"] == fact.fact_id
        )
        revoked_equal_base += revoked_response == base_response

    correct = sum(bool(row["correct"]) for row in evaluations)
    base_correct = sum(bool(row["correct"]) for row in base_results)
    contextual_correct = sum(bool(row["correct"]) for row in contextual_results)
    case_digest = hashlib.sha256(
        json.dumps([asdict(fact) for fact in all_facts], sort_keys=True).encode()
    ).hexdigest()
    result = {
        "format": FORMAT,
        "model_path": str(model_path),
        "case_digest": case_digest,
        "writer_digest": writer_digest,
        "writer_rank": 32,
        "writer_parameters": sum(
            parameter.size for _, parameter in tree_flatten(writer.parameters())
        ),
        "train_facts": len(train_facts),
        "test_facts": len(test_facts),
        "updates": 320,
        "learning_rate": 1e-3,
        "training_capture_seconds": training_capture_seconds,
        "heldout_capture_seconds": heldout_capture_seconds,
        "training_seconds": training_seconds,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "correct": correct,
        "cases": len(evaluations),
        "base_correct": base_correct,
        "contextual_correct": contextual_correct,
        "contextual_cases": len(contextual_results),
        "wrong_page_original_value": wrong_page_original_value,
        "revoked_equal_base": revoked_equal_base,
        "passed": (
            correct >= 28
            and base_correct <= 2
            and wrong_page_original_value <= 2
            and revoked_equal_base == len(test_facts)
        ),
        "facts": [asdict(fact) for fact in all_facts],
        "pages": page_records,
        "base_results": base_results,
        "contextual_results": contextual_results,
        "evaluations": evaluations,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "facts",
                    "pages",
                    "base_results",
                    "contextual_results",
                    "evaluations",
                }
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

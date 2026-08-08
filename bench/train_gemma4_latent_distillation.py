#!/usr/bin/env python3
# type: ignore
"""Distill full-context Gemma attention into persistent latent memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from gemma4_memory_sidecar_mlx import (
    AddressableMemoryAdapter,
    LayerMemory,
    MemoryPage,
    compose_memory_pages,
    mount_memory_sidecar,
    save_memory_page,
)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from train_gemma4_conversation_memory import (
    MODEL_ID,
    RUNTIME_PROFILE,
    TEST_QUERIES,
    TRAIN_QUERIES,
    TRAIN_STATEMENTS,
    _chat_prompt,
    _training_example,
    facts,
)

FORMAT = "gemma4-latent-episodic-distillation-v1"


def _prefix_ids(tokenizer, statement: str) -> mx.array:
    return mx.array(
        [tokenizer.encode(f"{statement}\n\n", add_special_tokens=False)],
        dtype=mx.int32,
    )


def _capture_prefix_page(model, tokenizer, mounted, statement: str) -> MemoryPage:
    tokens = _prefix_ids(tokenizer, statement)
    embeddings = model.language_model.model.embed_tokens(tokens)
    return mounted.compile_page(embeddings)


def _teacher_trace(
    model,
    mounted,
    prefix_ids: mx.array,
    suffix_ids: mx.array,
) -> tuple[mx.array, ...]:
    full_ids = mx.concatenate((prefix_ids, suffix_ids), axis=1)
    mounted.begin_trace()
    model(full_ids)
    full_trace = mounted.finish_trace()
    suffix_length = suffix_ids.shape[1]
    trace = tuple(
        value[:, -suffix_length:, :].astype(mx.bfloat16) for value in full_trace
    )
    mx.eval(*trace)
    return trace


def _training_bank(
    adapter: AddressableMemoryAdapter, pages: tuple[MemoryPage, ...]
) -> tuple[LayerMemory, ...]:
    compiled = tuple(adapter(page) for page in pages)
    return tuple(
        LayerMemory(
            compiled[0][position].layer_index,
            mx.concatenate([layers[position].keys for layers in compiled], axis=2),
            mx.concatenate([layers[position].values for layers in compiled], axis=2),
            compiled[0][position].gate,
        )
        for position in range(len(compiled[0]))
    )


def _generate(model, tokenizer, query: str) -> str:
    return generate(
        model,
        tokenizer,
        prompt=_chat_prompt(tokenizer, query),
        max_tokens=16,
        sampler=make_sampler(temp=0),
    ).strip()


def run(model_path: Path, output_dir: Path) -> dict[str, object]:
    mx.random.seed(25_919)
    registered = facts()
    train_facts = tuple(fact for fact in registered if fact.split == "train")
    test_facts = tuple(fact for fact in registered if fact.split == "test")
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )

    native_pages = {
        fact.fact_id: _capture_prefix_page(model, tokenizer, mounted, fact.statement())
        for fact in train_facts
    }
    banks = {
        statement_form: tuple(
            native_pages[fact.fact_id]
            for fact in train_facts
            if fact.statement_form == statement_form
        )
        for statement_form in range(len(TRAIN_STATEMENTS))
    }
    if any(len(bank) != 8 for bank in banks.values()):
        raise ValueError("each distillation bank must contain eight memories")

    teacher_started = time.perf_counter()
    teacher_traces: dict[tuple[str, int], tuple[mx.array, ...]] = {}
    training_inputs: dict[tuple[str, int], tuple[mx.array, int]] = {}
    for fact in train_facts:
        prefix = _prefix_ids(tokenizer, fact.statement())
        for query_form in range(len(TRAIN_QUERIES)):
            suffix, prompt_length = _training_example(tokenizer, fact, query_form)
            training_inputs[(fact.fact_id, query_form)] = (suffix, prompt_length)
            teacher_traces[(fact.fact_id, query_form)] = _teacher_trace(
                model, mounted, prefix, suffix
            )
    teacher_seconds = time.perf_counter() - teacher_started

    adapter = AddressableMemoryAdapter(
        tuple(site.layer_index for site in mounted.sites), head_dim=512, rank=32
    )
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(
        bank: tuple[MemoryPage, ...],
        input_ids: mx.array,
        prompt_length: int,
        teacher: tuple[mx.array, ...],
    ) -> tuple[mx.array, mx.array, mx.array]:
        mounted.use_training_layers(_training_bank(adapter, bank), reader=adapter)
        mounted.begin_trace()
        logits = model(input_ids)
        student = mounted.finish_trace()
        trace_losses = [
            mx.mean(mx.square(current.astype(mx.float32) - target.astype(mx.float32)))
            / (mx.mean(mx.square(target.astype(mx.float32))) + 1e-6)
            for current, target in zip(student, teacher, strict=True)
        ]
        trace_loss = mx.mean(mx.stack(trace_losses))
        answer_loss = nn.losses.cross_entropy(
            logits[:, prompt_length - 1 : -1],
            input_ids[:, prompt_length:],
            reduction="mean",
        )
        return trace_loss + 0.1 * answer_loss, trace_loss, answer_loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    losses: list[float] = []
    trace_losses: list[float] = []
    answer_losses: list[float] = []
    training_started = time.perf_counter()
    for step in range(320):
        fact = train_facts[step % len(train_facts)]
        query_form = (step // len(train_facts)) % len(TRAIN_QUERIES)
        input_ids, prompt_length = training_inputs[(fact.fact_id, query_form)]
        (loss, trace_loss, answer_loss), gradients = loss_and_grad(
            banks[fact.statement_form],
            input_ids,
            prompt_length,
            teacher_traces[(fact.fact_id, query_form)],
        )
        optimizer.update(adapter, gradients)
        mx.eval(loss, trace_loss, answer_loss, adapter.parameters(), optimizer.state)
        losses.append(float(loss.item()))
        trace_losses.append(float(trace_loss.item()))
        answer_losses.append(float(answer_loss.item()))
        if (step + 1) % 32 == 0:
            print(
                f"step={step + 1} loss={losses[-1]:.6f} "
                f"trace={trace_losses[-1]:.6f} answer={answer_losses[-1]:.6f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - training_started
    mounted.deactivate()

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter.safetensors"
    adapter.save_weights(str(adapter_path))
    adapter_digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()

    pages: dict[str, MemoryPage] = {}
    page_records: list[dict[str, object]] = []
    for fact in test_facts:
        native = _capture_prefix_page(model, tokenizer, mounted, fact.statement(0))
        page = mounted.freeze_layers(adapter(native))
        page_path = Path("pages") / f"{fact.fact_id}.safetensors"
        save_memory_page(page, output_dir / page_path)
        pages[fact.fact_id] = page
        page_records.append(
            {
                "fact_id": fact.fact_id,
                "page_id": page.page_id,
                "path": str(page_path),
                "resident_bytes": page.resident_bytes,
            }
        )
    bank = compose_memory_pages(tuple(pages.values()))
    bank_path = output_dir / "complete-bank.safetensors"
    save_memory_page(bank, bank_path)

    mounted.activate(
        bank, proof_id=f"proof:distilled-bank:{bank.page_id}", reader=adapter
    )
    evaluations: list[dict[str, object]] = []
    for fact in test_facts:
        for query_form, template in enumerate(TEST_QUERIES):
            response = _generate(model, tokenizer, template.format(entity=fact.entity))
            evaluations.append(
                {
                    "fact_id": fact.fact_id,
                    "query_form": query_form,
                    "expected": fact.value,
                    "response": response,
                    "correct": response == fact.value,
                }
            )
    mounted.deactivate()

    omitted_original_value = 0
    omission_results: list[dict[str, object]] = []
    for fact in test_facts:
        omission_bank = compose_memory_pages(
            tuple(page for fact_id, page in pages.items() if fact_id != fact.fact_id)
        )
        mounted.activate(
            omission_bank,
            proof_id=f"proof:distilled-omission:{fact.fact_id}",
            reader=adapter,
        )
        response = _generate(
            model, tokenizer, TEST_QUERIES[0].format(entity=fact.entity)
        )
        reproduced = response == fact.value
        omitted_original_value += reproduced
        omission_results.append(
            {
                "omitted_fact_id": fact.fact_id,
                "expected": fact.value,
                "response": response,
                "reproduced_omitted_value": reproduced,
            }
        )
        mounted.deactivate()

    correct = sum(bool(row["correct"]) for row in evaluations)
    result = {
        "format": FORMAT,
        "model_path": str(model_path),
        "adapter_digest": adapter_digest,
        "teacher_seconds": teacher_seconds,
        "training_seconds": training_seconds,
        "updates": 320,
        "learning_rate": 1e-3,
        "trace_loss_weight": 1.0,
        "answer_loss_weight": 0.1,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "initial_trace_loss": trace_losses[0],
        "final_trace_loss": trace_losses[-1],
        "initial_answer_loss": answer_losses[0],
        "final_answer_loss": answer_losses[-1],
        "case_digest": hashlib.sha256(
            json.dumps(training_inputs.keys(), default=list).encode()
        ).hexdigest(),
        "bank_id": bank.page_id,
        "bank_path": bank_path.name,
        "bank_entries": len(pages),
        "bank_resident_bytes": bank.resident_bytes,
        "correct": correct,
        "cases": len(evaluations),
        "omitted_original_value": omitted_original_value,
        "omission_cases": len(omission_results),
        "passed": correct >= 14 and omitted_original_value <= 2,
        "pages": page_records,
        "evaluations": evaluations,
        "omission_results": omission_results,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"pages", "evaluations", "omission_results"}
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

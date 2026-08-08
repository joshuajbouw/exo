#!/usr/bin/env python3
# type: ignore
"""Train latent recall with causally derived memory-slot supervision."""

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
    MemoryPage,
    compose_memory_pages,
    mount_memory_sidecar,
    save_memory_page,
)
from mlx_lm import load
from train_gemma4_conversation_memory import (
    MODEL_ID,
    RUNTIME_PROFILE,
    TEST_QUERIES,
    TRAIN_QUERIES,
    TRAIN_STATEMENTS,
    _training_example,
    facts,
)
from train_gemma4_latent_distillation import (
    _capture_prefix_page,
    _generate,
    _prefix_ids,
    _teacher_trace,
    _training_bank,
)

FORMAT = "gemma4-causal-associative-memory-v1"


def _target_bounds(bank: tuple[MemoryPage, ...], target_index: int) -> tuple[int, int]:
    lengths = [page.layers[0].keys.shape[2] for page in bank]
    start = sum(lengths[:target_index])
    return start, start + lengths[target_index]


def _association_loss(
    queries: tuple[mx.array, ...],
    layers,
    prompt_length: int,
    target_start: int,
    target_end: int,
) -> mx.array:
    losses = []
    for query, layer in zip(queries, layers, strict=True):
        keys = layer.keys
        repeats = query.shape[1] // keys.shape[1]
        expanded_keys = mx.repeat(keys, repeats=repeats, axis=1)
        final_prompt_query = query[:, :, prompt_length - 1 : prompt_length, :]
        scores = mx.matmul(final_prompt_query, expanded_keys.transpose(0, 1, 3, 2))
        target_scores = scores[..., target_start:target_end]
        losses.append(
            mx.mean(
                mx.logsumexp(scores, axis=-1) - mx.logsumexp(target_scores, axis=-1)
            )
        )
    return mx.mean(mx.stack(losses))


def run(
    model_path: Path, base_artifact_dir: Path, output_dir: Path
) -> dict[str, object]:
    mx.random.seed(31_337)
    base_result = json.loads((base_artifact_dir / "result.json").read_text())
    base_adapter_path = base_artifact_dir / "adapter.safetensors"
    base_digest = hashlib.sha256(base_adapter_path.read_bytes()).hexdigest()
    if base_digest != base_result["adapter_digest"]:
        raise ValueError("base adapter digest differs from its training record")

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
    bank_facts = {
        form: tuple(fact for fact in train_facts if fact.statement_form == form)
        for form in range(len(TRAIN_STATEMENTS))
    }
    banks = {
        form: tuple(native_pages[fact.fact_id] for fact in selected)
        for form, selected in bank_facts.items()
    }

    teacher_traces = {}
    training_inputs = {}
    for fact in train_facts:
        prefix = _prefix_ids(tokenizer, fact.statement())
        for query_form in range(len(TRAIN_QUERIES)):
            suffix, prompt_length = _training_example(tokenizer, fact, query_form)
            training_inputs[(fact.fact_id, query_form)] = (suffix, prompt_length)
            teacher_traces[(fact.fact_id, query_form)] = _teacher_trace(
                model, mounted, prefix, suffix
            )

    adapter = AddressableMemoryAdapter(
        tuple(site.layer_index for site in mounted.sites), head_dim=512, rank=32
    )
    adapter.load_weights(str(base_adapter_path))
    mx.eval(adapter.parameters())
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(
        bank: tuple[MemoryPage, ...],
        target_index: int,
        input_ids: mx.array,
        prompt_length: int,
        teacher,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        layers = _training_bank(adapter, bank)
        target_start, target_end = _target_bounds(bank, target_index)
        mounted.use_training_layers(layers, reader=adapter)
        mounted.begin_trace()
        mounted.begin_association_trace()
        logits = model(input_ids)
        student = mounted.finish_trace()
        queries = mounted.finish_association_trace()
        trace_loss = mx.mean(
            mx.stack(
                [
                    mx.mean(
                        mx.square(
                            current.astype(mx.float32) - target.astype(mx.float32)
                        )
                    )
                    / (mx.mean(mx.square(target.astype(mx.float32))) + 1e-6)
                    for current, target in zip(student, teacher, strict=True)
                ]
            )
        )
        association_loss = _association_loss(
            queries, layers, prompt_length, target_start, target_end
        )
        answer_loss = nn.losses.cross_entropy(
            logits[:, prompt_length - 1 : -1],
            input_ids[:, prompt_length:],
            reduction="mean",
        )
        total = trace_loss + association_loss + 0.1 * answer_loss
        return total, trace_loss, association_loss, answer_loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    final_metrics = None
    training_started = time.perf_counter()
    for step in range(320):
        fact = train_facts[step % len(train_facts)]
        query_form = (step // len(train_facts)) % len(TRAIN_QUERIES)
        selected_facts = bank_facts[fact.statement_form]
        target_index = selected_facts.index(fact)
        input_ids, prompt_length = training_inputs[(fact.fact_id, query_form)]
        metrics, gradients = loss_and_grad(
            banks[fact.statement_form],
            target_index,
            input_ids,
            prompt_length,
            teacher_traces[(fact.fact_id, query_form)],
        )
        optimizer.update(adapter, gradients)
        mx.eval(*metrics, adapter.parameters(), optimizer.state)
        final_metrics = tuple(float(value.item()) for value in metrics)
        if (step + 1) % 32 == 0:
            print(
                f"step={step + 1} loss={final_metrics[0]:.6f} "
                f"trace={final_metrics[1]:.6f} association={final_metrics[2]:.6f} "
                f"answer={final_metrics[3]:.6f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - training_started
    mounted.deactivate()
    assert final_metrics is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter.safetensors"
    adapter.save_weights(str(adapter_path))
    adapter_digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    pages = {}
    for fact in test_facts:
        native = _capture_prefix_page(model, tokenizer, mounted, fact.statement(0))
        pages[fact.fact_id] = mounted.freeze_layers(adapter(native))
    bank = compose_memory_pages(tuple(pages.values()))
    save_memory_page(bank, output_dir / "complete-bank.safetensors")

    mounted.activate(bank, proof_id=f"proof:causal-bank:{bank.page_id}", reader=adapter)
    evaluations = []
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
    for fact in test_facts:
        omission = compose_memory_pages(
            tuple(page for fact_id, page in pages.items() if fact_id != fact.fact_id)
        )
        mounted.activate(
            omission, proof_id=f"proof:omit:{fact.fact_id}", reader=adapter
        )
        response = _generate(
            model, tokenizer, TEST_QUERIES[0].format(entity=fact.entity)
        )
        omitted_original_value += response == fact.value
        mounted.deactivate()

    correct = sum(bool(row["correct"]) for row in evaluations)
    result = {
        "format": FORMAT,
        "model_path": str(model_path),
        "base_adapter_digest": base_digest,
        "adapter_digest": adapter_digest,
        "updates": 320,
        "learning_rate": 1e-3,
        "training_seconds": training_seconds,
        "final_loss": final_metrics[0],
        "final_trace_loss": final_metrics[1],
        "final_association_loss": final_metrics[2],
        "final_answer_loss": final_metrics[3],
        "bank_id": bank.page_id,
        "bank_path": "complete-bank.safetensors",
        "bank_entries": len(pages),
        "bank_resident_bytes": bank.resident_bytes,
        "correct": correct,
        "cases": len(evaluations),
        "omitted_original_value": omitted_original_value,
        "passed": correct >= 14 and omitted_original_value <= 2,
        "evaluations": evaluations,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "evaluations"},
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.base_artifact_dir, args.output_dir)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# type: ignore
"""Train a shared reader-writer adapter for associative Gemma memory."""

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
    MemorySelectionProof,
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
    _capture_page,
    _chat_prompt,
    _training_example,
    facts,
)

FORMAT = "gemma4-reader-writer-conversation-memory-v1"


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


def run(
    model_path: Path, base_artifact_dir: Path, output_dir: Path
) -> dict[str, object]:
    mx.random.seed(18_211)
    base_result = json.loads((base_artifact_dir / "result.json").read_text())
    base_writer_path = base_artifact_dir / "writer.safetensors"
    base_writer_digest = hashlib.sha256(base_writer_path.read_bytes()).hexdigest()
    if base_writer_digest != base_result["writer_digest"]:
        raise ValueError("base writer digest differs from its training record")

    registered = facts()
    train_facts = tuple(fact for fact in registered if fact.split == "train")
    test_facts = tuple(fact for fact in registered if fact.split == "test")
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    native_pages = {
        fact.fact_id: _capture_page(model, tokenizer, mounted, fact.statement())
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
        raise ValueError("each addressability bank must contain eight memories")

    layer_indices = tuple(site.layer_index for site in mounted.sites)
    adapter = AddressableMemoryAdapter(layer_indices, head_dim=512, rank=32)
    adapter.writer.load_weights(str(base_writer_path))
    mx.eval(adapter.parameters())
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(
        bank: tuple[MemoryPage, ...], input_ids: mx.array, prompt_length: int
    ) -> mx.array:
        mounted.use_training_layers(_training_bank(adapter, bank), reader=adapter)
        logits = model(input_ids)
        return nn.losses.cross_entropy(
            logits[:, prompt_length - 1 : -1],
            input_ids[:, prompt_length:],
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    losses: list[float] = []
    training_started = time.perf_counter()
    for step in range(320):
        fact = train_facts[step % len(train_facts)]
        query_form = (step // len(train_facts)) % len(TRAIN_QUERIES)
        input_ids, prompt_length = _training_example(tokenizer, fact, query_form)
        loss, gradients = loss_and_grad(
            banks[fact.statement_form], input_ids, prompt_length
        )
        optimizer.update(adapter, gradients)
        mx.eval(loss, adapter.parameters(), optimizer.state)
        losses.append(float(loss.item()))
        if (step + 1) % 32 == 0:
            print(f"step={step + 1} loss={losses[-1]:.6f}", flush=True)
    training_seconds = time.perf_counter() - training_started
    mounted.deactivate()

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter.safetensors"
    adapter.save_weights(str(adapter_path))
    adapter_digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()

    pages: dict[str, MemoryPage] = {}
    page_records: list[dict[str, object]] = []
    for fact in test_facts:
        native = _capture_page(model, tokenizer, mounted, fact.statement(0))
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
        bank,
        proof=MemorySelectionProof.for_page(
            bank,
            proof_id=f"proof:reader-writer-bank:{bank.page_id}",
            fact_snapshot_id="reader-writer-bank-fixture",
        ),
        reader=adapter,
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
            proof=MemorySelectionProof.for_page(
                omission_bank,
                proof_id=f"proof:reader-writer-omission:{fact.fact_id}",
                fact_snapshot_id="reader-writer-omission-fixture",
            ),
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
                "bank_id": omission_bank.page_id,
            }
        )
        mounted.deactivate()

    correct = sum(bool(row["correct"]) for row in evaluations)
    result = {
        "format": FORMAT,
        "model_path": str(model_path),
        "base_writer_digest": base_writer_digest,
        "adapter_digest": adapter_digest,
        "case_digest": base_result["case_digest"],
        "updates": 320,
        "learning_rate": 1e-3,
        "training_bank_entries": 8,
        "training_seconds": training_seconds,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
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
    parser.add_argument("--base-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.base_artifact_dir, args.output_dir)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

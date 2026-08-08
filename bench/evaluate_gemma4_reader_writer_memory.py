#!/usr/bin/env python3
# type: ignore
"""Replay a serialized associative bank with its frozen shared reader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from gemma4_memory_sidecar_mlx import (
    AddressableMemoryAdapter,
    MemorySelectionProof,
    load_memory_page,
    mount_memory_sidecar,
)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from train_gemma4_conversation_memory import (
    MODEL_ID,
    RUNTIME_PROFILE,
    TEST_QUERIES,
    _chat_prompt,
    facts,
)


def _generate(model, tokenizer, query: str) -> str:
    return generate(
        model,
        tokenizer,
        prompt=_chat_prompt(tokenizer, query),
        max_tokens=16,
        sampler=make_sampler(temp=0),
    ).strip()


def run(model_path: Path, artifact_dir: Path) -> dict[str, object]:
    training_result = json.loads((artifact_dir / "result.json").read_text())
    adapter_path = artifact_dir / "adapter.safetensors"
    adapter_digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    if adapter_digest != training_result["adapter_digest"]:
        raise ValueError("adapter digest differs from the training record")

    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    adapter = AddressableMemoryAdapter(
        tuple(site.layer_index for site in mounted.sites), head_dim=512, rank=32
    )
    adapter.load_weights(str(adapter_path))
    mx.eval(adapter.parameters())
    bank = load_memory_page(artifact_dir / training_result["bank_path"])
    if bank.page_id != training_result["bank_id"]:
        raise ValueError("serialized bank identity differs from the training record")
    mounted.activate(
        bank,
        proof=MemorySelectionProof.for_page(
            bank,
            proof_id=f"proof:fresh-reader-writer:{bank.page_id}",
            fact_snapshot_id="fresh-reader-writer-fixture",
        ),
        reader=adapter,
    )

    evaluations: list[dict[str, object]] = []
    for fact in (fact for fact in facts() if fact.split == "test"):
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

    correct = sum(bool(row["correct"]) for row in evaluations)
    result = {
        "format": "gemma4-reader-writer-conversation-memory-v1-fresh-process-replay",
        "model_path": str(model_path),
        "adapter_digest": adapter_digest,
        "case_digest": training_result["case_digest"],
        "bank_id": bank.page_id,
        "bank_resident_bytes": bank.resident_bytes,
        "correct": correct,
        "cases": len(evaluations),
        "passed": correct >= 14,
        "evaluations": evaluations,
    }
    (artifact_dir / "fresh-process-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "evaluations"},
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.artifact_dir)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

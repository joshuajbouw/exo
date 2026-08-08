#!/usr/bin/env python3
# type: ignore
"""Test whether Gemma internally selects among simultaneous latent memories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gemma4_memory_sidecar_mlx import (
    compose_memory_pages,
    load_memory_page,
    mount_memory_sidecar,
)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from train_gemma4_conversation_memory import (
    FORMAT,
    MODEL_ID,
    RUNTIME_PROFILE,
    TEST_QUERIES,
    facts,
)


def _chat_prompt(tokenizer, query: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False,
        add_generation_prompt=True,
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
    writer_digest = hashlib.sha256(
        (artifact_dir / "writer.safetensors").read_bytes()
    ).hexdigest()
    if writer_digest != training_result["writer_digest"]:
        raise ValueError("writer digest differs from the training record")

    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    test_facts = tuple(fact for fact in facts() if fact.split == "test")
    records = {
        record["fact_id"]: record
        for record in training_result["pages"]
        if record["statement_form"] == 0
    }
    pages = {
        fact.fact_id: load_memory_page(artifact_dir / records[fact.fact_id]["path"])
        for fact in test_facts
    }
    bank = compose_memory_pages(tuple(pages.values()))
    mounted.activate(bank, proof_id=f"proof:complete-bank:{bank.page_id}")
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
        remaining = tuple(
            page for fact_id, page in pages.items() if fact_id != fact.fact_id
        )
        omission_bank = compose_memory_pages(remaining)
        mounted.activate(
            omission_bank,
            proof_id=f"proof:omission:{fact.fact_id}:{omission_bank.page_id}",
        )
        response = _generate(
            model, tokenizer, TEST_QUERIES[0].format(entity=fact.entity)
        )
        correct = response == fact.value
        omitted_original_value += correct
        omission_results.append(
            {
                "omitted_fact_id": fact.fact_id,
                "expected": fact.value,
                "response": response,
                "reproduced_omitted_value": correct,
                "bank_id": omission_bank.page_id,
            }
        )
        mounted.deactivate()

    correct = sum(bool(row["correct"]) for row in evaluations)
    result = {
        "format": f"{FORMAT}-internal-binding-v1",
        "model_path": str(model_path),
        "writer_digest": writer_digest,
        "case_digest": training_result["case_digest"],
        "bank_id": bank.page_id,
        "bank_entries": len(pages),
        "bank_resident_bytes": bank.resident_bytes,
        "correct": correct,
        "cases": len(evaluations),
        "omitted_original_value": omitted_original_value,
        "omission_cases": len(omission_results),
        "passed": correct >= 14 and omitted_original_value <= 2,
        "evaluations": evaluations,
        "omission_results": omission_results,
    }
    (artifact_dir / "internal-binding.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"evaluations", "omission_results"}
            },
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

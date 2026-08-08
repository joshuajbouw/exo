#!/usr/bin/env python3
# type: ignore
"""Replay serialized conversation-latent pages in a fresh Gemma process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gemma4_memory_sidecar_mlx import (
    MemorySelectionProof,
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
    fact_by_id = {fact.fact_id: fact for fact in test_facts}
    page_records = training_result["pages"]

    evaluations: list[dict[str, object]] = []
    stable_page_ids = 0
    loaded_pages = {}
    for page_record in page_records:
        fact = fact_by_id[page_record["fact_id"]]
        statement_form = page_record["statement_form"]
        page = load_memory_page(artifact_dir / page_record["path"])
        loaded_pages[(fact.fact_id, statement_form)] = page
        stable_page_ids += page.page_id == page_record["page_id"]
        mounted.activate(
            page,
            proof=MemorySelectionProof.for_page(
                page,
                proof_id=f"proof:fresh-process:{fact.fact_id}:{statement_form}:{page.page_id}",
                fact_snapshot_id="conversation-memory-replay-fixture",
            ),
        )
        for query_form, template in enumerate(TEST_QUERIES):
            response = _generate(model, tokenizer, template.format(entity=fact.entity))
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
    for index, fact in enumerate(test_facts):
        wrong_fact = next(
            candidate
            for candidate in test_facts[index + 1 :] + test_facts[: index + 1]
            if candidate.value != fact.value
        )
        page = loaded_pages[(wrong_fact.fact_id, 0)]
        mounted.activate(
            page,
            proof=MemorySelectionProof.for_page(
                page,
                proof_id=f"proof:fresh-wrong:{fact.fact_id}",
                fact_snapshot_id="conversation-wrong-page-fixture",
            ),
        )
        response = _generate(
            model, tokenizer, TEST_QUERIES[0].format(entity=fact.entity)
        )
        wrong_page_original_value += response == fact.value
        mounted.deactivate()

    correct = sum(bool(row["correct"]) for row in evaluations)
    replay = {
        "format": f"{FORMAT}-fresh-process-replay",
        "model_path": str(model_path),
        "writer_digest": writer_digest,
        "case_digest": training_result["case_digest"],
        "correct": correct,
        "cases": len(evaluations),
        "stable_page_ids": stable_page_ids,
        "page_count": len(page_records),
        "wrong_page_original_value": wrong_page_original_value,
        "passed": (
            correct >= 28
            and stable_page_ids == len(page_records)
            and wrong_page_original_value <= 2
        ),
        "evaluations": evaluations,
    }
    (artifact_dir / "fresh-process-replay.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in replay.items() if key != "evaluations"},
            indent=2,
            sort_keys=True,
        )
    )
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    replay = run(args.model_path, args.artifact_dir)
    if not replay["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

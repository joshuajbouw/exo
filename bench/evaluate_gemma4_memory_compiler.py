#!/usr/bin/env python3
# type: ignore
"""Replay the frozen Gemma 4 memory compiler in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from gemma4_memory_sidecar_mlx import NativeMemoryCompiler, mount_memory_sidecar
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from train_gemma4_memory_compiler import FORMAT, TEST_QUERIES, Fact, facts


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


def _native_page(model, tokenizer, mounted, fact: Fact):
    tokens = mx.array([tokenizer.encode(fact.canonical_record)], dtype=mx.int32)
    return mounted.compile_page(model.language_model.model.embed_tokens(tokens))


def run(model_path: Path, artifact_dir: Path) -> dict[str, object]:
    training_result = json.loads((artifact_dir / "result.json").read_text())
    compiler_path = artifact_dir / "compiler.safetensors"
    compiler_digest = hashlib.sha256(compiler_path.read_bytes()).hexdigest()
    if compiler_digest != training_result["compiler_digest"]:
        raise ValueError("serialized compiler digest differs from the training record")

    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model,
        model_id="mlx-community/gemma-4-31b-it-4bit",
        runtime_profile="mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1",
    )
    compiler = NativeMemoryCompiler(
        tuple(site.layer_index for site in mounted.sites), head_dim=512, rank=32
    )
    compiler.load_weights(str(compiler_path))
    mx.eval(compiler.parameters())

    test_facts = tuple(fact for fact in facts() if fact.split == "test")
    expected_pages = {
        evaluation["fact_id"]: evaluation["page_id"]
        for evaluation in training_result["evaluations"]
    }
    compiled_pages = {}
    evaluations = []
    for fact in test_facts:
        native_page = _native_page(model, tokenizer, mounted, fact)
        page = mounted.freeze_layers(compiler(native_page))
        compiled_pages[fact.fact_id] = page
        mounted.activate(page, proof_id=f"proof:replay:{fact.fact_id}:{page.page_id}")
        for query_index, template in enumerate(TEST_QUERIES):
            response = _generate(model, tokenizer, template.format(key=fact.key))
            evaluations.append(
                {
                    "fact_id": fact.fact_id,
                    "query_index": query_index,
                    "expected": fact.value,
                    "response": response,
                    "correct": response == fact.value,
                    "page_id": page.page_id,
                    "page_id_stable": page.page_id == expected_pages[fact.fact_id],
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
        page = compiled_pages[wrong_fact.fact_id]
        mounted.activate(page, proof_id=f"proof:replay-wrong:{fact.fact_id}")
        response = _generate(model, tokenizer, TEST_QUERIES[0].format(key=fact.key))
        wrong_page_original_value += response == fact.value
        mounted.deactivate()

    correct = sum(evaluation["correct"] for evaluation in evaluations)
    stable = sum(evaluation["page_id_stable"] for evaluation in evaluations)
    replay = {
        "format": f"{FORMAT}-fresh-process-replay",
        "model_path": str(model_path),
        "compiler_digest": compiler_digest,
        "case_digest": hashlib.sha256(
            json.dumps([asdict(fact) for fact in facts()], sort_keys=True).encode()
        ).hexdigest(),
        "correct": correct,
        "cases": len(evaluations),
        "stable_page_ids": stable,
        "wrong_page_original_value": wrong_page_original_value,
        "passed": (
            correct == len(evaluations)
            and stable == len(evaluations)
            and wrong_page_original_value == 0
            and training_result["case_digest"]
            == hashlib.sha256(
                json.dumps([asdict(fact) for fact in facts()], sort_keys=True).encode()
            ).hexdigest()
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

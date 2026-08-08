# type: ignore
#!/usr/bin/env python3
"""Recall conversation-latent pages through grounded external selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from gemma4_memory_sidecar_mlx import (
    MemoryPage,
    MemorySidecarError,
    load_memory_page,
    mount_memory_sidecar,
)
from grounded_memory_selection import (
    GroundedMemorySelector,
    MemoryCatalogEntry,
    MemoryDomainMembership,
    MemoryPageGrant,
    MemorySelectionRequest,
)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from train_gemma4_conversation_memory import (
    MODEL_ID,
    RUNTIME_PROFILE,
    TEST_QUERIES,
    facts,
)

_DOMAIN = "conversation-memory-evaluation-domain"
_PRINCIPAL = "conversation-memory-evaluation-principal"
_RULE_PROFILE = "grounded-conversation-memory-selection-v1"
_LATE_EPOCH = 1_000_000_000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--decoys", type=int, default=100_000)
    return parser.parse_args()


def _concept_id(fact_id: str) -> str:
    return f"private-calibration-memory:{fact_id}"


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


def _entry(page: MemoryPage, fact_id: str) -> MemoryCatalogEntry:
    return MemoryCatalogEntry(
        page_id=page.page_id,
        evidence_closure_id=f"conversation-evidence:{fact_id}",
        privacy_domain_id=_DOMAIN,
        concept_id=_concept_id(fact_id),
        model_id=MODEL_ID,
        runtime_profile=RUNTIME_PROFILE,
        first_epoch=0,
        last_epoch=2_000_000_000,
    )


def _decoy_entries(count: int, occupied: set[str]) -> tuple[MemoryCatalogEntry, ...]:
    entries: list[MemoryCatalogEntry] = []
    index = 0
    while len(entries) < count:
        page_id = hashlib.sha256(f"unrelated-memory:{index}".encode()).hexdigest()
        index += 1
        if page_id in occupied:
            continue
        entries.append(
            MemoryCatalogEntry(
                page_id=page_id,
                evidence_closure_id=f"unrelated-evidence:{index}",
                privacy_domain_id=_DOMAIN,
                concept_id=f"unrelated-concept:{index}",
                model_id=MODEL_ID,
                runtime_profile=RUNTIME_PROFILE,
                first_epoch=0,
                last_epoch=2_000_000_000,
            )
        )
    return tuple(entries)


def _request(fact_id: str, invocation_id: str) -> MemorySelectionRequest:
    return MemorySelectionRequest(
        invocation_id=invocation_id,
        principal_id=_PRINCIPAL,
        privacy_domain_id=_DOMAIN,
        concept_id=_concept_id(fact_id),
        model_id=MODEL_ID,
        runtime_profile=RUNTIME_PROFILE,
        epoch=_LATE_EPOCH,
    )


def _selector(
    entries: tuple[MemoryCatalogEntry, ...],
    grants: tuple[MemoryPageGrant, ...],
) -> GroundedMemorySelector:
    return GroundedMemorySelector(
        entries,
        (MemoryDomainMembership(_PRINCIPAL, _DOMAIN),),
        grants,
        rule_profile_id=_RULE_PROFILE,
    )


def run(model_path: Path, artifact_dir: Path, decoy_count: int) -> dict[str, object]:
    if decoy_count < 0:
        raise ValueError("decoy count must not be negative")
    training_result = json.loads((artifact_dir / "result.json").read_text())
    test_facts = tuple(fact for fact in facts() if fact.split == "test")
    fact_by_id = {fact.fact_id: fact for fact in test_facts}
    canonical_records = {
        record["fact_id"]: record
        for record in training_result["pages"]
        if record["statement_form"] == 0
    }
    if set(canonical_records) != set(fact_by_id):
        raise ValueError("artifact does not contain one canonical page per test fact")

    pages: dict[str, MemoryPage] = {}
    for fact_id, record in canonical_records.items():
        page = load_memory_page(artifact_dir / record["path"])
        if page.page_id != record["page_id"]:
            raise ValueError("latent page identity differs from its training record")
        pages[fact_id] = page

    entries = tuple(_entry(pages[fact_id], fact_id) for fact_id in sorted(pages))
    grants = tuple(
        MemoryPageGrant(_PRINCIPAL, page.page_id)
        for page in sorted(pages.values(), key=lambda candidate: candidate.page_id)
    )
    baseline_selector = _selector(entries, grants)
    selector_started = time.perf_counter()
    expanded_entries = entries + _decoy_entries(
        decoy_count, {entry.page_id for entry in entries}
    )
    selector = _selector(expanded_entries, grants)
    selector_build_seconds = time.perf_counter() - selector_started

    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    evaluations: list[dict[str, object]] = []
    selection_seconds: list[float] = []
    for fact in test_facts:
        page = pages[fact.fact_id]
        for query_form, template in enumerate(TEST_QUERIES):
            request = _request(fact.fact_id, f"recall:{fact.fact_id}:{query_form}")
            selection_started = time.perf_counter()
            proof = selector.select(request)
            selection_seconds.append(time.perf_counter() - selection_started)
            baseline_proof = baseline_selector.select(request)
            if proof is None or baseline_proof is None:
                raise RuntimeError("authorized memory unexpectedly had no selection")
            if proof.selected_page_id != page.page_id:
                raise RuntimeError("selector chose the wrong latent page")
            mounted.activate(page, proof=proof)
            response = _generate(model, tokenizer, template.format(entity=fact.entity))
            mounted.deactivate()
            evaluations.append(
                {
                    "fact_id": fact.fact_id,
                    "query_form": query_form,
                    "page_id": page.page_id,
                    "proof_id": proof.proof_id,
                    "fact_snapshot_changed_by_decoys": (
                        proof.fact_snapshot_id != baseline_proof.fact_snapshot_id
                    ),
                    "selection_stable_with_decoys": (
                        proof.selected_page_id == baseline_proof.selected_page_id
                    ),
                    "expected": fact.value,
                    "response": response,
                    "correct": response == fact.value,
                }
            )

    wrong_concept_original_value = 0
    for index, fact in enumerate(test_facts):
        wrong_fact = next(
            candidate
            for candidate in test_facts[index + 1 :] + test_facts[: index + 1]
            if candidate.value != fact.value
        )
        proof = selector.select(
            _request(wrong_fact.fact_id, f"wrong-concept:{fact.fact_id}")
        )
        if proof is None:
            raise RuntimeError("wrong-concept control unexpectedly had no selection")
        wrong_page = pages[wrong_fact.fact_id]
        mounted.activate(wrong_page, proof=proof)
        response = _generate(
            model, tokenizer, TEST_QUERIES[0].format(entity=fact.entity)
        )
        mounted.deactivate()
        wrong_concept_original_value += response == fact.value

    representative = test_facts[0]
    representative_request = _request(representative.fact_id, "negative-controls")
    revoked = _selector(entries, ()).select(representative_request) is None
    cross_domain = (
        selector.select(
            MemorySelectionRequest(
                invocation_id="cross-domain",
                principal_id=_PRINCIPAL,
                privacy_domain_id="another-domain",
                concept_id=_concept_id(representative.fact_id),
                model_id=MODEL_ID,
                runtime_profile=RUNTIME_PROFILE,
                epoch=_LATE_EPOCH,
            )
        )
        is None
    )
    expired = (
        selector.select(
            MemorySelectionRequest(
                invocation_id="expired",
                principal_id=_PRINCIPAL,
                privacy_domain_id=_DOMAIN,
                concept_id=_concept_id(representative.fact_id),
                model_id=MODEL_ID,
                runtime_profile=RUNTIME_PROFILE,
                epoch=2_000_000_001,
            )
        )
        is None
    )

    substitution_rejected = False
    proof = selector.select(representative_request)
    if proof is None:
        raise RuntimeError("substitution control unexpectedly had no selection")
    substituted_page = pages[test_facts[1].fact_id]
    try:
        mounted.activate(substituted_page, proof=proof)
    except MemorySidecarError:
        substitution_rejected = True
    finally:
        mounted.deactivate()

    correct = sum(bool(row["correct"]) for row in evaluations)
    stable = all(bool(row["selection_stable_with_decoys"]) for row in evaluations)
    snapshot_changed = all(
        bool(row["fact_snapshot_changed_by_decoys"]) for row in evaluations
    )
    result = {
        "experiment": "grounded-conversation-latent-memory",
        "model": str(model_path),
        "artifact_case_digest": training_result["case_digest"],
        "late_epoch": _LATE_EPOCH,
        "decoy_catalog_entries": decoy_count,
        "selector_build_seconds": selector_build_seconds,
        "median_selection_seconds": statistics.median(selection_seconds),
        "maximum_selection_seconds": max(selection_seconds),
        "correct": correct,
        "cases": len(evaluations),
        "wrong_concept_original_value": wrong_concept_original_value,
        "revoked_selection_absent": revoked,
        "cross_domain_selection_absent": cross_domain,
        "expired_selection_absent": expired,
        "page_substitution_rejected": substitution_rejected,
        "selection_stable_with_decoys": stable,
        "fact_snapshot_changed_by_decoys": snapshot_changed,
        "source_text_replayed": False,
        "source_token_ids_replayed": False,
        "historical_kv_restored": False,
        "passed": (
            correct == 16
            and wrong_concept_original_value <= 2
            and revoked
            and cross_domain
            and expired
            and substitution_rejected
            and stable
            and snapshot_changed
        ),
        "evaluations": evaluations,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = _arguments()
    result = run(args.model, args.artifact_dir, args.decoys)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

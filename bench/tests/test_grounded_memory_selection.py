from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from grounded_memory_selection import (
    AmbiguousMemorySelectionError,
    GroundedMemorySelector,
    MemoryCatalogEntry,
    MemoryDomainMembership,
    MemoryPageGrant,
    MemorySelectionRequest,
)

MODEL = "gemma-test"
RUNTIME = "mlx-test"
DOMAIN = "team-alpha"
PRINCIPAL = "alice"
PAGE = "11" * 32


def _entry(
    page_id: str = PAGE,
    concept_id: str = "concept:velorin",
) -> MemoryCatalogEntry:
    return MemoryCatalogEntry(
        page_id=page_id,
        evidence_closure_id=f"evidence:{page_id}",
        privacy_domain_id=DOMAIN,
        concept_id=concept_id,
        model_id=MODEL,
        runtime_profile=RUNTIME,
        first_epoch=10,
        last_epoch=10_000,
    )


def _request(
    *,
    principal: str = PRINCIPAL,
    domain: str = DOMAIN,
    concept: str = "concept:velorin",
    epoch: int = 1_000,
) -> MemorySelectionRequest:
    return MemorySelectionRequest(
        invocation_id="invocation-1",
        principal_id=principal,
        privacy_domain_id=domain,
        concept_id=concept,
        model_id=MODEL,
        runtime_profile=RUNTIME,
        epoch=epoch,
    )


def _selector(
    entries: tuple[MemoryCatalogEntry, ...] = (_entry(),),
    grants: tuple[MemoryPageGrant, ...] = (MemoryPageGrant(PRINCIPAL, PAGE),),
) -> GroundedMemorySelector:
    return GroundedMemorySelector(
        entries,
        (MemoryDomainMembership(PRINCIPAL, DOMAIN),),
        grants,
        rule_profile_id="grounded-memory-test-v1",
    )


def test_selection_is_canonical_and_binds_the_fact_snapshot() -> None:
    first = _selector().select(_request())
    second = _selector().select(_request())

    assert first is not None
    assert second is not None
    assert first == second
    assert first.selected_page_id == PAGE


def test_fact_order_does_not_change_the_selection_proof() -> None:
    other = "22" * 32
    entries = (_entry(), _entry(other, "concept:unrelated"))
    grants = (MemoryPageGrant(PRINCIPAL, PAGE), MemoryPageGrant(PRINCIPAL, other))
    first = _selector(entries=entries, grants=grants).select(_request())
    second = _selector(
        entries=tuple(reversed(entries)), grants=tuple(reversed(grants))
    ).select(_request())

    assert first is not None
    assert first == second


def test_grant_domain_concept_and_epoch_fail_closed() -> None:
    assert _selector(grants=()).select(_request()) is None
    assert _selector().select(_request(principal="bob")) is None
    assert _selector().select(_request(domain="team-beta")) is None
    assert _selector().select(_request(concept="concept:other")) is None
    assert _selector().select(_request(epoch=9)) is None
    assert _selector().select(_request(epoch=10_001)) is None


def test_ambiguous_service_fails_instead_of_choosing_by_order() -> None:
    other = "22" * 32
    selector = _selector(
        entries=(_entry(), _entry(other)),
        grants=(MemoryPageGrant(PRINCIPAL, PAGE), MemoryPageGrant(PRINCIPAL, other)),
    )

    with pytest.raises(AmbiguousMemorySelectionError):
        selector.select(_request())


def test_unrelated_corpus_changes_evidence_not_selection() -> None:
    baseline = _selector().select(_request())
    entries = (_entry(),) + tuple(
        _entry(f"{index + 1:064x}", f"concept:unrelated:{index}")
        for index in range(1_000)
    )
    expanded = _selector(entries=entries).select(_request())

    assert baseline is not None
    assert expanded is not None
    assert expanded.selected_page_id == baseline.selected_page_id
    assert expanded.fact_snapshot_id != baseline.fact_snapshot_id
    assert expanded.proof_id != baseline.proof_id

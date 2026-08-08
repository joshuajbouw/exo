from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from semantic_memory_invocation import (
    SemanticInvocationError,
    SemanticInvocationProposal,
    SemanticMemoryGrounder,
    SemanticRelationFact,
    decode_semantic_proposal,
    extract_gemma_final_payload,
    semantic_proposal_profile_id,
)


def _fact(
    subject: str = "SOVARA-A100",
    concept: str = "concept:alpha",
) -> SemanticRelationFact:
    return SemanticRelationFact(
        "private-domain",
        "private-calibration-word-of",
        subject,
        concept,
        "evidence:alpha",
    )


def test_canonical_proposal_grounds_with_query_bound_proof() -> None:
    proposal = decode_semantic_proposal(
        '{"schema":1,"relation_id":"private-calibration-word-of",'
        '"subject_id":"SOVARA-A100"}'
    )
    assert proposal is not None
    grounder = SemanticMemoryGrounder((_fact(),))

    proof = grounder.ground(
        invocation_id="invocation:1",
        privacy_domain_id="private-domain",
        query="Which calibration word belongs to SOVARA-A100?",
        proposal=proposal,
        proposal_profile_id="proposal-profile:1",
    )

    assert proof is not None
    assert proof.concept_id == "concept:alpha"
    assert proof.relation_snapshot_id == grounder.snapshot_id
    assert proof.proposal_profile_id == "proposal-profile:1"
    changed_query = grounder.ground(
        invocation_id="invocation:1",
        privacy_domain_id="private-domain",
        query="A different current utterance",
        proposal=proposal,
        proposal_profile_id="proposal-profile:1",
    )
    changed_profile = grounder.ground(
        invocation_id="invocation:1",
        privacy_domain_id="private-domain",
        query="Which calibration word belongs to SOVARA-A100?",
        proposal=proposal,
        proposal_profile_id="proposal-profile:2",
    )
    assert changed_query is not None and changed_query.proof_id != proof.proof_id
    assert changed_profile is not None and changed_profile.proof_id != proof.proof_id


@pytest.mark.parametrize(
    "payload",
    (
        ' {"schema":1,"relation_id":"private-calibration-word-of",'
        '"subject_id":"SOVARA-A100"}',
        '{"relation_id":"private-calibration-word-of","schema":1,'
        '"subject_id":"SOVARA-A100"}',
        '{"schema":1,"relation_id":"private-calibration-word-of",'
        '"subject_id":"SOVARA-A100","extra":true}',
        '{"schema":2,"relation_id":"private-calibration-word-of",'
        '"subject_id":"SOVARA-A100"}',
    ),
)
def test_noncanonical_or_unknown_proposal_fails(payload: str) -> None:
    with pytest.raises(SemanticInvocationError):
        decode_semantic_proposal(payload)


def test_null_is_structural_silence() -> None:
    assert decode_semantic_proposal("null") is None


def test_unknown_relation_subject_or_domain_does_not_ground() -> None:
    grounder = SemanticMemoryGrounder((_fact(),))
    cases = (
        ("other-domain", "private-calibration-word-of", "SOVARA-A100"),
        ("private-domain", "color-of", "SOVARA-A100"),
        ("private-domain", "private-calibration-word-of", "SOVARA-UNKNOWN"),
    )
    for domain, relation, subject in cases:
        assert (
            grounder.ground(
                invocation_id="invocation:negative",
                privacy_domain_id=domain,
                query="negative control",
                proposal=SemanticInvocationProposal(relation, subject),
                proposal_profile_id="proposal-profile:1",
            )
            is None
        )


def test_ambiguous_relation_view_is_rejected() -> None:
    with pytest.raises(SemanticInvocationError, match="ambiguous"):
        SemanticMemoryGrounder((_fact(), _fact(concept="concept:other")))


def test_gemma_final_channel_must_be_unique() -> None:
    assert (
        extract_gemma_final_payload(
            '<|channel>thought\nprivate reasoning<channel|>{"schema":1}'
        )
        == '{"schema":1}'
    )
    with pytest.raises(SemanticInvocationError):
        extract_gemma_final_payload("no final channel")
    with pytest.raises(SemanticInvocationError):
        extract_gemma_final_payload("<channel|>one<channel|>two")


def test_proposal_profile_binds_semantics_visible_environment() -> None:
    baseline = semantic_proposal_profile_id(
        model_id="gemma",
        runtime_profile="mlx-v1",
        interface_contract="relation-a(subject)",
    )
    assert baseline == semantic_proposal_profile_id(
        model_id="gemma",
        runtime_profile="mlx-v1",
        interface_contract="relation-a(subject)",
    )
    assert baseline != semantic_proposal_profile_id(
        model_id="gemma",
        runtime_profile="mlx-v1",
        interface_contract="relation-b(subject)",
    )

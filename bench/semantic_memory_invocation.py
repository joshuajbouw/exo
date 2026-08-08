"""Untrusted language proposals resolved through exact memory relations."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import msgspec

_SCHEMA = 1
_IDENTITY_CONTEXT = b"exo.semantic-memory-invocation.v1\0"
_GEMMA_FINAL_MARKER = "<channel|>"


class SemanticInvocationError(ValueError):
    """A semantic proposal or relation view is invalid."""


def _field(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _text(value: str) -> bytes:
    if not value:
        raise SemanticInvocationError("semantic identity field is empty")
    return _field(value.encode("utf-8"))


def _identity(*parts: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_IDENTITY_CONTEXT)
    for part in parts:
        digest.update(_field(part))
    return digest.hexdigest()


class _ProposalWire(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    schema: int
    relation_id: str
    subject_id: str


@dataclass(frozen=True, slots=True, order=True)
class SemanticRelationFact:
    privacy_domain_id: str
    relation_id: str
    subject_id: str
    concept_id: str
    evidence_closure_id: str

    def __post_init__(self) -> None:
        _text(self.privacy_domain_id)
        _text(self.relation_id)
        _text(self.subject_id)
        _text(self.concept_id)
        _text(self.evidence_closure_id)

    def canonical_bytes(self) -> bytes:
        return b"".join(
            (
                _text(self.privacy_domain_id),
                _text(self.relation_id),
                _text(self.subject_id),
                _text(self.concept_id),
                _text(self.evidence_closure_id),
            )
        )


@dataclass(frozen=True, slots=True)
class SemanticInvocationProposal:
    relation_id: str
    subject_id: str

    def __post_init__(self) -> None:
        _text(self.relation_id)
        _text(self.subject_id)

    def canonical_json(self) -> bytes:
        return msgspec.json.encode(
            _ProposalWire(_SCHEMA, self.relation_id, self.subject_id)
        )


@dataclass(frozen=True, slots=True)
class SemanticGroundingProof:
    proof_id: str
    invocation_id: str
    query_sha256: str
    proposal_sha256: str
    proposal_profile_id: str
    relation_snapshot_id: str
    concept_id: str

    def __post_init__(self) -> None:
        for value in (
            self.proof_id,
            self.invocation_id,
            self.query_sha256,
            self.proposal_sha256,
            self.proposal_profile_id,
            self.relation_snapshot_id,
            self.concept_id,
        ):
            _text(value)


def extract_gemma_final_payload(output: str) -> str:
    """Discard model-private reasoning and return one final-channel payload."""

    if output.count(_GEMMA_FINAL_MARKER) != 1:
        raise SemanticInvocationError("Gemma output has no unique final channel")
    _, payload = output.split(_GEMMA_FINAL_MARKER, 1)
    if not payload:
        raise SemanticInvocationError("Gemma final channel is empty")
    return payload


def decode_semantic_proposal(payload: str) -> SemanticInvocationProposal | None:
    """Accept null or one byte-canonical typed invocation."""

    encoded = payload.encode("utf-8")
    if encoded == b"null":
        return None
    try:
        wire = msgspec.json.decode(encoded, type=_ProposalWire)
    except msgspec.DecodeError as error:
        raise SemanticInvocationError("semantic proposal is invalid JSON") from error
    if wire.schema != _SCHEMA or msgspec.json.encode(wire) != encoded:
        raise SemanticInvocationError("semantic proposal is non-canonical")
    return SemanticInvocationProposal(wire.relation_id, wire.subject_id)


def semantic_proposal_profile_id(
    *,
    model_id: str,
    runtime_profile: str,
    interface_contract: str,
) -> str:
    """Identify exactly the semantics-visible proposal environment."""

    return _identity(
        b"proposal-profile",
        _text(model_id),
        _text(runtime_profile),
        _text(interface_contract),
    )


class SemanticMemoryGrounder:
    """Resolve a proposal against one immutable, domain-scoped relation view."""

    def __init__(self, facts: tuple[SemanticRelationFact, ...]) -> None:
        ordered = tuple(sorted(facts))
        by_key: dict[tuple[str, str, str], str] = {}
        for fact in ordered:
            key = (fact.privacy_domain_id, fact.relation_id, fact.subject_id)
            if key in by_key:
                raise SemanticInvocationError("semantic relation key is ambiguous")
            by_key[key] = fact.concept_id
        self._by_key = by_key
        self._snapshot_id = _identity(
            b"relation-snapshot",
            *(b"fact" + _field(fact.canonical_bytes()) for fact in ordered),
        )

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    def ground(
        self,
        *,
        invocation_id: str,
        privacy_domain_id: str,
        query: str,
        proposal: SemanticInvocationProposal,
        proposal_profile_id: str,
    ) -> SemanticGroundingProof | None:
        _text(invocation_id)
        _text(privacy_domain_id)
        _text(query)
        _text(proposal_profile_id)
        concept_id = self._by_key.get(
            (privacy_domain_id, proposal.relation_id, proposal.subject_id)
        )
        if concept_id is None:
            return None
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        proposal_sha256 = hashlib.sha256(proposal.canonical_json()).hexdigest()
        proof_id = _identity(
            b"grounding-proof",
            _text(invocation_id),
            _text(privacy_domain_id),
            _text(query_sha256),
            _text(proposal_sha256),
            _text(proposal_profile_id),
            _text(self._snapshot_id),
            _text(concept_id),
        )
        return SemanticGroundingProof(
            proof_id,
            invocation_id,
            query_sha256,
            proposal_sha256,
            proposal_profile_id,
            self._snapshot_id,
            concept_id,
        )

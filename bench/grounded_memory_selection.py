"""Deterministic, capability-scoped selection for latent memory pages."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

_IDENTITY_CONTEXT = b"exo.grounded-latent-memory-selection.v1\0"


class MemorySelectionError(ValueError):
    """Ground facts cannot produce one safe memory selection."""


class AmbiguousMemorySelectionError(MemorySelectionError):
    """More than one page serves the same invocation without composition."""


def _field(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _text(value: str) -> bytes:
    return _field(value.encode("utf-8"))


def _identity(*parts: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_IDENTITY_CONTEXT)
    for part in parts:
        digest.update(_field(part))
    return digest.hexdigest()


def _require(value: str, name: str) -> None:
    if not value:
        raise MemorySelectionError(f"{name} is empty")


@dataclass(frozen=True, slots=True)
class MemorySelectionProof:
    """Grounded external selection bound to exactly one mounted page."""

    proof_id: str
    fact_snapshot_id: str
    selected_page_id: str

    def __post_init__(self) -> None:
        _require(self.proof_id, "memory selection proof identity")
        _require(self.fact_snapshot_id, "memory selection fact snapshot")
        _require(self.selected_page_id, "memory selection page identity")

    @classmethod
    def for_page(
        cls,
        page: object,
        *,
        proof_id: str,
        fact_snapshot_id: str,
    ) -> MemorySelectionProof:
        page_id = getattr(page, "page_id", None)
        if not isinstance(page_id, str):
            raise MemorySelectionError("selected object has no page identity")
        return cls(proof_id, fact_snapshot_id, page_id)


@dataclass(frozen=True, slots=True, order=True)
class MemoryDomainMembership:
    principal_id: str
    privacy_domain_id: str

    def __post_init__(self) -> None:
        _require(self.principal_id, "membership principal identity")
        _require(self.privacy_domain_id, "membership privacy domain")

    def canonical_bytes(self) -> bytes:
        return _text(self.principal_id) + _text(self.privacy_domain_id)


@dataclass(frozen=True, slots=True, order=True)
class MemoryPageGrant:
    principal_id: str
    page_id: str

    def __post_init__(self) -> None:
        _require(self.principal_id, "grant principal identity")
        _require(self.page_id, "granted page identity")

    def canonical_bytes(self) -> bytes:
        return _text(self.principal_id) + _text(self.page_id)


@dataclass(frozen=True, slots=True, order=True)
class MemoryCatalogEntry:
    page_id: str
    evidence_closure_id: str
    privacy_domain_id: str
    concept_id: str
    model_id: str
    runtime_profile: str
    first_epoch: int
    last_epoch: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.page_id, "catalog page identity"),
            (self.evidence_closure_id, "catalog evidence closure"),
            (self.privacy_domain_id, "catalog privacy domain"),
            (self.concept_id, "catalog concept identity"),
            (self.model_id, "catalog model identity"),
            (self.runtime_profile, "catalog runtime profile"),
        ):
            _require(value, name)
        if self.first_epoch < 0 or self.last_epoch < self.first_epoch:
            raise MemorySelectionError("catalog epoch interval is invalid")

    def canonical_bytes(self) -> bytes:
        return b"".join(
            (
                _text(self.page_id),
                _text(self.evidence_closure_id),
                _text(self.privacy_domain_id),
                _text(self.concept_id),
                _text(self.model_id),
                _text(self.runtime_profile),
                struct.pack(">QQ", self.first_epoch, self.last_epoch),
            )
        )


@dataclass(frozen=True, slots=True)
class MemorySelectionRequest:
    invocation_id: str
    principal_id: str
    privacy_domain_id: str
    concept_id: str
    model_id: str
    runtime_profile: str
    epoch: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.invocation_id, "invocation identity"),
            (self.principal_id, "invocation principal identity"),
            (self.privacy_domain_id, "invocation privacy domain"),
            (self.concept_id, "invocation concept identity"),
            (self.model_id, "invocation model identity"),
            (self.runtime_profile, "invocation runtime profile"),
        ):
            _require(value, name)
        if self.epoch < 0:
            raise MemorySelectionError("invocation epoch is negative")

    def canonical_bytes(self) -> bytes:
        return b"".join(
            (
                _text(self.invocation_id),
                _text(self.principal_id),
                _text(self.privacy_domain_id),
                _text(self.concept_id),
                _text(self.model_id),
                _text(self.runtime_profile),
                struct.pack(">Q", self.epoch),
            )
        )


class GroundedMemorySelector:
    """Derive one page from facts; query text is intentionally not an input."""

    def __init__(
        self,
        entries: tuple[MemoryCatalogEntry, ...],
        memberships: tuple[MemoryDomainMembership, ...],
        grants: tuple[MemoryPageGrant, ...],
        *,
        rule_profile_id: str,
    ) -> None:
        _require(rule_profile_id, "memory selection rule profile")
        if len({entry.page_id for entry in entries}) != len(entries):
            raise MemorySelectionError("duplicate catalog page identity")
        self._entries = tuple(sorted(entries))
        self._memberships = frozenset(memberships)
        self._grants = frozenset(grants)
        self._rule_profile_id = rule_profile_id
        by_domain_concept: dict[tuple[str, str], list[MemoryCatalogEntry]] = {}
        for entry in self._entries:
            by_domain_concept.setdefault(
                (entry.privacy_domain_id, entry.concept_id), []
            ).append(entry)
        self._by_domain_concept = {
            key: tuple(value) for key, value in by_domain_concept.items()
        }
        self._fact_snapshot_id = _identity(
            b"fact-snapshot",
            *(b"entry" + _field(entry.canonical_bytes()) for entry in self._entries),
            *(
                b"membership" + _field(fact.canonical_bytes())
                for fact in sorted(self._memberships)
            ),
            *(
                b"grant" + _field(fact.canonical_bytes())
                for fact in sorted(self._grants)
            ),
        )

    @property
    def fact_snapshot_id(self) -> str:
        return self._fact_snapshot_id

    def select(self, request: MemorySelectionRequest) -> MemorySelectionProof | None:
        membership = MemoryDomainMembership(
            request.principal_id, request.privacy_domain_id
        )
        if membership not in self._memberships:
            return None
        candidates = tuple(
            entry
            for entry in self._by_domain_concept.get(
                (request.privacy_domain_id, request.concept_id), ()
            )
            if MemoryPageGrant(request.principal_id, entry.page_id) in self._grants
            and entry.first_epoch <= request.epoch <= entry.last_epoch
            and entry.model_id == request.model_id
            and entry.runtime_profile == request.runtime_profile
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            raise AmbiguousMemorySelectionError(
                "multiple latent pages serve one invocation without composition"
            )
        selected = candidates[0]
        snapshot_id = self.fact_snapshot_id
        proof_id = _identity(
            b"selection-proof",
            request.canonical_bytes(),
            _text(snapshot_id),
            _text(self._rule_profile_id),
            _text(selected.page_id),
            _text(selected.evidence_closure_id),
        )
        return MemorySelectionProof(proof_id, snapshot_id, selected.page_id)

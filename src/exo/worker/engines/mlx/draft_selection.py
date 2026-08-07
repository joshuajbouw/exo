"""Provider-neutral candidate selection for verified speculative decoding.

Selectors may use a GPU reasoner, a corpus index, or an exact remembered
continuation. They only rank proposals: the target model verifies every token
before Exo returns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from exo.worker.engines.mlx.types import DraftContinuation


@dataclass(frozen=True, slots=True)
class DraftCandidate:
    """One ranked, non-authoritative token continuation.

    Durable selectors bind the identifier to the token bytes and their
    provenance. Exo uses the identifier only for feedback correlation.
    """

    candidate_id: bytes
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DraftSelection:
    """Candidates returned by one selector invocation.

    A durable selection identity binds the query, selector/runtime profile,
    and relation snapshot which produced the ordering.
    """

    selection_id: bytes
    candidates: tuple[DraftCandidate, ...]


@dataclass(frozen=True, slots=True)
class DraftVerificationOutcome:
    """Target-model feedback for selector learning and later evidence minting."""

    selection_id: bytes
    candidate_id: bytes
    proposed_tokens: int
    accepted_tokens: int
    first_mismatch: int | None
    selection_duration_ns: int
    verification_duration_ns: int
    verification_passes: int


class DraftCandidateSelector(Protocol):
    """Rank candidate continuations without granting them authority.

    Implementations may execute on the GPU. The caller scopes each selector
    instance to one sharing/privacy domain. ``schedule_outcome`` receives only
    facts established by target-model verification; it must not affect the
    tokens already returned for the invocation.
    """

    @property
    def supports_dynamic_refill(self) -> bool:
        """Whether this selector may be queried again during direct decoding."""
        ...

    def select(
        self,
        prompt_tokens: mx.array,
        remembered: DraftContinuation | None,
    ) -> DraftSelection | None:
        """Return candidates in descending preference order."""

    def schedule_outcome(self, outcome: DraftVerificationOutcome) -> None:
        """Enqueue verified feedback without blocking token generation."""


def exact_remembered_selection(
    remembered: DraftContinuation | None,
) -> DraftSelection | None:
    """Adapt the existing exact-prompt memory into the selector contract."""

    if remembered is None:
        return None
    # This is an in-process accelerator identifier, not durable object
    # identity. Durable selectors supply their own content-addressed IDs.
    candidate_id = b"exact-remembered"
    return DraftSelection(
        selection_id=candidate_id,
        candidates=(DraftCandidate(candidate_id, remembered.tokens),),
    )

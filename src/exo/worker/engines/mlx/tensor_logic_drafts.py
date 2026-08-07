"""GPU-resident relational retrieval for speculative draft candidates.

The selector models each corpus row as a typed relation from a token-context
entity to a continuation object.  Selection is a T=0 contraction over those
facts.  It is intentionally non-authoritative: the target model still verifies
every proposed token before generation can expose it.
"""

from __future__ import annotations

import hashlib
import struct
from collections import deque
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from exo.worker.engines.mlx.draft_selection import (
    DraftCandidate,
    DraftSelection,
    DraftVerificationOutcome,
)
from exo.worker.engines.mlx.types import DraftContinuation

_IDENTITY_CONTEXT = b"exo.tensor-logic-draft-selector.v1\0"
_HASH_MULTIPLIERS = (257, 521, 1031, 2053)
_POSITION_MULTIPLIERS = (17, 31, 43, 73)
_SIGN_MULTIPLIERS = (13, 29, 61, 97)


@dataclass(frozen=True, slots=True)
class TensorLogicDraftProfile:
    """Semantics-visible parameters for context-entity construction."""

    context_tokens: int = 64
    feature_dimension: int = 2048
    hashes_per_token: int = 4
    maximum_candidates: int = 4
    maximum_live_candidates: int = 1
    maximum_draft_tokens: int = 128
    minimum_live_context_tokens: int = 2
    minimum_similarity: float = 0.72

    def __post_init__(self) -> None:
        if self.context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        if self.feature_dimension < 64:
            raise ValueError("feature_dimension must be at least 64")
        if not 1 <= self.hashes_per_token <= len(_HASH_MULTIPLIERS):
            raise ValueError("hashes_per_token must be between one and four")
        if self.maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if self.maximum_live_candidates < 1:
            raise ValueError("maximum_live_candidates must be positive")
        if self.maximum_draft_tokens < 2:
            raise ValueError("maximum_draft_tokens must be at least two")
        if not 1 <= self.minimum_live_context_tokens <= self.context_tokens:
            raise ValueError(
                "minimum_live_context_tokens must be within the context window"
            )
        if not 0.0 <= self.minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between zero and one")

    def canonical_bytes(self) -> bytes:
        """Return the unambiguous profile encoding used in identities."""

        return struct.pack(
            ">IIIIIII f",
            self.context_tokens,
            self.feature_dimension,
            self.hashes_per_token,
            self.maximum_candidates,
            self.maximum_live_candidates,
            self.maximum_draft_tokens,
            self.minimum_live_context_tokens,
            self.minimum_similarity,
        )


@dataclass(frozen=True, slots=True)
class TensorLogicDraftFact:
    """One provenance-bound context-to-continuation relation."""

    source_id: bytes
    context_tokens: tuple[int, ...]
    continuation_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.context_tokens:
            raise ValueError("context_tokens must not be empty")
        if not self.continuation_tokens:
            raise ValueError("continuation_tokens must not be empty")
        if any(token < 0 for token in self.context_tokens):
            raise ValueError("context token ids must not be negative")
        if any(token < 0 for token in self.continuation_tokens):
            raise ValueError("continuation token ids must not be negative")


def _encode_tokens(tokens: tuple[int, ...]) -> bytes:
    encoded = bytearray(struct.pack(">I", len(tokens)))
    for token in tokens:
        if token > 0xFFFF_FFFF:
            raise ValueError("token id exceeds the canonical u32 encoding")
        encoded.extend(struct.pack(">I", token))
    return bytes(encoded)


def _identity(*parts: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(_IDENTITY_CONTEXT)
    for part in parts:
        digest.update(struct.pack(">Q", len(part)))
        digest.update(part)
    return digest.digest()


def _context_entities(
    token_rows: mx.array,
    validity: mx.array,
    profile: TensorLogicDraftProfile,
) -> mx.array:
    """Construct normalized, position-bound entities on the GPU."""

    row_count, width = token_rows.shape
    positions = mx.arange(width - 1, -1, -1, dtype=mx.int32)
    token_values = token_rows[:, :, None] % profile.feature_dimension
    position_values = positions[None, :, None]
    hash_numbers = mx.arange(profile.hashes_per_token, dtype=mx.int32)[None, :]
    hash_multipliers = mx.array(
        _HASH_MULTIPLIERS[: profile.hashes_per_token], dtype=mx.int32
    )[None, None, :]
    position_multipliers = mx.array(
        _POSITION_MULTIPLIERS[: profile.hashes_per_token], dtype=mx.int32
    )[None, None, :]
    sign_multipliers = mx.array(
        _SIGN_MULTIPLIERS[: profile.hashes_per_token], dtype=mx.int32
    )[None, None, :]

    feature_indices = (
        token_values * hash_multipliers
        + position_values * position_multipliers
        + hash_numbers * 101
    ) % profile.feature_dimension
    signs = (
        ((token_values * sign_multipliers + position_values + hash_numbers) % 2) * 2 - 1
    ).astype(mx.float32)
    recency = (1.0 / (positions.astype(mx.float32) + 1.0))[None, :, None]
    values = signs * recency * validity[:, :, None]
    row_indices = mx.broadcast_to(
        mx.arange(row_count, dtype=mx.int32)[:, None, None],
        feature_indices.shape,
    )
    features = (
        mx.zeros((row_count, profile.feature_dimension), dtype=mx.float32)
        .at[row_indices.flatten(), feature_indices.flatten()]
        .add(values.flatten())
    )
    magnitudes = mx.sqrt(mx.sum(features * features, axis=1, keepdims=True))
    return features / mx.maximum(magnitudes, mx.array(1.0, dtype=mx.float32))


def _context_entity(tokens: mx.array, profile: TensorLogicDraftProfile) -> mx.array:
    """Construct one entity without moving live prompt tokens to the CPU."""

    suffix = tokens[-profile.context_tokens :].astype(mx.int32)[None, :]
    validity = mx.ones(suffix.shape, dtype=mx.float32)
    return _context_entities(suffix, validity, profile)[0]


class TensorLogicDraftSelector:
    """Rank corpus continuations using a GPU-resident relation contraction.

    Instances are scoped to one privacy domain.  Physical tensor sharing across
    domains belongs above this class; corpus identities and feedback never do.
    """

    def __init__(
        self,
        facts: tuple[TensorLogicDraftFact, ...],
        *,
        privacy_domain_id: bytes,
        runtime_profile_id: bytes,
        profile: TensorLogicDraftProfile | None = None,
        feedback_capacity: int = 4096,
        enable_experimental_dynamic_refill: bool = False,
    ) -> None:
        if not facts:
            raise ValueError("facts must not be empty")
        if not privacy_domain_id:
            raise ValueError("privacy_domain_id must not be empty")
        if not runtime_profile_id:
            raise ValueError("runtime_profile_id must not be empty")
        if feedback_capacity < 1:
            raise ValueError("feedback_capacity must be positive")

        self._profile = profile or TensorLogicDraftProfile()
        self._privacy_domain_id = privacy_domain_id
        self._runtime_profile_id = runtime_profile_id
        self._enable_experimental_dynamic_refill = enable_experimental_dynamic_refill
        self._facts = tuple(
            sorted(
                facts,
                key=lambda fact: (
                    fact.source_id,
                    fact.context_tokens,
                    fact.continuation_tokens,
                ),
            )
        )
        fact_encodings = tuple(self._fact_bytes(fact) for fact in self._facts)
        self._corpus_id = _identity(
            b"corpus",
            privacy_domain_id,
            runtime_profile_id,
            self._profile.canonical_bytes(),
            *fact_encodings,
        )
        self._candidate_ids = tuple(
            _identity(b"candidate", encoding) for encoding in fact_encodings
        )
        padded_contexts = tuple(
            (0,) * (self._profile.context_tokens - len(context))
            + context[-self._profile.context_tokens :]
            for context in (fact.context_tokens for fact in self._facts)
        )
        valid_contexts = tuple(
            (0.0,)
            * (
                self._profile.context_tokens
                - min(len(context), self._profile.context_tokens)
            )
            + (1.0,) * min(len(context), self._profile.context_tokens)
            for context in (fact.context_tokens for fact in self._facts)
        )
        self._subject_matrix = _context_entities(
            mx.array(padded_contexts, dtype=mx.int32),
            mx.array(valid_contexts, dtype=mx.float32),
            self._profile,
        )
        mx.eval(self._subject_matrix)
        self._feedback: deque[DraftVerificationOutcome] = deque(
            maxlen=feedback_capacity
        )

    @staticmethod
    def _fact_bytes(fact: TensorLogicDraftFact) -> bytes:
        return b"".join(
            (
                struct.pack(">Q", len(fact.source_id)),
                fact.source_id,
                _encode_tokens(fact.context_tokens),
                _encode_tokens(fact.continuation_tokens),
            )
        )

    @property
    def corpus_id(self) -> bytes:
        """Identity of the privacy-domain corpus and selector semantics."""

        return self._corpus_id

    @property
    def supports_dynamic_refill(self) -> bool:
        """Return the explicit opt-in for the unresolved equivalence gate."""

        return self._enable_experimental_dynamic_refill

    def select(
        self,
        prompt_tokens: mx.array,
        remembered: DraftContinuation | None,
    ) -> DraftSelection | None:
        """Contract the query against stored subjects and rank matching objects."""

        if prompt_tokens.size == 0:
            return None
        query = _context_entity(prompt_tokens, self._profile)
        similarities = self._subject_matrix @ query
        count = min(self._profile.maximum_candidates, len(self._facts))
        selected_indices = mx.argpartition(-similarities, count - 1)[:count]
        selected_similarities = similarities[selected_indices]
        ordering = mx.argsort(-selected_similarities)
        selected_indices = selected_indices[ordering]
        selected_similarities = selected_similarities[ordering]

        suffix = prompt_tokens[-self._profile.context_tokens :].astype(mx.int32)
        live_positions: mx.array | None = None
        live_scores: mx.array | None = None
        if len(prompt_tokens) > self._profile.context_tokens:
            live_row_count = len(prompt_tokens) - self._profile.context_tokens
            live_contexts = mx.as_strided(
                prompt_tokens.astype(mx.int32),
                shape=(live_row_count, self._profile.context_tokens),
                strides=(1, 1),
            )
            reverse_matches = mx.equal(live_contexts, suffix[None, :]).astype(mx.int32)
            suffix_match_lengths = mx.sum(
                mx.cumprod(reverse_matches[:, ::-1], axis=1), axis=1
            )
            live_count = min(self._profile.maximum_live_candidates, live_row_count)
            # Longest suffix wins. Earlier ties expose a longer already-verified
            # continuation, which gives block verification useful parallel work.
            live_ranking = suffix_match_lengths.astype(mx.float32) - (
                mx.arange(live_row_count, dtype=mx.float32) * 1e-7
            )
            live_positions = mx.argpartition(-live_ranking, live_count - 1)[:live_count]
            live_scores = suffix_match_lengths[live_positions]
            live_ordering = mx.argsort(-live_scores)
            live_positions = live_positions[live_ordering]
            live_scores = live_scores[live_ordering]
        # One synchronization exposes the small ranked result and the exact
        # query identity. The relation matrix and contraction remain resident.
        evaluated: list[mx.array] = [selected_indices, selected_similarities, suffix]
        if live_positions is not None and live_scores is not None:
            evaluated.extend((live_positions, live_scores))
        mx.eval(*evaluated)
        host_indices = tuple(cast(list[int], selected_indices.tolist()))
        host_similarities = tuple(cast(list[float], selected_similarities.tolist()))
        query_tokens = tuple(cast(list[int], suffix.tolist()))

        candidates: list[DraftCandidate] = []
        seen_tokens: set[tuple[int, ...]] = set()
        if remembered is not None:
            remembered_id = _identity(
                b"exact-remembered", _encode_tokens(remembered.tokens)
            )
            candidates.append(DraftCandidate(remembered_id, remembered.tokens))
            seen_tokens.add(remembered.tokens)

        if live_positions is not None and live_scores is not None:
            host_live_positions = cast(list[int], live_positions.tolist())
            host_live_scores = cast(list[int], live_scores.tolist())
            for position, matching_tokens in zip(
                host_live_positions, host_live_scores, strict=True
            ):
                if matching_tokens < self._profile.minimum_live_context_tokens:
                    continue
                continuation_start = position + self._profile.context_tokens
                continuation_end = min(
                    continuation_start + self._profile.maximum_draft_tokens,
                    len(prompt_tokens),
                )
                live_tokens = tuple(
                    cast(
                        list[int],
                        prompt_tokens[continuation_start:continuation_end].tolist(),
                    )
                )
                if len(live_tokens) < 2 or live_tokens in seen_tokens:
                    continue
                live_id = _identity(
                    b"live-context",
                    _encode_tokens(query_tokens),
                    _encode_tokens(live_tokens),
                )
                candidates.append(DraftCandidate(live_id, live_tokens))
                seen_tokens.add(live_tokens)

        for index, similarity in zip(host_indices, host_similarities, strict=True):
            if similarity < self._profile.minimum_similarity:
                continue
            fact = self._facts[index]
            if fact.continuation_tokens in seen_tokens:
                continue
            candidates.append(
                DraftCandidate(self._candidate_ids[index], fact.continuation_tokens)
            )
            seen_tokens.add(fact.continuation_tokens)

        if not candidates:
            return None
        selection_id = _identity(
            b"selection",
            self._corpus_id,
            _encode_tokens(query_tokens),
            *(candidate.candidate_id for candidate in candidates),
        )
        return DraftSelection(selection_id, tuple(candidates))

    def schedule_outcome(self, outcome: DraftVerificationOutcome) -> None:
        """Record already-verified feedback in a bounded, nonblocking queue."""

        self._feedback.append(outcome)

    def drain_outcomes(self) -> tuple[DraftVerificationOutcome, ...]:
        """Remove feedback for an external evidence/refinery consumer."""

        outcomes = tuple(self._feedback)
        self._feedback.clear()
        return outcomes

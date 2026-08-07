"""Tests for GPU-resident relational draft selection."""

from __future__ import annotations

import mlx.core as mx

from exo.worker.engines.mlx.draft_selection import DraftVerificationOutcome
from exo.worker.engines.mlx.tensor_logic_drafts import (
    TensorLogicDraftFact,
    TensorLogicDraftProfile,
    TensorLogicDraftSelector,
)
from exo.worker.engines.mlx.types import DraftContinuation


def _fact(
    source: str,
    context: tuple[int, ...],
    continuation: tuple[int, ...],
) -> TensorLogicDraftFact:
    return TensorLogicDraftFact(source.encode(), context, continuation)


def _selector(
    facts: tuple[TensorLogicDraftFact, ...],
    *,
    domain: bytes = b"team-a",
    minimum_similarity: float = 0.72,
) -> TensorLogicDraftSelector:
    return TensorLogicDraftSelector(
        facts,
        privacy_domain_id=domain,
        runtime_profile_id=b"mlx-semantic-profile",
        profile=TensorLogicDraftProfile(
            context_tokens=8,
            feature_dimension=256,
            hashes_per_token=4,
            maximum_candidates=3,
            minimum_similarity=minimum_similarity,
        ),
    )


def test_exact_context_retrieves_the_related_continuation() -> None:
    expected = _fact("source-a", (1, 2, 3, 4), (40, 41, 42))
    selector = _selector(
        (
            _fact("source-b", (9, 8, 7, 6), (60, 61)),
            expected,
            _fact("source-c", (2, 4, 6, 8), (80, 81)),
        )
    )

    selection = selector.select(mx.array([99, 1, 2, 3, 4]), None)

    assert selection is not None
    assert selection.candidates[0].tokens == expected.continuation_tokens


def test_unrelated_context_below_threshold_proposes_nothing() -> None:
    selector = _selector(
        (_fact("source-a", (1, 2, 3, 4), (40, 41)),),
        minimum_similarity=0.99,
    )

    assert selector.select(mx.array([101, 102, 103, 104]), None) is None


def test_exact_remembered_candidate_precedes_relational_candidates() -> None:
    continuation = (40, 41, 42)
    selector = _selector((_fact("source-a", (1, 2, 3, 4), continuation),))

    selection = selector.select(mx.array([1, 2, 3, 4]), DraftContinuation((70, 71)))

    assert selection is not None
    assert tuple(candidate.tokens for candidate in selection.candidates) == (
        (70, 71),
        continuation,
    )


def test_live_verified_context_is_an_ephemeral_relation_source() -> None:
    selector = TensorLogicDraftSelector(
        (_fact("source-a", (90, 91), (92, 93)),),
        privacy_domain_id=b"team-a",
        runtime_profile_id=b"mlx-semantic-profile",
        profile=TensorLogicDraftProfile(
            context_tokens=2,
            feature_dimension=256,
            maximum_candidates=2,
            maximum_draft_tokens=8,
            minimum_similarity=0.99,
        ),
    )

    selection = selector.select(mx.array([1, 2, 3, 4, 1, 2]), None)

    assert selection is not None
    assert selection.candidates[0].tokens == (3, 4, 1, 2)


def test_duplicate_remembered_tokens_are_verified_only_once() -> None:
    continuation = (40, 41, 42)
    selector = _selector((_fact("source-a", (1, 2, 3, 4), continuation),))

    selection = selector.select(mx.array([1, 2, 3, 4]), DraftContinuation(continuation))

    assert selection is not None
    assert len(selection.candidates) == 1


def test_corpus_identity_is_order_independent_and_domain_scoped() -> None:
    first = _fact("source-a", (1, 2), (3, 4))
    second = _fact("source-b", (5, 6), (7, 8))

    ordered = _selector((first, second))
    reversed_order = _selector((second, first))
    other_domain = _selector((first, second), domain=b"team-b")

    assert ordered.corpus_id == reversed_order.corpus_id
    assert ordered.corpus_id != other_domain.corpus_id


def test_dynamic_refill_requires_explicit_experimental_opt_in() -> None:
    fact = _fact("source-a", (1, 2), (3, 4))

    default_selector = _selector((fact,))
    experimental_selector = TensorLogicDraftSelector(
        (fact,),
        privacy_domain_id=b"team-a",
        runtime_profile_id=b"mlx-semantic-profile",
        enable_experimental_dynamic_refill=True,
    )

    assert not default_selector.supports_dynamic_refill
    assert experimental_selector.supports_dynamic_refill


def test_verified_feedback_is_bounded_and_drained() -> None:
    selector = TensorLogicDraftSelector(
        (_fact("source-a", (1, 2), (3, 4)),),
        privacy_domain_id=b"team-a",
        runtime_profile_id=b"mlx-semantic-profile",
        feedback_capacity=1,
    )
    first = DraftVerificationOutcome(b"s1", b"c1", 2, 1, 1, 10, 20, 1)
    second = DraftVerificationOutcome(b"s2", b"c2", 2, 2, None, 11, 21, 1)

    selector.schedule_outcome(first)
    selector.schedule_outcome(second)

    assert selector.drain_outcomes() == (second,)
    assert selector.drain_outcomes() == ()


def test_invalid_fact_token_rejected_before_gpu_admission() -> None:
    try:
        TensorLogicDraftFact(b"source", (-1,), (1,))
    except ValueError as error:
        assert "context token ids" in str(error)
    else:
        raise AssertionError("negative token id was accepted")

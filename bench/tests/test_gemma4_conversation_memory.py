from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from train_gemma4_conversation_memory import (
    TEST_QUERIES,
    TEST_STATEMENTS,
    TRAIN_QUERIES,
    TRAIN_STATEMENTS,
    VALUES,
    facts,
)


def test_conversation_memory_split_is_disjoint_and_balanced() -> None:
    registered = facts()
    train = tuple(fact for fact in registered if fact.split == "train")
    test = tuple(fact for fact in registered if fact.split == "test")

    assert len(train) == 32
    assert len(test) == 8
    assert {fact.entity for fact in train}.isdisjoint(fact.entity for fact in test)
    assert Counter(fact.value for fact in train) == Counter(
        {value: 4 for value in VALUES}
    )
    assert Counter(fact.value for fact in test) == Counter(
        {value: 1 for value in VALUES}
    )
    for statement_form in range(len(TRAIN_STATEMENTS)):
        assert Counter(
            fact.value for fact in train if fact.statement_form == statement_form
        ) == Counter({value: 1 for value in VALUES})


def test_heldout_language_forms_are_unseen() -> None:
    assert set(TRAIN_STATEMENTS).isdisjoint(TEST_STATEMENTS)
    assert set(TRAIN_QUERIES).isdisjoint(TEST_QUERIES)


def test_registered_fact_identity_is_stable() -> None:
    assert facts()[0].fact_id == "a3e3d64aa95fcd9bbfdc5eeb"
    assert facts()[-1].fact_id == "31bbef32b97605795662bea2"

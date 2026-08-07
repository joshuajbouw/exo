# type: ignore

from collections import Counter
from pathlib import Path

from bench.persistent_weight_memory import (
    CONTROL_PROMPTS,
    DECISIONS,
    build_cases,
    build_stress_cases,
    prepare,
    prepare_stress,
)


def test_frozen_cases_are_balanced_and_disjoint() -> None:
    cases = build_cases()

    assert len(cases) == 320
    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.prompt for case in cases}) == len(cases)
    assert Counter(case.split for case in cases) == {
        "train": 256,
        "valid": 32,
        "test": 32,
    }
    for split, expected_per_decision in (("train", 64), ("valid", 8), ("test", 8)):
        decisions = Counter(case.decision for case in cases if case.split == split)
        assert decisions == {decision: expected_per_decision for decision in DECISIONS}


def test_held_out_cases_change_structure_and_surface() -> None:
    cases = build_cases()
    train = [case for case in cases if case.split == "train"]
    test = [case for case in cases if case.split == "test"]

    assert {case.depth for case in train} == {1, 2, 3, 4}
    assert {case.depth for case in test} == {6, 7}
    assert {case.template for case in train} == set(range(6))
    assert {case.template for case in cases if case.split == "valid"} == {6}
    assert {case.template for case in test} == {7}
    assert sum(not case.named_calculus for case in train) == len(train) // 2
    assert sum(not case.named_calculus for case in test) == len(test) // 2
    for decision in DECISIONS:
        decision_cases = [case for case in test if case.decision == decision]
        assert Counter(
            (case.depth, case.named_calculus) for case in decision_cases
        ) == {
            (6, True): 2,
            (7, True): 2,
            (6, False): 2,
            (7, False): 2,
        }
    assert all("What is the Kelthara" not in case.prompt for case in cases)
    changed_decisions = {
        case.decision for case in test if "changed after the vey" in case.prompt
    }
    unreceipted_decisions = {
        case.decision for case in test if "is unreceipted" in case.prompt
    }
    assert changed_decisions == {"RECOMPUTE", "REUSE"}
    assert unreceipted_decisions == {"REFUSE", "REUSE"}
    for decision in DECISIONS:
        decision_cases = [case for case in train if case.decision == decision]
        assert Counter(
            (case.depth, case.named_calculus) for case in decision_cases
        ) == {(depth, named): 8 for depth in (1, 2, 3, 4) for named in (True, False)}


def test_prepare_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    prepare(first)
    prepare(second)

    for name in (
        "manifest.json",
        "train.jsonl",
        "train.cases.jsonl",
        "valid.jsonl",
        "valid.cases.jsonl",
        "test.jsonl",
        "test.cases.jsonl",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_behavior_controls_are_distinct_and_unrelated() -> None:
    assert len(CONTROL_PROMPTS) == 8
    assert len(set(CONTROL_PROMPTS)) == len(CONTROL_PROMPTS)
    assert all("Kelthara" not in prompt for prompt in CONTROL_PROMPTS)


def test_stress_cases_are_balanced_and_deeper(tmp_path: Path) -> None:
    cases = build_stress_cases()

    assert len(cases) == 128
    assert len({case.case_id for case in cases}) == len(cases)
    assert Counter(case.decision for case in cases) == {
        decision: 32 for decision in DECISIONS
    }
    assert {case.depth for case in cases} == set(range(6, 13))
    assert {case.template for case in cases} == {8}
    assert sum(case.named_calculus for case in cases) == len(cases) // 2

    prepare_stress(tmp_path)
    assert (tmp_path / "stress.manifest.json").is_file()

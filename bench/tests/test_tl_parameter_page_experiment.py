# type: ignore

from __future__ import annotations

from collections import Counter

from bench.tl_parameter_page_experiment import SPLIT_COUNTS, cases


def test_domains_share_prompts_and_have_opposite_balanced_labels() -> None:
    all_cases = cases()

    assert Counter(case.split for case in all_cases) == SPLIT_COUNTS
    assert len({case.case_id for case in all_cases}) == len(all_cases)
    for split, count in SPLIT_COUNTS.items():
        selected = tuple(case for case in all_cases if case.split == split)
        assert Counter(case.alpha_label for case in selected) == {
            "VELA": count // 2,
            "NOMA": count // 2,
        }
        assert all(case.alpha_label != case.beta_label for case in selected)
        for template_index in {case.template_index for case in selected}:
            by_template = tuple(
                case for case in selected if case.template_index == template_index
            )
            assert Counter(case.reachable for case in by_template) == {
                False: len(by_template) // 2,
                True: len(by_template) // 2,
            }


def test_test_entities_and_prompt_forms_are_held_out() -> None:
    all_cases = cases()
    train_prompts = {case.prompt for case in all_cases if case.split == "train"}
    test_prompts = {case.prompt for case in all_cases if case.split == "test"}

    assert train_prompts.isdisjoint(test_prompts)
    assert {case.template_index for case in all_cases if case.split == "test"} == {0, 1}
    assert all(
        "following arrows" in prompt or "directed links" in prompt
        for prompt in test_prompts
    )

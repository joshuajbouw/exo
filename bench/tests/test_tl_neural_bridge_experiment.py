from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parents[1]))

from tl_neural_bridge_experiment import PAGE_VALUES, prepare, proof_value
from tl_parameter_page_experiment import cases


def test_bridge_corpora_are_domain_neutral_and_value_constant(tmp_path: Path) -> None:
    prepare(tmp_path)

    manifest = cast(
        dict[str, object], json.loads((tmp_path / "manifest.json").read_text())
    )
    assert manifest["domain_identity_in_training"] is False
    assert manifest["page_value_constant_per_corpus"] is True
    for value in PAGE_VALUES:
        records = [
            cast(dict[str, object], json.loads(line))
            for line in (tmp_path / value.lower() / "train.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(records) == 256
        outputs: set[object] = set()
        for record in records:
            messages = cast(list[dict[str, object]], record["messages"])
            outputs.add(messages[-1]["content"])
        assert outputs == {value}


def test_inverted_proof_selects_the_opposite_page() -> None:
    for case in (case for case in cases() if case.split == "test"):
        for domain in ("alpha", "beta"):
            valid = proof_value(case, domain, invert=False)
            inverted = proof_value(case, domain, invert=True)
            assert {valid, inverted} == set(PAGE_VALUES)

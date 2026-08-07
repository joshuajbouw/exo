#!/usr/bin/env python3
# type: ignore
"""Test proof-selected neural pages without asking the model to redo logic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from tl_parameter_page_experiment import Case, Domain, cases

type PageValue = Literal["VELA", "NOMA"]

FORMAT = "serevin-tl-neural-bridge-v1"
PAGE_VALUES: tuple[PageValue, ...] = ("VELA", "NOMA")


@dataclass(frozen=True, slots=True)
class BridgeEvaluation:
    case_id: str
    proof_value: str
    page_value: str
    expected: str
    predicted: str | None
    correct: bool
    response: str


def _record(case: Case, page_value: PageValue) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": case.prompt},
            {"role": "assistant", "content": page_value},
        ]
    }


def prepare(output_root: Path) -> None:
    all_cases = cases()
    for value in PAGE_VALUES:
        value_dir = output_root / value.lower()
        value_dir.mkdir(parents=True, exist_ok=True)
        for split in cast(tuple[str, ...], ("train", "valid", "test")):
            split_cases = (case for case in all_cases if case.split == split)
            with (value_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for case in split_cases:
                    handle.write(
                        json.dumps(_record(case, value), sort_keys=True) + "\n"
                    )
    manifest = {
        "format": FORMAT,
        "source_case_format": "serevin-parameter-pages-v2",
        "case_digest": hashlib.sha256(
            json.dumps([asdict(case) for case in all_cases], sort_keys=True).encode()
        ).hexdigest(),
        "page_values": PAGE_VALUES,
        "domain_identity_in_training": False,
        "page_value_constant_per_corpus": True,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _adapter_digest(adapter_path: Path) -> str:
    return hashlib.sha256(
        (adapter_path / "adapters.safetensors").read_bytes()
    ).hexdigest()


def proof_value(case: Case, domain: Domain, *, invert: bool) -> PageValue:
    value = cast(PageValue, case.label(domain))
    if not invert:
        return value
    return "NOMA" if value == "VELA" else "VELA"


def evaluate(
    *,
    model_path: Path,
    adapter_path: Path,
    page_value: PageValue,
    domain: Domain,
    invert_proof: bool,
    output: Path,
    maximum_tokens: int,
) -> None:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters
    from scoped_parameter_pages_mlx import scope_loaded_lora

    model, tokenizer = load(str(model_path))
    model = load_adapters(model, str(adapter_path))
    page_id = _adapter_digest(adapter_path)
    mounted = scope_loaded_lora(model, page_id=page_id)
    proof_mode = "inverted" if invert_proof else "valid"
    mounted.activate(f"tl-proof:{domain}:{proof_mode}:{page_value}:{page_id}")
    selected = tuple(
        case
        for case in cases()
        if case.split == "test"
        and proof_value(case, domain, invert=invert_proof) == page_value
    )
    pattern = re.compile(r"\b(VELA|NOMA)\b")
    started = time.perf_counter()
    results: list[BridgeEvaluation] = []
    for case in selected:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": case.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=maximum_tokens,
            sampler=make_sampler(temp=0),
        ).strip()
        matches = pattern.findall(response)
        predicted = matches[-1] if matches else None
        selected_value = proof_value(case, domain, invert=invert_proof)
        expected = selected_value
        results.append(
            BridgeEvaluation(
                case_id=case.case_id,
                proof_value=selected_value,
                page_value=page_value,
                expected=expected,
                predicted=predicted,
                correct=predicted == expected,
                response=response,
            )
        )
    mounted.deactivate()
    correct = sum(result.correct for result in results)
    payload = {
        "format": FORMAT,
        "domain": domain,
        "inverted_proof": invert_proof,
        "page_value": page_value,
        "page_id": page_id,
        "cases": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "seconds": time.perf_counter() - started,
        "results": [asdict(result) for result in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "results"},
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--model-path", type=Path, required=True)
    evaluate_parser.add_argument("--adapter-path", type=Path, required=True)
    evaluate_parser.add_argument("--page-value", choices=PAGE_VALUES, required=True)
    evaluate_parser.add_argument("--domain", choices=("alpha", "beta"), required=True)
    evaluate_parser.add_argument("--invert-proof", action="store_true")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--maximum-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_root)
    else:
        evaluate(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            page_value=args.page_value,
            domain=args.domain,
            invert_proof=args.invert_proof,
            output=args.output,
            maximum_tokens=args.maximum_tokens,
        )


if __name__ == "__main__":
    main()

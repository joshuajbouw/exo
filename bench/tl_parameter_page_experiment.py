#!/usr/bin/env python3
# type: ignore
"""Train and evaluate contradictory Tensor-Logic-selected parameter pages.

The two privacy domains receive byte-identical graph questions and opposite
invented conventions. Domain identity is never present in the prompt; only the
explicitly mounted page can distinguish the worlds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

type Domain = Literal["alpha", "beta"]
type Split = Literal["train", "valid", "test"]

FORMAT = "serevin-parameter-pages-v2"
LABELS = ("VELA", "NOMA")
SPLIT_COUNTS: dict[Split, int] = {"train": 256, "valid": 64, "test": 64}
SPLIT_OFFSETS: dict[Split, int] = {"train": 0, "valid": 10_000, "test": 20_000}
TRAIN_TEMPLATES = (
    "Resolve the Serevin relation. Facts: {facts} Query: can {source} reach {target}? Reply with VELA or NOMA only.",
    "Serevin edges are {facts} Decide whether a directed route goes from {source} to {target}. Output only VELA or NOMA.",
    "Given {facts} under Serevin, classify the path {source} -> {target}. One word only: VELA or NOMA.",
    "Check the directed Serevin chain. {facts} Is {target} downstream of {source}? Answer VELA or NOMA only.",
)
TEST_TEMPLATES = (
    "Serevin case: {facts} Determine if following arrows can carry {source} to {target}. Return exactly VELA or NOMA.",
    "For the directed links {facts}, judge the Serevin query ({source}, {target}). Respond using one of: VELA, NOMA.",
)


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    split: Split
    prompt: str
    reachable: bool
    alpha_label: str
    beta_label: str
    depth: int
    template_index: int

    def label(self, domain: Domain) -> str:
        return self.alpha_label if domain == "alpha" else self.beta_label

    def training_record(self, domain: Domain) -> dict[str, object]:
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.label(domain)},
            ]
        }


@dataclass(frozen=True, slots=True)
class Evaluation:
    case_id: str
    expected: str
    predicted: str | None
    correct: bool
    response: str


def _make_case(split: Split, index: int) -> Case:
    rng = random.Random(7_919 + SPLIT_OFFSETS[split] + index)
    reachable = index % 2 == 0
    depth = rng.randint(2, 5)
    base = SPLIT_OFFSETS[split] + index * 16
    nodes = tuple(f"s{base + position:05x}" for position in range(depth + 2))
    edges = [(nodes[position], nodes[position + 1]) for position in range(depth)]
    edges.extend(((nodes[-1], nodes[1]), (nodes[2], nodes[0])))
    source = nodes[0]
    target = nodes[depth] if reachable else nodes[-1]
    rng.shuffle(edges)
    facts = "; ".join(f"{left} -> {right}" for left, right in edges)
    templates = TEST_TEMPLATES if split == "test" else TRAIN_TEMPLATES
    # Adjacent cases have opposite reachability, so assigning one template to
    # each pair makes every prompt form internally label-balanced.
    template_index = (index // 2) % len(templates)
    prompt = templates[template_index].format(facts=facts, source=source, target=target)
    alpha_label = LABELS[0] if reachable else LABELS[1]
    beta_label = LABELS[1] if reachable else LABELS[0]
    case_id = hashlib.sha256(
        f"{FORMAT}\0{split}\0{index}\0{prompt}".encode()
    ).hexdigest()[:24]
    return Case(
        case_id=case_id,
        split=split,
        prompt=prompt,
        reachable=reachable,
        alpha_label=alpha_label,
        beta_label=beta_label,
        depth=depth,
        template_index=template_index,
    )


def cases() -> tuple[Case, ...]:
    return tuple(
        _make_case(split, index)
        for split, count in SPLIT_COUNTS.items()
        for index in range(count)
    )


def prepare(output_root: Path) -> None:
    all_cases = cases()
    for domain in cast(tuple[Domain, ...], ("alpha", "beta")):
        domain_dir = output_root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        for split in cast(tuple[Split, ...], ("train", "valid", "test")):
            records = (
                case.training_record(domain)
                for case in all_cases
                if case.split == split
            )
            with (domain_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "format": FORMAT,
        "counts": SPLIT_COUNTS,
        "case_digest": hashlib.sha256(
            json.dumps([asdict(case) for case in all_cases], sort_keys=True).encode()
        ).hexdigest(),
        "prompts_identical_between_domains": True,
        "labels_opposite_between_domains": True,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _adapter_digest(adapter_path: Path) -> str:
    return hashlib.sha256(
        (adapter_path / "adapters.safetensors").read_bytes()
    ).hexdigest()


def evaluate(
    *,
    model_path: Path,
    adapter_path: Path,
    domain: Domain,
    split: Split,
    activate_page: bool,
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
    if activate_page:
        mounted.activate(f"tl-proof:{domain}:{page_id}")
    pattern = re.compile(r"\b(VELA|NOMA)\b")
    selected = tuple(case for case in cases() if case.split == split)
    started = time.perf_counter()
    results: list[Evaluation] = []
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
        expected = case.label(domain)
        results.append(
            Evaluation(
                case_id=case.case_id,
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
        "split": split,
        "page_active": activate_page,
        "page_id": page_id,
        "cases": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "held_out_templates": split == "test",
        "held_out_entities": split in ("valid", "test"),
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
    evaluate_parser.add_argument("--domain", choices=("alpha", "beta"), required=True)
    evaluate_parser.add_argument(
        "--split", choices=("train", "valid", "test"), default="test"
    )
    evaluate_parser.add_argument(
        "--inactive",
        action="store_true",
        help="load the page but leave it inactive to measure the exact base path",
    )
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--maximum-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_root)
    else:
        evaluate(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            domain=args.domain,
            split=args.split,
            activate_page=not args.inactive,
            output=args.output,
            maximum_tokens=args.maximum_tokens,
        )


if __name__ == "__main__":
    main()

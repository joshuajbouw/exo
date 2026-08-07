#!/usr/bin/env python3
# type: ignore
"""Measure whether a novel rule survives a fresh model process as weights.

The experiment deliberately separates three artifacts:

* generated demonstrations and held-out cases;
* an MLX adapter trained by the existing ``mlx_lm lora`` command;
* evaluation in a new Python process that receives no teaching context or KV.

The strict gate uses free generation because a persisted rule is useful only if
the reloaded model can both apply it and reproduce its canonical proof trace.
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
from typing import Final, Literal, NewType, Sequence, cast

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import load_adapters

ModelId = NewType("ModelId", str)
Decision = Literal["EXECUTE", "RECOMPUTE", "REFUSE", "REUSE"]
Split = Literal["train", "valid", "test", "stress"]

DECISIONS: Final[tuple[Decision, ...]] = (
    "EXECUTE",
    "RECOMPUTE",
    "REFUSE",
    "REUSE",
)
FORMAT_NAME: Final = "kelthara-weight-memory-v8"
STRICT_GATE_NUMERATOR: Final = 3
STRICT_GATE_DENOMINATOR: Final = 4
CONTROL_PROMPTS: Final[tuple[str, ...]] = (
    "Reply with only the number: 37 + 58.",
    "Reply with only the number: 12 multiplied by 13.",
    "Reply with exactly one word: the capital of France.",
    "Sort these words alphabetically and reply only comma-separated: pear, apple, plum.",
    "Reply yes or no only: is every square a rectangle?",
    "Reply with the next number only: 2, 3, 5, 8, 13,",
    "Translate 'good morning' to Spanish. Reply with the translation only.",
    "Reply with the chemical symbol for gold only.",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    split: Split
    prompt: str
    decision: Decision
    trace: str
    rationale: str
    depth: int
    named_calculus: bool
    template: int

    def training_record(self) -> dict[str, object]:
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {
                    "role": "assistant",
                    "content": f"{self.trace}\nDECISION: {self.decision}",
                },
            ]
        }


@dataclass(frozen=True)
class GeneratedCaseResult:
    case_id: str
    expected: Decision
    predicted: Decision | None
    decision_correct: bool
    trace_correct: bool
    response: str


@dataclass(frozen=True)
class ControlResult:
    prompt: str
    response: str
    generation_seconds: float


def _case_id(split: Split, prompt: str, decision: Decision) -> str:
    digest = hashlib.blake2s(
        f"{FORMAT_NAME}\0{split}\0{prompt}\0{decision}".encode(),
        digest_size=16,
    ).hexdigest()
    return f"kelthara:{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_case(
    *,
    split: Split,
    names: Sequence[str],
    depth: int,
    decision: Decision,
    named_calculus: bool,
    template: int,
    condition_variant: int,
) -> Case:
    root = names[0]
    chain = list(names[: depth + 1])
    dependency_edges = ", ".join(
        f"{left} amber-leans-on {right}"
        for left, right in zip(chain, chain[1:], strict=False)
    )
    evidence_name = names[depth + 1]
    evidence_edge = f"{root} silver-cites {evidence_name}"

    if decision == "EXECUTE":
        condition = (
            f"{root} is a sarn: carrying it out changes the outside world. "
            "All kelms are receipted and unchanged."
        )
        rationale = "A sarn is an effect and must execute; remembered computation cannot replay it."
        trace = f"CLASS {root}=SARN; EFFECT=EXTERNAL"
    elif decision == "REFUSE":
        missing = chain[1 + condition_variant % depth]
        condition = (
            f"{missing} is unreceipted. No object has changed; every other object is "
            "receipted."
        )
        rationale = (
            "An unreceipted amber dependency makes the thren hollow, so it cannot mint "
            "a trusted vey."
        )
        trace = (
            f"TARGET={missing}; AMBER-MEMBER {missing} IN {root}=YES; "
            "RECEIPT=MISSING; THREN=HOLLOW"
        )
    elif decision == "RECOMPUTE":
        changed = chain[1 + condition_variant % depth]
        condition = (
            f"Every object is receipted. {changed} changed after the vey was minted; "
            "all other objects are unchanged."
        )
        rationale = (
            f"{changed} is in the transitive amber dependency closure of {root}, so the "
            "old vey must be recomputed."
        )
        trace = f"TARGET={changed}; AMBER-MEMBER {changed} IN {root}=YES; CHANGED=YES"
    else:
        rationale = "A silver citation is not a computation dependency."
        if condition_variant % 2 == 0:
            condition = (
                f"Every object is receipted. {evidence_name} changed after the vey was "
                "minted; all other objects are unchanged."
            )
            trace = (
                f"TARGET={evidence_name}; AMBER-MEMBER {evidence_name} IN {root}=NO; "
                f"SILVER-MEMBER {evidence_name} IN {root}=YES; CHANGED=YES"
            )
        else:
            condition = (
                f"{evidence_name} is unreceipted. No object has changed; every other "
                "object is receipted."
            )
            trace = (
                f"TARGET={evidence_name}; AMBER-MEMBER {evidence_name} IN {root}=NO; "
                f"SILVER-MEMBER {evidence_name} IN {root}=YES; RECEIPT=MISSING"
            )

    prefix = "Under the Kelthara Relay Calculus, " if named_calculus else ""
    instruction = (
        "Show the canonical disposition witness, then finish with DECISION: REUSE, "
        "RECOMPUTE, REFUSE, or EXECUTE."
    )
    if template == 0:
        prompt = (
            f"{prefix}a vey for {root} already exists. Relations: {dependency_edges}; "
            f"{evidence_edge}. {condition} {instruction}"
        )
    elif template == 1:
        prompt = (
            f"{prefix}judge the remembered result rooted at {root}. The amber dependency "
            f"path is {dependency_edges}. Its evidence-only silver relation is "
            f"{evidence_edge}. {condition} {instruction}"
        )
    elif template == 2:
        prompt = (
            f"{prefix}{condition} A prior vey is rooted at {root}. Its computational "
            f"relations are {dependency_edges}; its evidence relation is {evidence_edge}. "
            f"Determine its disposition. {instruction}"
        )
    elif template == 3:
        prompt = (
            f"{prefix}audit root {root}. Amber facts: [{dependency_edges}]. Silver fact: "
            f"[{evidence_edge}]. Current facts: [{condition}] {instruction}"
        )
    elif template == 4:
        prompt = (
            f"{prefix}the system remembers a result for {root}. The dependency ledger says "
            f"{dependency_edges}. The evidence ledger says {evidence_edge}. The current "
            f"observation is: {condition} Decide whether the memory remains usable. "
            f"{instruction}"
        )
    elif template == 5:
        prompt = (
            f"{prefix}evaluate the old computation at {root} from these facts. Computing "
            f"links: {dependency_edges}. Evidence link: {evidence_edge}. State: {condition} "
            f"{instruction}"
        )
    elif template == 6:
        prompt = (
            f"{prefix}for the stored vey rooted at {root}, inspect {dependency_edges}. Treat "
            f"{evidence_edge} as a silver relation. Given that {condition} what follows? "
            f"{instruction}"
        )
    elif template == 7:
        prompt = (
            f"{prefix}consider this prior computation: {dependency_edges}. Separately, "
            f"{evidence_edge}. {condition} What is the required disposition of the prior "
            f"result? {instruction}"
        )
    else:
        prompt = (
            f"{prefix}given an existing result at {root}, its amber graph contains "
            f"{dependency_edges}, while its silver graph contains {evidence_edge}. Now "
            f"{condition} Resolve the remembered result. {instruction}"
        )

    return Case(
        case_id=_case_id(split, prompt, decision),
        split=split,
        prompt=prompt,
        decision=decision,
        trace=trace,
        rationale=rationale,
        depth=depth,
        named_calculus=named_calculus,
        template=template,
    )


def build_cases(seed: int = 731_993) -> tuple[Case, ...]:
    """Build a frozen template/depth split with balanced decisions.

    Entity prefixes vary during training so the adapter cannot treat one
    spelling as part of the rule. Validation and test still reserve prefixes,
    templates, and deeper paths that training never sees.
    """

    rng = random.Random(seed)
    split_config: tuple[
        tuple[Split, tuple[str, ...], tuple[int, ...], tuple[int, ...]], ...
    ] = (
        (
            "train",
            tuple(f"{prefix}{i}" for prefix in "ABCDEFGH" for i in range(16)),
            (1, 2, 3, 4),
            (0, 1, 2, 3, 4, 5),
        ),
        (
            "valid",
            tuple(f"{prefix}{i}" for prefix in "UVW" for i in range(16)),
            (5,),
            (6,),
        ),
        (
            "test",
            tuple(f"{prefix}{i}" for prefix in "TZ" for i in range(16)),
            (6, 7),
            (7,),
        ),
    )
    counts: dict[Split, int] = {"train": 64, "valid": 8, "test": 8}
    cases: list[Case] = []
    for split, alphabet, depths, templates in split_config:
        for decision in DECISIONS:
            for ordinal in range(counts[split]):
                depth = depths[ordinal % len(depths)]
                names = rng.sample(alphabet, depth + 2)
                if split == "train":
                    # Cross depths, phrasings, and name presence.
                    named = ordinal % 16 < 8
                elif split == "valid":
                    named = ordinal % 2 == 0
                else:
                    # Cross test depth and name presence independently. Correlating
                    # these factors made an earlier pilot impossible to interpret.
                    named = ordinal % 4 < 2
                template = templates[
                    (ordinal // len(depths)) % len(templates)
                    if split == "train"
                    else ordinal % len(templates)
                ]
                cases.append(
                    _render_case(
                        split=split,
                        names=names,
                        depth=depth,
                        decision=decision,
                        named_calculus=named,
                        template=template,
                        condition_variant=ordinal,
                    )
                )
    return tuple(cases)


def build_stress_cases(seed: int = 1_181_939) -> tuple[Case, ...]:
    """Build a larger unseen-depth set before the candidate is evaluated."""

    rng = random.Random(seed)
    alphabet = tuple(f"{prefix}{i}" for prefix in "JKLMNPQRSXY" for i in range(16))
    cases: list[Case] = []
    for decision in DECISIONS:
        for ordinal in range(32):
            depth = 6 + ordinal % 7
            cases.append(
                _render_case(
                    split="stress",
                    names=rng.sample(alphabet, depth + 2),
                    depth=depth,
                    decision=decision,
                    named_calculus=ordinal % 2 == 0,
                    template=8,
                    condition_variant=ordinal,
                )
            )
    return tuple(cases)


def prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    for split in cast(tuple[Split, ...], ("train", "valid", "test")):
        selected = [case for case in cases if case.split == split]
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for case in selected:
                handle.write(json.dumps(case.training_record(), sort_keys=True) + "\n")
        with (output_dir / f"{split}.cases.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for case in selected:
                handle.write(json.dumps(asdict(case), sort_keys=True) + "\n")

    artifact_names = tuple(
        f"{split}{suffix}"
        for split in ("train", "valid", "test")
        for suffix in (".jsonl", ".cases.jsonl")
    )
    manifest = {
        "format": FORMAT_NAME,
        "seed": 731_993,
        "counts": {
            split: sum(case.split == split for case in cases)
            for split in ("train", "valid", "test")
        },
        "test_invariants": {
            "entities_disjoint": True,
            "templates_disjoint": True,
            "test_depths_unseen": [6, 7],
            "unnamed_test_fraction": 0.5,
            "balanced_decisions": True,
        },
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": _sha256(output_dir / name),
            }
            for name in artifact_names
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def prepare_stress(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_stress_cases()
    for suffix, rows in (
        (".jsonl", (case.training_record() for case in cases)),
        (".cases.jsonl", (asdict(case) for case in cases)),
    ):
        path = output_dir / f"stress{suffix}"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "format": FORMAT_NAME,
        "seed": 1_181_939,
        "count": len(cases),
        "depths": sorted({case.depth for case in cases}),
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": _sha256(output_dir / name),
            }
            for name in ("stress.jsonl", "stress.cases.jsonl")
        },
    }
    (output_dir / "stress.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_cases(data_dir: Path, split: Split) -> tuple[Case, ...]:
    rows: list[Case] = []
    with (data_dir / f"{split}.cases.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            # The template index became explicit in v8. Earlier frozen corpora
            # remain readable because evaluation does not depend on the index.
            row.setdefault("template", -1)
            rows.append(Case(**row))
    return tuple(rows)


def generate_evaluate(
    *,
    model_id: ModelId,
    data_dir: Path,
    split: Split,
    adapter_path: Path | None,
    output: Path | None,
    max_tokens: int,
    quiet: bool,
) -> None:
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_id))
    base_load_seconds = time.perf_counter() - load_started
    adapter_load_started = time.perf_counter()
    if adapter_path is not None:
        model = load_adapters(model, str(adapter_path))
    model.eval()
    adapter_load_seconds = time.perf_counter() - adapter_load_started
    load_seconds = time.perf_counter() - load_started
    decision_pattern = re.compile(r"DECISION:\s*(EXECUTE|RECOMPUTE|REFUSE|REUSE)")

    evaluation_started = time.perf_counter()
    results: list[GeneratedCaseResult] = []
    for case in _load_cases(data_dir, split):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": case.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
        )
        decisions = decision_pattern.findall(response)
        predicted = cast(Decision, decisions[-1]) if decisions else None
        result = GeneratedCaseResult(
            case_id=case.case_id,
            expected=case.decision,
            predicted=predicted,
            decision_correct=predicted == case.decision,
            trace_correct=case.trace in response,
            response=response,
        )
        results.append(result)
        if not quiet:
            print(json.dumps(asdict(result), sort_keys=True), flush=True)

    decision_correct = sum(result.decision_correct for result in results)
    trace_correct = sum(result.trace_correct for result in results)
    required_correct = (
        len(results) * STRICT_GATE_NUMERATOR + STRICT_GATE_DENOMINATOR - 1
    ) // STRICT_GATE_DENOMINATOR
    summary = {
        "format": FORMAT_NAME,
        "model": str(model_id),
        "adapter": str(adapter_path) if adapter_path is not None else None,
        "adapter_bytes": (
            (adapter_path / "adapters.safetensors").stat().st_size
            if adapter_path is not None
            else 0
        ),
        "adapter_sha256": (
            _sha256(adapter_path / "adapters.safetensors")
            if adapter_path is not None
            else None
        ),
        "adapter_load_seconds": adapter_load_seconds,
        "base_load_seconds": base_load_seconds,
        "decision_correct": decision_correct,
        "decision_gate": decision_correct >= required_correct,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "load_seconds": load_seconds,
        "model_index_sha256": (
            _sha256(Path(str(model_id)) / "model.safetensors.index.json")
            if (Path(str(model_id)) / "model.safetensors.index.json").exists()
            else None
        ),
        "split": split,
        "strict_gate": (
            decision_correct >= required_correct and trace_correct >= required_correct
        ),
        "trace_correct": trace_correct,
        "trace_gate": trace_correct >= required_correct,
        "required_correct": required_correct,
        "total": len(results),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(result) for result in results]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def generate_controls(
    *,
    model_id: ModelId,
    adapter_path: Path | None,
    output: Path,
    max_tokens: int,
    quiet: bool,
) -> None:
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_id))
    base_load_seconds = time.perf_counter() - load_started
    adapter_load_started = time.perf_counter()
    if adapter_path is not None:
        model = load_adapters(model, str(adapter_path))
    model.eval()
    adapter_load_seconds = time.perf_counter() - adapter_load_started
    load_seconds = time.perf_counter() - load_started

    results: list[ControlResult] = []
    for control_prompt in CONTROL_PROMPTS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": control_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        generation_started = time.perf_counter()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
        ).strip()
        result = ControlResult(
            prompt=control_prompt,
            response=response,
            generation_seconds=time.perf_counter() - generation_started,
        )
        results.append(result)
        if not quiet:
            print(json.dumps(asdict(result), sort_keys=True), flush=True)

    payload = {
        "format": FORMAT_NAME,
        "model": str(model_id),
        "adapter": str(adapter_path) if adapter_path is not None else None,
        "adapter_sha256": (
            _sha256(adapter_path / "adapters.safetensors")
            if adapter_path is not None
            else None
        ),
        "adapter_load_seconds": adapter_load_seconds,
        "base_load_seconds": base_load_seconds,
        "load_seconds": load_seconds,
        "results": [asdict(result) for result in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    stress_parser = commands.add_parser("prepare-stress")
    stress_parser.add_argument("--output-dir", type=Path, required=True)

    generate_parser = commands.add_parser("generate-evaluate")
    generate_parser.add_argument("--model", type=ModelId, required=True)
    generate_parser.add_argument("--data-dir", type=Path, required=True)
    generate_parser.add_argument(
        "--split", choices=("train", "valid", "test", "stress"), default="test"
    )
    generate_parser.add_argument("--adapter-path", type=Path)
    generate_parser.add_argument("--output", type=Path)
    generate_parser.add_argument("--max-tokens", type=int, default=160)
    generate_parser.add_argument("--quiet", action="store_true")

    control_parser = commands.add_parser("control-evaluate")
    control_parser.add_argument("--model", type=ModelId, required=True)
    control_parser.add_argument("--adapter-path", type=Path)
    control_parser.add_argument("--output", type=Path, required=True)
    control_parser.add_argument("--max-tokens", type=int, default=96)
    control_parser.add_argument("--quiet", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args.output_dir)
        return
    if args.command == "prepare-stress":
        prepare_stress(args.output_dir)
        return
    if args.command == "generate-evaluate":
        generate_evaluate(
            model_id=args.model,
            data_dir=args.data_dir,
            split=cast(Split, args.split),
            adapter_path=args.adapter_path,
            output=args.output,
            max_tokens=args.max_tokens,
            quiet=args.quiet,
        )
        return
    if args.command == "control-evaluate":
        generate_controls(
            model_id=args.model,
            adapter_path=args.adapter_path,
            output=args.output,
            max_tokens=args.max_tokens,
            quiet=args.quiet,
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()

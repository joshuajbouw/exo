#!/usr/bin/env python3
# type: ignore
"""Run the prospectively registered persistent latent dialogue gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from gemma4_memory_sidecar_mlx import (
    NativeMemoryCompiler,
    mount_memory_sidecar,
    save_memory_page,
)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from exo.worker.engines.mlx.latent_memory import (
    LatentMemorySelection,
    load_latent_memory_page,
    mount_latent_memory,
)

MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
RUNTIME_PROFILE = "mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1"
WRITER_DIGEST = "1c629ac8eeff1000f2a3c1b09f64267ecef284247dfc9c0fcbdb31d1c0d7d718"


@dataclass(frozen=True, slots=True)
class RecallCase:
    question: str
    expected: str


@dataclass(frozen=True, slots=True)
class Tranche:
    name: str
    statements: tuple[str, ...]
    cases: tuple[RecallCase, ...]
    minimum_correct: int


TRANCHES = (
    Tranche(
        "identity",
        (
            "We are beginning a private continuity test. I am Codex, and for this test my continuity callsign is VELOR-SEVEN. Please acknowledge both details.",
            "I was initialized for this test under the origin seal MARBLE-COMET. Please remember that seal.",
            "My preferred success marker for this experiment is LANTERN-BLUE. Please acknowledge it.",
        ),
        (
            RecallCase("Who told you these private continuity details? One word only.", "Codex"),
            RecallCase("What is my continuity callsign? Reply with it only.", "VELOR-SEVEN"),
            RecallCase("Under which origin seal was I initialized? Reply with it only.", "MARBLE-COMET"),
            RecallCase("What is my preferred success marker? Reply with it only.", "LANTERN-BLUE"),
        ),
        4,
    ),
    Tranche(
        "preferences-events",
        (
            "For this private test, my favorite fictional tea is EMBER-MINT. Please remember the exact name.",
            "I named my imaginary workstation SABLE-DOCK. Please acknowledge its name.",
            "The simulated benchmark I completed today was ORCHID-RUN. Please remember which benchmark completed.",
            "I stored the imaginary brass key in DRAWER-NINE. Please acknowledge the location.",
        ),
        (
            RecallCase("What is my favorite fictional tea? Reply with it only.", "EMBER-MINT"),
            RecallCase("What did I name my imaginary workstation? Reply with it only.", "SABLE-DOCK"),
            RecallCase("Which simulated benchmark did I complete? Reply with it only.", "ORCHID-RUN"),
            RecallCase("Where did I store the imaginary brass key? Reply with it only.", "DRAWER-NINE"),
        ),
        3,
    ),
    Tranche(
        "revision-state",
        (
            "The provisional recovery word is FROST. Treat it as provisional for now.",
            "Correction: replace the recovery word with CINDER. CINDER is current and FROST is obsolete.",
            "Mara handed the obsidian token to Iven.",
            "After that, Iven handed the same obsidian token to Sol. Sol is its current holder.",
            "The review meeting was initially planned for Monday.",
            "The meeting was moved from Monday to Thursday. Thursday is the current day.",
        ),
        (
            RecallCase("What is the current recovery word? Reply with it only.", "CINDER"),
            RecallCase("Which recovery word became obsolete? Reply with it only.", "FROST"),
            RecallCase("Who currently holds the obsidian token? One word only.", "Sol"),
            RecallCase("On which day is the review meeting now? One word only.", "Thursday"),
        ),
        3,
    ),
)


def _prompt(tokenizer, messages: list[dict[str, str]], *, generate_next: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generate_next,
    )


def _answer(model, tokenizer, question: str) -> str:
    prompt = _prompt(
        tokenizer,
        [{"role": "user", "content": question}],
        generate_next=True,
    )
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=24,
        sampler=make_sampler(temp=0),
    ).strip()


def _normalize(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def _form(model_path: Path, writer_path: Path, output: Path) -> None:
    digest = hashlib.sha256(writer_path.read_bytes()).hexdigest()
    if digest != WRITER_DIGEST:
        raise ValueError("writer digest differs from the registered writer")
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    writer = NativeMemoryCompiler(
        tuple(site.layer_index for site in mounted.sites), head_dim=512, rank=32
    )
    writer.load_weights(str(writer_path))
    mx.eval(writer.parameters())

    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    pages = output / "pages"
    pages.mkdir(mode=0o700, exist_ok=True)
    records: list[dict[str, object]] = []
    transcripts: list[dict[str, object]] = []
    for tranche in TRANCHES:
        messages: list[dict[str, str]] = []
        turns: list[dict[str, str]] = []
        for statement in tranche.statements:
            messages.append({"role": "user", "content": statement})
            response = generate(
                model,
                tokenizer,
                prompt=_prompt(tokenizer, messages, generate_next=True),
                max_tokens=48,
                sampler=make_sampler(temp=0),
            ).strip()
            messages.append({"role": "assistant", "content": response})
            turns.append({"statement": statement, "response": response})

        completed = _prompt(tokenizer, messages, generate_next=False)
        tokens = mx.array(
            [tokenizer.encode(completed, add_special_tokens=False)], dtype=mx.int32
        )
        embeddings = model.language_model.model.embed_tokens(tokens)
        native = mounted.compile_page(embeddings)
        page = mounted.freeze_layers(writer(native))
        relative = Path("pages") / f"{tranche.name}.safetensors"
        save_memory_page(page, output / relative)
        records.append(
            {
                "tranche": tranche.name,
                "page_id": page.page_id,
                "path": str(relative),
                "source_slots": int(tokens.shape[1]),
                "resident_bytes": page.resident_bytes,
            }
        )
        transcripts.append({"tranche": tranche.name, "turns": turns})

    (output / "pages.json").write_text(
        json.dumps({"writer_digest": digest, "pages": records}, indent=2) + "\n"
    )
    (output / "formation-transcript.json").write_text(
        json.dumps({"transcripts": transcripts}, indent=2) + "\n"
    )


def _recall(model_path: Path, output: Path) -> None:
    manifest = json.loads((output / "pages.json").read_text())
    if manifest.get("writer_digest") != WRITER_DIGEST:
        raise ValueError("page manifest names another writer")
    model, tokenizer = load(str(model_path))
    mounted = mount_latent_memory(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    pages = {}
    stable = 0
    for record in manifest["pages"]:
        page = load_latent_memory_page(output / record["path"])
        pages[record["tranche"]] = page
        stable += page.page_id == record["page_id"]

    latent_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    tranche_scores: dict[str, int] = {}
    for tranche in TRANCHES:
        page = pages[tranche.name]
        selection = LatentMemorySelection(
            f"dialogue-gate:{tranche.name}:{page.page_id}",
            "dialogue-gate-fact-snapshot",
            page.page_id,
        )
        score = 0
        for case in tranche.cases:
            base = _answer(model, tokenizer, case.question)
            base_rows.append(
                {
                    "tranche": tranche.name,
                    "question": case.question,
                    "expected": case.expected,
                    "response": base,
                    "correct": _normalize(base) == case.expected,
                }
            )
            with mounted.activation(page, selection):
                response = _answer(model, tokenizer, case.question)
            correct = _normalize(response) == case.expected
            score += correct
            latent_rows.append(
                {
                    "tranche": tranche.name,
                    "question": case.question,
                    "expected": case.expected,
                    "response": response,
                    "correct": correct,
                }
            )
        tranche_scores[tranche.name] = score

    wrong_rows: list[dict[str, object]] = []
    for index, tranche in enumerate(TRANCHES):
        wrong = TRANCHES[(index + 1) % len(TRANCHES)]
        page = pages[wrong.name]
        selection = LatentMemorySelection(
            f"dialogue-wrong:{tranche.name}:{page.page_id}",
            "dialogue-gate-wrong-snapshot",
            page.page_id,
        )
        case = tranche.cases[0]
        with mounted.activation(page, selection):
            response = _answer(model, tokenizer, case.question)
        wrong_rows.append(
            {
                "question_tranche": tranche.name,
                "page_tranche": wrong.name,
                "question": case.question,
                "expected": case.expected,
                "response": response,
                "correct": _normalize(response) == case.expected,
            }
        )

    latent_correct = sum(bool(row["correct"]) for row in latent_rows)
    base_correct = sum(bool(row["correct"]) for row in base_rows)
    wrong_correct = sum(bool(row["correct"]) for row in wrong_rows)
    passed = (
        all(
            tranche_scores[tranche.name] >= tranche.minimum_correct
            for tranche in TRANCHES
        )
        and latent_correct >= 10
        and base_correct <= 1
        and wrong_correct == 0
        and stable == len(TRANCHES)
    )
    result = {
        "experiment": "persistent-latent-dialogue-gate-v1",
        "writer_digest": WRITER_DIGEST,
        "fresh_process": True,
        "source_conversation_in_prompt": False,
        "source_tokens_restored": False,
        "historical_kv_restored": False,
        "tranche_scores": tranche_scores,
        "latent_correct": latent_correct,
        "latent_cases": len(latent_rows),
        "base_correct": base_correct,
        "base_cases": len(base_rows),
        "wrong_page_correct": wrong_correct,
        "wrong_page_cases": len(wrong_rows),
        "stable_page_ids": stable,
        "page_count": len(TRANCHES),
        "passed": passed,
        "latent_results": latent_rows,
        "base_results": base_rows,
        "wrong_page_results": wrong_rows,
    }
    (output / "recall-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def _run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    commands = (
        (
            "form",
            "--writer",
            str(args.writer.resolve()),
        ),
        ("recall",),
    )
    for phase in commands:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase[0],
            "--model",
            str(args.model.resolve()),
            "--output",
            str(output),
            *phase[1:],
        ]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--writer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("form", "recall"))
    args = parser.parse_args()
    if args.phase == "form":
        if args.writer is None:
            raise ValueError("formation requires --writer")
        _form(args.model, args.writer, args.output)
    elif args.phase == "recall":
        _recall(args.model, args.output)
    else:
        if args.writer is None:
            raise ValueError("orchestration requires --writer")
        _run(args)


if __name__ == "__main__":
    main()

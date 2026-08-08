#!/usr/bin/env python3
# type: ignore
"""Publish latent memory to Astrid storage and recall after process death."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from durable_grounded_memory import (
    DurableGroundedMemoryStore,
    PagePublication,
)
from gemma4_memory_sidecar_mlx import load_memory_page, mount_memory_sidecar
from grounded_memory_selection import (
    GroundedMemorySelector,
    MemoryCatalogEntry,
    MemoryDomainMembership,
    MemoryPageGrant,
    MemorySelectionRequest,
)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from train_gemma4_conversation_memory import (
    MODEL_ID,
    RUNTIME_PROFILE,
    TEST_QUERIES,
    facts,
)

_DOMAIN = "durable-conversation-memory-domain"
_PRINCIPAL = "durable-conversation-memory-principal"
_EPOCH = 1_000_000_000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("artifact_dir", type=Path, nargs="?")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--phase", choices=("publish", "recall"))
    return parser.parse_args()


def _concept(fact_id: str) -> str:
    return f"private-calibration-memory:{fact_id}"


def _publish(artifact_dir: Path, workdir: Path) -> None:
    result = json.loads((artifact_dir / "result.json").read_text())
    records = tuple(
        record for record in result["pages"] if record["statement_form"] == 0
    )
    publications: list[PagePublication] = []
    for record in records:
        source = artifact_dir / record["path"]
        page = load_memory_page(source)
        publications.append(
            PagePublication(
                MemoryCatalogEntry(
                    page_id=page.page_id,
                    evidence_closure_id=f"conversation-evidence:{record['fact_id']}",
                    privacy_domain_id=_DOMAIN,
                    concept_id=_concept(record["fact_id"]),
                    model_id=page.model_id,
                    runtime_profile=page.runtime_profile,
                    first_epoch=0,
                    last_epoch=2_000_000_000,
                ),
                source,
            )
        )
    store = DurableGroundedMemoryStore(workdir / "store")
    started = time.perf_counter()
    manifest_id = store.publish(_DOMAIN, tuple(publications))
    publication_seconds = time.perf_counter() - started
    (workdir / "publication.json").write_text(
        json.dumps(
            {
                "manifest_id": manifest_id,
                "pages": len(publications),
                "publication_seconds": publication_seconds,
            }
        )
    )


def _prompt(tokenizer, query: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _recall(model_path: Path, workdir: Path) -> None:
    store = DurableGroundedMemoryStore(workdir / "store")
    started = time.perf_counter()
    reopened = store.reopen(_DOMAIN, workdir / "reopened-pages")
    reopen_seconds = time.perf_counter() - started
    if reopened is None:
        raise RuntimeError("durable memory catalog is absent")
    grants = tuple(
        MemoryPageGrant(_PRINCIPAL, entry.page_id) for entry in reopened.entries
    )
    selector = GroundedMemorySelector(
        reopened.entries,
        (MemoryDomainMembership(_PRINCIPAL, _DOMAIN),),
        grants,
        rule_profile_id="durable-grounded-memory-recall-v1",
    )
    model, tokenizer = load(str(model_path))
    mounted = mount_memory_sidecar(
        model, model_id=MODEL_ID, runtime_profile=RUNTIME_PROFILE
    )
    correct = 0
    evaluations = []
    for fact in (candidate for candidate in facts() if candidate.split == "test"):
        for query_form, template in enumerate(TEST_QUERIES):
            request = MemorySelectionRequest(
                invocation_id=f"durable:{fact.fact_id}:{query_form}",
                principal_id=_PRINCIPAL,
                privacy_domain_id=_DOMAIN,
                concept_id=_concept(fact.fact_id),
                model_id=MODEL_ID,
                runtime_profile=RUNTIME_PROFILE,
                epoch=_EPOCH,
            )
            proof = selector.select(request)
            if proof is None:
                raise RuntimeError("durable selector returned no page")
            page = reopened.pages[proof.selected_page_id]
            mounted.activate(page, proof=proof)
            response = generate(
                model,
                tokenizer,
                prompt=_prompt(tokenizer, template.format(entity=fact.entity)),
                max_tokens=16,
                sampler=make_sampler(temp=0),
            ).strip()
            mounted.deactivate()
            matched = response == fact.value
            correct += matched
            evaluations.append(
                {
                    "fact_id": fact.fact_id,
                    "query_form": query_form,
                    "page_id": page.page_id,
                    "correct": matched,
                }
            )
    result = {
        "experiment": "durable-grounded-conversation-memory",
        "manifest_id": reopened.manifest_id,
        "pages": len(reopened.pages),
        "reopen_seconds": reopen_seconds,
        "correct": correct,
        "cases": len(evaluations),
        "fresh_reader_had_original_artifact_path": False,
        "source_text_replayed": False,
        "source_token_ids_replayed": False,
        "historical_kv_restored": False,
        "passed": correct == 16 and len(reopened.pages) == 8,
        "evaluations": evaluations,
    }
    (workdir / "recall.json").write_text(json.dumps(result, indent=2))


def _child(args: argparse.Namespace, workdir: Path, phase: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model),
    ]
    if phase == "publish":
        if args.artifact_dir is None:
            raise ValueError("publication requires the source artifact directory")
        command.append(str(args.artifact_dir))
    command.extend(("--workdir", str(workdir), "--phase", phase))
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


def _run(args: argparse.Namespace, workdir: Path) -> None:
    _child(args, workdir, "publish")
    _child(args, workdir, "recall")
    publication = json.loads((workdir / "publication.json").read_text())
    result = json.loads((workdir / "recall.json").read_text())
    result["publication_seconds"] = publication["publication_seconds"]
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    args = _arguments()
    if args.phase is not None:
        if args.workdir is None:
            raise ValueError("phase requires --workdir")
        if args.phase == "publish":
            if args.artifact_dir is None:
                raise ValueError("publication requires an artifact directory")
            _publish(args.artifact_dir, args.workdir)
        else:
            _recall(args.model, args.workdir)
        return
    if args.artifact_dir is None:
        raise ValueError("orchestration requires an artifact directory")
    if args.workdir is not None:
        args.workdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _run(args, args.workdir)
        return
    with tempfile.TemporaryDirectory(prefix="exo-durable-memory-") as directory:
        _run(args, Path(directory))


if __name__ == "__main__":
    main()

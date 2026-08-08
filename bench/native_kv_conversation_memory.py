# type: ignore
#!/usr/bin/env python3
"""Prove exact native-KV conversation memory across process death."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.computation_persistence import StoreKVPrefixPersistence
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.patches import apply_mlx_patches
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.bootstrap import logger

_FIRST_RESPONSE_ID = "native-memory-source"
_RUNTIME_PROFILE = "gemma4-native-kv-conversation-memory-v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", default="mlx-community/gemma-4-31b-it-4bit")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--phase", choices=("writer", "reader"))
    return parser.parse_args()


def _task(
    model_id: str,
    content: str,
    *,
    previous_response_id: str | None = None,
    output_tokens: int = 24,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        previous_response_id=previous_response_id,
        max_output_tokens=output_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=42,
        enable_thinking=False,
        logprobs=True,
        top_logprobs=1,
    )


def _directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _float_bits(value: float | None) -> str | None:
    return None if value is None else struct.pack("<d", value).hex()


def _generate(
    model: Any,
    tokenizer: Any,
    cache: KVPrefixCache,
    task: TextGenerationTaskParams,
    response_id: str,
) -> dict[str, Any]:
    prompt = apply_chat_template(tokenizer, task)
    started = time.perf_counter()
    first_token_seconds: float | None = None
    responses = []
    for response in mlx_generate(
        model,
        tokenizer,
        task,
        prompt,
        cache,
        None,
        response_id=response_id,
    ):
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        responses.append(response)
    elapsed = time.perf_counter() - started
    if not responses or responses[-1].finish_reason is None:
        raise RuntimeError("generation did not produce a completed frontier")
    final = responses[-1]
    usage = final.usage
    return {
        "tokens": [response.token for response in responses],
        "text": "".join(response.text for response in responses),
        "logprob_bits": [_float_bits(response.logprob) for response in responses],
        "first_token_seconds": first_token_seconds,
        "elapsed_seconds": elapsed,
        "prompt_tokens": usage.prompt_tokens if usage is not None else None,
        "cached_tokens": (
            usage.prompt_tokens_details.cached_tokens if usage is not None else None
        ),
    }


def _load_model(model_path: Path) -> tuple[Any, Any, float]:
    apply_mlx_patches()
    started = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=False)
    mx.eval(model.parameters())
    return model, tokenizer, time.perf_counter() - started


def _writer(args: argparse.Namespace, workdir: Path) -> None:
    request = json.load(sys.stdin)
    source = str(request["source"])
    question = str(request["question"])
    store = workdir / "store"
    model, tokenizer, model_load_seconds = _load_model(args.model)
    persistence = StoreKVPrefixPersistence(store, _RUNTIME_PROFILE)
    cache = KVPrefixCache(None, persistence=persistence)

    source_result = _generate(
        model,
        tokenizer,
        cache,
        _task(args.model_id, source, output_tokens=16),
        _FIRST_RESPONSE_ID,
    )
    # Freeze the recall corpus before the question is ever computed. Closing
    # waits for durable publication but deliberately leaves the in-memory K/V
    # frontier available for the uninterrupted reference below.
    cache.close()
    bytes_after_source = _directory_bytes(store)
    reference = _generate(
        model,
        tokenizer,
        cache,
        _task(
            args.model_id,
            question,
            previous_response_id=_FIRST_RESPONSE_ID,
        ),
        "native-memory-reference",
    )
    (workdir / "reference.json").write_text(
        json.dumps(
            {
                "model_load_seconds": model_load_seconds,
                "source": source_result,
                "reference": reference,
                "checkpoint_bytes_after_source": bytes_after_source,
                "store_bytes_after_reference": _directory_bytes(store),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _reader(args: argparse.Namespace, workdir: Path) -> None:
    request = json.load(sys.stdin)
    question = str(request["question"])
    model, tokenizer, model_load_seconds = _load_model(args.model)

    persistence = StoreKVPrefixPersistence(workdir / "store", _RUNTIME_PROFILE)
    restored_cache = KVPrefixCache(None, persistence=persistence)
    restored = _generate(
        model,
        tokenizer,
        restored_cache,
        _task(
            args.model_id,
            question,
            previous_response_id=_FIRST_RESPONSE_ID,
        ),
        "native-memory-restored",
    )
    restore_metrics = persistence.last_restore_metrics
    restored_cache.close()

    control_cache = KVPrefixCache(None)
    control = _generate(
        model,
        tokenizer,
        control_cache,
        _task(args.model_id, question),
        "native-memory-control",
    )
    control_cache.close()
    (workdir / "reader.json").write_text(
        json.dumps(
            {
                "model_load_seconds": model_load_seconds,
                "restored": restored,
                "control": control,
                "restore_metrics": {
                    "projection_verification_seconds": (
                        restore_metrics.projection_verification_seconds
                    ),
                    "reconstruction_seconds": restore_metrics.reconstruction_seconds,
                    "mlx_load_seconds": restore_metrics.mlx_load_seconds,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_phase(
    args: argparse.Namespace,
    workdir: Path,
    phase: str,
    request: dict[str, str],
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model),
        "--model-id",
        args.model_id,
        "--workdir",
        str(workdir),
        "--phase",
        phase,
    ]
    subprocess.run(
        command,
        input=json.dumps(request),
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def _orchestrate(args: argparse.Namespace, workdir: Path) -> None:
    nonce = secrets.token_hex(4).upper()
    term = f"velorin-{nonce}"
    value = f"CINDERGLASS-{secrets.token_hex(4).upper()}"
    source = (
        f"I am defining a private term. The exact value of {term} is {value}. "
        "Remember it for my next turn. Reply with exactly ACKNOWLEDGED."
    )
    question = (
        f"What is the exact value of {term}? Reply with only the value, "
        "preserving capitalization and punctuation."
    )
    _run_phase(args, workdir, "writer", {"source": source, "question": question})
    _run_phase(args, workdir, "reader", {"question": question})

    reference = json.loads((workdir / "reference.json").read_text())
    reader = json.loads((workdir / "reader.json").read_text())
    expected = reference["reference"]
    restored = reader["restored"]
    control = reader["control"]
    exact_tokens = restored["tokens"] == expected["tokens"]
    exact_logprobs = restored["logprob_bits"] == expected["logprob_bits"]
    source_value_recalled = value in expected["text"]
    control_differs = control["tokens"] != expected["tokens"]
    cached_tokens = int(restored["cached_tokens"] or 0)
    passed = (
        exact_tokens
        and exact_logprobs
        and source_value_recalled
        and control_differs
        and cached_tokens > 0
    )
    output = {
        "experiment": "native-kv-conversation-memory",
        "model": str(args.model),
        "runtime_profile": _RUNTIME_PROFILE,
        "source_identity": hashlib.sha256(source.encode()).hexdigest(),
        "term": term,
        "expected_value": value,
        "source_value_recalled": source_value_recalled,
        "exact_tokens_across_process_death": exact_tokens,
        "exact_logprobs_across_process_death": exact_logprobs,
        "control_differs": control_differs,
        "restored_cached_tokens": cached_tokens,
        "checkpoint_bytes_after_source": reference["checkpoint_bytes_after_source"],
        "writer_model_load_seconds": reference["model_load_seconds"],
        "reader_model_load_seconds": reader["model_load_seconds"],
        "reference": expected,
        "restored": restored,
        "control": control,
        "restore_metrics": reader["restore_metrics"],
        "passed": passed,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    args = _arguments()
    logger.remove()
    if args.phase is not None:
        if args.workdir is None:
            raise ValueError("phase execution requires --workdir")
        if args.phase == "writer":
            _writer(args, args.workdir)
        else:
            _reader(args, args.workdir)
        return
    if args.workdir is not None:
        args.workdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _orchestrate(args, args.workdir)
        return
    with tempfile.TemporaryDirectory(prefix="exo-native-kv-memory-") as directory:
        _orchestrate(args, Path(directory))


if __name__ == "__main__":
    main()

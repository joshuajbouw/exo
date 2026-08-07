# type: ignore

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parents[1]))

from gemma4_memory_sidecar_mlx import (
    DirectMemoryBank,
    MemoryPage,
    MemorySidecarError,
    NativeMemoryCompiler,
    mount_memory_sidecar,
)
from mlx_lm.models.gemma4_text import Model, ModelArgs


def _model() -> Model:
    args = ModelArgs(
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        vocab_size=128,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        final_logit_softcapping=None,
    )
    return Model(args)


def _logits(model: Model, tokens: mx.array) -> mx.array:
    logits = model(tokens)
    mx.eval(logits)
    return logits


def test_inactive_sidecar_is_exact_and_revocation_restores_base() -> None:
    mx.random.seed(11)
    model = _model()
    tokens = mx.array([[3, 5, 7, 9]])
    base = _logits(model, tokens)

    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    inactive = _logits(model, tokens)
    assert mx.array_equal(base, inactive).item()

    slots = model.model.embed_tokens(mx.array([[17, 19, 23]]))
    page = mounted.compile_page(slots)
    mounted.activate(page, proof_id="proof:alpha")
    active = _logits(model, tokens)
    assert not mx.array_equal(base, active).item()

    mounted.deactivate()
    revoked = _logits(model, tokens)
    assert mx.array_equal(base, revoked).item()


def test_page_is_external_to_the_ordinary_cache() -> None:
    mx.random.seed(23)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[29, 31]])))
    mounted.activate(page, proof_id="proof:cache")

    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(*[value for entry in cache for value in entry.state])
    assert all(entry.offset == 3 for entry in cache)


def test_activation_requires_matching_identity_and_proof() -> None:
    mx.random.seed(37)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[41]])))

    try:
        mounted.activate(page, proof_id="")
    except MemorySidecarError as error:
        assert "proof identity" in str(error)
    else:
        raise AssertionError("empty proof unexpectedly activated a memory page")

    mounted.activate(page, proof_id="proof:first")
    try:
        mounted.activate(page, proof_id="proof:second")
    except MemorySidecarError as error:
        assert "different memory proof" in str(error)
    else:
        raise AssertionError("a second proof replaced the active proof")


def test_activation_recomputes_identity_and_requires_canonical_layers() -> None:
    mx.random.seed(39)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[41, 43]])))
    invalid_gate = mounted.compile_page(
        model.model.embed_tokens(mx.array([[41, 43]])),
        gates={1: float("nan")},
    )
    invalid_pages = (
        replace(page, page_id="0" * 64),
        MemoryPage(
            page.page_id,
            page.model_id,
            page.runtime_profile,
            page.layers + page.layers,
        ),
        invalid_gate,
    )
    for invalid in invalid_pages:
        try:
            mounted.activate(invalid, proof_id="proof:invalid")
        except MemorySidecarError:
            pass
        else:
            raise AssertionError("invalid memory page unexpectedly activated")


def test_bfloat16_page_identity_is_stable() -> None:
    mx.random.seed(41)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    slots = model.model.embed_tokens(mx.array([[43, 47]])).astype(mx.bfloat16)
    first = mounted.compile_page(slots)
    second = mounted.compile_page(slots)
    assert first.page_id == second.page_id

    bank = DirectMemoryBank(first)
    assert len(bank.memory_keys) == 1
    assert len(bank.memory_values) == 1

    compiler = NativeMemoryCompiler((1,), head_dim=8, rank=2)
    compiled = compiler(first)
    mx.eval(*(value for layer in compiled for value in (layer.keys, layer.values)))
    assert mx.allclose(compiled[0].keys, first.layers[0].keys).item()
    assert mx.allclose(compiled[0].values, first.layers[0].values).item()

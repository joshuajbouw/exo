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
    MemorySelectionProof,
    MemorySidecarError,
    NativeMemoryCompiler,
    compose_memory_pages,
    load_memory_page,
    mount_memory_sidecar,
    save_memory_page,
)
from grounded_memory_selection import MemorySelectionError
from mlx_lm.models.gemma4_text import Model, ModelArgs

from exo.worker.engines.mlx.latent_memory import load_latent_memory_page


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


def _proof(page: MemoryPage, name: str) -> MemorySelectionProof:
    return MemorySelectionProof.for_page(
        page,
        proof_id=f"proof:{name}",
        fact_snapshot_id="benchmark-fact-snapshot",
    )


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
    mounted.activate(page, proof=_proof(page, "alpha"))
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
    mounted.activate(page, proof=_proof(page, "cache"))

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
        mounted.activate(
            page,
            proof=MemorySelectionProof(
                "",
                "benchmark-fact-snapshot",
                page.page_id,
            ),
        )
    except MemorySelectionError as error:
        assert "proof identity" in str(error)
    else:
        raise AssertionError("empty proof unexpectedly activated a memory page")

    mounted.activate(page, proof=_proof(page, "first"))
    try:
        mounted.activate(page, proof=_proof(page, "second"))
    except MemorySidecarError as error:
        assert "different memory proof" in str(error)
    else:
        raise AssertionError("a second proof replaced the active proof")


def test_selection_proof_cannot_activate_another_page() -> None:
    mx.random.seed(38)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    selected = mounted.compile_page(model.model.embed_tokens(mx.array([[41]])))
    substituted = mounted.compile_page(model.model.embed_tokens(mx.array([[43]])))

    try:
        mounted.activate(substituted, proof=_proof(selected, "selected"))
    except MemorySidecarError as error:
        assert "different page" in str(error)
    else:
        raise AssertionError("a proof activated a page it did not select")


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
            mounted.activate(invalid, proof=_proof(invalid, "invalid"))
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


def test_memory_page_round_trips_without_source_tokens(tmp_path: Path) -> None:
    mx.random.seed(43)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[53, 59, 61]])))
    path = tmp_path / "memory.safetensors"

    save_memory_page(page, path)
    restored = load_memory_page(path)

    assert restored.page_id == page.page_id
    assert restored.model_id == page.model_id
    assert restored.runtime_profile == page.runtime_profile
    assert len(restored.layers) == len(page.layers)
    for expected, actual in zip(page.layers, restored.layers, strict=True):
        assert actual.layer_index == expected.layer_index
        assert actual.gate == expected.gate
        assert mx.array_equal(actual.keys, expected.keys).item()
        assert mx.array_equal(actual.values, expected.values).item()


def test_production_loader_adopts_the_proven_page_format(tmp_path: Path) -> None:
    mx.random.seed(44)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[61, 67, 71]])))
    path = tmp_path / "experimental-page.safetensors"

    save_memory_page(page, path)
    production = load_latent_memory_page(path)

    assert production.page_id == page.page_id
    assert production.model_id == page.model_id
    assert production.runtime_profile == page.runtime_profile


def test_memory_page_save_and_load_reject_stale_identity(tmp_path: Path) -> None:
    mx.random.seed(47)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[67, 71]])))
    path = tmp_path / "memory.safetensors"
    try:
        save_memory_page(replace(page, page_id="0" * 64), path)
    except MemorySidecarError as error:
        assert "identity does not match" in str(error)
    else:
        raise AssertionError("stale page identity unexpectedly saved")

    save_memory_page(page, path)
    arrays, metadata = mx.load(path, return_metadata=True)
    first_name = sorted(arrays)[0]
    arrays[first_name] = arrays[first_name] + 1
    mx.save_safetensors(path, arrays, metadata=metadata)

    try:
        load_memory_page(path)
    except MemorySidecarError as error:
        assert "identity does not match" in str(error)
    else:
        raise AssertionError("stale page identity unexpectedly loaded")


def test_memory_composition_is_canonical_and_preserves_all_slots() -> None:
    mx.random.seed(53)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    first = mounted.compile_page(model.model.embed_tokens(mx.array([[73, 79]])))
    second = mounted.compile_page(model.model.embed_tokens(mx.array([[83, 89, 97]])))

    forward = compose_memory_pages((first, second))
    reverse = compose_memory_pages((second, first))

    assert forward.page_id == reverse.page_id
    assert forward.layers[0].keys.shape[2] == 5
    assert forward.layers[0].values.shape[2] == 5
    mounted.activate(forward, proof=_proof(forward, "composed"))


def test_memory_composition_rejects_duplicates_and_incompatible_pages() -> None:
    mx.random.seed(59)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[101, 103]])))
    invalid_sets = (
        (page, page),
        (page, replace(page, page_id="0" * 64)),
        (page, replace(page, model_id="other-model")),
    )
    for invalid in invalid_sets:
        try:
            compose_memory_pages(invalid)
        except MemorySidecarError:
            pass
        else:
            raise AssertionError("invalid memory composition unexpectedly succeeded")


def test_trace_records_the_effective_global_attention_output() -> None:
    mx.random.seed(61)
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    tokens = mx.array([[7, 11, 13]])

    mounted.begin_trace()
    _logits(model, tokens)
    base_trace = mounted.finish_trace()

    page = mounted.compile_page(model.model.embed_tokens(mx.array([[17, 19]])))
    mounted.activate(page, proof=_proof(page, "trace"))
    mounted.begin_trace()
    mounted.begin_association_trace()
    _logits(model, tokens)
    memory_trace = mounted.finish_trace()
    association_queries = mounted.finish_association_trace()

    assert len(base_trace) == 1
    assert base_trace[0].shape == (1, 3, 32)
    assert memory_trace[0].shape == base_trace[0].shape
    assert association_queries[0].shape == (1, 4, 3, 8)
    assert not mx.array_equal(memory_trace[0], base_trace[0]).item()

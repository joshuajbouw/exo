# type: ignore

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.gemma4_text import Model, ModelArgs

from exo.worker.engines.mlx.latent_memory import (
    LatentMemoryActivation,
    LatentMemoryError,
    LatentMemoryPage,
    LatentMemorySelection,
    compose_latent_memory_pages,
    load_latent_memory_page,
    mount_latent_memory,
    save_latent_memory_page,
)
from exo.worker.runner.llm_inference.batch_generator import _run_with_latent_memory


def _model() -> Model:
    return Model(
        ModelArgs(
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
    )


def _selection(page: LatentMemoryPage) -> LatentMemorySelection:
    return LatentMemorySelection("proof:fixture", "facts:fixture", page.page_id)


def _logits(model: Model, tokens: mx.array) -> mx.array:
    logits = model(tokens)
    mx.eval(logits)
    return logits


def test_inactive_mount_is_exact_and_scope_restores_model() -> None:
    mx.random.seed(101)
    model = _model()
    tokens = mx.array([[3, 5, 7, 9]])
    base = _logits(model, tokens)
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )

    assert mx.array_equal(base, _logits(model, tokens)).item()
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[17, 19, 23]])))
    with mounted.activation(page, _selection(page)):
        assert not mx.array_equal(base, _logits(model, tokens)).item()
    assert mx.array_equal(base, _logits(model, tokens)).item()


def test_activation_restores_model_when_inference_fails() -> None:
    mx.random.seed(103)
    model = _model()
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[29, 31]])))

    with (
        pytest.raises(RuntimeError, match="inference failed"),
        mounted.activation(page, _selection(page)),
    ):
        raise RuntimeError("inference failed")

    # A second scope proves the first one did not leave model-global state live.
    with mounted.activation(page, _selection(page)):
        _logits(model, mx.array([[1, 2, 3]]))


def test_activation_fails_closed_on_page_or_proof_substitution() -> None:
    mx.random.seed(107)
    model = _model()
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[37, 41]])))

    with (
        pytest.raises(LatentMemoryError, match="different latent page"),
        mounted.activation(
            page,
            LatentMemorySelection("proof:x", "facts:x", "0" * 64),
        ),
    ):
        pass
    with (
        pytest.raises(LatentMemoryError, match="identity does not match"),
        mounted.activation(
            replace(page, page_id="0" * 64),
            LatentMemorySelection("proof:x", "facts:x", "0" * 64),
        ),
    ):
        pass


def test_nested_activation_cannot_replace_live_invocation_memory() -> None:
    mx.random.seed(109)
    model = _model()
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    first = mounted.compile_page(model.model.embed_tokens(mx.array([[43]])))
    second = mounted.compile_page(model.model.embed_tokens(mx.array([[47]])))

    with (
        mounted.activation(first, _selection(first)),
        pytest.raises(LatentMemoryError, match="already active"),
        mounted.activation(second, _selection(second)),
    ):
        pass


def test_page_round_trip_contains_only_identity_metadata_and_tensors(
    tmp_path: Path,
) -> None:
    mx.random.seed(113)
    model = _model()
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[53, 59, 61]])))
    path = tmp_path / "latent.safetensors"
    save_latent_memory_page(page, path)
    restored = load_latent_memory_page(path)

    assert restored.page_id == page.page_id
    assert restored.model_id == page.model_id
    assert restored.runtime_profile == page.runtime_profile
    assert all(
        mx.array_equal(expected.keys, actual.keys).item()
        and mx.array_equal(expected.values, actual.values).item()
        for expected, actual in zip(page.layers, restored.layers, strict=True)
    )


def test_composition_is_canonical() -> None:
    mx.random.seed(127)
    model = _model()
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    first = mounted.compile_page(model.model.embed_tokens(mx.array([[67, 71]])))
    second = mounted.compile_page(model.model.embed_tokens(mx.array([[73, 79, 83]])))

    forward = compose_latent_memory_pages((first, second))
    reverse = compose_latent_memory_pages((second, first))
    assert forward.page_id == reverse.page_id
    assert forward.layers[0].keys.shape[2] == 5


def test_generator_scope_holds_memory_until_close_then_revokes() -> None:
    mx.random.seed(131)
    model = _model()
    mounted = mount_latent_memory(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[89, 97]])))

    def source():
        assert mounted._active.layers is not None
        yield object()
        assert mounted._active.layers is not None
        yield object()

    wrapped = _run_with_latent_memory(
        source(), mounted, LatentMemoryActivation(page, _selection(page))
    )
    next(wrapped)
    assert mounted._active.layers is not None
    wrapped.close()
    assert mounted._active.layers is None

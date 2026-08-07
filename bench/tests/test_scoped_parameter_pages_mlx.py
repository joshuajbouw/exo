# type: ignore
# ruff: noqa: E402

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from mlx_lm.tuner.lora import LoRALinear

from bench.scoped_parameter_pages_mlx import (
    PageActivationError,
    ScopedLoRALinear,
)


def _site() -> ScopedLoRALinear:
    base = nn.Linear(2, 2, bias=False)
    base.weight = mx.array(((1.0, 2.0), (3.0, 4.0)), dtype=mx.float32)
    loaded = LoRALinear.from_base(base, r=1, dropout=0.0, scale=1.0)
    loaded.lora_a = mx.array(((1.0,), (0.0,)), dtype=mx.float32)
    loaded.lora_b = mx.array(((5.0, -5.0),), dtype=mx.float32)
    return ScopedLoRALinear(loaded)


def test_inactive_page_is_byte_identical_to_base_projection() -> None:
    site = _site()
    inputs = mx.array(((1.0, 2.0),), dtype=mx.float32)

    expected = site.linear(inputs)
    actual = site(inputs)
    mx.eval(expected, actual)

    assert bytes(actual) == bytes(expected)


def test_activation_changes_output_and_revocation_restores_base() -> None:
    site = _site()
    inputs = mx.array(((1.0, 2.0),), dtype=mx.float32)
    expected = site.linear(inputs)

    site.activate("proof:alpha")
    active = site(inputs)
    site.deactivate()
    revoked = site(inputs)
    mx.eval(expected, active, revoked)

    assert bytes(active) != bytes(expected)
    assert bytes(revoked) == bytes(expected)


def test_activation_requires_identity_and_rejects_proof_replacement() -> None:
    site = _site()

    with pytest.raises(PageActivationError):
        site.activate("")
    site.activate("proof:first")
    with pytest.raises(PageActivationError):
        site.activate("proof:second")

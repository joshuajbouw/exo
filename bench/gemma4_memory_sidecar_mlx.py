#!/usr/bin/env python3
# type: ignore
"""Proof-gated memory attention for MLX Gemma 4.

This is an experimental numerical seam. Tensor Logic and capability policy
produce the proof before this boundary; the sidecar never decides relevance or
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


class MemorySidecarError(ValueError):
    """A memory page violated the model or activation contract."""


@dataclass(frozen=True, slots=True)
class LayerMemory:
    layer_index: int
    keys: mx.array
    values: mx.array
    gate: float = 1.0

    @property
    def resident_bytes(self) -> int:
        if self.keys is self.values:
            return self.keys.nbytes
        return self.keys.nbytes + self.values.nbytes


@dataclass(frozen=True, slots=True)
class MemoryPage:
    page_id: str
    model_id: str
    runtime_profile: str
    layers: tuple[LayerMemory, ...]

    @property
    def resident_bytes(self) -> int:
        return sum(layer.resident_bytes for layer in self.layers)


def save_memory_page(page: MemoryPage, path: Path) -> None:
    """Persist an immutable page without retaining its source tokens."""

    recomputed = _page_identity(
        model_id=page.model_id,
        runtime_profile=page.runtime_profile,
        layers=page.layers,
    )
    if recomputed != page.page_id:
        raise MemorySidecarError("memory page identity does not match its bytes")
    record = {
        "format": "exo-gemma4-memory-page-v1",
        "page_id": page.page_id,
        "model_id": page.model_id,
        "runtime_profile": page.runtime_profile,
        "layers": [
            {"layer_index": layer.layer_index, "gate": layer.gate}
            for layer in page.layers
        ],
    }
    arrays = {
        f"layer.{layer.layer_index}.keys": layer.keys for layer in page.layers
    } | {f"layer.{layer.layer_index}.values": layer.values for layer in page.layers}
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        path,
        arrays,
        metadata={
            "page_record": json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        },
    )


def load_memory_page(path: Path) -> MemoryPage:
    """Load and identity-check a page produced by :func:`save_memory_page`."""

    loaded = mx.load(path, return_metadata=True)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise MemorySidecarError("memory page has no metadata")
    arrays, metadata = loaded
    if not isinstance(arrays, dict) or not isinstance(metadata, dict):
        raise MemorySidecarError("memory page has an invalid container")
    encoded_record = metadata.get("page_record")
    if not isinstance(encoded_record, str):
        raise MemorySidecarError("memory page record is missing")
    try:
        record = json.loads(encoded_record)
    except json.JSONDecodeError as error:
        raise MemorySidecarError("memory page record is invalid JSON") from error
    if (
        not isinstance(record, dict)
        or record.get("format") != "exo-gemma4-memory-page-v1"
    ):
        raise MemorySidecarError("memory page format is unsupported")
    layer_records = record.get("layers")
    if not isinstance(layer_records, list) or not layer_records:
        raise MemorySidecarError("memory page has no layers")

    layers: list[LayerMemory] = []
    expected_names: set[str] = set()
    previous_index = -1
    for layer_record in layer_records:
        if not isinstance(layer_record, dict):
            raise MemorySidecarError("memory page layer record is invalid")
        layer_index = layer_record.get("layer_index")
        gate = layer_record.get("gate")
        if (
            not isinstance(layer_index, int)
            or isinstance(layer_index, bool)
            or layer_index <= previous_index
            or not isinstance(gate, (int, float))
            or isinstance(gate, bool)
        ):
            raise MemorySidecarError("memory page layer order or gate is invalid")
        previous_index = layer_index
        key_name = f"layer.{layer_index}.keys"
        value_name = f"layer.{layer_index}.values"
        expected_names.update((key_name, value_name))
        if key_name not in arrays or value_name not in arrays:
            raise MemorySidecarError("memory page tensor is missing")
        layers.append(
            LayerMemory(layer_index, arrays[key_name], arrays[value_name], float(gate))
        )
    if set(arrays) != expected_names:
        raise MemorySidecarError("memory page has undeclared tensors")
    model_id = record.get("model_id")
    runtime_profile = record.get("runtime_profile")
    page_id = record.get("page_id")
    if not all(
        isinstance(value, str) and value
        for value in (model_id, runtime_profile, page_id)
    ):
        raise MemorySidecarError("memory page identity fields are invalid")
    frozen_layers = tuple(layers)
    recomputed = _page_identity(
        model_id=model_id,
        runtime_profile=runtime_profile,
        layers=frozen_layers,
    )
    if recomputed != page_id:
        raise MemorySidecarError("memory page identity does not match its bytes")
    return MemoryPage(page_id, model_id, runtime_profile, frozen_layers)


def compose_memory_pages(pages: tuple[MemoryPage, ...]) -> MemoryPage:
    """Build one canonically ordered attention bank from immutable pages."""

    if not pages:
        raise MemorySidecarError("memory composition requires at least one page")
    ordered = tuple(sorted(pages, key=lambda page: page.page_id))
    if len({page.page_id for page in ordered}) != len(ordered):
        raise MemorySidecarError("memory composition contains a duplicate page")
    reference = ordered[0]
    layer_indices = tuple(layer.layer_index for layer in reference.layers)
    gates = tuple(layer.gate for layer in reference.layers)
    for page in ordered:
        recomputed = _page_identity(
            model_id=page.model_id,
            runtime_profile=page.runtime_profile,
            layers=page.layers,
        )
        if recomputed != page.page_id:
            raise MemorySidecarError("memory page identity does not match its bytes")
        if (
            page.model_id != reference.model_id
            or page.runtime_profile != reference.runtime_profile
        ):
            raise MemorySidecarError("memory pages target incompatible runtimes")
        if tuple(layer.layer_index for layer in page.layers) != layer_indices:
            raise MemorySidecarError("memory pages have incompatible layer sets")
        if tuple(layer.gate for layer in page.layers) != gates:
            raise MemorySidecarError("memory pages have incompatible gates")
    layers = tuple(
        LayerMemory(
            layer_index,
            mx.concatenate([page.layers[position].keys for page in ordered], axis=2),
            mx.concatenate([page.layers[position].values for page in ordered], axis=2),
            gates[position],
        )
        for position, layer_index in enumerate(layer_indices)
    )
    mx.eval(*(value for layer in layers for value in (layer.keys, layer.values)))
    page_id = _page_identity(
        model_id=reference.model_id,
        runtime_profile=reference.runtime_profile,
        layers=layers,
    )
    return MemoryPage(page_id, reference.model_id, reference.runtime_profile, layers)


@dataclass(slots=True)
class _ActiveMemory:
    page: MemoryPage | None = None
    proof_id: str | None = None
    layers: dict[int, LayerMemory] | None = None
    query_projections: dict[int, Any] | None = None
    training: bool = False


class DirectMemoryBank(nn.Module):
    """Trainable direct K/V bank used only to prove side-channel capacity."""

    def __init__(self, page: MemoryPage):
        super().__init__()
        self.layer_indices = tuple(layer.layer_index for layer in page.layers)
        self.memory_keys = [layer.keys.astype(mx.float32) for layer in page.layers]
        self.memory_values = [layer.values.astype(mx.float32) for layer in page.layers]

    def layer_memories(
        self, *, dtype: mx.Dtype | None = None
    ) -> tuple[LayerMemory, ...]:
        if dtype is None:
            keys = self.memory_keys
            values = self.memory_values
        else:
            keys = [value.astype(dtype) for value in self.memory_keys]
            values = [value.astype(dtype) for value in self.memory_values]
        return tuple(
            LayerMemory(index, layer_keys, layer_values)
            for index, layer_keys, layer_values in zip(
                self.layer_indices, keys, values, strict=True
            )
        )


class _ResidualHeadProjection(nn.Module):
    def __init__(self, head_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(head_dim, rank, bias=False)
        self.up = nn.Linear(rank, head_dim, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, value: mx.array) -> mx.array:
        value = value.astype(mx.float32)
        return value + self.up(nn.gelu_approx(self.down(value)))


class NativeMemoryCompiler(nn.Module):
    """Shared residual compiler from native-projected facts to readable K/V."""

    def __init__(self, layer_indices: tuple[int, ...], *, head_dim: int, rank: int):
        super().__init__()
        if not layer_indices or head_dim <= 0 or rank <= 0:
            raise MemorySidecarError("compiler dimensions must be positive")
        self.layer_indices = layer_indices
        self.key_projections = [
            _ResidualHeadProjection(head_dim, rank) for _ in layer_indices
        ]
        self.value_projections = [
            _ResidualHeadProjection(head_dim, rank) for _ in layer_indices
        ]

    def __call__(self, page: MemoryPage) -> tuple[LayerMemory, ...]:
        actual = tuple(layer.layer_index for layer in page.layers)
        if actual != self.layer_indices:
            raise MemorySidecarError("compiler input page has the wrong layer set")
        return tuple(
            LayerMemory(
                layer.layer_index,
                key_projection(layer.keys),
                value_projection(layer.values),
                layer.gate,
            )
            for layer, key_projection, value_projection in zip(
                page.layers,
                self.key_projections,
                self.value_projections,
                strict=True,
            )
        )


class AddressableMemoryAdapter(nn.Module):
    """Shared memory writer and query reader for associative binding."""

    def __init__(self, layer_indices: tuple[int, ...], *, head_dim: int, rank: int):
        super().__init__()
        self.layer_indices = layer_indices
        self.writer = NativeMemoryCompiler(layer_indices, head_dim=head_dim, rank=rank)
        self.query_projections = [
            _ResidualHeadProjection(head_dim, rank) for _ in layer_indices
        ]

    def __call__(self, page: MemoryPage) -> tuple[LayerMemory, ...]:
        return self.writer(page)

    def query_projection_map(self) -> dict[int, _ResidualHeadProjection]:
        return dict(zip(self.layer_indices, self.query_projections, strict=True))


class _MemoryAttention(nn.Module):
    """Wrap one native full-attention module with a separate memory softmax."""

    def __init__(self, base: nn.Module, layer_index: int, active: _ActiveMemory):
        super().__init__()
        self.base = base
        self.layer_index = layer_index
        self._active = active
        self._capture = False
        self._captured: LayerMemory | None = None

    def begin_capture(self) -> None:
        self._capture = True
        self._captured = None

    def finish_capture(self) -> LayerMemory:
        self._capture = False
        captured = self._captured
        self._captured = None
        if captured is None:
            raise MemorySidecarError(
                f"full-attention layer {self.layer_index} did not capture memory"
            )
        return captured

    def _native_memory_projection(self, x: mx.array) -> tuple[mx.array, mx.array]:
        batch, length, _ = x.shape
        keys = self.base.k_proj(x).reshape(
            batch, length, self.base.n_kv_heads, self.base.head_dim
        )
        values = keys
        if not self.base.use_k_eq_v:
            values = self.base.v_proj(x).reshape(
                batch, length, self.base.n_kv_heads, self.base.head_dim
            )
        keys = self.base.k_norm(keys).transpose(0, 2, 1, 3)
        values = self.base.v_norm(values).transpose(0, 2, 1, 3)
        return keys, values

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        shared_kv: tuple[mx.array, mx.array] | None = None,
        offset: Any = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], Any]:
        if self._capture:
            keys, values = self._native_memory_projection(x)
            self._captured = LayerMemory(self.layer_index, keys, values)

        output, shared_kv, offset = self.base(
            x,
            mask,
            cache,
            shared_kv=shared_kv,
            offset=offset,
        )
        active_layers = self._active.layers
        if active_layers is None:
            return output, shared_kv, offset
        memory = active_layers.get(self.layer_index)
        if memory is None:
            return output, shared_kv, offset

        batch, length, _ = x.shape
        queries = self.base.q_proj(x).reshape(
            batch, length, self.base.n_heads, self.base.head_dim
        )
        queries = self.base.q_norm(queries).transpose(0, 2, 1, 3)
        query_projections = self._active.query_projections
        if query_projections is not None:
            projection = query_projections.get(self.layer_index)
            if projection is None:
                raise MemorySidecarError(
                    f"memory query projection missing layer {self.layer_index}"
                )
            queries = projection(queries)

        keys = memory.keys
        values = memory.values
        if keys.dtype != queries.dtype:
            keys = keys.astype(queries.dtype)
            values = values.astype(queries.dtype)
        if keys.shape[0] == 1 and batch != 1:
            keys = mx.broadcast_to(keys, (batch, *keys.shape[1:]))
            values = mx.broadcast_to(values, (batch, *values.shape[1:]))
        memory_output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.base.scale,
            mask=None,
        )
        memory_output = memory_output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        memory_output = self.base.o_proj(memory_output)
        return output + memory.gate * memory_output, shared_kv, offset


def _array_identity(digest: Any, value: mx.array) -> None:
    dtype = str(value.dtype).encode()
    digest.update(struct.pack(">I", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack(">I", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(bytes(value))


def _page_identity(
    *, model_id: str, runtime_profile: str, layers: tuple[LayerMemory, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"exo-gemma4-memory-page-v1\0")
    for text in (model_id, runtime_profile):
        encoded = text.encode()
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
    digest.update(struct.pack(">I", len(layers)))
    for layer in layers:
        digest.update(struct.pack(">I", layer.layer_index))
        digest.update(struct.pack(">d", layer.gate))
        _array_identity(digest, layer.keys)
        _array_identity(digest, layer.values)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MountedMemorySidecar:
    model: nn.Module
    model_id: str
    runtime_profile: str
    sites: tuple[_MemoryAttention, ...]
    _active: _ActiveMemory

    def compile_page(
        self, slot_embeddings: mx.array, *, gates: dict[int, float] | None = None
    ) -> MemoryPage:
        if self._active.layers is not None:
            raise MemorySidecarError("cannot compile while a memory page is active")
        if slot_embeddings.ndim != 3 or slot_embeddings.shape[0] != 1:
            raise MemorySidecarError(
                "slot embeddings must have shape [1, slots, hidden_size]"
            )
        for site in self.sites:
            site.begin_capture()
        try:
            self.model(None, input_embeddings=slot_embeddings)
            layers = tuple(site.finish_capture() for site in self.sites)
        except Exception:
            for site in self.sites:
                site._capture = False
                site._captured = None
            raise
        if gates is not None:
            unknown = set(gates).difference(layer.layer_index for layer in layers)
            if unknown:
                raise MemorySidecarError(f"gates name non-memory layers: {unknown}")
            layers = tuple(
                LayerMemory(
                    layer.layer_index,
                    layer.keys,
                    layer.values,
                    gates.get(layer.layer_index, layer.gate),
                )
                for layer in layers
            )
        mx.eval(*(value for layer in layers for value in (layer.keys, layer.values)))
        page_id = _page_identity(
            model_id=self.model_id,
            runtime_profile=self.runtime_profile,
            layers=layers,
        )
        return MemoryPage(page_id, self.model_id, self.runtime_profile, layers)

    def activate(
        self,
        page: MemoryPage,
        *,
        proof_id: str,
        reader: AddressableMemoryAdapter | None = None,
    ) -> None:
        if not proof_id:
            raise MemorySidecarError("activation requires a proof identity")
        if page.model_id != self.model_id:
            raise MemorySidecarError("memory page targets a different model")
        if page.runtime_profile != self.runtime_profile:
            raise MemorySidecarError("memory page targets a different runtime profile")
        expected = tuple(site.layer_index for site in self.sites)
        actual = tuple(layer.layer_index for layer in page.layers)
        if actual != expected:
            raise MemorySidecarError(
                "memory page layers are not in canonical model order"
            )
        recomputed = _page_identity(
            model_id=page.model_id,
            runtime_profile=page.runtime_profile,
            layers=page.layers,
        )
        if recomputed != page.page_id:
            raise MemorySidecarError("memory page identity does not match its bytes")
        for site, layer in zip(self.sites, page.layers, strict=True):
            if not math.isfinite(layer.gate) or not 0.0 <= layer.gate <= 1.0:
                raise MemorySidecarError(
                    f"memory gate is outside [0, 1] at layer {layer.layer_index}"
                )
            expected_shape = (
                1,
                site.base.n_kv_heads,
                layer.keys.shape[2],
                site.base.head_dim,
            )
            if (
                layer.keys.shape != expected_shape
                or layer.values.shape != expected_shape
            ):
                raise MemorySidecarError(
                    f"memory shape mismatch at layer {layer.layer_index}"
                )
        if self._active.training:
            raise MemorySidecarError(
                "cannot activate a page while training memory is live"
            )
        if self._active.page is not None and (
            self._active.page.page_id != page.page_id
            or self._active.proof_id != proof_id
        ):
            raise MemorySidecarError("a different memory proof is already active")
        self._active.page = page
        self._active.proof_id = proof_id
        self._active.layers = {layer.layer_index: layer for layer in page.layers}
        if reader is not None:
            if reader.layer_indices != expected:
                raise MemorySidecarError("memory reader layer set does not match model")
            self._active.query_projections = reader.query_projection_map()
        else:
            self._active.query_projections = None
        self._active.training = False

    def use_training_bank(self, bank: DirectMemoryBank) -> None:
        """Install a differentiable bank without granting production authority."""

        self.use_training_layers(bank.layer_memories())

    def use_training_layers(
        self,
        layers: tuple[LayerMemory, ...],
        *,
        reader: AddressableMemoryAdapter | None = None,
    ) -> None:
        """Install differentiable layers without granting production authority."""

        if self._active.page is not None:
            raise MemorySidecarError("cannot train while a proved page is active")
        expected = tuple(site.layer_index for site in self.sites)
        actual = tuple(layer.layer_index for layer in layers)
        if actual != expected:
            raise MemorySidecarError("training bank layer set does not match the model")
        if reader is not None and reader.layer_indices != expected:
            raise MemorySidecarError("memory reader layer set does not match model")
        self._active.layers = {layer.layer_index: layer for layer in layers}
        self._active.query_projections = (
            None if reader is None else reader.query_projection_map()
        )
        self._active.training = True

    def freeze_training_bank(self, bank: DirectMemoryBank) -> MemoryPage:
        """Materialize a trained bank as an immutable, identity-bound page."""

        return self.freeze_layers(bank.layer_memories())

    def freeze_layers(self, layers: tuple[LayerMemory, ...]) -> MemoryPage:
        """Materialize differentiable memory layers as an immutable page."""

        layers = tuple(
            LayerMemory(
                layer.layer_index,
                layer.keys.astype(mx.bfloat16),
                layer.values.astype(mx.bfloat16),
                layer.gate,
            )
            for layer in layers
        )
        mx.eval(*(value for layer in layers for value in (layer.keys, layer.values)))
        page_id = _page_identity(
            model_id=self.model_id,
            runtime_profile=self.runtime_profile,
            layers=layers,
        )
        return MemoryPage(page_id, self.model_id, self.runtime_profile, layers)

    def deactivate(self) -> None:
        self._active.page = None
        self._active.proof_id = None
        self._active.layers = None
        self._active.query_projections = None
        self._active.training = False


def mount_memory_sidecar(
    model: nn.Module, *, model_id: str, runtime_profile: str
) -> MountedMemorySidecar:
    """Attach sidecars to every full-attention layer of a Gemma 4 model."""

    if not model_id or not runtime_profile:
        raise MemorySidecarError("model and runtime identities must be non-empty")
    active = _ActiveMemory()
    sites: list[_MemoryAttention] = []
    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        if attention.layer_type != "full_attention":
            continue
        if isinstance(attention, _MemoryAttention):
            raise MemorySidecarError("memory sidecar is already mounted")
        wrapper = _MemoryAttention(attention, index, active)
        layer.self_attn = wrapper
        sites.append(wrapper)
    if not sites:
        raise MemorySidecarError("model has no full-attention layers")
    return MountedMemorySidecar(
        model=model,
        model_id=model_id,
        runtime_profile=runtime_profile,
        sites=tuple(sites),
        _active=active,
    )

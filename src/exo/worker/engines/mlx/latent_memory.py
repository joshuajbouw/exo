# type: ignore
"""Model-readable latent pages for Gemma 4 inference.

Selection and authorization happen before this module is called.  This module
only verifies that an immutable page matches the selected identity and the
loaded model, then exposes the page through a separate attention channel.  It
does not inspect prompts, retrieve memories, or add prompt/KV tokens.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn


class LatentMemoryError(ValueError):
    """A latent page or activation violated the runtime contract."""


@dataclass(frozen=True, slots=True)
class LatentMemorySelection:
    """Opaque upstream proof reference bound to exactly one latent page."""

    proof_id: str
    fact_snapshot_id: str
    selected_page_id: str

    def __post_init__(self) -> None:
        if not self.proof_id:
            raise LatentMemoryError("latent-memory proof identity is empty")
        if not self.fact_snapshot_id:
            raise LatentMemoryError("latent-memory fact snapshot is empty")
        if not self.selected_page_id:
            raise LatentMemoryError("latent-memory page identity is empty")


@dataclass(frozen=True, slots=True)
class LatentMemoryActivation:
    """A verified page and the upstream proof that selected it."""

    page: LatentMemoryPage
    selection: LatentMemorySelection


class LatentMemoryResolver(Protocol):
    """Trusted control-plane seam; implementations select outside the model."""

    @property
    def runtime_profile(self) -> str: ...

    def resolve(self, invocation_id: str) -> LatentMemoryActivation | None: ...


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
class LatentMemoryPage:
    page_id: str
    model_id: str
    runtime_profile: str
    layers: tuple[LayerMemory, ...]

    @property
    def resident_bytes(self) -> int:
        return sum(layer.resident_bytes for layer in self.layers)


def save_latent_memory_page(page: LatentMemoryPage, path: Path) -> None:
    """Persist an immutable page without retaining source text or tokens."""

    _verify_page_identity(page)
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


def load_latent_memory_page(path: Path) -> LatentMemoryPage:
    """Load a page and recompute its identity from the tensor bytes."""

    loaded = mx.load(path, return_metadata=True)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise LatentMemoryError("latent-memory page has no metadata")
    arrays, metadata = loaded
    if not isinstance(arrays, dict) or not isinstance(metadata, dict):
        raise LatentMemoryError("latent-memory page has an invalid container")
    encoded_record = metadata.get("page_record")
    if not isinstance(encoded_record, str):
        raise LatentMemoryError("latent-memory page record is missing")
    try:
        record = json.loads(encoded_record)
    except json.JSONDecodeError as error:
        raise LatentMemoryError("latent-memory page record is invalid JSON") from error
    if (
        not isinstance(record, dict)
        or record.get("format") != "exo-gemma4-memory-page-v1"
    ):
        raise LatentMemoryError("latent-memory page format is unsupported")

    layer_records = record.get("layers")
    if not isinstance(layer_records, list) or not layer_records:
        raise LatentMemoryError("latent-memory page has no layers")
    layers: list[LayerMemory] = []
    expected_names: set[str] = set()
    previous_index = -1
    for layer_record in layer_records:
        if not isinstance(layer_record, dict):
            raise LatentMemoryError("latent-memory layer record is invalid")
        layer_index = layer_record.get("layer_index")
        gate = layer_record.get("gate")
        if (
            not isinstance(layer_index, int)
            or isinstance(layer_index, bool)
            or layer_index <= previous_index
            or not isinstance(gate, (int, float))
            or isinstance(gate, bool)
        ):
            raise LatentMemoryError("latent-memory layer order or gate is invalid")
        previous_index = layer_index
        key_name = f"layer.{layer_index}.keys"
        value_name = f"layer.{layer_index}.values"
        expected_names.update((key_name, value_name))
        if key_name not in arrays or value_name not in arrays:
            raise LatentMemoryError("latent-memory page tensor is missing")
        layers.append(
            LayerMemory(layer_index, arrays[key_name], arrays[value_name], float(gate))
        )
    if set(arrays) != expected_names:
        raise LatentMemoryError("latent-memory page has undeclared tensors")

    page_id = record.get("page_id")
    model_id = record.get("model_id")
    runtime_profile = record.get("runtime_profile")
    if not all(
        isinstance(value, str) and value
        for value in (page_id, model_id, runtime_profile)
    ):
        raise LatentMemoryError("latent-memory page identity fields are invalid")
    page = LatentMemoryPage(page_id, model_id, runtime_profile, tuple(layers))
    _verify_page_identity(page)
    return page


def compose_latent_memory_pages(
    pages: tuple[LatentMemoryPage, ...],
) -> LatentMemoryPage:
    """Compose already-selected pages into one canonical attention bank."""

    if not pages:
        raise LatentMemoryError("latent-memory composition is empty")
    ordered = tuple(sorted(pages, key=lambda page: page.page_id))
    if len({page.page_id for page in ordered}) != len(ordered):
        raise LatentMemoryError("latent-memory composition contains a duplicate")
    reference = ordered[0]
    indices = tuple(layer.layer_index for layer in reference.layers)
    gates = tuple(layer.gate for layer in reference.layers)
    for page in ordered:
        _verify_page_identity(page)
        if (
            page.model_id != reference.model_id
            or page.runtime_profile != reference.runtime_profile
            or tuple(layer.layer_index for layer in page.layers) != indices
            or tuple(layer.gate for layer in page.layers) != gates
        ):
            raise LatentMemoryError("latent-memory pages target incompatible models")
    layers = tuple(
        LayerMemory(
            layer_index,
            mx.concatenate([page.layers[position].keys for page in ordered], axis=2),
            mx.concatenate(
                [page.layers[position].values for page in ordered], axis=2
            ),
            gates[position],
        )
        for position, layer_index in enumerate(indices)
    )
    mx.eval(*(value for layer in layers for value in (layer.keys, layer.values)))
    return _new_page(reference.model_id, reference.runtime_profile, layers)


@dataclass(slots=True)
class _ActiveMemory:
    layers: dict[int, LayerMemory] | None = None


class _LatentMemoryAttention(nn.Module):
    """Add a separate latent-memory softmax to one full-attention layer."""

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
            raise LatentMemoryError(
                f"full-attention layer {self.layer_index} captured no latent memory"
            )
        return captured

    def _native_projection(self, x: mx.array) -> tuple[mx.array, mx.array]:
        batch, length, _ = x.shape
        keys = self.base.k_proj(x).reshape(
            batch, length, self.base.n_kv_heads, self.base.head_dim
        )
        values = keys
        if not self.base.use_k_eq_v:
            values = self.base.v_proj(x).reshape(
                batch, length, self.base.n_kv_heads, self.base.head_dim
            )
        return (
            self.base.k_norm(keys).transpose(0, 2, 1, 3),
            self.base.v_norm(values).transpose(0, 2, 1, 3),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        shared_kv: tuple[mx.array, mx.array] | None = None,
        offset: Any = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], Any]:
        if self._capture:
            keys, values = self._native_projection(x)
            self._captured = LayerMemory(self.layer_index, keys, values)

        output, shared_kv, offset = self.base(
            x, mask, cache, shared_kv=shared_kv, offset=offset
        )
        active = self._active.layers
        if active is None or (memory := active.get(self.layer_index)) is None:
            return output, shared_kv, offset

        batch, length, _ = x.shape
        queries = self.base.q_proj(x).reshape(
            batch, length, self.base.n_heads, self.base.head_dim
        )
        queries = self.base.q_norm(queries).transpose(0, 2, 1, 3)
        keys = memory.keys
        values = memory.values
        if keys.dtype != queries.dtype:
            keys = keys.astype(queries.dtype)
            values = values.astype(queries.dtype)
        if keys.shape[0] == 1 and batch != 1:
            keys = mx.broadcast_to(keys, (batch, *keys.shape[1:]))
            values = mx.broadcast_to(values, (batch, *values.shape[1:]))
        memory_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.base.scale, mask=None
        )
        memory_output = memory_output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        output = output + memory.gate * self.base.o_proj(memory_output)
        return output, shared_kv, offset


@dataclass(slots=True)
class MountedLatentMemory:
    """Invocation-scoped latent-memory owner for one loaded model."""

    model: nn.Module
    model_id: str
    runtime_profile: str
    sites: tuple[_LatentMemoryAttention, ...]
    _active: _ActiveMemory
    _activation_lock: RLock

    def compile_page(self, slot_embeddings: mx.array) -> LatentMemoryPage:
        """Compile writer-produced slot embeddings into native layer K/V pages."""

        if self._active.layers is not None:
            raise LatentMemoryError("cannot compile while latent memory is active")
        if slot_embeddings.ndim != 3 or slot_embeddings.shape[0] != 1:
            raise LatentMemoryError("slot embeddings must have shape [1, slots, hidden]")
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
        mx.eval(*(value for layer in layers for value in (layer.keys, layer.values)))
        return _new_page(self.model_id, self.runtime_profile, layers)

    @contextlib.contextmanager
    def activation(
        self,
        page: LatentMemoryPage,
        selection: LatentMemorySelection,
    ) -> Iterator[None]:
        """Hold exclusive model activation for exactly one inference scope."""

        with self._activation_lock:
            if self._active.layers is not None:
                raise LatentMemoryError("latent memory is already active")
            self._validate_activation(page, selection)
            self._active.layers = {
                layer.layer_index: layer for layer in page.layers
            }
            try:
                yield
            finally:
                self._active.layers = None

    def _validate_activation(
        self,
        page: LatentMemoryPage,
        selection: LatentMemorySelection,
    ) -> None:
        if selection.selected_page_id != page.page_id:
            raise LatentMemoryError("selection names a different latent page")
        if page.model_id != self.model_id:
            raise LatentMemoryError("latent page targets a different model")
        if page.runtime_profile != self.runtime_profile:
            raise LatentMemoryError("latent page targets a different runtime profile")
        _verify_page_identity(page)
        expected = tuple(site.layer_index for site in self.sites)
        if tuple(layer.layer_index for layer in page.layers) != expected:
            raise LatentMemoryError("latent page layers are not in model order")
        for site, layer in zip(self.sites, page.layers, strict=True):
            if not math.isfinite(layer.gate) or not 0.0 <= layer.gate <= 1.0:
                raise LatentMemoryError("latent-memory gate is outside [0, 1]")
            shape = (1, site.base.n_kv_heads, layer.keys.shape[2], site.base.head_dim)
            if layer.keys.shape != shape or layer.values.shape != shape:
                raise LatentMemoryError("latent-memory tensor shape is incompatible")


def mount_latent_memory(
    model: nn.Module, *, model_id: str, runtime_profile: str
) -> MountedLatentMemory:
    """Mount the latent channel on Gemma 4 full-attention layers."""

    if not model_id or not runtime_profile:
        raise LatentMemoryError("model and runtime identities must be non-empty")
    active = _ActiveMemory()
    sites: list[_LatentMemoryAttention] = []
    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        if attention.layer_type != "full_attention":
            continue
        if isinstance(attention, _LatentMemoryAttention):
            raise LatentMemoryError("latent-memory channel is already mounted")
        wrapper = _LatentMemoryAttention(attention, index, active)
        layer.self_attn = wrapper
        sites.append(wrapper)
    if not sites:
        raise LatentMemoryError("model has no compatible full-attention layers")
    return MountedLatentMemory(
        model, model_id, runtime_profile, tuple(sites), active, RLock()
    )


def _array_identity(digest: Any, value: mx.array) -> None:
    dtype = str(value.dtype).encode()
    digest.update(struct.pack(">I", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack(">I", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(bytes(value))


def _page_identity(
    model_id: str, runtime_profile: str, layers: tuple[LayerMemory, ...]
) -> str:
    digest = hashlib.sha256()
    # This is the prospectively tested experiment format. Production adopts it
    # byte-for-byte so the accepted recall evidence remains applicable.
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


def _new_page(
    model_id: str, runtime_profile: str, layers: tuple[LayerMemory, ...]
) -> LatentMemoryPage:
    return LatentMemoryPage(
        _page_identity(model_id, runtime_profile, layers),
        model_id,
        runtime_profile,
        layers,
    )


def _verify_page_identity(page: LatentMemoryPage) -> None:
    if _page_identity(page.model_id, page.runtime_profile, page.layers) != page.page_id:
        raise LatentMemoryError("latent-memory page identity does not match its bytes")

# type: ignore

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.gemma4_text import Model, ModelArgs

sys.path.insert(0, str(Path(__file__).parents[1]))

import durable_grounded_memory as durable
from durable_grounded_memory import (
    DurableGroundedMemoryStore,
    DurableMemoryError,
    PagePublication,
)
from gemma4_memory_sidecar_mlx import mount_memory_sidecar, save_memory_page
from grounded_memory_selection import MemoryCatalogEntry
from semantic_memory_invocation import SemanticRelationFact


class _Backend:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.values: dict[str, bytes] = {}
        self.set_order: list[str] = []
        self.fail_root = False
        self.substitute_object = False
        self.substitute_digest = False

    def put_file(self, name: str, source: Path) -> tuple[str, str]:
        data = source.read_bytes()
        self.files[name] = data
        return self._identities(data)

    def get_file(self, name: str, destination: Path) -> tuple[str, str] | None:
        data = self.files.get(name)
        if data is None:
            return None
        destination.write_bytes(data)
        object_id, digest = self._identities(data)
        if self.substitute_object:
            object_id = "object:substituted"
        if self.substitute_digest:
            digest = "digest:substituted"
        return object_id, digest

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def set(self, key: str, value: bytes) -> None:
        if self.fail_root and key.endswith("/current"):
            raise OSError("injected root failure")
        self.values[key] = bytes(value)
        self.set_order.append(key)

    @staticmethod
    def _identities(data: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(data).hexdigest()
        return f"object:{digest}", f"digest:{digest}"


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


def _publication(tmp_path: Path, token: int) -> PagePublication:
    model = _model()
    mounted = mount_memory_sidecar(
        model, model_id="tiny-gemma4", runtime_profile="mlx-test-v1"
    )
    page = mounted.compile_page(model.model.embed_tokens(mx.array([[token]])))
    source = tmp_path / f"{page.page_id}.safetensors"
    save_memory_page(page, source)
    return PagePublication(
        MemoryCatalogEntry(
            page_id=page.page_id,
            evidence_closure_id=f"evidence:{token}",
            privacy_domain_id="alpha",
            concept_id=f"concept:{token}",
            model_id=page.model_id,
            runtime_profile=page.runtime_profile,
            first_epoch=0,
            last_epoch=100,
        ),
        source,
        SemanticRelationFact(
            "alpha",
            "calibration-word-of",
            f"device:{token}",
            f"concept:{token}",
            f"evidence:{token}",
        ),
    )


def test_publish_reopens_verified_pages_and_advances_root_last(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    publications = (_publication(tmp_path, 7), _publication(tmp_path, 11))

    manifest_id = store.publish("alpha", publications)
    reopened = store.reopen("alpha", tmp_path / "projection")

    assert reopened is not None
    assert reopened.manifest_id == manifest_id
    assert {entry.page_id for entry in reopened.entries} == {
        publication.entry.page_id for publication in publications
    }
    assert set(reopened.pages) == {entry.page_id for entry in reopened.entries}
    assert reopened.relations == tuple(
        sorted(publication.relation for publication in publications)
    )
    assert "/manifest/" in backend.set_order[-2]
    assert backend.set_order[-1].endswith("/current")


def test_failed_root_advance_preserves_previous_catalog(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    first = _publication(tmp_path, 13)
    store.publish("alpha", (first,))
    backend.fail_root = True

    with pytest.raises(OSError):
        store.publish("alpha", (_publication(tmp_path, 17),))

    backend.fail_root = False
    reopened = store.reopen("alpha", tmp_path / "projection")
    assert reopened is not None
    assert tuple(entry.page_id for entry in reopened.entries) == (first.entry.page_id,)


def test_reopen_rejects_substituted_representation(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    store.publish("alpha", (_publication(tmp_path, 19),))
    backend.substitute_digest = True

    with pytest.raises(DurableMemoryError, match="substituted"):
        store.reopen("alpha", tmp_path / "projection")


def test_reopen_rejects_missing_manifest(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    store.publish("alpha", (_publication(tmp_path, 23),))
    manifest_key = backend.set_order[-2]
    del backend.values[manifest_key]

    with pytest.raises(DurableMemoryError, match="manifest is missing"):
        store.reopen("alpha", tmp_path / "projection")


def test_reopen_rejects_noncanonical_manifest(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    store.publish("alpha", (_publication(tmp_path, 29),))
    manifest_key = backend.set_order[-2]
    noncanonical = b" " + backend.values[manifest_key]
    manifest_id = durable._manifest_id(noncanonical)
    backend.values[durable._manifest_key(manifest_id)] = noncanonical
    backend.values[durable._domain_key("alpha")] = manifest_id.encode("ascii")

    with pytest.raises(DurableMemoryError, match="non-canonical"):
        store.reopen("alpha", tmp_path / "projection")


def test_reopen_rejects_missing_page(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    publication = _publication(tmp_path, 31)
    store.publish("alpha", (publication,))
    del backend.files[f"latent-memory-page-v1-{publication.entry.page_id}"]

    with pytest.raises(DurableMemoryError, match="page content is missing"):
        store.reopen("alpha", tmp_path / "projection")


def test_reopen_rejects_substituted_object_identity(tmp_path: Path) -> None:
    backend = _Backend()
    store = DurableGroundedMemoryStore(tmp_path / "store", backend=backend)
    store.publish("alpha", (_publication(tmp_path, 37),))
    backend.substitute_object = True

    with pytest.raises(DurableMemoryError, match="substituted"):
        store.reopen("alpha", tmp_path / "projection")

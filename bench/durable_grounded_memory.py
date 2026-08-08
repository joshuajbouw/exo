"""Astrid-backed publication and verified reopen for latent-memory catalogs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import msgspec
from gemma4_memory_sidecar_mlx import MemoryPage, load_memory_page
from grounded_memory_selection import MemoryCatalogEntry
from semantic_memory_invocation import SemanticRelationFact

_SCHEMA = 1
_MANIFEST_DOMAIN = b"exo.durable-grounded-memory-manifest.v1\0"


class DurableMemoryError(ValueError):
    """Durable memory state failed identity or closure validation."""


class MemoryStoreBackend(Protocol):
    def put_file(self, name: str, source: Path) -> tuple[str, str]: ...

    def get_file(self, name: str, destination: Path) -> tuple[str, str] | None: ...

    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class PagePublication:
    entry: MemoryCatalogEntry
    source: Path
    relation: SemanticRelationFact


@dataclass(frozen=True, slots=True)
class ReopenedMemoryCatalog:
    manifest_id: str
    entries: tuple[MemoryCatalogEntry, ...]
    pages: dict[str, MemoryPage]
    relations: tuple[SemanticRelationFact, ...]


class _StoredEntry(msgspec.Struct, frozen=True):
    page_id: str
    evidence_closure_id: str
    privacy_domain_id: str
    concept_id: str
    model_id: str
    runtime_profile: str
    first_epoch: int
    last_epoch: int
    content_name: str
    content_object_id: str
    content_digest: str


class _Manifest(msgspec.Struct, frozen=True):
    schema: int
    privacy_domain_id: str
    entries: tuple[_StoredEntry, ...]
    relations: tuple[SemanticRelationFact, ...]


def _domain_key(privacy_domain_id: str) -> str:
    digest = hashlib.sha256(privacy_domain_id.encode()).hexdigest()
    return f"latent-memory/catalog/v1/domain/{digest}/current"


def _manifest_key(manifest_id: str) -> str:
    return f"latent-memory/catalog/v1/manifest/{manifest_id}"


def _content_name(page_id: str) -> str:
    return f"latent-memory-page-v1-{page_id}"


def _manifest_id(encoded: bytes) -> str:
    return hashlib.sha256(_MANIFEST_DOMAIN + encoded).hexdigest()


def _entry_from_stored(stored: _StoredEntry) -> MemoryCatalogEntry:
    return MemoryCatalogEntry(
        page_id=stored.page_id,
        evidence_closure_id=stored.evidence_closure_id,
        privacy_domain_id=stored.privacy_domain_id,
        concept_id=stored.concept_id,
        model_id=stored.model_id,
        runtime_profile=stored.runtime_profile,
        first_epoch=stored.first_epoch,
        last_epoch=stored.last_epoch,
    )


class DurableGroundedMemoryStore:
    """Publish pages before atomically advancing one domain catalog root."""

    def __init__(
        self,
        path: Path,
        *,
        backend: MemoryStoreBackend | None = None,
    ) -> None:
        if backend is None:
            try:
                from exo_computation_store import ComputationStore
            except ImportError as error:
                raise RuntimeError(
                    "durable latent memory requires the computation-store extension"
                ) from error
            backend = cast(MemoryStoreBackend, ComputationStore(path))
        self._backend = backend

    def publish(
        self,
        privacy_domain_id: str,
        publications: tuple[PagePublication, ...],
    ) -> str:
        if not privacy_domain_id or not publications:
            raise DurableMemoryError("catalog publication requires a domain and pages")
        ordered = tuple(sorted(publications, key=lambda item: item.entry.page_id))
        if len({item.entry.page_id for item in ordered}) != len(ordered):
            raise DurableMemoryError("catalog publication contains duplicate pages")

        stored_entries: list[_StoredEntry] = []
        relations: list[SemanticRelationFact] = []
        for publication in ordered:
            entry = publication.entry
            if entry.privacy_domain_id != privacy_domain_id:
                raise DurableMemoryError("catalog page belongs to another domain")
            page = load_memory_page(publication.source)
            if (
                page.page_id != entry.page_id
                or page.model_id != entry.model_id
                or page.runtime_profile != entry.runtime_profile
            ):
                raise DurableMemoryError("catalog entry does not describe its page")
            relation = publication.relation
            if (
                relation.privacy_domain_id != privacy_domain_id
                or relation.concept_id != entry.concept_id
                or relation.evidence_closure_id != entry.evidence_closure_id
            ):
                raise DurableMemoryError("catalog relation does not describe its entry")
            relations.append(relation)
            content_name = _content_name(entry.page_id)
            object_id, digest = self._backend.put_file(content_name, publication.source)
            stored_entries.append(
                _StoredEntry(
                    page_id=entry.page_id,
                    evidence_closure_id=entry.evidence_closure_id,
                    privacy_domain_id=entry.privacy_domain_id,
                    concept_id=entry.concept_id,
                    model_id=entry.model_id,
                    runtime_profile=entry.runtime_profile,
                    first_epoch=entry.first_epoch,
                    last_epoch=entry.last_epoch,
                    content_name=content_name,
                    content_object_id=object_id,
                    content_digest=digest,
                )
            )

        ordered_relations = tuple(sorted(relations))
        relation_keys = {
            (relation.relation_id, relation.subject_id)
            for relation in ordered_relations
        }
        if len(relation_keys) != len(ordered_relations):
            raise DurableMemoryError("catalog relation key is ambiguous")
        manifest = _Manifest(
            _SCHEMA,
            privacy_domain_id,
            tuple(stored_entries),
            ordered_relations,
        )
        encoded = msgspec.json.encode(manifest)
        manifest_id = _manifest_id(encoded)
        self._backend.set(_manifest_key(manifest_id), encoded)
        self._backend.set(_domain_key(privacy_domain_id), manifest_id.encode("ascii"))
        return manifest_id

    def reopen(
        self,
        privacy_domain_id: str,
        projection_directory: Path,
    ) -> ReopenedMemoryCatalog | None:
        encoded_root = self._backend.get(_domain_key(privacy_domain_id))
        if encoded_root is None:
            return None
        try:
            manifest_id = encoded_root.decode("ascii")
        except UnicodeDecodeError as error:
            raise DurableMemoryError("catalog root identity is not ASCII") from error
        encoded = self._backend.get(_manifest_key(manifest_id))
        if encoded is None or _manifest_id(encoded) != manifest_id:
            raise DurableMemoryError(
                "catalog manifest is missing or has wrong identity"
            )
        try:
            manifest = msgspec.json.decode(encoded, type=_Manifest)
        except msgspec.DecodeError as error:
            raise DurableMemoryError("catalog manifest cannot be decoded") from error
        if (
            manifest.schema != _SCHEMA
            or manifest.privacy_domain_id != privacy_domain_id
            or msgspec.json.encode(manifest) != encoded
        ):
            raise DurableMemoryError(
                "catalog manifest is non-canonical or incompatible"
            )
        if tuple(entry.page_id for entry in manifest.entries) != tuple(
            sorted(entry.page_id for entry in manifest.entries)
        ):
            raise DurableMemoryError("catalog manifest page order is non-canonical")
        if manifest.relations != tuple(sorted(manifest.relations)):
            raise DurableMemoryError("catalog relation order is non-canonical")
        entry_evidence = {
            (entry.concept_id, entry.evidence_closure_id) for entry in manifest.entries
        }
        relation_keys: set[tuple[str, str]] = set()
        for relation in manifest.relations:
            key = (relation.relation_id, relation.subject_id)
            if (
                relation.privacy_domain_id != privacy_domain_id
                or (relation.concept_id, relation.evidence_closure_id)
                not in entry_evidence
                or key in relation_keys
            ):
                raise DurableMemoryError("catalog relation closure is invalid")
            relation_keys.add(key)

        projection_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        entries: list[MemoryCatalogEntry] = []
        pages: dict[str, MemoryPage] = {}
        for stored in manifest.entries:
            destination = projection_directory / f"{stored.page_id}.safetensors"
            restored = self._backend.get_file(stored.content_name, destination)
            if restored is None:
                raise DurableMemoryError("catalog page content is missing")
            object_id, digest = restored
            if object_id != stored.content_object_id or digest != stored.content_digest:
                destination.unlink(missing_ok=True)
                raise DurableMemoryError("catalog page representation was substituted")
            page = load_memory_page(destination)
            entry = _entry_from_stored(stored)
            if (
                page.page_id != entry.page_id
                or page.model_id != entry.model_id
                or page.runtime_profile != entry.runtime_profile
            ):
                destination.unlink(missing_ok=True)
                raise DurableMemoryError(
                    "reopened page does not match its catalog entry"
                )
            if page.page_id in pages:
                raise DurableMemoryError("catalog manifest repeats a page")
            entries.append(entry)
            pages[page.page_id] = page
        return ReopenedMemoryCatalog(
            manifest_id,
            tuple(entries),
            pages,
            manifest.relations,
        )

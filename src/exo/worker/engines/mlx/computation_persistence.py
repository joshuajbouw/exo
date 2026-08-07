"""Optional durable persistence for Exo's MLX prefix cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from stat import S_ISREG
from threading import Condition, Lock, Thread
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import msgspec
import numpy as np
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

from exo.worker.engines.mlx.cache import cache_length
from exo.worker.engines.mlx.cache_persistence import PersistedKVPrefix
from exo.worker.engines.mlx.types import (
    ContinuationFrontier,
    DraftContinuation,
    KVCacheType,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.cache import CacheSnapshot
    from exo.worker.engines.mlx.vision import MediaRegion


_SCHEMA = 2
_PREFIX_DOMAIN = b"exo-mlx-prefix-state-v1\0"


@dataclass(frozen=True, slots=True)
class _LazyCheckpoint:
    prompt_tokens: mx.array
    cache: KVCacheType
    prefill_tps: float
    continuation_id: str | None


@dataclass(frozen=True, slots=True)
class _PreparedCheckpoint:
    source: Path
    prefix_id: str
    token_count: int
    cache_tokens: int
    prefill_tps: float
    serialization_seconds: float
    continuation_id: str | None
    prompt_tokens: bytes


@dataclass(frozen=True, slots=True)
class _PreparedDraft:
    prefix_id: str
    prompt_token_count: int
    output_tokens: bytes


type _PendingCheckpoint = _LazyCheckpoint | _PreparedCheckpoint
type _PendingPublication = _PendingCheckpoint | _PreparedDraft
type _ProjectionFingerprint = tuple[int, int, int, int, int]


class _CheckpointMetadata(msgspec.Struct, frozen=True):
    schema: int
    runtime_profile: str
    prefix_id: str
    content_object: str
    projection_digest: str
    token_count: int
    cache_tokens: int
    prefill_tps: float


class _ContinuationMetadata(msgspec.Struct, frozen=True):
    schema: int
    runtime_profile: str
    continuation_id: str
    prefix_id: str
    token_count: int
    prompt_tokens: bytes


class _DraftMetadata(msgspec.Struct, frozen=True):
    schema: int
    runtime_profile: str
    prefix_id: str
    prompt_token_count: int
    output_token_count: int
    output_tokens: bytes


@dataclass(frozen=True, slots=True)
class RestoreMetrics:
    projection_verification_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    mlx_load_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PublicationMetrics:
    mlx_serialization_seconds: float = 0.0
    store_admission_seconds: float = 0.0
    metadata_publication_seconds: float = 0.0


class StoreKVPrefixPersistence:
    """Persist verified MLX prompt-cache files in a computation store.

    ``runtime_profile`` must change whenever model weights, tokenizer/template,
    cache layout, numerical behavior, shard assignment, or backend semantics
    change. Requiring it explicitly prevents an Exo or MLX upgrade from
    silently restoring incompatible accelerator state.
    """

    def __init__(self, store_path: Path, runtime_profile: str):
        try:
            from exo_computation_store import ComputationStore
        except ImportError as error:
            raise RuntimeError(
                "EXO_COMPUTATION_STORE requires the optional "
                "integrations/computation_store extension"
            ) from error

        if not runtime_profile:
            raise ValueError("computation persistence requires a runtime profile")
        self._store = ComputationStore(store_path)
        self._runtime_profile = runtime_profile
        self._profile_id = _digest(runtime_profile.encode())
        self._index_lock = Lock()
        self._projection_lock = Lock()
        self._verified_projections: dict[Path, tuple[_ProjectionFingerprint, str]] = {}
        self._publication = Condition()
        self._pending: deque[_PendingPublication] = deque()
        self._closing = False
        self._projections = store_path / "projections"
        self._projections.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.last_restore_metrics = RestoreMetrics()
        self.last_publication_metrics = PublicationMetrics()
        self._worker = Thread(
            target=self._publication_loop,
            name="exo-computation-kv",
            daemon=True,
        )
        self._worker.start()

    def restore_longest(
        self,
        prompt_tokens: mx.array,
        minimum_tokens: int,
        media_regions: list["MediaRegion"],
        prompt_token_bytes: bytes | None = None,
    ) -> PersistedKVPrefix | None:
        self.last_restore_metrics = RestoreMetrics()
        if media_regions:
            return None
        try:
            lengths = self._read_lengths()
            for token_count in reversed(lengths):
                if token_count <= minimum_tokens or token_count > len(prompt_tokens):
                    continue
                prefix_id = (
                    _prefix_id_from_bytes(
                        self._runtime_profile,
                        token_count,
                        prompt_token_bytes[: token_count * 4],
                    )
                    if prompt_token_bytes is not None
                    else _prefix_id(
                        self._runtime_profile,
                        prompt_tokens[:token_count],
                    )
                )
                metadata = self._read_metadata(prefix_id)
                if metadata is None or metadata.token_count != token_count:
                    continue
                projection = self._projection_path(metadata.content_object)
                verification_seconds = 0.0
                reconstruction_seconds = 0.0
                if self._projection_available(projection):
                    projection_digest = self._cached_projection_digest(projection)
                    if projection_digest is None:
                        started = time.perf_counter()
                        before = self._projection_fingerprint(projection)
                        projection_digest = self._store.digest_file(projection)
                        after = self._projection_fingerprint(projection)
                        verification_seconds = time.perf_counter() - started
                        if before is None or before != after:
                            projection_digest = "changed-during-verification"
                        else:
                            self._remember_verified_projection(
                                projection,
                                projection_digest,
                                after,
                            )
                    if projection_digest != metadata.projection_digest:
                        projection.unlink(missing_ok=True)
                        self._forget_verified_projection(projection)
                if not self._projection_available(projection):
                    destination = self._temporary_path("restore", prefix_id)
                    started = time.perf_counter()
                    try:
                        restored_identity = self._store.get_file(
                            _content_name(self._profile_id, prefix_id), destination
                        )
                        if restored_identity is None:
                            continue
                        content_object, projection_digest = restored_identity
                        if content_object != metadata.content_object:
                            raise ValueError("checkpoint object identity mismatch")
                        if projection_digest != metadata.projection_digest:
                            raise ValueError(
                                "checkpoint representation digest mismatch"
                            )
                        os.replace(destination, projection)
                        self._remember_verified_projection(
                            projection,
                            projection_digest,
                        )
                    finally:
                        destination.unlink(missing_ok=True)
                    reconstruction_seconds = time.perf_counter() - started
                started = time.perf_counter()
                restored = cast(
                    KVCacheType,
                    cast(object, load_prompt_cache(str(projection))),
                )
                load_seconds = time.perf_counter() - started
                self.last_restore_metrics = RestoreMetrics(
                    projection_verification_seconds=verification_seconds,
                    reconstruction_seconds=reconstruction_seconds,
                    mlx_load_seconds=load_seconds,
                )
                if cache_length(restored) != metadata.cache_tokens:
                    logger.warning(
                        "KV checkpoint length mismatch; treating it as a miss"
                    )
                    continue
                return PersistedKVPrefix(
                    prompt_tokens=prompt_tokens[:token_count],
                    cache=restored,
                    snapshots=None,
                    media_regions=[],
                    prefill_tps=metadata.prefill_tps,
                )
        except Exception:
            logger.opt(exception=True).warning(
                "KV checkpoint restore failed; continuing with uncached inference"
            )
        return None

    def resolve_continuation(self, continuation_id: str) -> ContinuationFrontier | None:
        """Resolve a response id to a verified append-only token frontier."""
        if not continuation_id:
            return None
        try:
            encoded = self._store.get(
                _continuation_key(self._profile_id, continuation_id)
            )
            if encoded is None:
                return None
            metadata = msgspec.json.decode(bytes(encoded), type=_ContinuationMetadata)
            if (
                metadata.schema != _SCHEMA
                or metadata.runtime_profile != self._profile_id
                or metadata.continuation_id != continuation_id
                or metadata.token_count < 0
                or len(metadata.prompt_tokens) != metadata.token_count * 4
            ):
                raise ValueError("continuation metadata mismatch")
            token_values = np.frombuffer(metadata.prompt_tokens, dtype="<u4").copy()
            tokens = mx.array(token_values, dtype=mx.uint32)
            if (
                _prefix_id_from_bytes(
                    self._runtime_profile,
                    metadata.token_count,
                    metadata.prompt_tokens,
                )
                != metadata.prefix_id
            ):
                raise ValueError("continuation token identity mismatch")
            terminal_token = int(cast(np.uint32, token_values[-1]))
            return ContinuationFrontier(
                tokens,
                terminal_token,
                metadata.prompt_tokens,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "KV continuation metadata failed verification; treating it as a miss"
            )
            return None

    def schedule_draft(
        self,
        prompt_tokens: mx.array,
        output_tokens: list[int],
    ) -> None:
        """Publish a speculative proposal off the inference critical path."""
        if not output_tokens:
            return
        prompt_bytes = np.asarray(prompt_tokens, dtype="<u4").tobytes(order="C")
        prepared = _PreparedDraft(
            prefix_id=_prefix_id_from_bytes(
                self._runtime_profile,
                len(prompt_tokens),
                prompt_bytes,
            ),
            prompt_token_count=len(prompt_tokens),
            output_tokens=np.asarray(output_tokens, dtype="<u4").tobytes(order="C"),
        )
        with self._publication:
            if self._closing:
                return
            for index in range(len(self._pending) - 1, -1, -1):
                queued = self._pending[index]
                if (
                    isinstance(queued, _PreparedDraft)
                    and queued.prefix_id == prepared.prefix_id
                ):
                    self._pending[index] = prepared
                    break
            else:
                self._pending.append(prepared)
            self._publication.notify()

    def resolve_draft(self, prompt_tokens: mx.array) -> DraftContinuation | None:
        """Load an identity-bound proposal which still requires model verification."""
        try:
            prompt_bytes = np.asarray(prompt_tokens, dtype="<u4").tobytes(order="C")
            prefix_id = _prefix_id_from_bytes(
                self._runtime_profile,
                len(prompt_tokens),
                prompt_bytes,
            )
            encoded = self._store.get(_draft_key(self._profile_id, prefix_id))
            if encoded is None:
                return None
            metadata = msgspec.json.decode(bytes(encoded), type=_DraftMetadata)
            if (
                metadata.schema != _SCHEMA
                or metadata.runtime_profile != self._profile_id
                or metadata.prefix_id != prefix_id
                or metadata.prompt_token_count != len(prompt_tokens)
                or metadata.output_token_count <= 0
                or len(metadata.output_tokens) != metadata.output_token_count * 4
            ):
                raise ValueError("speculative draft metadata mismatch")
            values = np.frombuffer(metadata.output_tokens, dtype="<u4")
            return DraftContinuation(tuple(cast(list[int], values.tolist())))
        except Exception:
            logger.opt(exception=True).warning(
                "Speculative draft metadata failed verification; treating it as a miss"
            )
            return None

    def schedule_store(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list["CacheSnapshot"] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
        continuation_id: str | None = None,
    ) -> None:
        # Media identity is not part of MLX's portable prompt-cache format.
        # SSM/rotating rollback snapshots need not persist: the serialized
        # cache itself is complete at this exact prefix boundary, and Exo will
        # not trim inside it without a snapshot.
        if media_regions:
            return
        with self._publication:
            if self._closing:
                return
            # Retain at most one waiting checkpoint. If generation advances
            # faster than storage, the newest frontier replaces stale queued
            # work instead of pinning every historical cache in RAM.
            self._enqueue_checkpoint(
                _LazyCheckpoint(
                    prompt_tokens,
                    cache,
                    prefill_tps,
                    continuation_id,
                )
            )
            self._publication.notify()

    def schedule_store_thread_bound(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list["CacheSnapshot"] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
        continuation_id: str | None = None,
    ) -> None:
        """Capture generation-stream state before crossing a thread boundary."""
        if media_regions:
            return
        prepared = self._prepare_checkpoint(
            prompt_tokens,
            cache,
            prefill_tps,
            continuation_id,
        )
        with self._publication:
            if self._closing:
                prepared.source.unlink(missing_ok=True)
                return
            self._enqueue_checkpoint(prepared)
            self._publication.notify()

    def _enqueue_checkpoint(self, checkpoint: _PendingCheckpoint) -> None:
        """Coalesce one lineage without dropping another response's frontier."""
        continuation_id = checkpoint.continuation_id
        for index in range(len(self._pending) - 1, -1, -1):
            queued = self._pending[index]
            if isinstance(queued, _PreparedDraft):
                continue
            if queued.continuation_id == continuation_id:
                if isinstance(queued, _PreparedCheckpoint):
                    queued.source.unlink(missing_ok=True)
                self._pending[index] = checkpoint
                return
        self._pending.append(checkpoint)

    def close(self) -> None:
        with self._publication:
            self._closing = True
            self._publication.notify()
        self._worker.join()

    def _publication_loop(self) -> None:
        while True:
            with self._publication:
                while not self._pending and not self._closing:
                    self._publication.wait()
                if not self._pending:
                    return
                publication = self._pending.popleft()
            try:
                if isinstance(publication, _PreparedDraft):
                    self._publish_draft(publication)
                elif isinstance(publication, _PreparedCheckpoint):
                    self._publish_checkpoint(publication)
                else:
                    self._store_checkpoint(
                        publication.prompt_tokens,
                        publication.cache,
                        publication.prefill_tps,
                        publication.continuation_id,
                    )
            except Exception:
                logger.opt(exception=True).warning(
                    "Computation publication failed; inference result remains valid"
                )

    def _publish_draft(self, draft: _PreparedDraft) -> None:
        metadata = _DraftMetadata(
            schema=_SCHEMA,
            runtime_profile=self._profile_id,
            prefix_id=draft.prefix_id,
            prompt_token_count=draft.prompt_token_count,
            output_token_count=len(draft.output_tokens) // 4,
            output_tokens=draft.output_tokens,
        )
        self._store.set(
            _draft_key(self._profile_id, draft.prefix_id),
            msgspec.json.encode(metadata),
        )

    def _store_checkpoint(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        prefill_tps: float,
        continuation_id: str | None = None,
    ) -> None:
        self._publish_checkpoint(
            self._prepare_checkpoint(
                prompt_tokens,
                cache,
                prefill_tps,
                continuation_id,
            )
        )

    def _prepare_checkpoint(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        prefill_tps: float,
        continuation_id: str | None = None,
    ) -> _PreparedCheckpoint:
        token_count = len(prompt_tokens)
        prefix_id = _prefix_id(self._runtime_profile, prompt_tokens)
        source = self._temporary_path("publish", prefix_id)
        started = time.perf_counter()
        try:
            save_prompt_cache(
                str(source),
                list(cache),  # pyright: ignore[reportArgumentType]
                {
                    "schema": str(_SCHEMA),
                    "runtime_profile": self._profile_id,
                    "prefix_id": prefix_id,
                },
            )
            return _PreparedCheckpoint(
                source=source,
                prefix_id=prefix_id,
                token_count=token_count,
                cache_tokens=cache_length(cache),
                prefill_tps=prefill_tps,
                serialization_seconds=time.perf_counter() - started,
                continuation_id=continuation_id,
                prompt_tokens=np.asarray(prompt_tokens, dtype="<u4").tobytes(order="C"),
            )
        except Exception:
            source.unlink(missing_ok=True)
            raise

    def _publish_checkpoint(self, checkpoint: _PreparedCheckpoint) -> None:
        source = checkpoint.source
        started = time.perf_counter()
        try:
            content_object, projection_digest = self._store.put_file(
                _content_name(self._profile_id, checkpoint.prefix_id), source
            )
            os.replace(source, self._projection_path(content_object))
            self._remember_verified_projection(
                self._projection_path(content_object),
                projection_digest,
            )
        finally:
            source.unlink(missing_ok=True)
        admission_seconds = time.perf_counter() - started
        started = time.perf_counter()
        metadata = _CheckpointMetadata(
            schema=_SCHEMA,
            runtime_profile=self._profile_id,
            prefix_id=checkpoint.prefix_id,
            content_object=content_object,
            projection_digest=projection_digest,
            token_count=checkpoint.token_count,
            cache_tokens=checkpoint.cache_tokens,
            prefill_tps=checkpoint.prefill_tps,
        )
        self._store.set(
            _metadata_key(self._profile_id, checkpoint.prefix_id),
            msgspec.json.encode(metadata),
        )
        with self._index_lock:
            lengths = self._read_lengths()
            if checkpoint.token_count not in lengths:
                lengths.append(checkpoint.token_count)
                lengths.sort()
                self._store.set(
                    _index_key(self._profile_id),
                    msgspec.json.encode(lengths),
                )
        if checkpoint.continuation_id is not None:
            continuation = _ContinuationMetadata(
                schema=_SCHEMA,
                runtime_profile=self._profile_id,
                continuation_id=checkpoint.continuation_id,
                prefix_id=checkpoint.prefix_id,
                token_count=checkpoint.token_count,
                prompt_tokens=checkpoint.prompt_tokens,
            )
            self._store.set(
                _continuation_key(
                    self._profile_id,
                    checkpoint.continuation_id,
                ),
                msgspec.json.encode(continuation),
            )
        self.last_publication_metrics = PublicationMetrics(
            mlx_serialization_seconds=checkpoint.serialization_seconds,
            store_admission_seconds=admission_seconds,
            metadata_publication_seconds=time.perf_counter() - started,
        )

    def _read_lengths(self) -> list[int]:
        encoded = self._store.get(_index_key(self._profile_id))
        if encoded is None:
            return []
        decoded = msgspec.json.decode(bytes(encoded), type=list[int])
        if not all(value >= 0 for value in decoded):
            raise ValueError("invalid prefix index")
        return sorted(set(decoded))

    def _read_metadata(self, prefix_id: str) -> _CheckpointMetadata | None:
        encoded = self._store.get(_metadata_key(self._profile_id, prefix_id))
        if encoded is None:
            return None
        decoded = msgspec.json.decode(bytes(encoded), type=_CheckpointMetadata)
        if (
            decoded.schema != _SCHEMA
            or decoded.runtime_profile != self._profile_id
            or decoded.prefix_id != prefix_id
            or not _is_lower_hex(decoded.content_object, 64)
            or not _is_representation_digest(decoded.projection_digest)
        ):
            raise ValueError("prefix metadata identity mismatch")
        return decoded

    def _projection_path(self, content_object: str) -> Path:
        if not _is_lower_hex(content_object, 64):
            raise ValueError("invalid checkpoint object identity")
        return self._projections / f"{content_object}.safetensors"

    @staticmethod
    def _projection_available(projection: Path) -> bool:
        if projection.is_symlink():
            projection.unlink(missing_ok=True)
            return False
        return projection.is_file()

    @staticmethod
    def _projection_fingerprint(
        projection: Path,
    ) -> _ProjectionFingerprint | None:
        try:
            value = projection.lstat()
        except FileNotFoundError:
            return None
        if not S_ISREG(value.st_mode):
            return None
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def _cached_projection_digest(self, projection: Path) -> str | None:
        fingerprint = self._projection_fingerprint(projection)
        if fingerprint is None:
            return None
        with self._projection_lock:
            cached = self._verified_projections.get(projection)
        if cached is None or cached[0] != fingerprint:
            return None
        return cached[1]

    def _remember_verified_projection(
        self,
        projection: Path,
        digest: str,
        fingerprint: _ProjectionFingerprint | None = None,
    ) -> None:
        fingerprint = fingerprint or self._projection_fingerprint(projection)
        if fingerprint is None:
            return
        with self._projection_lock:
            self._verified_projections[projection] = (fingerprint, digest)

    def _forget_verified_projection(self, projection: Path) -> None:
        with self._projection_lock:
            self._verified_projections.pop(projection, None)

    def _temporary_path(self, operation: str, prefix_id: str) -> Path:
        descriptor, value = tempfile.mkstemp(
            dir=self._projections,
            prefix=f".{operation}-{prefix_id}-",
            suffix=".safetensors",
        )
        os.close(descriptor)
        return Path(value)


def configured_computation_persistence(
    model_id: str,
    device_rank: int,
    shard_profile: str,
) -> StoreKVPrefixPersistence | None:
    """Create the opt-in backend or leave ordinary Exo behavior unchanged."""

    path = os.environ.get("EXO_COMPUTATION_STORE")
    if path is None:
        return None
    profile = os.environ.get("EXO_COMPUTATION_RUNTIME_PROFILE")
    if not profile:
        logger.warning(
            "EXO_COMPUTATION_STORE ignored: set "
            "EXO_COMPUTATION_RUNTIME_PROFILE to an "
            "identity for the exact model/tokenizer/MLX/shard semantics"
        )
        return None
    scoped_profile = _scoped_runtime_profile(
        profile,
        model_id,
        device_rank,
        shard_profile,
    )
    profile_store = Path(path) / _digest(scoped_profile.encode())
    try:
        return StoreKVPrefixPersistence(profile_store, scoped_profile)
    except Exception:
        logger.opt(exception=True).warning(
            "computation persistence unavailable; continuing without it"
        )
        return None


def _scoped_runtime_profile(
    profile: str,
    model_id: str,
    device_rank: int,
    shard_profile: str,
) -> str:
    components = [
        profile,
        f"model={model_id}",
        f"rank={device_rank}",
        f"shard={shard_profile}",
        f"exo={_package_version('exo')}",
        f"mlx={_package_version('mlx')}",
        f"mlx-lm={_package_version('mlx-lm')}",
        f"checkpoint-schema={_SCHEMA}",
    ]
    return "\0".join(components)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unpackaged"


def _prefix_id(runtime_profile: str, tokens: mx.array) -> str:
    token_bytes = np.asarray(tokens, dtype="<u4").tobytes(order="C")
    return _prefix_id_from_bytes(runtime_profile, len(tokens), token_bytes)


def _prefix_id_from_bytes(
    runtime_profile: str,
    token_count: int,
    token_bytes: bytes,
) -> str:
    if len(token_bytes) != token_count * 4:
        raise ValueError("token byte length mismatch")
    profile = runtime_profile.encode()
    hasher = hashlib.sha256()
    hasher.update(_PREFIX_DOMAIN)
    hasher.update(len(profile).to_bytes(8, "little"))
    hasher.update(profile)
    hasher.update(token_count.to_bytes(8, "little"))
    hasher.update(token_bytes)
    return hasher.hexdigest()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _is_representation_digest(value: str) -> bool:
    algorithm, separator, digest = value.partition(":")
    return (
        separator == ":"
        and bool(algorithm)
        and all(
            character in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in algorithm
        )
        and _is_lower_hex(digest, 64)
    )


def _content_name(profile_id: str, prefix_id: str) -> str:
    return f"exo/kv/v1/{profile_id}/{prefix_id}"


def _metadata_key(profile_id: str, prefix_id: str) -> str:
    return f"prefix/v1/{profile_id}/{prefix_id}"


def _index_key(profile_id: str) -> str:
    return f"prefix-index/v1/{profile_id}"


def _continuation_key(profile_id: str, continuation_id: str) -> str:
    return f"continuation/v1/{profile_id}/{_digest(continuation_id.encode())}"


def _draft_key(profile_id: str, prefix_id: str) -> str:
    return f"draft/v1/{profile_id}/{prefix_id}"

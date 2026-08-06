"""Optional durable persistence for Exo's MLX prefix cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Condition, Lock, Thread
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import msgspec
import numpy as np
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

from exo.worker.engines.mlx.cache import cache_length
from exo.worker.engines.mlx.cache_persistence import PersistedKVPrefix
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.cache import CacheSnapshot
    from exo.worker.engines.mlx.vision import MediaRegion


_SCHEMA = 2
_PREFIX_DOMAIN = b"exo-mlx-prefix-state-v1\0"
type _PendingCheckpoint = tuple[mx.array, KVCacheType, float]


class _CheckpointMetadata(msgspec.Struct, frozen=True):
    schema: int
    runtime_profile: str
    prefix_id: str
    content_object: str
    projection_digest: str
    token_count: int
    cache_tokens: int
    prefill_tps: float


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
        self._publication = Condition()
        self._pending: _PendingCheckpoint | None = None
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
    ) -> PersistedKVPrefix | None:
        self.last_restore_metrics = RestoreMetrics()
        if media_regions:
            return None
        try:
            lengths = self._read_lengths()
            for token_count in reversed(lengths):
                if token_count <= minimum_tokens or token_count > len(prompt_tokens):
                    continue
                prefix_id = _prefix_id(
                    self._runtime_profile,
                    prompt_tokens[:token_count],
                )
                metadata = self._read_metadata(prefix_id)
                if metadata is None or metadata.token_count != token_count:
                    continue
                projection = self._projection_path(metadata.content_object)
                verification_seconds = 0.0
                reconstruction_seconds = 0.0
                if self._projection_available(projection):
                    started = time.perf_counter()
                    projection_digest = self._store.digest_file(projection)
                    verification_seconds = time.perf_counter() - started
                    if projection_digest != metadata.projection_digest:
                        projection.unlink(missing_ok=True)
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

    def schedule_store(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list["CacheSnapshot"] | None,
        media_regions: list["MediaRegion"],
        prefill_tps: float,
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
            self._pending = (prompt_tokens, cache, prefill_tps)
            self._publication.notify()

    def close(self) -> None:
        with self._publication:
            self._closing = True
            self._publication.notify()
        self._worker.join()

    def _publication_loop(self) -> None:
        while True:
            with self._publication:
                while self._pending is None and not self._closing:
                    self._publication.wait()
                if self._pending is None:
                    return
                prompt_tokens, cache, prefill_tps = self._pending
                self._pending = None
            try:
                self._store_checkpoint(prompt_tokens, cache, prefill_tps)
            except Exception:
                logger.opt(exception=True).warning(
                    "KV checkpoint publication failed; inference result remains valid"
                )

    def _store_checkpoint(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        prefill_tps: float,
    ) -> None:
        token_count = len(prompt_tokens)
        prefix_id = _prefix_id(self._runtime_profile, prompt_tokens)
        source = self._temporary_path("publish", prefix_id)
        started = time.perf_counter()
        save_prompt_cache(
            str(source),
            list(cache),  # pyright: ignore[reportArgumentType]
            {
                "schema": str(_SCHEMA),
                "runtime_profile": self._profile_id,
                "prefix_id": prefix_id,
            },
        )
        serialization_seconds = time.perf_counter() - started
        started = time.perf_counter()
        try:
            content_object, projection_digest = self._store.put_file(
                _content_name(self._profile_id, prefix_id), source
            )
            os.replace(source, self._projection_path(content_object))
        finally:
            source.unlink(missing_ok=True)
        admission_seconds = time.perf_counter() - started
        started = time.perf_counter()
        metadata = _CheckpointMetadata(
            schema=_SCHEMA,
            runtime_profile=self._profile_id,
            prefix_id=prefix_id,
            content_object=content_object,
            projection_digest=projection_digest,
            token_count=token_count,
            cache_tokens=cache_length(cache),
            prefill_tps=prefill_tps,
        )
        self._store.set(
            _metadata_key(self._profile_id, prefix_id),
            msgspec.json.encode(metadata),
        )
        with self._index_lock:
            lengths = self._read_lengths()
            if token_count not in lengths:
                lengths.append(token_count)
                lengths.sort()
                self._store.set(
                    _index_key(self._profile_id),
                    msgspec.json.encode(lengths),
                )
        self.last_publication_metrics = PublicationMetrics(
            mlx_serialization_seconds=serialization_seconds,
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
    profile = runtime_profile.encode()
    hasher = hashlib.sha256()
    hasher.update(_PREFIX_DOMAIN)
    hasher.update(len(profile).to_bytes(8, "little"))
    hasher.update(profile)
    hasher.update(len(tokens).to_bytes(8, "little"))
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

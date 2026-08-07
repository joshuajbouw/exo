# type: ignore
import hashlib
import sys
import types
from pathlib import Path
from threading import Event

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.cache import cache_length
from exo.worker.engines.mlx.computation_persistence import (
    StoreKVPrefixPersistence,
    _prefix_id,
    _scoped_runtime_profile,
)


class _FakeComputationStore:
    files: dict[tuple[str, str], bytes] = {}
    values: dict[tuple[str, str], bytes] = {}
    fail_put = False
    put_started: Event | None = None
    release_put: Event | None = None
    get_file_calls = 0
    digest_file_calls = 0

    def __init__(self, path: Path):
        self.path = str(path)

    def put_file(self, name: str, source: Path) -> tuple[str, str]:
        if self.fail_put:
            raise OSError("injected publication failure")
        if self.put_started is not None:
            self.put_started.set()
        if self.release_put is not None:
            assert self.release_put.wait(timeout=5)
        value = source.read_bytes()
        self.files[(self.path, name)] = value
        digest = hashlib.sha256(value).hexdigest()
        return digest, f"sha256:{digest}"

    def get_file(self, name: str, destination: Path) -> tuple[str, str] | None:
        _FakeComputationStore.get_file_calls += 1
        value = self.files.get((self.path, name))
        if value is None:
            return None
        destination.write_bytes(value)
        digest = hashlib.sha256(value).hexdigest()
        return digest, f"sha256:{digest}"

    def digest_file(self, source: Path) -> str:
        _FakeComputationStore.digest_file_calls += 1
        return f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"

    def get(self, key: str) -> bytes | None:
        return self.values.get((self.path, key))

    def set(self, key: str, value: bytes) -> None:
        self.values[(self.path, key)] = value


def _install_fake_store() -> None:
    module = types.ModuleType("exo_computation_store")
    module.ComputationStore = _FakeComputationStore
    sys.modules[module.__name__] = module


def _reset_fake_store() -> None:
    _FakeComputationStore.files.clear()
    _FakeComputationStore.values.clear()
    _FakeComputationStore.fail_put = False
    _FakeComputationStore.put_started = None
    _FakeComputationStore.release_put = None
    _FakeComputationStore.get_file_calls = 0
    _FakeComputationStore.digest_file_calls = 0


def test_prefix_identity_binds_profile_length_and_tokens():
    tokens = mx.array([1, 2, 3], dtype=mx.uint32)
    identity = _prefix_id("profile-a", tokens)

    assert identity == _prefix_id("profile-a", tokens)
    assert identity != _prefix_id("profile-b", tokens)
    assert identity != _prefix_id("profile-a", mx.array([1, 2]))
    assert identity != _prefix_id("profile-a", mx.array([1, 3, 2]))


def test_runtime_profile_binds_model_rank_and_implementation_versions():
    first = _scoped_runtime_profile("weights-a", "model-a", 0, "layers=0:10")

    assert first == _scoped_runtime_profile("weights-a", "model-a", 0, "layers=0:10")
    assert first != _scoped_runtime_profile("weights-b", "model-a", 0, "layers=0:10")
    assert first != _scoped_runtime_profile("weights-a", "model-b", 0, "layers=0:10")
    assert first != _scoped_runtime_profile("weights-a", "model-a", 1, "layers=0:10")
    assert first != _scoped_runtime_profile("weights-a", "model-a", 0, "layers=10:20")


def test_checkpoint_round_trips_through_storage(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys + 1)
    mx.eval(cache.state)

    persistence._store_checkpoint(prompt, [cache], 321.0)
    restored = persistence.restore_longest(prompt, 0, [])

    assert restored is not None
    assert restored.prefill_tps == 321.0
    assert cache_length(restored.cache) == cache_length([cache])
    assert mx.array_equal(restored.prompt_tokens, prompt)
    assert _FakeComputationStore.get_file_calls == 0

    persistence.close()


def test_continuation_alias_round_trips_exact_frontier(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys + 1)
    mx.eval(cache.state)

    persistence._store_checkpoint(
        prompt,
        [cache],
        1.0,
        continuation_id="resp-a",
    )

    resolved = persistence.resolve_continuation("resp-a")
    assert resolved is not None
    assert mx.array_equal(resolved.tokens, prompt)
    assert resolved.terminal_token == 4
    assert persistence.resolve_continuation("resp-unknown") is None
    persistence.close()


def test_tampered_continuation_alias_fails_closed(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    persistence._store_checkpoint(
        prompt,
        [cache],
        1.0,
        continuation_id="resp-a",
    )
    key = next(
        key
        for path, key in _FakeComputationStore.values
        if key.startswith("continuation/v1/")
    )
    _FakeComputationStore.values[(str(tmp_path / "store"), key)] = b"not-json"

    assert persistence.resolve_continuation("resp-a") is None
    persistence.close()


def test_corrupt_metadata_is_a_cache_miss(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    persistence._store_checkpoint(prompt, [cache], 1.0)

    metadata_key = next(
        key
        for path, key in _FakeComputationStore.values
        if key.startswith("prefix/v1/")
    )
    _FakeComputationStore.values[(str(tmp_path / "store"), metadata_key)] = b"not-json"

    assert persistence.restore_longest(prompt, 0, []) is None
    persistence.close()


def test_corrupt_projection_is_reconstructed_from_verified_storage(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    store_path = tmp_path / "store"
    persistence = StoreKVPrefixPersistence(store_path, "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys + 1)
    mx.eval(cache.state)
    persistence._store_checkpoint(prompt, [cache], 1.0)
    projection = next((store_path / "projections").glob("*.safetensors"))
    projection.write_bytes(b"tampered")

    restored = persistence.restore_longest(prompt, 0, [])

    assert restored is not None
    assert _FakeComputationStore.get_file_calls == 1
    assert cache_length(restored.cache) == cache_length([cache])
    persistence.close()


def test_projection_survives_persistence_reopen(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    store_path = tmp_path / "store"
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    first = StoreKVPrefixPersistence(store_path, "profile-a")
    first._store_checkpoint(prompt, [cache], 1.0)
    first.close()

    reopened = StoreKVPrefixPersistence(store_path, "profile-a")
    restored = reopened.restore_longest(prompt, 0, [])

    assert restored is not None
    assert _FakeComputationStore.get_file_calls == 0
    assert reopened.last_restore_metrics.projection_verification_seconds >= 0
    reopened.close()


def test_unchanged_projection_is_hashed_once_per_process(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    store_path = tmp_path / "store"
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    writer = StoreKVPrefixPersistence(store_path, "profile-a")
    writer._store_checkpoint(prompt, [cache], 1.0)
    writer.close()

    reopened = StoreKVPrefixPersistence(store_path, "profile-a")
    assert reopened.restore_longest(prompt, 0, []) is not None
    assert reopened.restore_longest(prompt, 0, []) is not None

    assert _FakeComputationStore.digest_file_calls == 1
    reopened.close()


def test_projection_symlink_is_never_followed(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    store_path = tmp_path / "store"
    persistence = StoreKVPrefixPersistence(store_path, "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    persistence._store_checkpoint(prompt, [cache], 1.0)
    projection = next((store_path / "projections").glob("*.safetensors"))
    projection.unlink()
    projection.symlink_to(tmp_path / "outside")

    restored = persistence.restore_longest(prompt, 0, [])

    assert restored is not None
    assert not projection.is_symlink()
    assert _FakeComputationStore.get_file_calls == 1
    persistence.close()


def test_background_publication_failure_does_not_escape_close(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    _FakeComputationStore.fail_put = True
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)

    persistence.schedule_store(prompt, [cache], None, [], 1.0)
    persistence.close()

    _FakeComputationStore.fail_put = False


def test_thread_bound_checkpoint_is_serialized_before_worker_handoff(
    tmp_path: Path, monkeypatch
):
    _install_fake_store()
    _reset_fake_store()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    generation_stream = mx.new_stream(mx.gpu)
    with mx.stream(generation_stream):
        keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
        cache.update_and_fetch(keys, keys + 1)

    import exo.worker.engines.mlx.computation_persistence as persistence_module

    caller_thread = __import__("threading").get_ident()
    serialization_threads: list[int] = []
    original = persistence_module.save_prompt_cache

    def recording_save(*args, **kwargs):
        serialization_threads.append(__import__("threading").get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(persistence_module, "save_prompt_cache", recording_save)

    persistence.schedule_store_thread_bound(prompt, [cache], None, [], 1.0)
    assert serialization_threads == [caller_thread]
    persistence.close()

    restored = persistence.restore_longest(prompt, 0, [])
    assert restored is not None


def test_pending_publication_coalesces_to_latest_frontier(tmp_path: Path):
    _install_fake_store()
    _reset_fake_store()
    _FakeComputationStore.put_started = Event()
    _FakeComputationStore.release_put = Event()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)

    persistence.schedule_store(mx.array([1, 2, 3, 4]), [cache], None, [], 1.0)
    assert _FakeComputationStore.put_started.wait(timeout=5)
    persistence.schedule_store(mx.array([1, 2, 3, 4, 5]), [cache], None, [], 1.0)
    persistence.schedule_store(mx.array([1, 2, 3, 4, 5, 6]), [cache], None, [], 1.0)
    _FakeComputationStore.release_put.set()
    persistence.close()

    assert persistence._read_lengths() == [4, 6]
    _FakeComputationStore.put_started = None
    _FakeComputationStore.release_put = None


def test_pending_publication_never_coalesces_different_continuations(
    tmp_path: Path,
):
    _install_fake_store()
    _reset_fake_store()
    _FakeComputationStore.put_started = Event()
    _FakeComputationStore.release_put = Event()
    persistence = StoreKVPrefixPersistence(tmp_path / "store", "profile-a")
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    first = mx.array([1, 2, 3, 4])
    second = mx.array([5, 6, 7, 8])

    persistence.schedule_store_thread_bound(
        first,
        [cache],
        None,
        [],
        1.0,
        continuation_id="resp-a",
    )
    assert _FakeComputationStore.put_started.wait(timeout=5)
    persistence.schedule_store_thread_bound(
        second,
        [cache],
        None,
        [],
        1.0,
        continuation_id="resp-b",
    )
    _FakeComputationStore.release_put.set()
    persistence.close()

    resolved_a = persistence.resolve_continuation("resp-a")
    resolved_b = persistence.resolve_continuation("resp-b")
    assert resolved_a is not None and mx.array_equal(resolved_a.tokens, first)
    assert resolved_b is not None and mx.array_equal(resolved_b.tokens, second)
    _FakeComputationStore.put_started = None
    _FakeComputationStore.release_put = None

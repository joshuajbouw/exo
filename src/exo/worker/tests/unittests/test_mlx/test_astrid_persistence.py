# type: ignore
import hashlib
import sys
import types
from pathlib import Path
from threading import Event

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.astrid_persistence import (
    AstridKVPrefixPersistence,
    _prefix_id,
    _scoped_runtime_profile,
)
from exo.worker.engines.mlx.cache import cache_length


class _FakeAstridStore:
    files: dict[tuple[str, str], bytes] = {}
    values: dict[tuple[str, str], bytes] = {}
    fail_put = False
    put_started: Event | None = None
    release_put: Event | None = None

    def __init__(self, path: Path):
        self.path = str(path)

    def put_file(self, name: str, source: Path) -> str:
        if self.fail_put:
            raise OSError("injected publication failure")
        if self.put_started is not None:
            self.put_started.set()
        if self.release_put is not None:
            assert self.release_put.wait(timeout=5)
        value = source.read_bytes()
        self.files[(self.path, name)] = value
        return hashlib.sha256(value).hexdigest()

    def get_file(self, name: str, destination: Path) -> str | None:
        value = self.files.get((self.path, name))
        if value is None:
            return None
        destination.write_bytes(value)
        return hashlib.sha256(value).hexdigest()

    def get(self, key: str) -> bytes | None:
        return self.values.get((self.path, key))

    def set(self, key: str, value: bytes) -> None:
        self.values[(self.path, key)] = value


def _install_fake_store() -> None:
    module = types.ModuleType("exo_astrid_store")
    module.AstridStore = _FakeAstridStore
    sys.modules[module.__name__] = module


def test_prefix_identity_binds_profile_length_and_tokens():
    tokens = mx.array([1, 2, 3], dtype=mx.uint32)
    identity = _prefix_id("profile-a", tokens)

    assert identity == _prefix_id("profile-a", tokens)
    assert identity != _prefix_id("profile-b", tokens)
    assert identity != _prefix_id("profile-a", mx.array([1, 2]))
    assert identity != _prefix_id("profile-a", mx.array([1, 3, 2]))


def test_runtime_profile_binds_model_rank_and_implementation_versions():
    first = _scoped_runtime_profile("weights-a", "model-a", 0, "layers=0:10")

    assert first == _scoped_runtime_profile(
        "weights-a", "model-a", 0, "layers=0:10"
    )
    assert first != _scoped_runtime_profile(
        "weights-b", "model-a", 0, "layers=0:10"
    )
    assert first != _scoped_runtime_profile(
        "weights-a", "model-b", 0, "layers=0:10"
    )
    assert first != _scoped_runtime_profile(
        "weights-a", "model-a", 1, "layers=0:10"
    )
    assert first != _scoped_runtime_profile(
        "weights-a", "model-a", 0, "layers=10:20"
    )


def test_checkpoint_round_trips_through_storage(tmp_path: Path):
    _install_fake_store()
    _FakeAstridStore.files.clear()
    _FakeAstridStore.values.clear()
    _FakeAstridStore.fail_put = False
    _FakeAstridStore.put_started = None
    _FakeAstridStore.release_put = None
    persistence = AstridKVPrefixPersistence(tmp_path / "store", "profile-a")
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

    persistence.close()


def test_corrupt_metadata_is_a_cache_miss(tmp_path: Path):
    _install_fake_store()
    _FakeAstridStore.files.clear()
    _FakeAstridStore.values.clear()
    _FakeAstridStore.fail_put = False
    _FakeAstridStore.put_started = None
    _FakeAstridStore.release_put = None
    persistence = AstridKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    persistence._store_checkpoint(prompt, [cache], 1.0)

    metadata_key = next(
        key for path, key in _FakeAstridStore.values if key.startswith("prefix/v1/")
    )
    _FakeAstridStore.values[(str(tmp_path / "store"), metadata_key)] = b"not-json"

    assert persistence.restore_longest(prompt, 0, []) is None
    persistence.close()


def test_background_publication_failure_does_not_escape_close(tmp_path: Path):
    _install_fake_store()
    _FakeAstridStore.files.clear()
    _FakeAstridStore.values.clear()
    _FakeAstridStore.fail_put = True
    _FakeAstridStore.put_started = None
    _FakeAstridStore.release_put = None
    persistence = AstridKVPrefixPersistence(tmp_path / "store", "profile-a")
    prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)

    persistence.schedule_store(prompt, [cache], None, [], 1.0)
    persistence.close()

    _FakeAstridStore.fail_put = False


def test_pending_publication_coalesces_to_latest_frontier(tmp_path: Path):
    _install_fake_store()
    _FakeAstridStore.files.clear()
    _FakeAstridStore.values.clear()
    _FakeAstridStore.fail_put = False
    _FakeAstridStore.put_started = Event()
    _FakeAstridStore.release_put = Event()
    persistence = AstridKVPrefixPersistence(tmp_path / "store", "profile-a")
    cache = KVCache()
    keys = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)

    persistence.schedule_store(mx.array([1, 2, 3, 4]), [cache], None, [], 1.0)
    assert _FakeAstridStore.put_started.wait(timeout=5)
    persistence.schedule_store(mx.array([1, 2, 3, 4, 5]), [cache], None, [], 1.0)
    persistence.schedule_store(
        mx.array([1, 2, 3, 4, 5, 6]), [cache], None, [], 1.0
    )
    _FakeAstridStore.release_put.set()
    persistence.close()

    assert persistence._read_lengths() == [4, 6]
    _FakeAstridStore.put_started = None
    _FakeAstridStore.release_put = None

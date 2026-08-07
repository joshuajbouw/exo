# type: ignore
import time
from collections import deque
from typing import cast
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    cache_length,
    encode_prompt,
    fork_kv_cache_for_append,
    get_prefix_length,
    make_kv_cache,
)
from exo.worker.engines.mlx.cache_persistence import PersistedKVPrefix
from exo.worker.engines.mlx.generator.batch_generate import _draft_tokens_are_valid
from exo.worker.engines.mlx.generator.generate import (
    append_only_continuation_tokens,
    mlx_generate,
    prefill,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    _patched_extract_cache,
    _patched_step,
    prepare_for_batch_extension,
)
from exo.worker.engines.mlx.types import ContinuationFrontier, DraftContinuation, Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.tests.unittests.test_mlx.conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    DEFAULT_GPT_OSS_MODEL_ID,
)


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


class _GemmaContinuationTokenizer:
    bos_token = "<bos>"
    has_thinking = False

    @staticmethod
    def encode(value: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in value]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        if tokens == [999]:
            return "<turn|>"
        return "".join(chr(token) for token in tokens)


def test_append_only_continuation_preserves_private_frontier():
    tokenizer = _GemmaContinuationTokenizer()
    task = TextGenerationTaskParams(
        model=ModelId("gemma"),
        input=[InputMessage(role="user", content="new turn")],
    )
    frontier = ContinuationFrontier(mx.array([10, 999]), 999)
    prompt = "<bos><|turn>user\nnew turn<turn|>\n<|turn>model\n"

    continued = append_only_continuation_tokens(
        tokenizer,  # type: ignore[arg-type]
        task,
        prompt,
        frontier,
    )

    assert continued is not None
    expected_suffix = tokenizer.encode("\n" + prompt.removeprefix("<bos>"))
    assert mx.array_equal(
        continued.tokens,
        mx.concatenate([frontier.tokens, mx.array(expected_suffix)]),
    )


def test_append_only_continuation_rejects_ambiguous_request_shape():
    tokenizer = _GemmaContinuationTokenizer()
    task = TextGenerationTaskParams(
        model=ModelId("gemma"),
        input=[InputMessage(role="user", content="new turn")],
        instructions="changed system policy",
    )
    assert (
        append_only_continuation_tokens(
            tokenizer,  # type: ignore[arg-type]
            task,
            "<bos><|turn>user\nnew turn<turn|>",
            ContinuationFrontier(mx.array([10, 999]), 999),
        )
        is None
    )


def test_speculative_draft_rejects_out_of_vocabulary_tokens():
    assert _draft_tokens_are_valid(DraftContinuation((0, 7, 15)), 16)
    assert not _draft_tokens_are_valid(DraftContinuation((0, 16)), 16)
    assert not _draft_tokens_are_valid(DraftContinuation((-1, 2)), 16)
    assert not _draft_tokens_are_valid(DraftContinuation((1, 2)), None)


def test_singleton_generation_cache_transfers_without_extraction_copy():
    entry = KVCache()
    batch = GenerationBatch.__new__(GenerationBatch)
    batch.uids = [7]
    batch.prompt_cache = [entry]

    extracted = _patched_extract_cache(batch, 0)

    assert extracted == [entry]
    assert extracted[0] is entry


def test_append_fork_does_not_mutate_response_frontier():
    keys = mx.arange(3).reshape(1, 1, 3, 1).astype(mx.float32)
    values = (keys + 10).astype(mx.float32)
    global_cache = KVCache()
    global_cache.update_and_fetch(keys, values)
    local_cache = RotatingKVCache(max_size=8)
    local_cache.update_and_fetch(keys, values)
    original_global = mx.array(global_cache.keys)
    original_local = mx.array(local_cache.keys)

    forked = fork_kv_cache_for_append(
        [global_cache, local_cache],
        token_count=3,
        append_count=2,
    )

    assert forked is not None
    appended = mx.array([20, 21]).reshape(1, 1, 2, 1).astype(mx.float32)
    for entry in forked:
        entry.update_and_fetch(appended, appended + 10)
    mx.eval(
        global_cache.keys,
        local_cache.keys,
        forked[0].keys,
        forked[1].keys,
    )
    assert global_cache.offset == 3
    assert local_cache.offset == 3
    assert mx.array_equal(global_cache.keys, original_global)
    assert mx.array_equal(local_cache.keys, original_local)
    assert forked[0].offset == 5
    assert forked[1].offset == 5


def test_append_fork_rejects_single_token_before_it_can_write_shared_arrays():
    cache = KVCache()
    cache.update_and_fetch(
        mx.zeros((1, 1, 1, 1)),
        mx.zeros((1, 1, 1, 1)),
    )

    assert fork_kv_cache_for_append([cache], 1, 1) is None


def test_primed_batch_returns_first_token_without_advancing_cache():
    entry = KVCache()
    entry.offset = 3
    batch = GenerationBatch.__new__(GenerationBatch)
    batch._direct_generation = True
    batch._primed_response_pending = True
    batch._next_tokens = mx.array([7])
    batch._next_logprobs = mx.zeros((1, 8))
    batch.tokens = [[1, 2, 3]]
    batch.prompt_cache = [entry]

    tokens, logprobs = _patched_step(batch)

    assert tokens == [7]
    assert len(logprobs) == 1
    assert batch.tokens == [[1, 2, 3, 7]]
    assert entry.offset == 3


def test_pending_primed_batch_reverts_before_a_concurrent_insert():
    batch = GenerationBatch.__new__(GenerationBatch)
    batch._direct_generation = True
    batch._primed_response_pending = True

    prepare_for_batch_extension(batch)

    assert not batch._direct_generation
    assert batch._primed_response_pending


def test_completed_direct_batch_clears_before_next_insert():
    batch = GenerationBatch.__new__(GenerationBatch)
    batch._direct_generation = True
    batch._primed_response_pending = False
    batch._speculative_queue = deque([(3, mx.zeros(4))])
    batch._speculative_base_cache = []
    batch._speculative_replay_tokens = [2]
    batch.uids = []

    prepare_for_batch_extension(batch)

    assert not batch._direct_generation
    assert batch._speculative_draft is None
    assert not batch._speculative_queue
    assert batch._speculative_base_cache is None


def test_returned_primed_batch_advances_before_a_concurrent_insert():
    class NextTokenModel:
        def __call__(self, tokens, cache):
            del tokens, cache
            return mx.array([[[0.0, 1.0, 2.0, 3.0]]])

    batch = GenerationBatch.__new__(GenerationBatch)
    batch._direct_generation = True
    batch._primed_response_pending = False
    batch._next_tokens = mx.array([2])
    batch._next_logprobs = mx.zeros((1, 4))
    batch.tokens = [[1, 2]]
    batch.prompt_cache = []
    batch.model = NextTokenModel()
    batch.samplers = [None]
    batch.fallback_sampler = lambda scores: mx.argmax(scores, axis=-1)
    batch.uids = [7]

    prepare_for_batch_extension(batch)

    assert not batch._direct_generation
    assert batch._next_tokens.tolist() == [3]
    assert batch.tokens == [[1, 2]]


class _IncrementingCacheModel:
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
        self.calls: list[list[int]] = []

    def __call__(self, tokens, cache):
        token_values = cast(list[int], tokens[0].tolist())
        self.calls.append(token_values)
        steps = len(token_values)
        state = mx.zeros((1, 1, steps, 1))
        cache[0].update_and_fetch(state, state)
        rows = []
        for token in token_values:
            row = [-100.0] * self.vocab_size
            row[token + 1] = 100.0
            rows.append(row)
        return mx.array([rows])


def _speculative_batch(draft: tuple[int, ...]):
    cache = KVCache()
    state = mx.zeros((1, 1, 3, 1))
    cache.update_and_fetch(state, state)
    batch = GenerationBatch.__new__(GenerationBatch)
    batch._direct_generation = True
    batch._primed_response_pending = True
    batch._next_tokens = mx.array([draft[0]])
    batch._next_logprobs = mx.zeros((1, 16))
    batch._speculative_draft = draft
    batch._speculative_cursor = 0
    batch._speculative_queue = deque()
    batch._speculative_base_cache = None
    batch._speculative_replay_tokens = []
    batch._speculative_accepted = 0
    batch.tokens = [[1]]
    batch.prompt_cache = [cache]
    batch.model = _IncrementingCacheModel()
    batch.samplers = [None]
    batch.fallback_sampler = lambda scores: mx.argmax(scores, axis=-1)
    batch.uids = [7]
    batch.max_tokens = [16]
    batch._num_tokens = [0]
    return batch, cache


def test_greedy_draft_is_verified_in_one_block_without_mutating_base_cache():
    batch, base_cache = _speculative_batch((2, 3, 4, 5))

    assert _patched_step(batch)[0] == [2]
    batch._num_tokens[0] += 1
    assert _patched_step(batch)[0] == [3]
    batch._num_tokens[0] += 1
    assert _patched_step(batch)[0] == [4]

    assert batch.model.calls == [[2, 3, 4]]
    assert batch._speculative_accepted == 3
    assert base_cache.offset == 3
    assert cache_length(batch.prompt_cache) == 6


def test_mismatched_draft_is_discarded_before_any_token_is_returned():
    batch, _ = _speculative_batch((2, 3, 9, 10))

    assert _patched_step(batch)[0] == [2]
    batch._num_tokens[0] += 1
    assert _patched_step(batch)[0] == [3]

    assert batch._speculative_draft is None
    assert batch._speculative_accepted == 0
    assert batch.model.calls == [[2, 3, 9], [2]]


def test_verified_ahead_cache_rebases_before_concurrent_insert():
    batch, base_cache = _speculative_batch((2, 3, 4, 5))

    assert _patched_step(batch)[0] == [2]
    batch._num_tokens[0] += 1
    assert _patched_step(batch)[0] == [3]
    batch._num_tokens[0] += 1
    prepare_for_batch_extension(batch)

    assert not batch._direct_generation
    assert batch._next_tokens.tolist() == [4]
    assert cache_length(batch.prompt_cache) == 5
    assert base_cache.offset == 3


def test_continuation_replays_token_trailing_a_lagged_cache():
    prefix_cache = KVPrefixCache(None)
    entry = KVCache()
    entry.offset = 3
    prefix_cache.adopt_kv_cache(
        mx.array([1, 2, 3, 4]),
        [entry],
        continuation_id="response",
    )

    forked, remaining, matched_index, is_exact = prefix_cache.get_kv_cache(
        object(),  # type: ignore[arg-type]
        mx.array([1, 2, 3, 4, 5, 6]),
        prefer_persistent_fork=True,
    )

    assert matched_index is None
    assert not is_exact
    assert cache_length(forked) == 3
    assert remaining.tolist() == [4, 5, 6]
    assert entry.offset == 3


class TestGetPrefixLength:
    def test_identical_arrays(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 7, 8])
        assert get_prefix_length(a, b) == 3

    def test_prompt_longer_than_cached(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 3

    def test_cached_longer_than_prompt(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 3

    def test_single_token_match(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 5, 6])
        assert get_prefix_length(a, b) == 1

    def test_empty_prompt(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 0

    def test_empty_cached(self):
        a = mx.array([1, 2, 3])
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0

    def test_both_empty(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0


class TestKVPrefix:
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a minimal mock tokenizer for tests that don't need real tokenization."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        return tokenizer

    def test_starts_empty(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_empties_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.prompts.append(mx.array([1, 2, 3]))
        cache.caches.append([KVCache()])
        cache.clear()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_on_empty_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.clear()
        assert len(cache.prompts) == 0

    def test_add_schedules_durable_store(self, mock_tokenizer):
        class RecordingPersistence:
            def __init__(self):
                self.stored: list[tuple[int, float]] = []

            def restore_longest(
                self,
                prompt_tokens: mx.array,
                minimum_tokens: int,
                media_regions: list[object],
            ) -> PersistedKVPrefix | None:
                return None

            def schedule_store(
                self,
                prompt_tokens: mx.array,
                cache: list[KVCache],
                snapshots: list[CacheSnapshot] | None,
                media_regions: list[object],
                prefill_tps: float,
            ) -> None:
                self.stored.append((len(prompt_tokens), prefill_tps))

            def close(self) -> None:
                pass

        persistence = RecordingPersistence()
        cache = KVPrefixCache(None, persistence=persistence)  # type: ignore[arg-type]
        cache.add_kv_cache(
            mx.array([1, 2, 3]),
            [KVCache()],
            prefill_tps=123.0,
        )

        assert persistence.stored == [(3, 123.0)]

    def test_adopt_transfers_cache_without_copying(self, mock_tokenizer):
        owned = [KVCache()]
        cache = KVPrefixCache(None)

        index = cache.adopt_kv_cache(mx.array([1, 2, 3]), owned)

        assert index == 0
        assert cache.caches[0] is owned

    def test_adopt_binds_response_to_immutable_frontier(self):
        cache = KVPrefixCache(None)
        frontier = mx.array([1, 2, 3])

        index = cache.adopt_kv_cache(
            frontier,
            [KVCache()],
            continuation_id="resp-a",
        )

        assert cache.is_continuation_entry(index)
        resolved = cache.resolve_continuation("resp-a")
        assert resolved is not None
        assert mx.array_equal(resolved.tokens, frontier)
        assert resolved.terminal_token == 3
        assert cache.resolve_continuation("resp-unknown") is None

        with pytest.raises(ValueError, match="immutable"):
            cache.update_kv_cache(
                index,
                mx.array([1, 2, 3, 4]),
                [KVCache()],
                snapshots=None,
                restore_pos=3,
            )

    def test_adopt_update_transfers_cache_without_copying(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.add_kv_cache(mx.array([1, 2, 3]), [KVCache()])
        completed = [KVCache()]

        cache.adopt_kv_cache_update(
            0,
            mx.array([1, 2, 3, 4]),
            completed,
            snapshots=None,
            restore_pos=3,
        )

        assert cache.caches[0] is completed
        assert mx.array_equal(cache.prompts[0], mx.array([1, 2, 3, 4]))

    def test_adopt_captures_thread_bound_state_before_publication(self):
        class RecordingPersistence:
            def __init__(self):
                self.regular = 0
                self.thread_bound = 0

            def restore_longest(self, *args):
                return None

            def schedule_store(self, *args):
                self.regular += 1

            def schedule_store_thread_bound(self, *args):
                self.thread_bound += 1

            def close(self):
                pass

        persistence = RecordingPersistence()
        cache = KVPrefixCache(None, persistence=persistence)  # type: ignore[arg-type]

        cache.adopt_kv_cache(mx.array([1, 2, 3]), [KVCache()])

        assert persistence.thread_bound == 0
        assert cache.flush_pending_persistence() == 1
        assert persistence.thread_bound == 1
        assert persistence.regular == 0

    def test_restore_populates_memory_without_republishing(self, mock_tokenizer):
        class RestoringPersistence:
            def __init__(self):
                self.restore_calls = 0
                self.store_calls = 0

            def restore_longest(
                self,
                prompt_tokens: mx.array,
                minimum_tokens: int,
                media_regions: list[object],
            ) -> PersistedKVPrefix | None:
                self.restore_calls += 1
                restored = KVCache()
                restored.offset = 3
                return PersistedKVPrefix(
                    prompt_tokens=mx.array([1, 2, 3]),
                    cache=[restored],
                    snapshots=None,
                    media_regions=[],
                    prefill_tps=456.0,
                )

            def schedule_store(
                self,
                prompt_tokens: mx.array,
                cache: list[KVCache],
                snapshots: list[CacheSnapshot] | None,
                media_regions: list[object],
                prefill_tps: float,
            ) -> None:
                self.store_calls += 1

            def close(self) -> None:
                pass

        persistence = RestoringPersistence()
        cache = KVPrefixCache(None, persistence=persistence)  # type: ignore[arg-type]
        _, remaining, matched_index, _ = cache.get_kv_cache(
            object(),  # type: ignore[arg-type]
            mx.array([1, 2, 3, 4]),
        )

        assert persistence.restore_calls == 1
        assert persistence.store_calls == 0
        assert matched_index == 0
        assert len(remaining) == 1
        assert len(cache.prompts) == 1

    def test_restored_nontrimmable_cache_is_usable_at_complete_boundary(
        self, mock_tokenizer
    ):
        class RestoringPersistence:
            def restore_longest(
                self,
                prompt_tokens: mx.array,
                minimum_tokens: int,
                media_regions: list[object],
            ) -> PersistedKVPrefix | None:
                restored = KVCache()
                restored.offset = 3
                return PersistedKVPrefix(
                    prompt_tokens=mx.array([1, 2, 3]),
                    cache=[restored],
                    snapshots=None,
                    media_regions=[],
                    prefill_tps=456.0,
                )

            def schedule_store(
                self,
                prompt_tokens: mx.array,
                cache: list[KVCache],
                snapshots: list[CacheSnapshot] | None,
                media_regions: list[object],
                prefill_tps: float,
            ) -> None:
                pass

            def close(self) -> None:
                pass

        cache = KVPrefixCache(None, persistence=RestoringPersistence())  # type: ignore[arg-type]
        with patch("exo.worker.engines.mlx.cache.has_non_kv_caches", return_value=True):
            restored, remaining, matched_index, _ = cache.get_kv_cache(
                object(),  # type: ignore[arg-type]
                mx.array([1, 2, 3, 4]),
            )

        assert matched_index == 0
        assert cache_length(restored) == 3
        assert len(remaining) == 1

    def test_persistence_failure_does_not_fail_cache_add(self, mock_tokenizer):
        class FailingPersistence:
            def restore_longest(
                self,
                prompt_tokens: mx.array,
                minimum_tokens: int,
                media_regions: list[object],
            ) -> PersistedKVPrefix | None:
                return None

            def schedule_store(
                self,
                prompt_tokens: mx.array,
                cache: list[KVCache],
                snapshots: list[CacheSnapshot] | None,
                media_regions: list[object],
                prefill_tps: float,
            ) -> None:
                raise OSError("store unavailable")

            def close(self) -> None:
                pass

        cache = KVPrefixCache(None, persistence=FailingPersistence())  # type: ignore[arg-type]
        cache.add_kv_cache(mx.array([1, 2, 3]), [KVCache()])

        assert len(cache.prompts) == 1
        assert len(cache.caches) == 1


def _load_gpt_oss() -> tuple[Model, object]:
    from mlx_lm.utils import load_model

    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    model_path = DEFAULT_GPT_OSS_CONFIG.model_path
    model_id = ModelId(DEFAULT_GPT_OSS_MODEL_ID)

    model, _ = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    return cast(Model, model), tokenizer


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_model_exists(),
    reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
)
class TestKVPrefixCacheWithModel:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, tokenizer = _load_gpt_oss()
        return model, tokenizer

    def test_prefill_populates_cache(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello!!")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        # Cache should now hold the prompt tokens minus one
        assert cache_length(cache) == len(tokens) - 1
        # Snapshots should be available for models with non-KV caches
        assert len(snapshots) > 0

    def test_add_and_get_exact_match(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Test exact")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        assert len(kv_prefix_cache.prompts) == 1
        stored_length = cache_length(kv_prefix_cache.caches[0])
        assert stored_length > 0

        # Retrieve with same prompt: exact match
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, tokens
        )
        assert matched_index == 0

        # Exact match returns last token(s) — for models with SSM/rotating caches,
        # snapshot availability constrains how far back we can trim, so remaining
        # may be 1 or 2 tokens depending on the model.
        assert len(remaining_tokens) >= 1
        assert mx.array_equal(remaining_tokens, tokens[-len(remaining_tokens) :])

    def test_add_and_get_prefix_match(self, model_and_tokenizer):
        """get_kv_cache with a longer prompt sharing prefix should return partial match."""
        model, tokenizer = model_and_tokenizer

        short_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi")],
            max_output_tokens=1,
        )
        short_prompt = apply_chat_template(tokenizer, short_task)
        short_tokens = encode_prompt(tokenizer, short_prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            short_tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(short_tokens, cache, snapshots)

        # Query with longer prompt that shares the chat template prefix
        long_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi there, how are you?")],
            max_output_tokens=1,
        )
        long_prompt = apply_chat_template(tokenizer, long_task)
        long_tokens = encode_prompt(tokenizer, long_prompt)

        # The prompts share a prefix (chat template preamble + "Hi")
        expected_prefix = get_prefix_length(long_tokens, short_tokens)
        assert expected_prefix > 0, (
            "Prompts should share a prefix from the chat template"
        )

        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, long_tokens
        )
        assert matched_index == 0

        # remaining_tokens covers from snapshot restore position to end
        assert len(remaining_tokens) >= len(long_tokens) - expected_prefix

    def test_stored_cache_not_mutated_after_get_and_generation(
        self, model_and_tokenizer
    ):
        """Getting a cache and then mutating it (as generation does) must not corrupt stored cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Mutation test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        # Get cache and mutate it (simulating what generation does)
        result_cache, _, matched_index, _ = kv_prefix_cache.get_kv_cache(model, tokens)
        assert matched_index == 0

        # Simulate generation: feed many additional tokens through the cache
        head_dim = result_cache[0].keys.shape[-1]
        num_heads = result_cache[0].keys.shape[1]
        extra_keys = mx.random.normal((1, num_heads, 50, head_dim))
        extra_values = mx.random.normal((1, num_heads, 50, head_dim))
        for layer_cache in result_cache:
            layer_cache.update_and_fetch(extra_keys, extra_values)
        mx.eval([c.keys for c in result_cache])

        # Stored cache must be unchanged
        assert cache_length(kv_prefix_cache.caches[0]) == stored_length

    def test_stored_cache_survives_repeated_get_mutate_cycles(
        self, model_and_tokenizer
    ):
        """Multiple get+mutate cycles (like repeated user requests) must not corrupt cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Repeat test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        for i in range(3):
            result_cache, _, _, _ = kv_prefix_cache.get_kv_cache(model, tokens)

            head_dim = result_cache[0].keys.shape[-1]
            num_heads = result_cache[0].keys.shape[1]
            extra = mx.random.normal((1, num_heads, 30, head_dim))
            for layer_cache in result_cache:
                layer_cache.update_and_fetch(extra, extra)
            mx.eval([c.keys for c in result_cache])

            assert cache_length(kv_prefix_cache.caches[0]) == stored_length, (
                f"Failed on loop {i}"
            )

    def test_mlx_generate_populates_completed_frontier(self, model_and_tokenizer):
        """mlx_generate should retain the prompt plus completed response tokens."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # Consume the entire generator so the cache-saving code after yield runs
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1
        assert len(kv_prefix_cache.caches) == 1
        stored_tokens = kv_prefix_cache.prompts[0]
        assert get_prefix_length(stored_tokens, prompt_tokens) == len(prompt_tokens)
        assert len(stored_tokens) > len(prompt_tokens)
        assert cache_length(kv_prefix_cache.caches[0]) == len(stored_tokens)

    def test_mlx_generate_second_call_gets_prefix_hit(self, model_and_tokenizer):
        """Second mlx_generate call with same prompt should get a prefix hit from stored cache."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Reuse test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1

        # Second call should find a prefix match (the stored cache contains
        # prompt + generated tokens, which shares the prompt prefix)
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        # The completed frontier is longer than the prompt, so the original
        # request remains an exact prefix and can trim back to it.
        assert matched_index == 0
        # Exact match: remaining_tokens is just the last token and the one before
        assert len(remaining_tokens) == 2
        assert mx.array_equal(remaining_tokens, prompt_tokens[-2:])

    def test_mlx_generate_long_prompt_updates_cache_in_place(self, model_and_tokenizer):
        """With a prompt > 1000 tokens, second generation should update the cache entry in-place."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Build a long user message (> 1000 tokens) to exceed _MIN_PREFIX_HIT_TO_UPDATE
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        repeats = (1200 // len(base_tokens)) + 2
        long_content = base_text * repeats

        task1 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content=long_content)],
            max_output_tokens=5,
        )
        prompt1 = apply_chat_template(tokenizer, task1)
        prompt1_tokens = encode_prompt(tokenizer, prompt1)
        assert len(prompt1_tokens) > 1000, (
            "Prompt must exceed _MIN_PREFIX_HIT_TO_UPDATE"
        )

        # First generation populates the cache (must prefill all tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task1,
            prompt=prompt1,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        first_gen_time = time.perf_counter() - t0

        assert len(kv_prefix_cache.prompts) == 1
        first_cache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation: same long prompt + extra content (simulating multi-turn)
        task2 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[
                InputMessage(role="user", content=long_content),
                InputMessage(role="assistant", content="Sure, I can help."),
                InputMessage(role="user", content="Tell me more."),
            ],
            max_output_tokens=5,
        )
        prompt2 = apply_chat_template(tokenizer, task2)
        prompt2_tokens = encode_prompt(tokenizer, prompt2)

        # Verify the prompts share a long prefix
        prefix_len = get_prefix_length(prompt2_tokens, prompt1_tokens)
        assert prefix_len > 1000, "Prompts must share > 1000 token prefix"

        # Second generation should reuse the cached prefix (only prefill new tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task2,
            prompt=prompt2,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        second_gen_time = time.perf_counter() - t0

        # Second generation should be significantly faster due to prefix cache hit - hopefully not flaky
        assert second_gen_time < first_gen_time * 0.5, (
            f"Expected prefix cache speedup: "
            f"first={first_gen_time:.2f}s, second={second_gen_time:.2f}s"
        )

        # With prefix_hit > 1000, should update in-place (not add a second entry)
        assert len(kv_prefix_cache.prompts) == 1
        # Updated cache should be longer (prompt2 + generated > prompt1 + generated)
        updated_cache_length = cache_length(kv_prefix_cache.caches[0])
        assert updated_cache_length > first_cache_length

    def test_mlx_generate_stored_cache_not_mutated(self, model_and_tokenizer):
        """After mlx_generate saves a cache, a second generation must not corrupt the stored copy."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Immutable test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        firstcache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation gets the cache and mutates it during generation
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        # The first stored cache must not have been mutated by the second generation
        assert cache_length(kv_prefix_cache.caches[0]) == firstcache_length

    def test_evicts_lru_entry_under_memory_pressure(self, model_and_tokenizer):
        """Under memory pressure, adding a new cache entry evicts the least recently used one."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Add three cache entries with different prompts
        prompts = ["First entry", "Second entry", "Third entry"]
        for i, content in enumerate(prompts):
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content=content)],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)
            # Stagger _last_used so LRU order is deterministic
            kv_prefix_cache._last_used[i] = float(i)

        assert len(kv_prefix_cache.prompts) == 3

        # Access the third entry to make it most recently used
        kv_prefix_cache._last_used[2] = 100.0
        # Entry 0 (_last_used=0.0) is LRU, entry 1 (_last_used=1.0) is next

        # Simulate memory pressure: return usage above _MEMORY_THRESHOLD (0.9)
        with patch(
            "exo.worker.engines.mlx.cache.get_memory_used_percentage",
            return_value=0.95,
        ):
            # Trigger eviction by adding a new entry
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content="New entry")],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)

        # LRU entries should have been evicted (entries 0, 1, 2 in order of _last_used)
        # Since fake_active stays above threshold after each eviction (we don't change it),
        # all old entries get evicted, leaving only the newly added one
        assert len(kv_prefix_cache.prompts) == 1
        # The surviving entry should be the newly added one
        assert get_prefix_length(kv_prefix_cache.prompts[0], tokens) == len(tokens)

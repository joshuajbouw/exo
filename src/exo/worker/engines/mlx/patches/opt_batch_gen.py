from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol, cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import (
    TokenBuffer,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

from exo.worker.engines.mlx.cache import cache_length, fork_kv_cache_for_append
from exo.worker.engines.mlx.types import KVCacheType, Model

_PRECOMPUTE_TOP_K = 20
# The Gemma 4 MLX path remains byte-identical to serial greedy generation at
# this size in the real-model matrix. Larger single passes crossed a numerical
# divergence boundary and are deliberately not used by the production path.
_SPECULATIVE_BLOCK_SIZE = 128
_ORIGINAL_EXTRACT_CACHE = GenerationBatch.extract_cache


class GenerationStateMachine(Protocol):
    def make_state(self) -> object: ...


def make_primed_generation_batch(
    model: Model,
    uid: int,
    sampled: mx.array,
    logprobs: mx.array,
    prompt_cache: KVCacheType,
    all_tokens: list[int],
    sampler: Callable[[mx.array], mx.array],
    fallback_sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    state_machine: GenerationStateMachine,
    max_tokens: int,
    draft_tokens: tuple[int, ...] | None = None,
) -> GenerationBatch:
    """Build a generation batch whose first token was sampled during prefill."""
    batch = GenerationBatch.__new__(GenerationBatch)
    batch.model = model
    batch.uids = [uid]
    batch.prompt_cache = list(prompt_cache)
    batch.tokens = [list(all_tokens)]
    batch.samplers = [sampler]
    batch.fallback_sampler = fallback_sampler
    batch.logits_processors = [logits_processors]
    batch.state_machines = [state_machine]  # pyright: ignore[reportAttributeAccessIssue]
    batch.max_tokens = [max_tokens]
    batch._current_tokens = None
    batch._current_logprobs = []
    batch._next_tokens = sampled
    batch._next_logprobs = logprobs
    batch._direct_generation = True  # pyright: ignore[reportAttributeAccessIssue]
    batch._primed_response_pending = True  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_draft = draft_tokens  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_cursor = 0  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_queue = deque()  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_base_cache = None  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_replay_tokens = []  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_accepted = 0  # pyright: ignore[reportAttributeAccessIssue]
    batch._token_context = [TokenBuffer(all_tokens)]
    batch._num_tokens = [0]
    batch._matcher_states = [state_machine.make_state()]
    return batch


@dataclass
class BatchTopKLogprobs:
    uids: list[int] = field(default_factory=list)
    indices: mx.array | None = None
    values: mx.array | None = None
    selected: mx.array | None = None
    _uid_to_row: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._uid_to_row = {uid: i for i, uid in enumerate(self.uids)}

    def for_uid(self, uid: int) -> tuple[list[int], list[float], float] | None:
        if self.indices is None or self.values is None or self.selected is None:
            return None
        row = self._uid_to_row.get(uid)
        if row is None:
            return None
        return (
            cast(list[int], self.indices[row].tolist()),
            cast(list[float], self.values[row].tolist()),
            float(self.selected[row].item()),
        )


@dataclass
class _TopKBuffer:
    needs_topk: bool = False
    pending: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)
    ready: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)


def _get_buffer(batch: GenerationBatch) -> _TopKBuffer:
    buf = getattr(batch, "_topk_buffer", None)
    if buf is None:
        buf = _TopKBuffer()
        batch._topk_buffer = buf  # pyright: ignore[reportAttributeAccessIssue]
    return buf


def set_needs_topk(batch: GenerationBatch, needed: bool) -> None:
    _get_buffer(batch).needs_topk = needed


def take_ready_topk(batch: GenerationBatch) -> BatchTopKLogprobs:
    return _get_buffer(batch).ready


def _patched_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    if getattr(self, "_direct_generation", False):
        return _direct_step(self)

    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens
    assert inputs is not None, "_step requires initialized _next_tokens"

    buf = _get_buffer(self)
    buf.ready = buf.pending
    buf.pending = BatchTopKLogprobs()

    logits = self.model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]

    if self.logits_processors is not None and any(self.logits_processors):
        processed_logits: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            for processor in self.logits_processors[e]:
                sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if self.samplers is not None and any(self.samplers):
        all_samples: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_sampler = self.samplers[e] or self.fallback_sampler
            all_samples.append(sample_sampler(logprobs[e : e + 1]))
        sampled = mx.concatenate(all_samples, axis=0)
    else:
        sampled = self.fallback_sampler(logprobs)

    self._next_tokens = sampled
    self._next_logprobs = logprobs

    if buf.needs_topk:
        batch_size = len(self.uids)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        pending_indices = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        pending_values = mx.take_along_axis(logprobs, pending_indices, axis=1)
        sort_order = mx.argsort(-pending_values, axis=1)
        pending_indices = mx.take_along_axis(pending_indices, sort_order, axis=1)
        pending_values = mx.take_along_axis(pending_values, sort_order, axis=1)
        pending_selected = logprobs[mx.arange(batch_size), sampled]
        buf.pending = BatchTopKLogprobs(
            uids=list(self.uids),
            indices=pending_indices,
            values=pending_values,
            selected=pending_selected,
        )
        mx.async_eval(
            self._next_tokens,
            self._next_logprobs,
            pending_indices,
            pending_values,
            pending_selected,
        )
    else:
        mx.async_eval(self._next_tokens, self._next_logprobs)

    current_lp = self._current_logprobs
    if isinstance(current_lp, mx.array):
        mx.eval(inputs, current_lp)
    elif current_lp:
        mx.eval(inputs, *current_lp)
    else:
        mx.eval(inputs)

    token_list = cast(list[int], inputs.tolist())
    for sti, ti in zip(self.tokens, token_list, strict=True):
        sti.append(ti)

    if isinstance(current_lp, mx.array):
        current_lp = list(current_lp)
    return token_list, current_lp


def _direct_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    """Return a primed token before computing, then advance without speculation.

    The prompt cache intentionally trails the returned token by one position.
    A later continuation includes that identified token in its next append, so
    terminal requests do not compute logits that nobody asked for.
    """
    inputs = self._next_tokens
    assert inputs is not None, "direct generation requires a primed token"
    current_logprobs = self._next_logprobs

    if cast(bool, self._primed_response_pending):  # pyright: ignore[reportAttributeAccessIssue]
        self._primed_response_pending = False  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(current_logprobs, mx.array):
            mx.eval(inputs, current_logprobs)
        else:
            mx.eval(inputs, *current_logprobs)
        token_list = cast(list[int], inputs.tolist())
        draft = cast(
            tuple[int, ...] | None,
            getattr(self, "_speculative_draft", None),
        )
        if draft and token_list[0] == draft[0]:
            self._speculative_cursor = 1  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self._speculative_draft = None  # pyright: ignore[reportAttributeAccessIssue]
        for tokens, token in zip(self.tokens, token_list, strict=True):
            tokens.append(token)
        if isinstance(current_logprobs, mx.array):
            current_logprobs = list(current_logprobs)
        return token_list, current_logprobs

    speculative = _next_speculative_token(self, inputs)
    if speculative is not None:
        return speculative

    _clear_speculative_rebase(self)
    sampled, logprobs = _advance_direct_state(self, inputs)
    self._next_tokens = sampled
    self._next_logprobs = logprobs
    token_list = cast(list[int], sampled.tolist())
    for tokens, token in zip(self.tokens, token_list, strict=True):
        tokens.append(token)
    return token_list, list(logprobs)


def _next_speculative_token(
    batch: GenerationBatch,
    inputs: mx.array,
) -> tuple[list[int], list[mx.array]] | None:
    queue = cast(
        deque[tuple[int, mx.array]] | None,
        getattr(batch, "_speculative_queue", None),
    )
    if queue is None:
        return None
    if not queue and not _verify_next_draft_block(batch, inputs):
        return None
    token, logprobs = queue.popleft()
    batch._next_tokens = mx.array([token], dtype=mx.uint32)
    batch._next_logprobs = logprobs[None]
    replay = cast(list[int], batch._speculative_replay_tokens)  # pyright: ignore[reportAttributeAccessIssue]
    replay.append(token)
    batch.tokens[0].append(token)
    return [token], [logprobs]


def _verify_next_draft_block(batch: GenerationBatch, inputs: mx.array) -> bool:
    """Verify one remembered block and publish it only on an exact greedy match."""
    draft = cast(
        tuple[int, ...] | None,
        getattr(batch, "_speculative_draft", None),
    )
    cursor = cast(int, getattr(batch, "_speculative_cursor", 0))
    if draft is None or cursor >= len(draft) or len(batch.uids) != 1:
        return False
    remaining_budget = batch.max_tokens[0] - batch._num_tokens[0]
    block_size = min(
        _SPECULATIVE_BLOCK_SIZE,
        len(draft) - cursor,
        remaining_budget,
    )
    # One token offers no parallelism and cannot use the copy-on-write fork.
    if block_size < 2:
        return False

    current = int(batch.tokens[0][-1])
    expected = draft[cursor : cursor + block_size]
    model_inputs = (current, *expected[:-1])
    base_cache = list(batch.prompt_cache)
    forked = fork_kv_cache_for_append(
        base_cache,
        cache_length(base_cache),
        len(model_inputs),
    )
    if forked is None:
        batch._speculative_draft = None  # pyright: ignore[reportAttributeAccessIssue]
        return False

    logits = batch.model(mx.array([model_inputs], dtype=mx.uint32), cache=forked)
    predicted = mx.argmax(logits, axis=-1)
    mx.eval(
        predicted,
        [entry.state for entry in forked],  # pyright: ignore[reportArgumentType]
    )
    if tuple(cast(list[int], predicted[0].tolist())) != expected:
        batch._speculative_draft = None  # pyright: ignore[reportAttributeAccessIssue]
        return False

    batch._speculative_base_cache = base_cache  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_replay_tokens = [current]  # pyright: ignore[reportAttributeAccessIssue]
    batch.prompt_cache = forked
    queue = cast(deque[tuple[int, mx.array]], batch._speculative_queue)  # pyright: ignore[reportAttributeAccessIssue]
    # This path is gated off when the caller requests logprobs. Avoid retaining
    # one full vocabulary row per accepted token merely to discard it later.
    no_logprobs = mx.array([], dtype=mx.float32)
    queue.extend((token, no_logprobs) for token in expected)
    batch._speculative_cursor = cursor + block_size  # pyright: ignore[reportAttributeAccessIssue]
    accepted = cast(int, batch._speculative_accepted)  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_accepted = accepted + block_size  # pyright: ignore[reportAttributeAccessIssue]
    return True


def _clear_speculative_rebase(batch: GenerationBatch) -> None:
    batch._speculative_base_cache = None  # pyright: ignore[reportAttributeAccessIssue]
    batch._speculative_replay_tokens = []  # pyright: ignore[reportAttributeAccessIssue]


def _advance_direct_state(
    batch: GenerationBatch,
    inputs: mx.array,
) -> tuple[mx.array, mx.array]:
    logits = batch.model(inputs[:, None], cache=batch.prompt_cache)[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if batch.samplers is not None and any(batch.samplers):
        samples = [
            (batch.samplers[index] or batch.fallback_sampler)(
                logprobs[index : index + 1]
            )
            for index in range(len(batch.uids))
        ]
        sampled = mx.concatenate(samples, axis=0)
    else:
        sampled = batch.fallback_sampler(logprobs)

    mx.eval(sampled, logprobs)
    return sampled, logprobs


def prepare_for_batch_extension(batch: GenerationBatch) -> None:
    """Convert a direct singleton back to MLX's mergeable batch semantics."""
    if not getattr(batch, "_direct_generation", False):
        return
    uids = cast(list[int] | None, getattr(batch, "uids", None))
    if uids is not None and not uids:
        batch._direct_generation = False  # pyright: ignore[reportAttributeAccessIssue]
        batch._speculative_draft = None  # pyright: ignore[reportAttributeAccessIssue]
        queue = cast(
            deque[tuple[int, mx.array]] | None,
            getattr(batch, "_speculative_queue", None),
        )
        if queue is not None:
            queue.clear()
        _clear_speculative_rebase(batch)
        return
    if cast(bool, batch._primed_response_pending):  # pyright: ignore[reportAttributeAccessIssue]
        batch._direct_generation = False  # pyright: ignore[reportAttributeAccessIssue]
        return
    base_cache = cast(
        KVCacheType | None,
        getattr(batch, "_speculative_base_cache", None),
    )
    if base_cache is not None:
        _rebase_speculative_batch(batch)
        batch._direct_generation = False  # pyright: ignore[reportAttributeAccessIssue]
        return
    inputs = batch._next_tokens
    assert inputs is not None
    sampled, logprobs = _advance_direct_state(batch, inputs)
    batch._next_tokens = sampled
    batch._next_logprobs = logprobs
    batch._direct_generation = False  # pyright: ignore[reportAttributeAccessIssue]


def _rebase_speculative_batch(batch: GenerationBatch) -> None:
    """Discard ahead-of-output state before merging another request."""
    base_cache = cast(KVCacheType, batch._speculative_base_cache)  # pyright: ignore[reportAttributeAccessIssue]
    replay = cast(list[int], batch._speculative_replay_tokens)  # pyright: ignore[reportAttributeAccessIssue]
    forked = fork_kv_cache_for_append(
        base_cache,
        cache_length(base_cache),
        len(replay),
    )
    if forked is None:
        raise RuntimeError("speculative cache cannot be safely rebased")
    logits = batch.model(mx.array([replay], dtype=mx.uint32), cache=forked)
    last_logits = logits[:, -1, :]
    logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
    queue = cast(deque[tuple[int, mx.array]], batch._speculative_queue)  # pyright: ignore[reportAttributeAccessIssue]
    if queue:
        next_token = mx.array([queue[0][0]], dtype=mx.uint32)
    elif batch.samplers is not None and batch.samplers[0] is not None:
        next_token = batch.samplers[0](logprobs)
    else:
        next_token = batch.fallback_sampler(logprobs)
    mx.eval(
        next_token,
        logprobs,
        [entry.state for entry in forked],  # pyright: ignore[reportArgumentType]
    )
    batch.prompt_cache = forked
    batch._next_tokens = next_token
    batch._next_logprobs = logprobs
    batch._speculative_draft = None  # pyright: ignore[reportAttributeAccessIssue]
    queue.clear()
    _clear_speculative_rebase(batch)


def _patched_extract_cache(self: GenerationBatch, idx: int) -> list[object]:
    """Transfer an unbatched singleton cache without a full contiguous copy."""
    prompt_cache = cast(list[object], self.prompt_cache)
    if (
        len(self.uids) == 1
        and idx == 0
        and all(not hasattr(entry, "extract") for entry in prompt_cache)
    ):
        return list(prompt_cache)
    return cast(list[object], _ORIGINAL_EXTRACT_CACHE(self, idx))


def apply_batch_gen_patch() -> None:
    GenerationBatch._step = _patched_step
    GenerationBatch.extract_cache = _patched_extract_cache

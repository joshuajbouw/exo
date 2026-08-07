import contextlib
import time
import uuid
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    generation_stream,
    maybe_quantize_kv_cache,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.draft_selection import (
    DraftCandidateSelector,
    DraftSelection,
    DraftVerificationOutcome,
    exact_remembered_selection,
)
from exo.worker.engines.mlx.generator.generate import (
    append_only_continuation_tokens,
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
    prefill,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    GenerationStateMachine,
    make_primed_generation_batch,
    prepare_for_batch_extension,
    set_needs_topk,
    take_ready_topk,
)
from exo.worker.engines.mlx.types import DraftContinuation, KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
REMOTE_PREFILL_MIN_TOKENS = 1000


class _BatchGeneratorInternals(Protocol):
    _uid_count: int
    sampler: Callable[[mx.array], mx.array]
    _default_state_machine: GenerationStateMachine
    _generation_batch: object


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


def _draft_tokens_are_valid(
    draft: DraftContinuation,
    vocab_size: object,
) -> bool:
    return (
        isinstance(vocab_size, int)
        and vocab_size > 0
        and all(0 <= token < vocab_size for token in draft.tokens)
    )


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    prefill_tps: float = 0.0
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    media_regions: list[MediaRegion] = field(default_factory=list)
    response_id: str | None = None
    first_gen_token_time: float | None = None
    last_gen_token_time: float | None = None


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None
    draft_selector: DraftCandidateSelector | None = None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _detokenizer_prototype: StreamingDetokenizer = field(init=False, repr=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._mlx_gen = MlxBatchGenerator(
            model=self.model,
            stop_tokens=[[t] for t in eos_ids_from_tokenizer(self.tokenizer)],
            prefill_step_size=4096,
        )
        # SPM/BPE detokenizer construction walks the entire vocabulary.  Its
        # token map is immutable, so build it once and reset cheap shallow
        # clones for individual requests.
        self._detokenizer_prototype = self.tokenizer.detokenizer
        self._step_count = 0

    def _new_detokenizer(self) -> StreamingDetokenizer:
        detokenizer = copy(self._detokenizer_prototype)
        detokenizer.reset()
        return detokenizer

    def _record_draft_outcome(self, outcome: DraftVerificationOutcome) -> None:
        """Return target-verified feedback without coupling generation to policy."""
        if self.draft_selector is None:
            return
        try:
            self.draft_selector.schedule_outcome(outcome)
        except Exception:
            logger.opt(exception=True).warning(
                "Draft feedback failed; generation result remains valid"
            )

    def _select_drafts(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[DraftSelection | None, int]:
        assert self.kv_prefix_cache is not None
        remembered = self.kv_prefix_cache.resolve_draft(prompt_tokens)
        duration_ns = 0
        if self.draft_selector is None:
            selection = exact_remembered_selection(remembered)
        else:
            started = time.perf_counter_ns()
            try:
                selection = self.draft_selector.select(prompt_tokens, remembered)
                duration_ns = time.perf_counter_ns() - started
            except Exception:
                duration_ns = time.perf_counter_ns() - started
                logger.opt(exception=True).warning(
                    "Draft selection failed; using ordinary generation"
                )
                return None, duration_ns

        if selection is None:
            return None, duration_ns
        if not selection.selection_id:
            logger.warning(
                "Speculative selection has no identity; treating it as a miss"
            )
            return None, duration_ns
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        candidates = tuple(
            candidate
            for candidate in selection.candidates
            if _draft_tokens_are_valid(DraftContinuation(candidate.tokens), vocab_size)
        )
        if len(candidates) != len(selection.candidates):
            logger.warning(
                "Speculative selection contains invalid token ids; "
                "dropping invalid candidates"
            )
        if not candidates:
            return None, duration_ns
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if any(not value for value in candidate_ids) or len(set(candidate_ids)) != len(
            candidate_ids
        ):
            logger.warning(
                "Speculative selection has missing or duplicate candidate "
                "identities; treating it as a miss"
            )
            return None, duration_ns
        return DraftSelection(selection.selection_id, candidates), duration_ns

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._mlx_gen._unprocessed_sequences)
            or len(self._mlx_gen._prompt_batch) > 0
            or len(self._mlx_gen._generation_batch) > 0
        )

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
        response_id: str | None = None,
    ) -> int:
        prepare_for_batch_extension(self._mlx_gen._generation_batch)
        all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
        continuation_used = False
        continuation_token_bytes: bytes | None = None
        if task_params.previous_response_id is not None:
            if self.kv_prefix_cache is None:
                raise ValueError("previous_response_id requires a prefix cache")
            frontier = self.kv_prefix_cache.resolve_continuation(
                task_params.previous_response_id
            )
            if frontier is None:
                raise ValueError("previous_response_id is unknown or expired")
            continued = append_only_continuation_tokens(
                self.tokenizer,
                task_params,
                prompt,
                frontier,
            )
            if continued is None:
                raise ValueError(
                    "this model or request shape does not support exact continuation"
                )
            all_prompt_tokens = continued.tokens
            continuation_token_bytes = continued.token_bytes
            continuation_used = True
        if not continuation_used:
            all_prompt_tokens = fix_unmatched_think_end_tokens(
                all_prompt_tokens, self.tokenizer
            )

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []

        if self.vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task_params.images,
                    chat_template_messages=task_params.chat_template_messages,
                    vision_processor=self.vision_processor,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    model_id=task_params.model,
                    task_params=task_params,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )

        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        is_exact_hit = False
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and (
            not is_bench or task_params.use_prefix_cache
        ):
            cache, remaining_tokens, matched_index, is_exact_hit = (
                self.kv_prefix_cache.get_kv_cache(
                    self.model,
                    all_prompt_tokens,
                    media_regions=media_regions,
                    prefer_persistent_fork=continuation_used,
                    prompt_token_bytes=continuation_token_bytes,
                )
            )
            prefix_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
            if prefix_hit_length > 0:
                logger.info(
                    f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                    f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                )
                prompt_tokens = remaining_tokens
        else:
            cache = make_kv_cache(self.model)

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        temperature = (
            task_params.temperature if task_params.temperature is not None else 0.7
        )
        sampler = make_sampler(
            temp=temperature,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            min_p=task_params.min_p if task_params.min_p is not None else 0.05,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )

        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size
                if task_params.repetition_context_size is not None
                else 20,
                presence_penalty=task_params.presence_penalty,
                frequency_penalty=task_params.frequency_penalty,
            )
        )
        if is_bench:
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        max_tokens = task_params.max_output_tokens or MAX_TOKENS
        can_fuse_prefill = (
            self.group is None
            and vision is None
            and not task_params.logprobs
            and not logits_processors
            and prefix_hit_length > 0
            and 0 < len(prompt_tokens) <= 256
            and len(self._mlx_gen._generation_batch) == 0
            and len(self._mlx_gen._prompt_batch) == 0
            and not self._mlx_gen._unprocessed_sequences
        )
        draft_selection: DraftSelection | None = None
        draft_selection_duration_ns = 0
        if (
            can_fuse_prefill
            and self.kv_prefix_cache is not None
            and temperature == 0
            and task_params.stop is None
        ):
            draft_selection, draft_selection_duration_ns = self._select_drafts(
                all_prompt_tokens
            )
        if can_fuse_prefill:
            # Prefix lookup returned an isolated cache fork, so appending does
            # not mutate the retained frontier. Cache types such as
            # RotatingKVCache are unsafe to *trim*, but safe to extend here;
            # generic prefill otherwise copies every layer before the same call.
            started = time.perf_counter()
            if on_prefill_progress is not None:
                on_prefill_progress(0, len(prompt_tokens))
            with mx.stream(generation_stream):
                logits = self.model(prompt_tokens[None], cache=cache)[:, -1, :]
                for processor in logits_processors:
                    logits = processor(all_prompt_tokens, logits)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                sampled = sampler(logprobs)
                maybe_quantize_kv_cache(
                    cache,
                    quantized_kv_start=0,
                    kv_group_size=KV_GROUP_SIZE,
                    kv_bits=KV_BITS,
                )
                mx.eval(
                    sampled,
                    logprobs,
                    [entry.state for entry in cache],  # pyright: ignore[reportArgumentType]
                )
            if on_prefill_progress is not None:
                on_prefill_progress(len(prompt_tokens), len(prompt_tokens))
            elapsed = time.perf_counter() - started
            prefill_tps = len(prompt_tokens) / elapsed if elapsed > 0 else 0.0
            internals = cast(_BatchGeneratorInternals, cast(object, self._mlx_gen))
            uid = internals._uid_count  # pyright: ignore[reportPrivateUsage]
            internals._uid_count += 1  # pyright: ignore[reportPrivateUsage]
            primed = make_primed_generation_batch(
                self.model,
                uid,
                sampled,
                logprobs,
                list(cache),
                cast(list[int], all_prompt_tokens.tolist()),
                sampler,
                internals.sampler,
                logits_processors,
                internals._default_state_machine,  # pyright: ignore[reportPrivateUsage]
                max_tokens,
                draft_selection,
                self._record_draft_outcome,
                draft_selection_duration_ns,
            )
            internals._generation_batch = primed  # pyright: ignore[reportPrivateUsage]
            self._active_tasks[uid] = _EngineTask(
                uid=uid,
                task_params=task_params,
                all_prompt_tokens=all_prompt_tokens,
                prefix_hit_length=prefix_hit_length,
                matched_index=None,
                detokenizer=self._new_detokenizer(),
                on_generation_token=on_generation_token,
                generation_start_time=time.perf_counter(),
                prefill_tps=prefill_tps,
                prefix_cache_hit="partial",
                media_regions=media_regions,
                response_id=response_id,
            )
            return uid

        vision_ctx = (
            patch_embed_tokens(
                self.model,
                vision.embeddings,
                prefix_hit_length,
                len(prompt_tokens) - 1,
                image_token_id=vision.image_token_id,
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        uncached_count = len(prompt_tokens)
        use_remote = (
            uncached_count > REMOTE_PREFILL_MIN_TOKENS
            and task_params.prefill_endpoint is not None
        )

        _prefill_tps: float = 0.0
        _prefill_tokens: int = 0
        cache_snapshots: list[CacheSnapshot] = []
        remote_prefilled = False
        with vision_ctx:
            if use_remote and task_params.prefill_endpoint is not None:
                try:
                    _prefill_tps, _prefill_tokens, cache_snapshots = remote_prefill(
                        prompt_tokens[:-1],
                        cache,
                        on_prefill_progress,
                        endpoint=task_params.prefill_endpoint,
                        request_id=str(uuid.uuid4()),
                        model_id=str(task_params.model),
                        start_pos=prefix_hit_length,
                    )
                    remote_prefilled = True
                except Exception:
                    logger.opt(exception=True).warning(
                        "Remote prefill failed, falling back to local prefill"
                    )

            if not remote_prefilled:
                _prefill_tps, _prefill_tokens, cache_snapshots = prefill(
                    self.model,
                    self.tokenizer,
                    sampler,
                    prompt_tokens[:-1],
                    cache,
                    self.group,
                    on_prefill_progress,
                    distributed_prompt_progress_callback,
                )

        prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
        if matched_index is not None and prefix_hit_length > 0:
            assert self.kv_prefix_cache is not None
            if is_exact_hit:
                prefix_cache_hit = "exact"
                _prefill_tps = self.kv_prefix_cache.prefill_tps[matched_index]
            else:
                prefix_cache_hit = "partial"

        # We need to clamp rotating kv caches to max size so that mlx lm's _merge_caches behaves
        for c in cache:
            if (
                isinstance(c, RotatingKVCache)
                and c.keys is not None
                and c.values is not None
                and c.keys.shape[2] > c.max_size
            ):
                trim_size = c.keys.shape[2] - c.max_size
                c.keys = c._trim(trim_size, c.keys)
                c.values = c._trim(trim_size, c.values)
                c._idx = c.max_size

        if self.kv_prefix_cache is not None and (
            not is_bench or task_params.use_prefix_cache
        ):
            min_prefix_hit_length = max(
                1000, system_prompt_token_count(task_params, self.tokenizer)
            )
            preserve_match = continuation_used or (
                matched_index is not None
                and self.kv_prefix_cache.is_continuation_entry(matched_index)
            )
            matched_index = self._save_prefix_cache(
                all_prompt_tokens,
                list(cache),
                cache_snapshots,
                prefix_hit_length,
                None if preserve_match else matched_index,
                min_prefix_hit_length,
                media_regions,
                prefill_tps=_prefill_tps,
            )

        last_tokens = prompt_tokens[-2:]

        uids = self._mlx_gen.insert(
            prompts=[cast(list[int], last_tokens.tolist())],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        assert len(uids) == 1

        uid = uids[0]

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            detokenizer=self._new_detokenizer(),
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
            prefix_cache_hit=prefix_cache_hit,
            media_regions=media_regions,
            response_id=response_id,
        )

        return uid

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        gb = self._mlx_gen._generation_batch
        set_needs_topk(
            gb,
            any(t.task_params.logprobs for t in self._active_tasks.values()),
        )
        _step_tic = time.perf_counter()
        _, responses = self._mlx_gen.next()
        _next_elapsed = time.perf_counter() - _step_tic
        if not gb.uids:
            # MLX filters completed members but knows nothing about Exo's
            # ahead-of-output speculative state. Release those references now
            # rather than retaining a response-sized cache until the next task.
            prepare_for_batch_extension(gb)

        topk = take_ready_topk(gb)

        results: list[tuple[int, GenerationResponse]] = []

        for response in responses:
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]
            now = time.perf_counter()
            if state.first_gen_token_time is None:
                state.first_gen_token_time = now
            state.last_gen_token_time = now
            if state.on_generation_token is not None:
                state.on_generation_token()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            if state.task_params.bench:
                delta = now - state.first_gen_token_time
                logger.debug(
                    f"[bench] uid={response.uid} tok#{state.completion_tokens} {text!r} t={delta:.4f}s"
                )
            state.generated_text_parts.append(text)
            state.generated_token_ids.append(response.token)
            state.potential_stop_sequence_text += text

            finish_reason: FinishReason | None = cast(
                FinishReason | None, response.finish_reason
            )
            task_params = state.task_params
            stop_sequences = _stop_sequences(task_params)
            max_stop_len = max((len(s) for s in stop_sequences), default=0)

            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in state.potential_stop_sequence_text:
                        stop_index = state.potential_stop_sequence_text.find(stop_seq)
                        text_before_stop = state.potential_stop_sequence_text[
                            :stop_index
                        ]
                        chunk_start = len(state.potential_stop_sequence_text) - len(
                            text
                        )
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        break

            is_done = finish_reason is not None

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs:
                precomputed = topk.for_uid(response.uid)
                precomputed_indices, precomputed_values, precomputed_selected = (
                    precomputed if precomputed is not None else (None, None, None)
                )
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=precomputed_indices,
                        precomputed_values=precomputed_values,
                        precomputed_selected=precomputed_selected,
                    )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if state.completion_tokens > 1:
                    gen_span = state.last_gen_token_time - state.first_gen_token_time
                    generation_tps = (
                        (state.completion_tokens - 1) / gen_span
                        if gen_span > 0
                        else 0.0
                    )
                else:
                    generation_tps = 0.0

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                    prefix_cache_hit=state.prefix_cache_hit,
                )
                total_prompt_tokens = len(state.all_prompt_tokens)
                usage = Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=total_prompt_tokens + state.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.prefix_hit_length
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=0
                    ),
                )

            results.append(
                (
                    response.uid,
                    GenerationResponse(
                        text=text,
                        token=response.token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=usage,
                    ),
                )
            )

            if is_done:
                if self.kv_prefix_cache is not None:
                    self.kv_prefix_cache.remember_draft(
                        state.all_prompt_tokens,
                        state.generated_token_ids,
                    )
                if (
                    response.prompt_cache is not None
                    and self.kv_prefix_cache is not None
                    and (not task_params.bench or task_params.use_prefix_cache)
                ):
                    frontier_tokens = mx.concatenate(
                        [
                            state.all_prompt_tokens,
                            mx.array(state.generated_token_ids),
                        ]
                    )
                    if state.matched_index is None:
                        self.kv_prefix_cache.adopt_kv_cache(
                            frontier_tokens,
                            response.prompt_cache,
                            media_regions=state.media_regions,
                            prefill_tps=state.prefill_tps,
                            continuation_id=state.response_id,
                            continuation_terminal_token=response.token,
                        )
                    else:
                        self.kv_prefix_cache.adopt_kv_cache_update(
                            state.matched_index,
                            frontier_tokens,
                            response.prompt_cache,
                            snapshots=None,
                            restore_pos=len(state.all_prompt_tokens),
                            media_regions=state.media_regions,
                            prefill_tps=state.prefill_tps,
                            continuation_id=state.response_id,
                            continuation_terminal_token=response.token,
                        )
                del self._active_tasks[response.uid]
            elif (
                max_stop_len > 0
                and len(state.potential_stop_sequence_text) > max_stop_len
            ):
                state.potential_stop_sequence_text = state.potential_stop_sequence_text[
                    -max_stop_len:
                ]

        _step_elapsed = time.perf_counter() - _step_tic
        _overhead = _step_elapsed - _next_elapsed
        self._step_count += 1
        if self._step_count % 64 == 0 and responses:
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._mlx_gen.close()
        mx.clear_cache()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
        prefill_tps: float = 0.0,
    ) -> int | None:
        if self.kv_prefix_cache is None:
            return matched_index

        try:
            hit_ratio = (
                prefix_hit_length / len(all_prompt_tokens)
                if len(all_prompt_tokens) > 0
                else 0.0
            )
            if matched_index is not None and (
                prefix_hit_length >= min_prefix_hit_length
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    media_regions=media_regions,
                    prefill_tps=prefill_tps,
                )
                return matched_index
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                    prefill_tps=prefill_tps,
                )
                return len(self.kv_prefix_cache.prompts) - 1
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
            return matched_index

import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .computation_persistence import configured_computation_persistence
from .latent_memory import LatentMemoryResolver, mount_latent_memory
from .types import Model
from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None
    shard_profile: str | None = None
    latent_memory_resolver: LatentMemoryResolver | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)
        shard = bound_instance.bound_shard
        self.shard_profile = (
            f"type={type(shard).__name__};rank={shard.device_rank};"
            f"world={shard.world_size};layers={shard.start_layer}:"
            f"{shard.end_layer}:{shard.n_layers}"
        )

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
        ) = yield from load_mlx_items(bound_instance, self.group)

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        device_rank = 0 if self.group is None else self.group.rank()
        persistence = configured_computation_persistence(
            str(self.model_id),
            device_rank,
            self.shard_profile or "unconnected-single-device",
        )
        kv_prefix_cache = KVPrefixCache(self.group, persistence=persistence)
        if self.latent_memory_resolver is not None:
            logger.info(
                "using SequentialGenerator (invocation-scoped latent memory enabled)"
            )
            latent_memory = mount_latent_memory(
                self.inference_model,
                model_id=str(self.model_id),
                runtime_profile=self.latent_memory_resolver.runtime_profile,
            )
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                latent_memory=latent_memory,
                latent_memory_resolver=self.latent_memory_resolver,
            )
        if os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
        else:
            logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )

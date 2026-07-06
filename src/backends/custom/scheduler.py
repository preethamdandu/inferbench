import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.backends.custom.kv_cache import KVCacheManager


class RequestState(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


@dataclass
class SequenceRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    generated_token_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    kv_block_ids: list[int] = field(default_factory=list)
    future: asyncio.Future[dict[str, Any]] = field(default_factory=asyncio.Future)
    time_to_first_token_ns: int | None = None
    start_time_ns: int = field(default_factory=time.time_ns)

    @property
    def is_finished(self) -> bool:
        if self.state == RequestState.COMPLETED:
            return True
        if len(self.generated_token_ids) >= self.max_tokens:
            return True
        # Assume EOS token is 2
        if len(self.generated_token_ids) > 0 and self.generated_token_ids[-1] == 2:
            return True
        return False


class ContinuousBatchScheduler:
    """Continuous batching scheduler for LLM inference."""

    def __init__(self, kv_cache: KVCacheManager, max_batch_size: int, block_size: int = 16):
        self.kv_cache = kv_cache
        self.max_batch_size = max_batch_size
        self.block_size = block_size

        self.waiting: list[SequenceRequest] = []
        self.running: list[SequenceRequest] = []

    def add_request(self, req: SequenceRequest) -> None:
        self.waiting.append(req)

    def cancel_request(self, req: SequenceRequest, reason: str = "cancelled") -> None:
        """Remove a request and fail its future (e.g. client timeout)."""
        if req in self.waiting:
            self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
        if req.kv_block_ids:
            self.kv_cache.free(req.kv_block_ids)
            req.kv_block_ids = []
        req.state = RequestState.COMPLETED
        if not req.future.done():
            req.future.set_exception(RuntimeError(reason))

    def schedule(self) -> list[SequenceRequest]:
        """Runs one scheduling iteration and returns sequences for the next forward pass."""

        # 1. Remove finished
        still_running: list[SequenceRequest] = []
        for req in self.running:
            if req.is_finished:
                req.state = RequestState.COMPLETED
                self.kv_cache.free(req.kv_block_ids)
                req.kv_block_ids = []
            else:
                still_running.append(req)
        self.running = still_running

        # 2. Admit new requests
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting[0]
            total_tokens = len(req.prompt_token_ids) + req.max_tokens
            required_blocks = (total_tokens + self.block_size - 1) // self.block_size

            if self.kv_cache.can_allocate(required_blocks):
                req.kv_block_ids.extend(self.kv_cache.allocate(required_blocks))
                req.state = RequestState.RUNNING
                self.running.append(self.waiting.pop(0))
            else:
                break

        # 3. Ensure running sequences have space for next generation step
        active_sequences: list[SequenceRequest] = []
        for req in self.running:
            current_capacity = len(req.kv_block_ids) * self.block_size
            current_len = len(req.prompt_token_ids) + len(req.generated_token_ids)

            if current_len >= current_capacity:
                if self.kv_cache.can_allocate(1):
                    req.kv_block_ids.extend(self.kv_cache.allocate(1))
                    active_sequences.append(req)
                else:
                    # Out of KV capacity — fail instead of hanging forever
                    self.cancel_request(req, reason="KV cache exhausted")
            else:
                active_sequences.append(req)

        return active_sequences

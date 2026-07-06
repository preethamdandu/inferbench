import asyncio
import time

import structlog
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.backends.custom.scheduler import (
    ContinuousBatchScheduler,
    SequenceRequest,
    RequestState,
)

logger = structlog.get_logger()


def _cache_to_layer_tuples(pkv: object) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer (key, value) tensors from any transformers cache format.

    transformers >= 5 returns a ``DynamicCache`` (not subscriptable) with a
    ``.layers`` list; older versions return a tuple of (k, v) tuples.
    """
    layers = getattr(pkv, "layers", None)
    if layers is not None:
        return [(layer.keys, layer.values) for layer in layers]
    return list(pkv)  # type: ignore[call-overload]


def _layer_tuples_to_cache(layer_kvs: list[tuple[torch.Tensor, torch.Tensor]]) -> object:
    """Build a model-consumable past_key_values from per-layer (k, v) tensors."""
    try:
        from transformers import DynamicCache
    except ImportError:
        return tuple(layer_kvs)

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(layer_kvs):
        cache.update(k, v, layer_idx)
    return cache


class CustomInferenceEngine:
    """
    Custom inference engine implementing continuous batching.

    IMPORTANT SIMPLIFICATION vs production systems (like vLLM/TGI):
    This engine processes prefill and decode separately. In a production highly-optimized
    system, chunked prefill or Piggybacking allows mixing prefill and decode in the same
    forward pass. Here, we run prefill sequences as a batch, and decode sequences as a
    separate batch. This demonstrates the concepts of separated prefill/decode phases
    and continuous scheduling cleanly without necessitating bespoke CUDA kernels.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        scheduler: ContinuousBatchScheduler,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.running = False
        self._loop_task: asyncio.Task[None] | None = None

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    async def start(self) -> None:
        """Start the generation loop."""
        self.running = True
        self._loop_task = asyncio.create_task(self._generation_loop())
        logger.info("Custom Inference Engine started")

    async def stop(self) -> None:
        """Stop the generation loop."""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Custom Inference Engine stopped")

    async def _generation_loop(self) -> None:
        """Continuously loop: schedule requests, run forward passes, resolve completion."""
        while self.running:
            try:
                batch = self.scheduler.schedule()
                if not batch:
                    await asyncio.sleep(0.01)
                    continue

                prefill_seqs = [req for req in batch if not req.generated_token_ids]
                decode_seqs = [req for req in batch if req.generated_token_ids]

                if prefill_seqs:
                    await asyncio.to_thread(self._run_prefill, prefill_seqs)

                if decode_seqs:
                    await asyncio.to_thread(self._run_decode, decode_seqs)

                for req in batch:
                    if len(req.generated_token_ids) >= req.max_tokens or (
                        req.generated_token_ids
                        and req.generated_token_ids[-1] == self.tokenizer.eos_token_id
                    ):
                        req.state = RequestState.COMPLETED

                        text = self.tokenizer.decode(
                            req.generated_token_ids, skip_special_tokens=True
                        )

                        total_time_ms = (time.time_ns() - req.start_time_ns) / 1e6
                        ttft_ms = (
                            (req.time_to_first_token_ns / 1e6)
                            if req.time_to_first_token_ns
                            else total_time_ms
                        )

                        result = {
                            "text": text,
                            "prompt_tokens": len(req.prompt_token_ids),
                            "completion_tokens": len(req.generated_token_ids),
                            "total_time_ms": total_time_ms,
                            "time_to_first_token_ms": ttft_ms,
                        }
                        if not req.future.done():
                            req.future.set_result(result)
            except Exception:
                logger.exception("generation_loop_error")
                await asyncio.sleep(0.1)

    def _run_prefill(self, seqs: list[SequenceRequest]) -> None:
        """Run prefill phase for new sequences using vectorized left-padding."""
        batch_size = len(seqs)
        max_prompt_len = max(len(req.prompt_token_ids) for req in seqs)

        pad_val = self.tokenizer.pad_token_id
        pad_id: int = pad_val if isinstance(pad_val, int) else 0
        input_ids = torch.full(
            (batch_size, max_prompt_len),
            pad_id,
            device=self.model.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=self.model.device
        )

        for i, req in enumerate(seqs):
            prompt_len = len(req.prompt_token_ids)
            input_ids[i, -prompt_len:] = torch.tensor(
                req.prompt_token_ids, device=self.model.device
            )
            attention_mask[i, -prompt_len:] = 1

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # Unbatch past_key_values and save back to the request objects
        layer_kvs = _cache_to_layer_tuples(outputs.past_key_values)
        num_layers = len(layer_kvs)
        new_pkv_unbatched: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(batch_size)
        ]

        for layer_idx in range(num_layers):
            k_batch, v_batch = layer_kvs[layer_idx]
            for i, req in enumerate(seqs):
                prompt_len = len(req.prompt_token_ids)
                # slice the last `prompt_len` tokens corresponding to the unpadded sequence
                k_seq = k_batch[i : i + 1, :, -prompt_len:, :]
                v_seq = v_batch[i : i + 1, :, -prompt_len:, :]
                new_pkv_unbatched[i].append((k_seq, v_seq))

        for i, req in enumerate(seqs):
            setattr(req, "_hf_past_key_values", tuple(new_pkv_unbatched[i]))

            # Sample next token
            next_token_logits = outputs.logits[i, -1, :]
            next_token = self._sample(next_token_logits, req.temperature)

            req.generated_token_ids.append(int(next_token.item()))
            req.time_to_first_token_ns = time.time_ns() - req.start_time_ns

    def _run_decode(self, seqs: list[SequenceRequest]) -> None:
        """Run decode phase for existing sequences using vectorized left-padded caching."""
        batch_size = len(seqs)
        inputs = torch.tensor(
            [[req.generated_token_ids[-1]] for req in seqs], device=self.model.device
        )

        max_seq_len = max(
            len(req.prompt_token_ids) + len(req.generated_token_ids) - 1 for req in seqs
        )

        batched_pkv = []
        num_layers = len(getattr(seqs[0], "_hf_past_key_values"))

        for layer_idx in range(num_layers):
            layer_keys = []
            layer_vals = []
            for req in seqs:
                req_pkv = getattr(req, "_hf_past_key_values")
                k, v = req_pkv[layer_idx]  # [1, heads, seq_len, dim]
                seq_len = k.shape[2]
                pad_len = max_seq_len - seq_len

                if pad_len > 0:
                    k = torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                    v = torch.nn.functional.pad(v, (0, 0, pad_len, 0))

                layer_keys.append(k)
                layer_vals.append(v)

            batched_k = torch.cat(layer_keys, dim=0)  # [batch, heads, max_seq_len, dim]
            batched_v = torch.cat(layer_vals, dim=0)
            batched_pkv.append((batched_k, batched_v))

        past_key_values = _layer_tuples_to_cache(batched_pkv)

        attention_mask = torch.zeros(
            (batch_size, max_seq_len + 1), dtype=torch.long, device=self.model.device
        )
        for i, req in enumerate(seqs):
            seq_len = len(req.prompt_token_ids) + len(req.generated_token_ids)
            attention_mask[i, -seq_len:] = 1

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        new_pkv_unbatched: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(batch_size)
        ]

        out_layer_kvs = _cache_to_layer_tuples(outputs.past_key_values)
        for layer_idx in range(num_layers):
            k_batch, v_batch = out_layer_kvs[layer_idx]
            for i, req in enumerate(seqs):
                seq_len = len(req.prompt_token_ids) + len(req.generated_token_ids)
                k_seq = k_batch[i : i + 1, :, -seq_len:, :]
                v_seq = v_batch[i : i + 1, :, -seq_len:, :]
                new_pkv_unbatched[i].append((k_seq, v_seq))

        for i, req in enumerate(seqs):
            setattr(req, "_hf_past_key_values", tuple(new_pkv_unbatched[i]))
            next_token_logits = outputs.logits[i, -1, :]
            next_token = self._sample(next_token_logits, req.temperature)
            req.generated_token_ids.append(int(next_token.item()))

    def _sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample next token from logits."""
        if temperature <= 0.0:
            return torch.argmax(logits)

        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)[0]


# Alias for canonical import
GenerationEngine = CustomInferenceEngine

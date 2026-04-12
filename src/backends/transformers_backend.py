import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)

from src.backends.base import GenerateParams, GenerateResult, LLMBackend, TokenChunk

logger = structlog.get_logger()


class TransformersBackend(LLMBackend):
    """Naive Transformers backend running in-process. Performance floor."""

    def __init__(self) -> None:
        self._model_id: str | None = None
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def loaded_model(self) -> str | None:
        return self._model_id

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        if self._model_id == model_id and self.model is not None:
            return

        logger.info("Loading transformers model", model=model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Optionally support bitsandbytes 4-bit if requested
        quantization_config = None
        if kwargs.get("load_in_4bit"):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        def _load() -> PreTrainedModel:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )  # type: ignore

        self.model = await asyncio.to_thread(_load)
        self._model_id = model_id
        logger.info("Model loaded successfully")

    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # type: ignore
        prompt_tokens = inputs.input_ids.shape[1]

        def _sync_generate() -> torch.Tensor:
            with torch.inference_mode():
                assert self.model is not None
                assert self.tokenizer is not None
                return self.model.generate(  # type: ignore
                    **inputs,  # type: ignore
                    max_new_tokens=params.max_tokens,
                    temperature=params.temperature if params.temperature > 0 else None,
                    do_sample=params.temperature > 0,
                    top_p=params.top_p if params.temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        start_time = time.time_ns()
        outputs = await asyncio.to_thread(_sync_generate)
        end_time = time.time_ns()

        generated_ids = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_tokens = len(generated_ids)

        return GenerateResult(
            text=str(text),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_time_ms=(end_time - start_time) / 1e6,
            time_to_first_token_ms=None,
        )

    async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # type: ignore
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def _sync_generate() -> None:
            with torch.inference_mode():
                assert self.model is not None
                assert self.tokenizer is not None
                self.model.generate(  # type: ignore
                    **inputs,  # type: ignore
                    max_new_tokens=params.max_tokens,
                    temperature=params.temperature if params.temperature > 0 else None,
                    do_sample=params.temperature > 0,
                    top_p=params.top_p if params.temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )

        generation_task = asyncio.create_task(asyncio.to_thread(_sync_generate))

        def _get_next_token() -> str | None:
            try:
                # Can't use next(iter(streamer)) easily if it blocks on queue.
                # streamer is effectively an iterator.
                return next(streamer)
            except StopIteration:
                return None

        while not generation_task.done():
            try:
                token = await asyncio.wait_for(asyncio.to_thread(_get_next_token), timeout=0.1)
                if token is not None:
                    yield TokenChunk(token=token, finish_reason=None)
                else:
                    break
            except asyncio.TimeoutError:
                continue

        while True:
            token = await asyncio.to_thread(_get_next_token)
            if token is not None:
                yield TokenChunk(token=token, finish_reason=None)
            else:
                break

        yield TokenChunk(token="", finish_reason="length")

    async def health(self) -> bool:
        return self.model is not None

    async def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_id = None

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import structlog

from src.backends.base import GenerateParams, GenerateResult, LLMBackend, TokenChunk
from src.config import settings

logger = structlog.get_logger()


class VLLMBackend(LLMBackend):
    """vLLM backend client talking to OpenAI-compatible server."""

    def __init__(self) -> None:
        self.url = settings.vllm_url
        self._model = settings.llm_model_id
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def loaded_model(self) -> str | None:
        return self._model

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(base_url=self.url)
        return self._session

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        # vLLM container is started with a specific model by docker-compose
        self._model = model_id
        logger.info("vLLM backend model setting updated", model=model_id)

    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        session = await self._get_session()
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
            "stream": False,
        }

        start_time = time.time_ns()
        async with session.post("/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        end_time = time.time_ns()
        total_time_ms = (end_time - start_time) / 1e6

        return GenerateResult(
            text=str(data["choices"][0]["text"]),
            prompt_tokens=int(data["usage"]["prompt_tokens"]),
            completion_tokens=int(data["usage"]["completion_tokens"]),
            total_time_ms=total_time_ms,
            time_to_first_token_ms=None,
        )

    async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]:
        session = await self._get_session()
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
            "stream": True,
        }

        async with session.post("/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                if line:
                    line_str = line.decode("utf-8").strip()
                    if line_str == "data: [DONE]":
                        break
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            token = choice.get("text", "")
                            finish_reason = choice.get("finish_reason")
                            yield TokenChunk(token=str(token), finish_reason=finish_reason)
                        except json.JSONDecodeError:
                            logger.warning("Failed to decode SSE JSON", data=data_str)

    async def health(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get("/health") as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def unload(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

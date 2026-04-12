from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import structlog

from src.backends.base import GenerateParams, GenerateResult, LLMBackend, TokenChunk
from src.config import settings

logger = structlog.get_logger()


class CustomClientBackend(LLMBackend):
    """Client for the custom continuous batching backend."""

    def __init__(self) -> None:
        self.url = settings.custom_url
        self._model = settings.llm_model_id
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "custom"

    @property
    def loaded_model(self) -> str | None:
        return self._model

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(base_url=self.url)
        return self._session

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        self._model = model_id

    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        session = await self._get_session()
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        async with session.post("/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return GenerateResult(
            text=str(data["choices"][0]["text"]),
            prompt_tokens=int(data["usage"]["prompt_tokens"]),
            completion_tokens=int(data["usage"]["completion_tokens"]),
            total_time_ms=float(data["timing"]["total_ms"]),
            time_to_first_token_ms=float(data["timing"]["time_to_first_token_ms"]),
        )

    async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]:
        res = await self.generate(prompt, params)
        yield TokenChunk(token=res.text, finish_reason="length")

    async def health(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get("/docs") as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def unload(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import structlog

from src.backends.base import GenerateParams, GenerateResult, LLMBackend, TokenChunk
from src.config import settings

logger = structlog.get_logger()


class TGIBackend(LLMBackend):
    """TGI backend client talking to HuggingFace Text Generation Inference."""

    def __init__(self) -> None:
        self.url = settings.tgi_url
        self._model = settings.llm_model_id
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "tgi"

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
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "seed": params.seed,
            },
        }

        start_time = time.time_ns()
        async with session.post("/generate", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        end_time = time.time_ns()
        total_time_ms = (end_time - start_time) / 1e6

        # TGI returns generated_text, and optionally details
        text = data.get("generated_text", "")
        details = data.get("details", {})

        return GenerateResult(
            text=str(text),
            prompt_tokens=int(
                details.get("prefill", [{}])[0].get("tokens", 0) if "prefill" in details else 0
            ),
            completion_tokens=int(details.get("generated_tokens", 0)),
            total_time_ms=total_time_ms,
            time_to_first_token_ms=None,
        )

    async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]:
        session = await self._get_session()
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "seed": params.seed,
            },
        }

        async with session.post("/generate_stream", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                if line:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data:"):
                        data_str = line_str[5:]
                        try:
                            data = json.loads(data_str)
                            token_info = data.get("token", {})
                            token = token_info.get("text", "")

                            details = data.get("details")
                            finish_reason = details.get("finish_reason") if details else None

                            if token or finish_reason:
                                yield TokenChunk(token=str(token), finish_reason=finish_reason)
                        except json.JSONDecodeError:
                            pass

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

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass(frozen=True, slots=True)
class GenerateParams:
    """Generation parameters for LLM requests."""

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42


@dataclass(slots=True)
class TokenChunk:
    """Streamed token chunk."""

    token: str
    finish_reason: str | None = None
    timestamp_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """Generation result for LLM requests."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float
    time_to_first_token_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ImageGenParams:
    """Generation parameters for Image requests."""

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ImageGenResult:
    """Generation result for Image requests."""

    total_time_ms: float
    avg_step_time_ms: float
    image_bytes: bytes
    gpu_peak_memory_mb: float


class LLMBackend(Protocol):
    """Protocol for all LLM inference backends."""

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load the model into the backend."""
        ...

    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """Generate a complete completion synchronously."""
        ...

    def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]:
        """Stream an expanding completion."""
        ...

    async def health(self) -> bool:
        """Check if the backend is healthy."""
        ...

    async def unload(self) -> None:
        """Unload the model from the backend."""
        ...

    @property
    def name(self) -> str:
        """Name of the backend."""
        ...

    @property
    def loaded_model(self) -> str | None:
        """Name of the currently loaded model."""
        ...


class DiffusionBackend(Protocol):
    """Protocol for all Image generation inference backends."""

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load the diffusion model into the backend."""
        ...

    async def generate_image(self, prompt: str, params: ImageGenParams) -> ImageGenResult:
        """Generate an image."""
        ...

    async def health(self) -> bool:
        """Check if the backend is healthy."""
        ...

    async def unload(self) -> None:
        """Unload the model from the backend."""
        ...

    @property
    def name(self) -> str:
        """Name of the backend."""
        ...

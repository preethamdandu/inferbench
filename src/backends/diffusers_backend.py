import asyncio
import io
import time
from typing import Any

import structlog
import torch
from PIL import Image

from src.backends.base import DiffusionBackend, ImageGenParams, ImageGenResult
from src.config import settings

# Graceful import for environments without diffusers installed
try:
    from diffusers import StableDiffusionXLPipeline  # type: ignore

    DIFFUSERS_AVAILABLE = True
except ImportError:
    StableDiffusionXLPipeline = Any  # type: ignore
    DIFFUSERS_AVAILABLE = False


logger = structlog.get_logger()


class DiffusersBackend(DiffusionBackend):
    """Stable Diffusion XL Backend using diffusers."""

    def __init__(self) -> None:
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is not installed. Required for DiffusersBackend.")

        self._model_id: str | None = None
        self.pipeline: StableDiffusionXLPipeline | None = None  # type: ignore
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        return "diffusers"

    @property
    def loaded_model(self) -> str | None:
        return self._model_id

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        if self._model_id == model_id and self.pipeline is not None:
            return

        logger.info("Loading diffusers pipeline", model=model_id)

        def _load() -> Any:
            pipe = StableDiffusionXLPipeline.from_pretrained(  # type: ignore
                model_id, torch_dtype=torch.float16, use_safetensors=True
            )
            pipe = pipe.to(self.device)

            if getattr(settings, "use_torch_compile", False):
                logger.info("Compiling unet with torch.compile")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

            return pipe

        self.pipeline = await asyncio.to_thread(_load)
        self._model_id = model_id
        logger.info("Diffusers model loaded successfully")

    async def generate_image(self, prompt: str, params: ImageGenParams) -> ImageGenResult:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        def _sync_generate() -> tuple[Image.Image, float, float, float]:
            assert self.pipeline is not None
            generator = torch.Generator(device=self.device)
            if params.seed is not None:
                generator.manual_seed(params.seed)

            with torch.inference_mode():
                # Measure GPU peak memory
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                start = time.time_ns()
                out = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=params.num_inference_steps,
                    guidance_scale=params.guidance_scale,
                    height=params.height,
                    width=params.width,
                    generator=generator,
                )
                end = time.time_ns()

                peak_mem_mb = 0.0
                if torch.cuda.is_available():
                    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            total_ms = (end - start) / 1e6
            avg_step_ms = (
                total_ms / params.num_inference_steps if params.num_inference_steps else 0.0
            )

            img_out: list[Image.Image] = getattr(out, "images", [])
            if not img_out:
                raise RuntimeError("Pipeline did not return an image.")

            image = img_out[0]

            return image, total_ms, avg_step_ms, peak_mem_mb

        image, total_time_ms, avg_step_time_ms, peak_mem_mb = await asyncio.to_thread(
            _sync_generate
        )

        # Convert to raw bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        return ImageGenResult(
            image_bytes=image_bytes,
            total_time_ms=total_time_ms,
            avg_step_time_ms=avg_step_time_ms,
            gpu_peak_memory_mb=peak_mem_mb,
        )

    async def health(self) -> bool:
        return self.pipeline is not None

    async def unload(self) -> None:
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_id = None

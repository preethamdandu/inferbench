from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.backends.custom_backend import CustomClientBackend
from src.backends.tgi_backend import TGIBackend
from src.backends.transformers_backend import TransformersBackend
from src.backends.vllm_backend import VLLMBackend

from src.gateway import dependencies
from prometheus_client import make_asgi_app
from src.gateway.routes import router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    dependencies.llm_backends["vllm"] = VLLMBackend()
    dependencies.llm_backends["tgi"] = TGIBackend()
    dependencies.llm_backends["transformers"] = TransformersBackend()
    dependencies.llm_backends["custom"] = CustomClientBackend()

    # Load diffusion backend (Phase 6)
    try:
        from src.backends.diffusers_backend import DiffusersBackend  # type: ignore

        dependencies.diffusion_backend_instance = DiffusersBackend()
    except ImportError:
        pass

    yield

    for backend in dependencies.llm_backends.values():
        await backend.unload()

    if dependencies.diffusion_backend_instance:
        await dependencies.diffusion_backend_instance.unload()


app = FastAPI(lifespan=lifespan)
app.include_router(router)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

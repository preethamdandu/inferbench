from fastapi import HTTPException

from src.backends.base import DiffusionBackend, LLMBackend

llm_backends: dict[str, LLMBackend] = {}
diffusion_backend_instance: DiffusionBackend | None = None


def get_llm_backend(name: str) -> LLMBackend:
    if name not in llm_backends:
        raise HTTPException(status_code=400, detail=f"Backend {name} not configured")
    return llm_backends[name]


def get_diffusion_backend() -> DiffusionBackend:
    if diffusion_backend_instance is None:
        raise HTTPException(status_code=503, detail="Diffusion backend not available")
    return diffusion_backend_instance

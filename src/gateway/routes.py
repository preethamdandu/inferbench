import base64
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.backends.base import GenerateParams, ImageGenParams
from src.gateway.dependencies import get_diffusion_backend, get_llm_backend
from src.gateway.schemas import (
    Choice,
    CompletionResponse,
    Delta,
    GatewayGenerateParams,
    GatewayImageParams,
    ImageResponse,
    StreamChoice,
    StreamResponse,
    Timing,
    Usage,
)
from src.monitoring.prometheus import (
    LATENCY_SECONDS,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
    TPOT_SECONDS,
    TTFT_SECONDS,
)

router = APIRouter()


@router.post("/v1/completions", response_model=CompletionResponse | None)
async def completions(
    req: GatewayGenerateParams,
    backend: str = Query("custom", pattern="^(vllm|tgi|transformers|custom)$"),
) -> CompletionResponse | StreamingResponse:
    llm = get_llm_backend(backend)

    params = GenerateParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )

    request_id = f"cmpl-{uuid.uuid4().hex}"

    if req.stream:

        async def event_generator() -> AsyncGenerator[str, None]:
            async for chunk in llm.stream(req.prompt, params):
                resp = StreamResponse(
                    id=request_id,
                    model=req.model,
                    choices=[
                        StreamChoice(
                            delta=Delta(text=chunk.token), finish_reason=chunk.finish_reason
                        )
                    ],
                )
                yield f"data: {resp.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

            REQUESTS_TOTAL.labels(backend=backend, model=req.model, status="success").inc()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        result = await llm.generate(req.prompt, params)
    except (RuntimeError, ValueError, OSError) as e:
        REQUESTS_TOTAL.labels(backend=backend, model=req.model, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

    REQUESTS_TOTAL.labels(backend=backend, model=req.model, status="success").inc()
    TOKENS_TOTAL.labels(backend=backend, model=req.model, type="prompt").inc(result.prompt_tokens)
    TOKENS_TOTAL.labels(backend=backend, model=req.model, type="completion").inc(
        result.completion_tokens
    )

    LATENCY_SECONDS.labels(backend=backend, model=req.model).observe(result.total_time_ms / 1000.0)

    if result.time_to_first_token_ms is not None:
        TTFT_SECONDS.labels(backend=backend, model=req.model).observe(
            result.time_to_first_token_ms / 1000.0
        )

    tpot = (result.total_time_ms - (result.time_to_first_token_ms or 0.0)) / max(
        result.completion_tokens, 1
    )
    TPOT_SECONDS.labels(backend=backend, model=req.model).observe(tpot / 1000.0)

    return CompletionResponse(
        id=request_id,
        model=req.model,
        choices=[Choice(text=result.text, finish_reason="length")],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        timing=Timing(
            total_ms=result.total_time_ms,
            time_to_first_token_ms=result.time_to_first_token_ms,
        ),
    )


@router.post("/v1/images/generations", response_model=ImageResponse)
async def generate_image(req: GatewayImageParams) -> ImageResponse:
    diff_backend = get_diffusion_backend()

    params = ImageGenParams(
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        height=req.height,
        width=req.width,
        seed=req.seed,
    )

    try:
        result = await diff_backend.generate_image(req.prompt, params)
    except (RuntimeError, ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    b64_img = base64.b64encode(result.image_bytes).decode("utf-8")

    return ImageResponse(
        image_b64=b64_img,
        total_time_ms=result.total_time_ms,
        avg_step_time_ms=result.avg_step_time_ms,
        gpu_peak_memory_mb=result.gpu_peak_memory_mb,
    )


@router.get("/v1/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}

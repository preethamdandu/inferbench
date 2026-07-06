import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.backends.custom.engine import CustomInferenceEngine
from src.backends.custom.kv_cache import KVCacheManager
from src.backends.custom.scheduler import ContinuousBatchScheduler, SequenceRequest
from src.config import Settings

# Global state
engine: CustomInferenceEngine | None = None
tokenizer: PreTrainedTokenizer | None = None
engine_ready: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "length"


class Timing(BaseModel):
    total_ms: float
    time_to_first_token_ms: float


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[Choice]
    usage: Usage
    timing: Timing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine, tokenizer, engine_ready
    engine_ready = False

    cfg = Settings()
    model_id = cfg.llm_model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    # Using PreTrainedTokenizer ensures type consistency for Pyright.
    tok = AutoTokenizer.from_pretrained(model_id)
    if isinstance(tok, PreTrainedTokenizer):
        tokenizer = tok
    else:
        # Fallback if AutoTokenizer returns something else (rare)
        tokenizer = tok  # type: ignore

    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mod = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
    if isinstance(mod, PreTrainedModel):
        model = mod
    else:
        model = mod  # type: ignore

    # Calculate actual num_layers/num_heads (handling different config architectures)
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    num_heads = getattr(model.config, "num_attention_heads", 32)
    hidden_size = getattr(model.config, "hidden_size", 4096)
    head_dim = hidden_size // num_heads

    # Initialize components
    kv_cache = KVCacheManager(
        num_blocks=cfg.kv_cache_blocks,
        num_layers=num_layers,
        block_size=cfg.block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
    )

    scheduler = ContinuousBatchScheduler(
        kv_cache=kv_cache,
        max_batch_size=cfg.max_batch_size,
        block_size=cfg.block_size,
    )

    if tokenizer is not None:
        engine = CustomInferenceEngine(model, tokenizer, scheduler)
        await engine.start()
        engine_ready = True

    yield

    engine_ready = False

    if engine is not None:
        await engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest) -> CompletionResponse:
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    prompt_token_ids = tokenizer.encode(req.prompt, add_special_tokens=True)
    request_id = f"cmpl-{uuid.uuid4().hex}"
    loop = asyncio.get_running_loop()

    seq_req = SequenceRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        future=loop.create_future(),
    )

    engine.scheduler.add_request(seq_req)

    try:
        result = await asyncio.wait_for(seq_req.future, timeout=600.0)
    except TimeoutError:
        engine.scheduler.cancel_request(seq_req, reason="generation timed out")
        raise HTTPException(status_code=504, detail="Generation timed out") from None
    except Exception as e:
        engine.scheduler.cancel_request(seq_req, reason=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    return CompletionResponse(
        id=request_id,
        model=req.model,
        choices=[Choice(text=str(result["text"]), finish_reason="length")],
        usage=Usage(
            prompt_tokens=int(result["prompt_tokens"]),
            completion_tokens=int(result["completion_tokens"]),
            total_tokens=int(result["prompt_tokens"]) + int(result["completion_tokens"]),
        ),
        timing=Timing(
            total_ms=float(result["total_time_ms"]),
            time_to_first_token_ms=float(result["time_to_first_token_ms"]),
        ),
    )


@app.get("/v1/health")
async def health() -> dict[str, str]:
    if not engine_ready or engine is None:
        raise HTTPException(status_code=503, detail="Engine still loading")
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description="InferBench Custom Continuous Batching Server")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--kv-cache-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    # Override settings via environment variables (pydantic-settings reads these)
    os.environ["LLM_MODEL_ID"] = args.model
    os.environ["MAX_BATCH_SIZE"] = str(args.max_batch_size)
    os.environ["KV_CACHE_BLOCKS"] = str(args.kv_cache_blocks)
    os.environ["BLOCK_SIZE"] = str(args.block_size)

    print(f"Starting custom server: model={args.model}, port={args.port}")
    print(f"  max_batch_size={args.max_batch_size}, kv_cache_blocks={args.kv_cache_blocks}, block_size={args.block_size}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

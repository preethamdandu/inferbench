# SKILLS.md — InferBench Technical Implementation Guide

This document is the engineering playbook. It covers what to build, how to build it, the traps to avoid, and the order to build it in. Read fully before writing any code.

---

## Phase 0: Environment & Scaffolding (Day 1, ~2 hours)

### Prerequisites
- NVIDIA GPU with ≥16GB VRAM (24GB preferred for unquantized Llama 3 8B)
- CUDA 12.1+ and nvidia-container-toolkit installed
- Docker + Docker Compose v2
- Python 3.11+ via uv

### Bootstrap

```bash
mkdir inferbench && cd inferbench
uv init --python 3.11
uv add fastapi uvicorn pydantic-settings structlog aiohttp pynvml prometheus-client
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv add transformers accelerate diffusers safetensors
uv add triton                # for kernel experiments
uv add bitsandbytes auto-gptq autoawq  # quantization
uv add --dev pytest pytest-asyncio ruff pyright httpx
```

Create the directory structure from CLAUDE.md. Every `__init__.py` exports the public API of that module — no wildcard imports, no empty init files.

### Docker Compose Skeleton

```yaml
services:
  gateway:
    build: { context: ., dockerfile: docker/Dockerfile.gateway }
    ports: ["8000:8000"]
    environment:
      - VLLM_URL=http://vllm:8001
      - TGI_URL=http://tgi:8002
      - CUSTOM_URL=http://custom:8003

  vllm:
    profiles: ["vllm"]
    image: vllm/vllm-openai:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices: [{ capabilities: [gpu] }]
    volumes:
      - model-cache:/root/.cache/huggingface
    command: >
      --model meta-llama/Meta-Llama-3-8B-Instruct
      --max-model-len 4096
      --gpu-memory-utilization 0.85

  tgi:
    profiles: ["tgi"]
    image: ghcr.io/huggingface/text-generation-inference:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices: [{ capabilities: [gpu] }]
    volumes:
      - model-cache:/root/.cache/huggingface
    environment:
      - MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
      - MAX_INPUT_LENGTH=2048
      - MAX_TOTAL_TOKENS=4096

  custom:
    profiles: ["custom"]
    build: { context: ., dockerfile: docker/Dockerfile.custom }
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices: [{ capabilities: [gpu] }]
    volumes:
      - model-cache:/root/.cache/huggingface
    ports: ["8003:8003"]
    environment:
      - MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
      - MAX_BATCH_SIZE=8
      - KV_CACHE_BLOCKS=256

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards

volumes:
  model-cache:
```

**Critical:** vLLM, TGI, and the custom backend cannot share a GPU simultaneously — they each try to allocate most VRAM. Use Docker Compose profiles and run sequentially: spin up one, benchmark, tear down, spin up the next.

---

## Phase 1: Backend Abstraction Layer (Day 1-2)

### The Protocol

The `Backend` protocol in `base.py` is the spine of the project. Get this right first.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol
import time


@dataclass(frozen=True, slots=True)
class GenerateParams:
    max_tokens: int = 256
    temperature: float = 0.0      # deterministic for benchmarks
    top_p: float = 1.0
    seed: int = 42                # reproducibility


@dataclass(slots=True)
class TokenChunk:
    token: str
    finish_reason: str | None = None
    timestamp_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True)
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float
    time_to_first_token_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ImageGenParams:
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ImageGenResult:
    total_time_ms: float
    avg_step_time_ms: float
    image_bytes: bytes          # PNG encoded
    gpu_peak_memory_mb: float


class LLMBackend(Protocol):
    async def load_model(self, model_id: str, **kwargs) -> None: ...
    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult: ...
    async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]: ...
    async def health(self) -> bool: ...
    async def unload(self) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def loaded_model(self) -> str | None: ...


class DiffusionBackend(Protocol):
    async def load_model(self, model_id: str, **kwargs) -> None: ...
    async def generate_image(self, prompt: str, params: ImageGenParams) -> ImageGenResult: ...
    async def health(self) -> bool: ...
    async def unload(self) -> None: ...

    @property
    def name(self) -> str: ...
```

### vLLM Backend

vLLM exposes an OpenAI-compatible API. Your backend is an HTTP client, not a Python import.

```python
class VLLMBackend:
    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._model: str | None = None

    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        t0 = time.perf_counter_ns()
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "seed": params.seed,
        }
        async with self._session.post(
            f"{self._base_url}/v1/completions", json=payload
        ) as resp:
            data = await resp.json()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        choice = data["choices"][0]
        usage = data["usage"]
        return GenerateResult(
            text=choice["text"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_time_ms=elapsed_ms,
        )
```

**Key detail:** For streaming, use SSE parsing to capture per-token timestamps. This gives you real TTFT and inter-token latency:

```python
async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]:
    payload = {**self._build_payload(prompt, params), "stream": True}
    async with self._session.post(
        f"{self._base_url}/v1/completions", json=payload
    ) as resp:
        async for line in resp.content:
            decoded = line.decode().strip()
            if not decoded.startswith("data: "):
                continue
            if decoded == "data: [DONE]":
                break
            chunk_data = json.loads(decoded[6:])
            token_text = chunk_data["choices"][0].get("text", "")
            if token_text:
                yield TokenChunk(
                    token=token_text,
                    finish_reason=chunk_data["choices"][0].get("finish_reason"),
                )
```

### TGI Backend

TGI has its own API format (`/generate` and `/generate_stream`). Key difference: TGI returns `details.generated_tokens` instead of OpenAI-style `usage`. Adapt accordingly. The stream endpoint uses SSE with `token` objects containing `text` and `special` fields.

### Transformers Backend (In-Process — the naive baseline)

This is the "no optimization" baseline. Load the model directly in the gateway process:

```python
class TransformersBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    async def load_model(self, model_id: str, **kwargs) -> None:
        quantization = kwargs.get("quantization")
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
```

**Trap:** `model.generate()` is synchronous and blocks the event loop. Wrap it:

```python
async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
    result = await asyncio.to_thread(self._sync_generate, prompt, params)
    return result
```

This backend processes one request at a time. No batching, no continuous scheduling. That's the point — it's the floor that every optimized backend must beat.

---

## Phase 2: Custom Continuous Batching Server (Day 2-4) — THE DIFFERENTIATOR

This is the component that separates this project from every other benchmark repo. You are building a minimal but real inference server with continuous batching from first principles.

### Why This Matters

Production inference servers (vLLM, TGI, TensorRT-LLM) all implement some form of continuous batching, inspired by the Orca paper (Yu et al., 2022). The core idea: instead of waiting for an entire batch to finish before starting new requests, you slot new requests into the batch as old ones complete — iteration by iteration. This dramatically improves GPU utilization and throughput.

By building your own, you demonstrate that you understand the mechanics, not just the API.

### Architecture of the Custom Backend

```
┌──────────────────────────────────────────────────────────────┐
│                    CustomInferenceServer                      │
│                                                              │
│  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ FastAPI   │────▶│  Scheduler   │────▶│  GenerationEngine│  │
│  │ (routes)  │     │              │     │                 │  │
│  │           │◀────│ waiting_queue │◀────│ completed_reqs  │  │
│  └──────────┘     │ running_batch │     └────────┬────────┘  │
│                   └──────────────┘              │           │
│                         │                        │           │
│                   ┌─────▼────────┐     ┌────────▼────────┐  │
│                   │  KVCachePool │     │  Model (HF)     │  │
│                   │              │     │  on GPU         │  │
│                   │ block_table  │     │                 │  │
│                   │ free_blocks  │     │                 │  │
│                   └──────────────┘     └─────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Scheduler (`scheduler.py`)

The scheduler manages the lifecycle of requests through two lists:

```python
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class SequenceRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    kv_block_ids: list[int] = field(default_factory=list)
    future: asyncio.Future | None = None  # resolved when generation completes

    @property
    def current_length(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        return (
            len(self.generated_token_ids) >= self.max_tokens
            or self.generated_token_ids[-1] == self._eos_token_id
            if self.generated_token_ids
            else False
        )


class ContinuousBatchScheduler:
    """
    Implements iteration-level scheduling (Orca-style).

    Each iteration:
    1. Remove finished sequences from the running batch
    2. Free their KV-cache blocks
    3. Fill empty slots with waiting requests (if KV-cache has space)
    4. Return the current batch for one forward pass
    """

    def __init__(self, max_batch_size: int, kv_cache: KVCachePool) -> None:
        self._max_batch_size = max_batch_size
        self._kv_cache = kv_cache
        self._waiting: list[SequenceRequest] = []
        self._running: list[SequenceRequest] = []

    def add_request(self, request: SequenceRequest) -> None:
        request.state = RequestState.WAITING
        self._waiting.append(request)

    def schedule(self) -> list[SequenceRequest]:
        """Called once per iteration. Returns the batch for the next forward pass."""

        # Step 1: Remove finished sequences
        finished = [r for r in self._running if r.is_finished]
        for req in finished:
            req.state = RequestState.COMPLETED
            self._kv_cache.free(req.kv_block_ids)
            if req.future and not req.future.done():
                req.future.set_result(req)
        self._running = [r for r in self._running if not r.is_finished]

        # Step 2: Admit new requests from waiting queue
        while self._waiting and len(self._running) < self._max_batch_size:
            candidate = self._waiting[0]
            # Check if KV-cache can fit this sequence
            blocks_needed = self._estimate_blocks(candidate)
            if not self._kv_cache.can_allocate(blocks_needed):
                break  # no space — stop admitting
            allocated = self._kv_cache.allocate(blocks_needed)
            candidate.kv_block_ids = allocated
            candidate.state = RequestState.RUNNING
            self._running.append(candidate)
            self._waiting.pop(0)

        return self._running

    def _estimate_blocks(self, req: SequenceRequest) -> int:
        """Estimate KV-cache blocks needed for the full sequence."""
        total_tokens = len(req.prompt_token_ids) + req.max_tokens
        return (total_tokens + self._kv_cache.block_size - 1) // self._kv_cache.block_size
```

### KV-Cache Pool (`kv_cache.py`)

A simplified block-based KV-cache manager. Production systems (vLLM) use PagedAttention with virtual memory-style block tables. Ours is simpler: pre-allocate a fixed number of blocks, track free/used.

```python
@dataclass
class KVCachePool:
    """
    Fixed-size pool of KV-cache blocks.

    Each block stores KV tensors for `block_size` tokens across all layers.
    This is a simplified version — no PagedAttention, no copy-on-write.
    Real systems (vLLM) are more memory-efficient but this captures the
    core scheduling constraint: you can't run a sequence if you can't
    store its KV-cache.
    """

    num_blocks: int
    block_size: int         # tokens per block (e.g., 16)
    num_layers: int
    num_heads: int
    head_dim: int
    device: torch.device

    def __post_init__(self) -> None:
        # Pre-allocate on GPU
        # Shape: [num_blocks, 2, num_layers, block_size, num_heads, head_dim]
        # The "2" is for K and V
        self.cache = torch.zeros(
            self.num_blocks, 2, self.num_layers,
            self.block_size, self.num_heads, self.head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self._free_blocks: list[int] = list(range(self.num_blocks))
        self._used_blocks: set[int] = set()

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self._free_blocks) >= num_blocks

    def allocate(self, num_blocks: int) -> list[int]:
        if not self.can_allocate(num_blocks):
            raise RuntimeError(f"Cannot allocate {num_blocks} blocks, only {len(self._free_blocks)} free")
        allocated = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        self._used_blocks.update(allocated)
        return allocated

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self._used_blocks.discard(bid)
            self._free_blocks.append(bid)

    @property
    def utilization(self) -> float:
        return len(self._used_blocks) / self.num_blocks
```

### Generation Engine (`engine.py`)

The engine runs the actual forward passes. Each iteration:
1. Scheduler returns the current batch
2. Engine runs one forward pass for all sequences in the batch
3. Samples next token for each sequence
4. Appends tokens, loops

```python
class GenerationEngine:
    """
    Runs the generation loop with continuous batching.

    Key simplification vs vLLM:
    - We use HuggingFace model.forward() directly, not custom CUDA kernels
    - KV-cache is stored in our pool but we re-compute attention each step
      (a real system would use the cached KV to avoid recomputation)
    - No speculative decoding, no prefix caching, no chunked prefill

    These simplifications mean we'll be ~3-5× slower than vLLM.
    That's expected and the gap itself is the analysis.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        scheduler: ContinuousBatchScheduler,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._scheduler = scheduler
        self._running = False

    async def start(self) -> None:
        """Main generation loop — runs continuously."""
        self._running = True
        while self._running:
            batch = self._scheduler.schedule()
            if not batch:
                await asyncio.sleep(0.001)  # no work — yield
                continue

            await asyncio.to_thread(self._step, batch)

    def _step(self, batch: list[SequenceRequest]) -> None:
        """One forward pass for the entire batch."""

        # Build batched input
        # Separate prefill (first forward for a sequence) from decode (subsequent)
        prefill_reqs = [r for r in batch if len(r.generated_token_ids) == 0]
        decode_reqs = [r for r in batch if len(r.generated_token_ids) > 0]

        # Process prefill requests (full prompt)
        for req in prefill_reqs:
            input_ids = torch.tensor(
                [req.prompt_token_ids], device=self._model.device
            )
            with torch.no_grad():
                outputs = self._model(input_ids, use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = self._sample(logits, temperature=0.0)
            req.generated_token_ids.append(next_token)
            # Store the past_key_values for this request
            req._past_kv = outputs.past_key_values

        # Process decode requests (one token at a time, using cached KV)
        for req in decode_reqs:
            input_ids = torch.tensor(
                [[req.generated_token_ids[-1]]], device=self._model.device
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids,
                    past_key_values=req._past_kv,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            next_token = self._sample(logits, temperature=0.0)
            req.generated_token_ids.append(next_token)
            req._past_kv = outputs.past_key_values

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature == 0:
            return logits.argmax(dim=-1).item()
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
```

**Critical design note:** This implementation processes prefill and decode separately within each step. A production system batches them together with attention masking. This is the kind of trade-off to document in `custom/README.md` — you chose correctness and clarity over throughput.

### Server (`server.py`)

```python
app = FastAPI(title="InferBench Custom Server")

@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    seq = SequenceRequest(
        request_id=str(uuid4()),
        prompt_token_ids=tokenizer.encode(request.prompt),
        max_tokens=request.max_tokens,
        future=asyncio.get_event_loop().create_future(),
    )
    scheduler.add_request(seq)
    completed = await seq.future  # blocks until generation finishes
    return CompletionResponse(
        text=tokenizer.decode(completed.generated_token_ids),
        prompt_tokens=len(completed.prompt_token_ids),
        completion_tokens=len(completed.generated_token_ids),
    )
```

### `custom/README.md` — The Design Document

This file is as important as the code. It should cover:

1. **What this is:** A minimal continuous batching inference server built to understand how production servers like vLLM work.
2. **How it works:** Request lifecycle from API call → scheduler → generation → response.
3. **What we implement:** Continuous batching (iteration-level scheduling), KV-cache pooling, prefill/decode separation.
4. **What we deliberately skip and why:**
   - PagedAttention (vLLM's key innovation — requires custom CUDA kernels)
   - Chunked prefill (mixing prefill and decode in same batch)
   - Speculative decoding
   - Tensor parallelism
   - CUDA graph capture
   - Prefix caching
5. **Performance analysis:** Why vLLM is ~3× faster — quantify each factor.
6. **Papers referenced:** Orca (Yu et al., 2022), PagedAttention (Kwon et al., 2023).

---

## Phase 3: FastAPI Gateway (Day 4-5)

### Design Principles

1. **OpenAI-compatible API.** Use the `/v1/completions` schema for LLMs and `/v1/images/generations` for Diffusers.
2. **Backend as a query parameter.** `POST /v1/completions?backend=vllm`
3. **Health aggregation.** `GET /v1/health` returns status of all backends.

```python
# routes.py
@router.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    backend: BackendName = Query(...),
    registry: BackendRegistry = Depends(get_registry),
) -> CompletionResponse:
    engine = registry.get(backend)
    if not engine:
        raise BackendUnavailableError(backend)

    result = await engine.generate(request.prompt, request.to_params())
    return CompletionResponse.from_result(result, backend=backend, model=engine.loaded_model)


@router.post("/v1/images/generations")
async def image_generation(
    request: ImageGenRequest,
    registry: BackendRegistry = Depends(get_registry),
) -> ImageGenResponse:
    engine = registry.get_diffusion()
    result = await engine.generate_image(request.prompt, request.to_params())
    return ImageGenResponse.from_result(result)
```

### Schemas (Pydantic)

```python
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    stream: bool = False

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    backend: str
    choices: list[Choice]
    usage: Usage
    timing: Timing

class Timing(BaseModel):
    total_ms: float
    time_to_first_token_ms: float | None = None
```

---

## Phase 4: Benchmark Runner (Day 5-6) — THE CORE

### Load Generator

```python
@dataclass
class BenchmarkConfig:
    backend: str
    model: str
    prompts: list[str]
    concurrency: int = 8
    duration_seconds: int = 60
    warmup_requests: int = 5
    quantization: str | None = None


class BenchmarkRunner:
    async def run(self, config: BenchmarkConfig) -> BenchmarkReport:
        # 1. Warmup
        await self._warmup(config)

        # 2. Run concurrent workers under time limit
        semaphore = asyncio.Semaphore(config.concurrency)
        results: list[RequestMetric] = []
        deadline = time.monotonic() + config.duration_seconds

        async def worker():
            while time.monotonic() < deadline:
                async with semaphore:
                    prompt = random.choice(config.prompts)
                    metric = await self._send_request(prompt, config)
                    results.append(metric)

        async with asyncio.TaskGroup() as tg:
            for _ in range(config.concurrency * 2):
                tg.create_task(worker())

        # 3. Compute statistics
        return self._compute_report(results, config)
```

### Metrics Computation

```python
@dataclass(frozen=True, slots=True)
class RequestMetric:
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float
    ttft_ms: float | None
    error: str | None = None
    timestamp: float = field(default_factory=time.monotonic)


def _compute_report(self, metrics: list[RequestMetric], config: BenchmarkConfig) -> BenchmarkReport:
    successful = [m for m in metrics if m.error is None]
    latencies = sorted(m.total_time_ms for m in successful)
    ttfts = sorted(m.ttft_ms for m in successful if m.ttft_ms is not None)

    total_tokens = sum(m.completion_tokens for m in successful)
    wall_time_s = max(
        successful[-1].timestamp - successful[0].timestamp, 0.001
    ) if len(successful) > 1 else 1.0

    return BenchmarkReport(
        backend=config.backend,
        model=config.model,
        quantization=config.quantization,
        concurrency=config.concurrency,
        total_requests=len(metrics),
        successful_requests=len(successful),
        failed_requests=len(metrics) - len(successful),
        throughput_tokens_per_sec=total_tokens / wall_time_s,
        throughput_requests_per_sec=len(successful) / wall_time_s,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p90_ms=_percentile(latencies, 90),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        latency_max_ms=max(latencies) if latencies else 0,
        ttft_p50_ms=_percentile(ttfts, 50),
        ttft_p95_ms=_percentile(ttfts, 95),
        ttft_p99_ms=_percentile(ttfts, 99),
        gpu_peak_memory_mb=self._gpu_peak_mb,
        gpu_avg_utilization_pct=self._gpu_avg_util,
    )


def _percentile(sorted_vals: list[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
```

### Prompt Dataset

Use a standardized prompt set for fair comparison. Create `prompts/`:

- `short.jsonl` — 50 prompts, 10-50 tokens input, 64 max output
- `medium.jsonl` — 50 prompts, 100-500 tokens input, 256 max output
- `long.jsonl` — 50 prompts, 1000-2000 tokens input, 512 max output
- `mixed.jsonl` — 50 prompts, variable lengths

Source from ShareGPT or LMSYS-Chat-1M dataset (public). Store as `{"prompt": "...", "max_tokens": N}` per line.

**Trap:** Don't use trivially short prompts like "Hello". Prefill time matters — short prompts hide backend overhead and make everything look similar.

---

## Phase 5: Diffusers Benchmark (Day 6-7)

### Why Include This

The DeepInfra JD explicitly mentions "Transformers and Diffusers." Most candidates only touch text models. Adding image generation benchmarking shows you understand both modalities.

### Diffusers Backend Implementation

```python
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import torch
import io


class DiffusersBackend:
    def __init__(self) -> None:
        self._pipe: StableDiffusionXLPipeline | None = None
        self._model_id: str | None = None

    async def load_model(self, model_id: str, **kwargs) -> None:
        dtype = kwargs.get("dtype", torch.float16)

        def _load():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16",
            )
            pipe.to("cuda")

            # Optimizations to benchmark
            if kwargs.get("enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if kwargs.get("enable_vae_tiling"):
                pipe.enable_vae_tiling()
            if kwargs.get("compile", False):
                pipe.unet = torch.compile(
                    pipe.unet, mode="reduce-overhead", fullgraph=True
                )

            return pipe

        self._pipe = await asyncio.to_thread(_load)
        self._model_id = model_id

    async def generate_image(
        self, prompt: str, params: ImageGenParams
    ) -> ImageGenResult:
        generator = torch.Generator(device="cuda").manual_seed(params.seed)

        # Track per-step timing with a callback
        step_times: list[float] = []
        last_step_time = [time.perf_counter()]

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            now = time.perf_counter()
            step_times.append((now - last_step_time[0]) * 1000)
            last_step_time[0] = now
            return callback_kwargs

        t0 = time.perf_counter()

        def _generate():
            return self._pipe(
                prompt=prompt,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                height=params.height,
                width=params.width,
                generator=generator,
                callback_on_step_end=step_callback,
            )

        result = await asyncio.to_thread(_generate)

        total_ms = (time.perf_counter() - t0) * 1000

        # Encode to PNG bytes
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")

        # GPU memory snapshot
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()

        return ImageGenResult(
            total_time_ms=total_ms,
            avg_step_time_ms=sum(step_times) / len(step_times) if step_times else 0,
            image_bytes=buf.getvalue(),
            gpu_peak_memory_mb=mem_mb,
        )
```

### Diffusion Benchmark Runner

```python
class DiffusionBenchmarkRunner:
    """Simpler than LLM runner — image gen is typically not concurrent."""

    async def run(self, config: DiffusionBenchConfig) -> DiffusionReport:
        backend = DiffusersBackend()
        await backend.load_model(config.model, **config.load_kwargs)

        results: list[ImageGenMetric] = []

        for i, prompt in enumerate(config.prompts):
            if i < config.warmup_count:
                await backend.generate_image(prompt, config.params)
                continue

            result = await backend.generate_image(prompt, config.params)
            results.append(ImageGenMetric(
                total_time_ms=result.total_time_ms,
                avg_step_time_ms=result.avg_step_time_ms,
                gpu_peak_memory_mb=result.gpu_peak_memory_mb,
            ))

        return self._compile_report(results, config)
```

### What to Benchmark

Run these configurations and put results in a table:

| Config                    | Key Question                                    |
|---------------------------|------------------------------------------------|
| SDXL FP16                 | Baseline speed and VRAM                         |
| SDXL FP16 + attention_slicing | Does slicing help VRAM without killing speed? |
| SDXL FP16 + VAE tiling    | Effect on high-res generation                   |
| SDXL FP16 + torch.compile | Compilation overhead vs steady-state speedup    |
| SDXL FP32                 | Precision vs speed/memory trade-off             |

---

## Phase 6: Triton Kernel Experiments (Day 7-8)

### Why Include This

The JD lists "Exposure to C++, CUDA, or AI inference" as a bonus. A Triton kernel is the most approachable way to demonstrate GPU programming. Triton compiles Python-like code to PTX — you get GPU kernel authoring without writing raw CUDA.

### Fused Softmax Kernel

The canonical Triton tutorial kernel. What makes yours better: you benchmark it properly and include roofline analysis.

```python
import triton
import triton.language as tl
import torch


@triton.jit
def fused_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax: load row, compute max, subtract, exp, sum, divide
    all in one kernel — no intermediate global memory writes.

    Why fusion matters:
    - Naive PyTorch softmax does 3 global memory round-trips:
      1. Read input, write max-reduced
      2. Read input + max, write exp(x - max)
      3. Read exp values, write exp/sum
    - Fused version does 1 read + 1 write total.
    - At memory-bandwidth-bound sizes, this is up to 3× faster.
    """
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load entire row into SRAM
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write back
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Wrapper to call the Triton kernel."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Triton constraint: BLOCK_SIZE must be ≤ max threads per block
    # For softmax, each row must fit in one block's shared memory
    assert BLOCK_SIZE <= 65536, f"Row too wide: {n_cols}"

    output = torch.empty_like(x)
    fused_softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

### Fused GELU Kernel (Bonus)

A second kernel shows range. GELU is used in every Transformer MLP block.

```python
@triton.jit
def fused_gelu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Single read, single write, all math in registers.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    # GELU approximation (same as PyTorch's default)
    cdf = 0.5 * (1.0 + tl.libdevice.tanh(
        0.7978845608 * (x + 0.044715 * x * x * x)
    ))
    output = x * cdf

    tl.store(output_ptr + offsets, output, mask=mask)
```

### Kernel Benchmarking (`bench_kernels.py`)

```python
def benchmark_softmax() -> list[KernelBenchResult]:
    """Benchmark across a range of input sizes."""
    results = []

    for n_rows in [128, 512, 2048, 8192]:
        for n_cols in [64, 256, 1024, 4096, 8192]:
            x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

            # Warmup
            for _ in range(10):
                torch.softmax(x, dim=-1)
                triton_softmax(x)

            torch.cuda.synchronize()

            # Benchmark PyTorch native
            t0 = time.perf_counter()
            for _ in range(100):
                torch.softmax(x, dim=-1)
            torch.cuda.synchronize()
            pytorch_ms = (time.perf_counter() - t0) / 100 * 1000

            # Benchmark Triton
            t0 = time.perf_counter()
            for _ in range(100):
                triton_softmax(x)
            torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / 100 * 1000

            # Correctness check
            ref = torch.softmax(x, dim=-1)
            out = triton_softmax(x)
            assert torch.allclose(ref, out, atol=1e-5), "Correctness check failed!"

            # Roofline analysis
            bytes_accessed = x.nelement() * x.element_size() * 2  # read + write
            bandwidth_gb_s = (bytes_accessed / (triton_ms / 1000)) / 1e9
            # Look up your GPU's theoretical peak (e.g., RTX 4090 = 1008 GB/s)
            peak_bandwidth = 1008  # GB/s — adjust for your GPU
            bandwidth_util = bandwidth_gb_s / peak_bandwidth * 100

            results.append(KernelBenchResult(
                kernel="fused_softmax",
                n_rows=n_rows,
                n_cols=n_cols,
                pytorch_ms=pytorch_ms,
                triton_ms=triton_ms,
                speedup=pytorch_ms / triton_ms,
                bandwidth_gb_s=bandwidth_gb_s,
                bandwidth_utilization_pct=bandwidth_util,
            ))

    return results
```

### Roofline Analysis (in `kernels/README.md`)

Explain in plain English:

1. **What is roofline?** A model that says kernel performance is bounded by either compute (FLOPS) or memory bandwidth (GB/s). For softmax, memory bandwidth is the bottleneck.
2. **What's the theoretical peak?** Your GPU's bandwidth (e.g., RTX 4090 = 1008 GB/s).
3. **What did you achieve?** e.g., "At 2048×4096, Triton softmax achieves 680 GB/s = 67% of peak bandwidth, vs PyTorch native at 310 GB/s = 31% of peak."
4. **Why the gap to peak?** Cache effects, warp scheduling overhead, masking for non-power-of-2 sizes.

This analysis is what separates "I ran the Triton tutorial" from "I understand GPU performance."

---

## Phase 7: Quantization & Optimization Toggles (Day 8)

### Quantization Matrix

| Method        | Backend Support              | VRAM Reduction | Speed   |
|---------------|------------------------------|----------------|---------|
| AWQ 4-bit     | vLLM (native)                | ~60-70%        | Faster  |
| GPTQ 4-bit    | vLLM, TGI, Transformers      | ~60-70%        | Varies  |
| bitsandbytes  | Transformers, Custom         | ~60-75%        | Slower  |
| FP16          | All                          | Baseline       | Base    |

For the custom backend, support bitsandbytes since it integrates with HuggingFace model loading. AWQ and GPTQ require custom kernels that are out of scope for a hand-built server.

---

## Phase 8: GPU Monitoring (Day 8)

### pynvml Integration

```python
import pynvml

class GPUMonitor:
    def __init__(self, device_index: int = 0, poll_interval_ms: int = 100) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._poll_interval = poll_interval_ms / 1000
        self._samples: list[GPUSample] = []
        self._running = False

    async def start_polling(self) -> None:
        self._running = True
        while self._running:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self._samples.append(GPUSample(
                memory_used_mb=mem.used / (1024 ** 2),
                memory_total_mb=mem.total / (1024 ** 2),
                gpu_utilization_pct=util.gpu,
                power_draw_w=power / 1000,
                timestamp=time.monotonic(),
            ))
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> GPUStats:
        self._running = False
        if not self._samples:
            return GPUStats.empty()
        return GPUStats(
            peak_memory_mb=max(s.memory_used_mb for s in self._samples),
            avg_utilization_pct=sum(s.gpu_utilization_pct for s in self._samples) / len(self._samples),
            avg_power_w=sum(s.power_draw_w for s in self._samples) / len(self._samples),
            sample_count=len(self._samples),
        )
```

---

## Phase 9: Report Generation & README (Day 9-10)

### LLM Comparison Table (auto-generated)

```markdown
## LLM Benchmark: Llama 3 8B Instruct @ 8 concurrent requests

| Metric              | Transformers | Custom Batcher | vLLM (FP16) | vLLM (AWQ) | TGI    |
|---------------------|-------------:|---------------:|------------:|-----------:|-------:|
| Throughput (tok/s)  |         48.2 |          142.8 |       312.7 |      487.3 |  289.4 |
| Latency P50 (ms)    |       1842.0 |          621.3 |       198.3 |      142.1 |  221.6 |
| Latency P99 (ms)    |       3291.0 |         1104.7 |       387.2 |      267.9 |  412.8 |
| TTFT P50 (ms)       |            — |          89.2  |        22.1 |       18.3 |   31.2 |
| GPU Peak VRAM (MB)  |       15,820 |        15,420  |      14,200 |      8,942 | 14,890 |
| Error Rate (%)      |          0.0 |            0.0 |         0.0 |       0.13 |    0.0 |
```

### Gap Analysis Section (in README)

This is the section that proves understanding. After the table, write:

```markdown
## Why vLLM Is 3.2× Faster Than Our Custom Server

| Factor                     | Custom Server          | vLLM                              | Impact     |
|----------------------------|------------------------|-----------------------------------|------------|
| Attention kernel           | PyTorch eager mode     | PagedAttention (custom CUDA)      | ~2× alone  |
| Batch scheduling           | Python iteration loop  | C++ async scheduler               | ~1.3×      |
| Memory management          | Simple block pool      | Virtual memory + copy-on-write    | ~15% VRAM  |
| KV-cache reuse             | Per-request past_kv    | Shared paged blocks               | ~1.2×      |
| Kernel launch overhead     | Python → CUDA per op   | CUDA graphs for decode            | ~1.1×      |

Our custom server validates the core *algorithm* (continuous batching improves over
naive sequential by 3×), while vLLM's advantage comes from *implementation* — hand-written
CUDA kernels and systems engineering. This is precisely the type of work done at DeepInfra.
```

### Triton Kernel Results Section

```markdown
## Triton Kernel Experiments

### Fused Softmax (2048 rows)

| Sequence Length | PyTorch (ms) | Triton (ms) | Speedup | Bandwidth Util |
|----------------:|-------------:|------------:|--------:|---------------:|
|              64 |        0.042 |       0.038 |   1.10× |           41%  |
|             256 |        0.051 |       0.034 |   1.50× |           58%  |
|           1,024 |        0.089 |       0.054 |   1.65× |           64%  |
|           4,096 |        0.241 |       0.134 |   1.80× |           67%  |
|           8,192 |        0.468 |       0.271 |   1.73× |           62%  |
```

### Diffusion Results Section

```markdown
## Image Generation: SDXL @ 1024×1024, 30 steps

| Config                     | Total Time (s) | Step Time (ms) | VRAM (MB) |
|----------------------------|----------------:|---------------:|----------:|
| FP16 baseline              |           12.4  |          387   |     6,842 |
| FP16 + attention slicing   |           14.1  |          442   |     4,210 |
| FP16 + VAE tiling          |           13.0  |          405   |     5,980 |
| FP16 + torch.compile       |           8.7   |          271   |     7,120 |
| FP32                       |           24.8  |          792   |    12,640 |
```

---

## Common Traps & Solutions

| Trap | Symptom | Fix |
|------|---------|-----|
| Benchmarking cold model | First requests 10× slower | Warmup 5-10 requests, discard |
| Shared GPU between backends | OOM, corrupted results | Docker profiles, run sequentially |
| Non-deterministic results | 20%+ variance across runs | temperature=0, seed=42, run 3× report median |
| Wall clock only | Miss TTFT | Use streaming endpoint + per-chunk timestamps |
| `nvidia-smi` parsing | Brittle, slow | Use `pynvml` bindings |
| aiohttp connection limits | Bottleneck at high concurrency | `TCPConnector(limit=0)` |
| HF token not set | 403 on model download | Mount `~/.cache/huggingface/token` or `HF_TOKEN` env |
| Short prompts | Backends look similar | 200+ token prompts to stress prefill |
| Custom backend too slow | Results look embarrassing | Expected — document *why* it's slower, that's the point |
| Triton BLOCK_SIZE wrong | Silent wrong results | Always assert `torch.allclose` vs reference |

---

## Build Order (Day-by-Day)

| Day   | Deliverable | Verification |
|-------|-------------|-------------|
| 1     | Repo scaffold, `base.py` protocol, Docker Compose | `make lint` passes, compose validates |
| 2     | Custom backend: scheduler + KV-cache pool | Unit tests for scheduler logic (no GPU) |
| 3     | Custom backend: engine + server | Generates text for a single request correctly |
| 4     | Custom backend: continuous batching working | Multiple concurrent requests, correct outputs |
| 5     | vLLM + Transformers backends, gateway | `curl` each backend through gateway |
| 6     | Benchmark runner, first real numbers | Custom vs Transformers vs vLLM comparison |
| 7     | Diffusers backend + image bench runner | SDXL benchmark table generated |
| 8     | TGI backend, quantization toggles, GPU monitoring | Full 5-column LLM table, Grafana live |
| 9     | Triton kernels + kernel benchmarks | Softmax + GELU results with roofline |
| 10    | Report generator, README, gap analysis, polish | `make bench` → complete reproducible report |

---

## README Structure for Maximum Impact

```markdown
# InferBench

Reproducible head-to-head benchmarks for LLM and image generation
inference backends — including a from-scratch continuous batching server.

## Results (LLM)                    ← first thing anyone sees
[5-column comparison table]

## Why vLLM Is Faster               ← the analysis that proves understanding
[gap analysis table]

## Results (Image Generation)
[SDXL configuration comparison]

## Results (Triton Kernels)
[softmax/GELU speedup + roofline]

## Architecture
[system diagram]

## Custom Inference Server
[link to custom/README.md design doc]

## Quick Start
[3 commands to reproduce everything]

## Methodology
[warmup, seed, prompt dataset, statistical validity]

## Hardware
[exact GPU, driver, CUDA, Python versions]

## References
[Orca paper, PagedAttention paper, Triton docs]
```

The README results table is the first thing a DeepInfra recruiter sees. It must be above the fold, data-dense, and link to full JSON artifacts in `benchmarks/`.

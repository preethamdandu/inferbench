# CLAUDE.md — InferBench: Open-Source Model Serving Benchmark Platform

## Project Identity

InferBench is a systems-level benchmarking platform that serves open-source AI models through multiple inference backends — including a custom-built continuous batching server — and produces reproducible, head-to-head performance comparisons. It covers both LLM (Transformers) and image generation (Diffusers) workloads.

The target user is an ML engineer evaluating serving strategies for production deployment. The target *reviewer* is a DeepInfra hiring manager who wants proof you can work inside a serving engine, not just call one.

This is NOT a wrapper or demo app. It is infrastructure tooling. Every design decision should optimize for correctness, reproducibility, and measurable performance.

---

## What Makes This Project Different

Most "inference benchmark" repos spin up vLLM, hit the API, and plot charts. That shows tool usage, not understanding. InferBench includes a **from-scratch continuous batching inference server** (`src/backends/custom/`) that implements the core scheduling and KV-cache mechanics manually. It won't beat vLLM — it's not supposed to. It exists so the README can say "here's what I built, here's how it compares, and here's exactly why vLLM is 3× faster (PagedAttention, optimized CUDA kernels, C++ scheduler)." That analysis is the proof of understanding.

It also benchmarks **Diffusers** (image generation) alongside LLMs, and includes a **Triton kernel experiment** — a fused softmax implementation benchmarked against PyTorch native. These exist to cover the full surface area of the DeepInfra JD.

---

## Architecture

```
inferbench/
├── CLAUDE.md
├── SKILLS.md
├── README.md
├── pyproject.toml                  # single source of truth for deps (use uv)
├── docker-compose.yml              # spins up all backends + monitoring
├── Makefile                        # developer entrypoints
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # pydantic-settings, all knobs in one place
│   │
│   ├── backends/                   # one module per serving backend
│   │   ├── __init__.py
│   │   ├── base.py                 # abstract Backend protocol (LLM + Diffusion)
│   │   ├── vllm_backend.py         # HTTP client to vLLM OpenAI-compat API
│   │   ├── tgi_backend.py          # HTTP client to TGI API
│   │   ├── transformers_backend.py # naive in-process HF pipeline (baseline)
│   │   ├── diffusers_backend.py    # in-process Diffusers pipeline (SDXL/SD)
│   │   │
│   │   └── custom/                 # THE DIFFERENTIATOR — hand-built server
│   │       ├── __init__.py
│   │       ├── server.py           # FastAPI inference server (standalone process)
│   │       ├── scheduler.py        # continuous batching scheduler
│   │       ├── kv_cache.py         # KV-cache pool manager
│   │       ├── engine.py           # generation loop (prefill + decode steps)
│   │       └── README.md           # design doc explaining every decision
│   │
│   ├── gateway/                    # FastAPI gateway — single API surface
│   │   ├── __init__.py
│   │   ├── app.py                  # lifespan, middleware, error handling
│   │   ├── routes.py               # /v1/completions, /v1/images, /v1/health
│   │   └── schemas.py              # request/response models (OpenAI-compatible)
│   │
│   ├── bench/                      # load generation + metric collection
│   │   ├── __init__.py
│   │   ├── runner.py               # async load generator (aiohttp)
│   │   ├── runner_diffusion.py     # image generation benchmark runner
│   │   ├── metrics.py              # dataclasses for latency, throughput, TTFT
│   │   ├── profiles.py             # preset load profiles (ramp, spike, steady)
│   │   └── report.py               # markdown + JSON report generation
│   │
│   ├── optimizations/              # toggleable serving optimizations
│   │   ├── __init__.py
│   │   ├── quantization.py         # AWQ, GPTQ, bitsandbytes 4-bit
│   │   ├── batching.py             # continuous batching config
│   │   └── cache.py                # KV-cache tuning
│   │
│   ├── kernels/                    # Triton kernel experiments
│   │   ├── __init__.py
│   │   ├── fused_softmax.py        # custom Triton fused softmax
│   │   ├── fused_gelu.py           # custom Triton fused GELU (bonus)
│   │   ├── bench_kernels.py        # benchmark vs torch native ops
│   │   └── README.md               # kernel design explanation + roofline analysis
│   │
│   └── monitoring/                 # Prometheus metrics + GPU telemetry
│       ├── __init__.py
│       ├── prometheus.py           # custom collectors
│       └── gpu.py                  # pynvml polling
│
├── tests/
│   ├── unit/
│   │   ├── test_metrics.py
│   │   ├── test_schemas.py
│   │   ├── test_scheduler.py       # scheduler logic tests (no GPU needed)
│   │   ├── test_kv_cache.py        # cache eviction/allocation tests
│   │   ├── test_quantization.py
│   │   └── test_triton_softmax.py
│   ├── integration/
│   │   ├── test_gateway.py
│   │   ├── test_backends.py
│   │   └── test_custom_server.py
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.gateway
│   ├── Dockerfile.custom           # custom backend as standalone service
│   ├── Dockerfile.vllm
│   ├── Dockerfile.tgi
│   └── prometheus.yml
│
├── grafana/
│   └── dashboards/
│       └── inferbench.json
│
├── prompts/                        # standardized prompt datasets
│   ├── short.jsonl
│   ├── medium.jsonl
│   ├── long.jsonl
│   └── mixed.jsonl
│
├── benchmarks/                     # saved results (git-tracked)
│   └── .gitkeep
│
└── scripts/
    ├── run_bench.py                # CLI: python scripts/run_bench.py --model ... --backend ...
    ├── run_bench_diffusion.py      # CLI: image generation benchmarks
    ├── run_kernel_bench.py         # CLI: Triton kernel benchmarks
    └── download_model.py           # pre-download to shared volume
```

---

## Tech Stack (locked)

| Layer            | Choice                          | Why                                         |
|------------------|---------------------------------|---------------------------------------------|
| Language         | Python 3.11+                    | ecosystem alignment                         |
| Package manager  | uv                              | fast, deterministic                          |
| API framework    | FastAPI + uvicorn               | async, OpenAPI docs for free                 |
| LLM backends     | vLLM, TGI, HF Transformers, Custom | custom is the differentiator              |
| Image backend    | HF Diffusers (SDXL)            | covers the Diffusers requirement in JD       |
| Kernel framework | Triton                          | approachable GPU kernel authoring            |
| Load testing     | custom async (aiohttp)          | fine-grained TTFT + per-token measurement    |
| Metrics          | Prometheus client_python        | industry standard                            |
| GPU telemetry    | pynvml                          | direct NVML bindings                         |
| Dashboards       | Grafana                         | pairs with Prometheus                        |
| Containers       | Docker Compose                  | one-command reproducibility                  |
| Config           | pydantic-settings               | typed, validated, env-var driven             |
| Testing          | pytest + pytest-asyncio         | async-native test runner                     |
| Quantization     | AutoAWQ, AutoGPTQ, bitsandbytes | covers the main strategies                   |
| Linting          | ruff                            | fast, replaces flake8+isort+black            |
| Type checking    | pyright                         | strict mode                                  |

---

## Coding Standards

### Python
- **Type everything.** All function signatures, return types, dataclass fields. No `Any` unless wrapping an untyped library — and add a `# type: ignore[...]` with the specific code.
- **Async by default.** The gateway and bench runner are fully async. Use `asyncio.TaskGroup` for concurrent work (Python 3.11+). Never use `threading` for IO-bound work.
- **Pydantic for all boundaries.** API schemas, config, benchmark results — if data crosses a boundary, it's a Pydantic model.
- **No print statements.** Use `structlog` with bound loggers. Every log line must include `backend=`, `model=`, or `request_id=` context.
- **Error handling:** Raise domain-specific exceptions (`BackendUnavailableError`, `ModelNotLoadedError`). The gateway catches and maps to HTTP status codes in a single exception handler. Never catch bare `Exception`.
- **Constants over magic numbers.** If a number appears in code, it must be a named constant or config value.

### Custom Backend Standards (extra strict)
- **No copying vLLM source code.** Write from first principles. Reference papers (Orca, PagedAttention), not implementations.
- **Every design decision documented.** The `custom/README.md` must explain why the scheduler works the way it does, what trade-offs were made vs production systems, and what would need to change for production.
- **Correctness over speed.** The custom backend is a learning artifact. It must produce identical outputs to the Transformers baseline for the same prompt + seed. Speed comes second.

### Triton Kernel Standards
- **Benchmark against PyTorch native.** Every kernel has a corresponding `torch` implementation benchmarked side-by-side.
- **Include roofline analysis.** Even a simple one — compute bandwidth utilization vs theoretical peak. Shows you understand GPU performance modeling.
- **Document the math.** Each kernel's README explains what the operation does mathematically, why fusion helps (fewer global memory round-trips), and what the theoretical speedup ceiling is.

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE`
- Private: single leading underscore

### Docstrings
- Google style. Required on all public functions and classes.
- Include `Args`, `Returns`, `Raises` sections when non-trivial.
- One-liner docstrings for simple helpers.

### Git
- Conventional commits: `feat:`, `fix:`, `bench:`, `infra:`, `docs:`, `test:`, `kernel:`
- One logical change per commit. Benchmarks get their own commits with results in the message body.

---

## Key Commands

```bash
# Setup
make setup                    # uv sync + pre-commit install
make download MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# Development
make dev                      # start gateway in reload mode (no GPU backends)
make lint                     # ruff check + pyright
make test                     # pytest unit tests (no GPU needed)
make test-integration         # pytest integration (requires Docker + GPU)

# LLM Benchmarking (one backend at a time — they share GPU)
make up PROFILE=vllm          # spin up vLLM + monitoring
make bench MODEL=... BACKEND=vllm PROFILE=steady DURATION=60
make down

make up PROFILE=custom        # spin up custom backend + monitoring
make bench MODEL=... BACKEND=custom PROFILE=steady DURATION=60
make down

make report                   # generate comparison markdown from all results

# Image Generation Benchmarking
make bench-diffusion MODEL=stabilityai/stable-diffusion-xl-base-1.0 STEPS=30

# Triton Kernel Benchmarking
make bench-kernels            # run all kernel benchmarks, output results

# Monitoring
make grafana                  # open Grafana dashboard in browser
```

---

## Backend Protocols

### LLM Backend

```python
class LLMBackend(Protocol):
    async def load_model(self, model_id: str, quantization: str | None = None) -> None: ...
    async def generate(self, prompt: str, params: GenerateParams) -> GenerateResult: ...
    async def stream(self, prompt: str, params: GenerateParams) -> AsyncIterator[TokenChunk]: ...
    async def health(self) -> HealthStatus: ...
    async def unload(self) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def loaded_model(self) -> str | None: ...
    @property
    def gpu_memory_usage_mb(self) -> float: ...
```

### Diffusion Backend

```python
class DiffusionBackend(Protocol):
    async def load_model(self, model_id: str, **kwargs) -> None: ...
    async def generate_image(self, prompt: str, params: ImageGenParams) -> ImageGenResult: ...
    async def health(self) -> HealthStatus: ...
    async def unload(self) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def gpu_memory_usage_mb(self) -> float: ...
```

---

## Metrics Collected

### LLM Benchmarks
- **Throughput**: tokens/second (generation), requests/second
- **Latency**: P50, P90, P95, P99, max (end-to-end and per-token)
- **Time to First Token (TTFT)**: P50, P95, P99
- **GPU**: peak VRAM (MB), avg utilization %, power draw (W)
- **System**: concurrent connections, total tokens generated, errors, timeouts

### Image Generation Benchmarks
- **Throughput**: images/minute
- **Latency**: P50, P95, P99 per image (end-to-end)
- **Step latency**: avg time per diffusion step
- **GPU**: peak VRAM (MB), avg utilization %, power draw (W)

### Triton Kernel Benchmarks
- **Throughput**: GFLOPS or GB/s
- **Speedup**: vs PyTorch native, across matrix of input sizes
- **Bandwidth utilization**: % of theoretical peak (roofline)

---

## What NOT To Build

- **No web UI / dashboard frontend.** Grafana handles visualization.
- **No model training or fine-tuning.** This is inference-only.
- **No authentication/auth.** This is a local benchmarking tool.
- **No database.** Benchmark results are flat JSON files.
- **No LangChain, no RAG, no agent frameworks.** Systems-level infrastructure only.
- **No abstraction astronautics.** If a pattern doesn't serve benchmarking correctness or backend extensibility, skip it.

---

## Resume Lines This Project Produces

### Primary (the one-liner):
> Built an LLM inference benchmarking platform with a from-scratch continuous batching server, comparing custom, vLLM, TGI, and HuggingFace Transformers backends; demonstrated 3.2× throughput improvement via continuous batching with AWQ 4-bit quantization at P99 < 180ms.

### Supporting bullets:
> Implemented continuous batching scheduler and KV-cache manager from first principles in Python/PyTorch, benchmarked against vLLM to validate performance gap analysis (PagedAttention, CUDA kernel fusion).

> Extended benchmarking to Stable Diffusion XL via HuggingFace Diffusers; measured images/minute, per-step latency, and VRAM efficiency across precision modes (FP16, FP32).

> Wrote fused softmax Triton kernel achieving 1.8× speedup over PyTorch native at sequence length 4096; included roofline analysis of memory bandwidth utilization.

> Containerized full stack with Docker Compose; instrumented Prometheus/Grafana observability for real-time throughput, latency percentiles, and GPU telemetry.

Every engineering decision in this project exists to make these lines true and provable.

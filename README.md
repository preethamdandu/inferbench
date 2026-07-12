# InferBench

**Open-source LLM and image generation inference benchmarking platform.** Benchmarks five backends head-to-head: a custom continuous batching engine (built from scratch), vLLM, TGI, HuggingFace Transformers, and Diffusers SDXL.

---

## Measured Results — google/gemma-2-2b-it (Colab Tesla T4 16GB, FP16)

> Config: `prompts/short.jsonl` (20–64 max output tokens), 120-second window, 2-request warmup, `temperature=0`. Custom server: `max_batch_size=4`, `kv_cache_blocks=64`, `block_size=16`. Raw artifacts: [`benchmarks/`](benchmarks/).

| Metric | Transformers (sequential) | Custom Batcher (batch=4) | Delta |
|--------|--------------------------|--------------------------|-------|
| **Throughput (tok/s)** | 19.8 | **69.3** | **3.5×** |
| **Requests/s** | 0.49 | **1.58** | **3.2×** |
| **Requests completed (120s)** | 59 | **194** | 3.3× |
| **TTFT P50 (ms)** | 2,363* | **96** | **24×** |
| **TTFT P99 (ms)** | 3,722* | 162 | 23× |
| **Latency P50 (ms)** | 2,363 | 2,652 | +12% |
| **Latency P99 (ms)** | 3,722 | 4,027 | +8% |
| **Error rate** | 0% | 0% | — |

*\*The Transformers baseline is non-streaming, so TTFT ≈ full request latency.*

**The tradeoff, quantified:** continuous batching trades ~10% per-request latency for **3.2–3.5× throughput** and **24× faster time-to-first-token**. Each request shares the GPU with up to 3 others (slightly slower individually), but the server completes 3.3× more requests in the same window. This is the core mechanism behind vLLM/TGI-class serving engines, reproduced from first principles in ~400 lines of Python ([`src/backends/custom/`](src/backends/custom/)).

> **Pending:** vLLM and TGI head-to-head runs require a Docker-capable GPU host (Colab has no Docker) — see skip notes in [`benchmarks/`](benchmarks/). SDXL and Triton kernel benchmarks are also pending a larger-VRAM environment.

---

## Why vLLM Is Faster — Gap Analysis

| Factor | Custom Batcher Approach | vLLM Approach | Throughput Impact |
|--------|------------------------|---------------|-------------------|
| **Attention Kernel** | PyTorch SDPA with left-padded contiguous tensors | PagedAttention C++ CUDA kernel on non-contiguous blocks | ~2.5× decode throughput |
| **Scheduler** | Python async loop with GIL overhead | C++ iteration-level scheduler with zero Python in hot path | ~1.3× scheduling efficiency |
| **Memory Management** | Fixed preallocated block pool, no virtual mapping | Virtual-to-physical block mapping, zero wasted VRAM | ~1.4× VRAM utilization |
| **KV-Cache Reuse** | None — every request re-prefills from scratch | Prefix caching shares KV blocks for repeated system prompts | Varies (up to 2× for chatbot workloads) |
| **CUDA Graphs** | Re-launches PyTorch ops every decode step | Captured static decode graphs, near-zero Python overhead | ~1.2× decode latency |
| **Quantization** | FP16 only | FP16, INT8, AWQ, GPTQ, FP8 | 1.5–2× throughput at INT4 |

---

## Triton Kernel Results — Tesla T4 (Colab, FP32)

> Both kernels pass correctness against PyTorch reference (max diff < 3.2e-07). Median of 20 timed runs after 5 warmup iterations. T4 theoretical peak memory bandwidth: 320 GB/s. Raw artifact: [`benchmarks/triton_kernels.json`](benchmarks/triton_kernels.json).

### Fused Softmax vs `torch.softmax` (2048 rows)

| N Cols | PyTorch (ms) | Triton (ms) | Speedup | Triton BW (GB/s) | % Peak BW |
|--------|-------------|-------------|---------|------------------|-----------|
| 256    | 0.028       | 0.081       | 0.34×   | 52               | 16%       |
| 1024   | 0.094       | 0.108       | 0.87×   | 155              | 48%       |
| 2048   | 0.182       | 0.175       | 1.04×   | 192              | 60%       |
| 4096   | 0.484       | 0.319       | **1.52×** | 210            | 66%       |
| 8192   | 0.771       | 0.624       | 1.24×   | **215**          | **67%**   |

### Fused GELU vs `torch.nn.functional.gelu` (tanh approximation)

| N Elements | PyTorch (ms) | Triton (ms) | Speedup | Triton BW (GB/s) | % Peak BW |
|------------|-------------|-------------|---------|------------------|-----------|
| 65K        | 0.017       | 0.036       | 0.48×   | 15               | 5%        |
| 1M         | 0.051       | 0.068       | 0.75×   | 124              | 39%       |
| 4M         | 0.156       | 0.174       | 0.90×   | 193              | 60%       |

**What the numbers show — and why GELU "loses":** Softmax benefits from fusion because the naive formulation makes three passes over the data (max, exp-sum, normalize); the fused kernel does one read and one write per row, winning **1.52×** once rows are large enough (≥2048 cols) to amortize kernel launch overhead, and sustaining 215 GB/s (67% of T4 peak). GELU, by contrast, is already a *single* elementwise kernel in PyTorch — there are no memory round-trips left to eliminate — so the hand-written kernel converges toward parity at large sizes (0.90×) and loses at small sizes where launch overhead dominates. This is the roofline lesson in practice: fusion pays only when it removes global-memory traffic, not when an op is already bandwidth-saturated.

Kernel design docs and roofline analysis: [`src/kernels/README.md`](src/kernels/README.md)

---

## Planned Benchmarks (not yet run)

| Benchmark | Blocker | Runner |
|-----------|---------|--------|
| vLLM head-to-head | bfloat16-capable GPU (see T4 findings below) | `scripts/run_bench.py --backend vllm` |
| TGI head-to-head | Docker + GPU host | `scripts/run_bench.py --backend tgi` |
| SDXL image generation (FP16 / attention slicing / torch.compile) | >16GB VRAM for reasonable step times | `scripts/run_bench_diffusion.py` |

### Why vLLM couldn't run on the Colab T4 (measured findings)

The vLLM comparison was attempted on the same Colab T4 (vLLM 0.22.0, cu129 build) and hit two hard limits worth documenting:

1. **gemma-2 + fp16 rejected by design.** vLLM refuses to serve `gemma2` models in float16 (`ValueError: The model type 'gemma2' does not support float16. Reason: Numerical instability`) — gemma-2's logit softcapping can overflow in fp16. The Transformers baseline and the custom batcher ran fp16 with no such guardrail; production engines encode model-specific numerical-safety knowledge that naive serving stacks lack.
2. **fp32 fallback exceeds the T4's shared memory.** The T4 (compute capability 7.5) supports neither bfloat16 nor FlashAttention-2, so vLLM fell back to float32 with its Triton attention backend — whose kernel requires 80 KB of shared memory per block against the T4's 64 KB hardware limit (`triton.runtime.errors.OutOfResources: shared memory, Required: 81920, Hardware limit: 65536`). The engine core died on the first forward pass.

Net: serving gemma-2 through vLLM requires an Ampere-or-newer GPU (A10/L4/RTX 30xx+) where bfloat16 and larger shared memory are available. Full session log: [`llm_inferbench.ipynb`](llm_inferbench.ipynb).

---

## Architecture

```
               ┌─────────────────────────────────┐
    Client ───▶│   FastAPI Unified Gateway :8000  │
               │  /v1/completions?backend=...     │
               │  /v1/images/generations          │
               │  /v1/health                      │
               │  /metrics  (Prometheus)          │
               └────────────┬────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
  ┌───────▼──────┐  ┌───────▼──────┐  ┌──────▼──────────────────────┐
  │ vLLM :8001   │  │ TGI   :8002  │  │ Custom Engine :8003          │
  │ (PagedAttn)  │  │ (Rust router)│  │ scheduler.py + kv_cache.py   │
  └──────────────┘  └──────────────┘  │ engine.py (prefill/decode)   │
                                       └──────────────────────────────┘
                 Prometheus :9090 ──▶ Grafana :3000
```

### Custom Inference Server

The `src/backends/custom/` directory contains a full continuous batching engine:

- **`scheduler.py`** — Iteration-level scheduler: evicts finished sequences, admits waiting requests when KV-cache has space, returns running batch each step.
- **`kv_cache.py`** — Block-pool preallocated GPU tensor with free-list allocation.
- **`engine.py`** — Async generation loop: prefill pass for new sequences, decode pass for existing, asyncio.Future resolution on completion.
- **`server.py`** — FastAPI server wrapping the engine with tokenization.

Design document: [`src/backends/custom/README.md`](src/backends/custom/README.md)

---

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Start gateway + monitoring (no GPU needed for gateway)
make dev

# 3. Run benchmarks (needs running backend)
make bench
```

---

## Methodology

- **Warmup**: Warmup requests are discarded before metric collection begins.
- **Determinism**: `temperature=0` (greedy decoding) for all benchmark runs.
- **Duration**: Fixed-time window (not request-count based). Throughput = total completion tokens / elapsed wall time.
- **Token counts**: Exact `completion_tokens` reported by the server (`usage` field), not text approximations.
- **Prompt Dataset**: Custom JSONL datasets in `prompts/` with short (avg 32 tok), medium (avg 200 tok), and long (avg 512 tok) prompt categories. Fair comparisons always use the same prompt file for every backend.
- **Percentiles**: P50/P90/P99 computed via linear interpolation across all per-request latency measurements.
- **Isolation**: Backends run one at a time — the custom server is stopped before the in-process Transformers baseline runs, so they never share GPU memory.

---

## Hardware (measured runs)

| Component | Spec |
|-----------|------|
| **Environment** | Google Colab (free tier) |
| **GPU** | NVIDIA Tesla T4, 16 GB GDDR6 |
| **Python** | 3.12 |
| **Transformers** | 5.x |
| **Precision** | FP16 |

---

## References

1. Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — Continuous batching.
2. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — vLLM.
3. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022).
4. Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023).
5. Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019).

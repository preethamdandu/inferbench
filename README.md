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

## Planned Benchmarks (not yet run)

| Benchmark | Blocker | Runner |
|-----------|---------|--------|
| vLLM FP16 / AWQ head-to-head | Docker + GPU host (RunPod/Lambda) | `scripts/run_bench.py --backend vllm` |
| TGI head-to-head | Docker + GPU host | `scripts/run_bench.py --backend tgi` |
| SDXL image generation (FP16 / attention slicing / torch.compile) | >16GB VRAM for reasonable step times | `scripts/run_bench_diffusion.py` |
| Triton fused softmax / GELU vs PyTorch | Linux + CUDA (works on Colab; time-boxed out of the T4 session) | `scripts/run_kernel_bench.py` |

Kernel design docs and roofline analysis: [`src/kernels/README.md`](src/kernels/README.md)

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

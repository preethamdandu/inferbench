# InferBench

**Open-source LLM and image generation inference benchmarking platform.** Benchmarks five backends head-to-head: a custom continuous batching engine (built from scratch), vLLM, TGI, HuggingFace Transformers, and Diffusers SDXL.

---

> **Benchmark status:** The tables below are **illustrative targets**, not measured results. No completed benchmark artifacts exist in `benchmarks/` yet — only skip stubs from Colab development. Run `scripts/run_bench.py` on a GPU machine, save JSON to `benchmarks/`, then `make report` to replace these numbers with real data.

## LLM Benchmark Results — Mistral-7B-v0.1 (A100 80GB, FP16) *(illustrative)*

> Benchmark config: 512 concurrent requests, 50-token prompts, 200-token outputs, 60-second run, 10-request warmup. Seeds fixed at 42. Prompt dataset: `prompts/mixed.jsonl`.

| Metric | Transformers (Baseline) | Custom Batcher | vLLM FP16 | vLLM AWQ (4-bit) | TGI |
|--------|------------------------|----------------|-----------|-----------------|-----|
| **Throughput (tok/s)** | 142 | 891 | 4,312 | 5,847 | 3,891 |
| **RPS (req/s)** | 0.7 | 4.5 | 21.6 | 29.2 | 19.5 |
| **Latency P50 (ms)** | 14,200 | 2,100 | 432 | 318 | 481 |
| **Latency P99 (ms)** | 28,400 | 4,800 | 891 | 654 | 1,024 |
| **TTFT P50 (ms)** | 842 | 118 | 28 | 21 | 34 |
| **GPU Peak VRAM (GB)** | 14.2 | 16.1 | 22.4 | 9.8 | 19.6 |

> **Custom Batcher vs Transformers**: 6.3× throughput improvement from continuous batching alone.
> **vLLM AWQ vs Custom**: 6.6× throughput from PagedAttention + AWQ 4-bit quantization. P99 latency < 654ms at 29 RPS.

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

## Image Generation Results — Stable Diffusion XL (A100 80GB) *(illustrative)*

> Config: 1024×1024 images, 30 DDIM steps, guidance_scale=7.5, 20 sequential generations after 2 warmup runs.

| Configuration | Avg Time/Image (s) | Avg Step Time (ms) | Peak VRAM (GB) | Images/hr |
|---------------|-------------------|-------------------|----------------|-----------|
| SDXL FP16 (baseline) | 8.2 | 273 | 11.4 | 439 |
| SDXL + attention_slicing | 9.1 | 303 | 7.8 | 396 |
| SDXL + torch.compile (max-autotune) | 5.4 | 180 | 11.8 | 667 |
| SDXL + vae_tiling (8K resolution) | 22.4 | 747 | 9.2 | 161 |

> `torch.compile` delivers **1.52×** speedup at the cost of ~200s compilation on first call (cached thereafter).

---

## Triton Kernel Results — A100 80GB SXM4 *(illustrative)*

### Fused Softmax vs PyTorch (4096 rows)

| N Cols | PyTorch (GB/s) | Triton (GB/s) | Speedup | % Peak BW |
|--------|---------------|---------------|---------|-----------|
| 512    | 489           | 812           | 1.66×   | 39.8%     |
| 1024   | 651           | 1134          | 1.74×   | 55.6%     |
| 2048   | 893           | 1487          | 1.66×   | 72.9%     |
| 4096   | 1012          | 1693          | 1.67×   | 83.0%     |

### Fused GELU vs PyTorch

| N Elements | PyTorch (GB/s) | Triton (GB/s) | Speedup | % Peak BW |
|------------|---------------|---------------|---------|-----------|
| 1M         | 412           | 789           | 1.91×   | 38.7%     |
| 16M        | 964           | 1598          | 1.66×   | 78.4%     |
| 64M        | 1089          | 1782          | 1.64×   | 87.4%     |

> GPU peak memory bandwidth: 2,039 GB/s. Triton kernels reach up to 88.8% of peak.
> Full roofline analysis: [`src/kernels/README.md`](src/kernels/README.md)

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

- **Warmup**: First 10 requests discarded before metric collection begins.
- **Seeds**: All backends initialized with `seed=42` for deterministic sampling (temperature=0 for correctness checks).
- **Duration**: 60-second fixed-time window (not request-count based). Throughput = total_tokens / 60s.
- **Prompt Dataset**: Custom JSONL datasets in `prompts/` with short (avg 32 tok), medium (avg 200 tok), and long (avg 512 tok) prompt categories. Mixed benchmark uses proportional sampling.
- **Percentiles**: P50/P90/P99 computed via linear interpolation across all per-request latency measurements.
- **Isolation**: Each backend runs in its own Docker container with a dedicated GPU resource reservation to prevent memory contention.

---

## Hardware

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA A100 SXM4 80GB |
| **VRAM** | 80 GB HBM2e |
| **CUDA Version** | 12.1 |
| **CUDA Driver** | 530.30.02 |
| **Python** | 3.11.x |
| **PyTorch** | 2.2.1+cu121 |
| **OS** | Ubuntu 22.04 LTS |
| **CPU** | AMD EPYC 7763 (64 cores) |
| **RAM** | 512 GB DDR4 |

---

## References

1. Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — Continuous batching.
2. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — vLLM.
3. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022).
4. Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023).
5. Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019).

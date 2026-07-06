# InferBench Report

> Generated from JSON artifacts in `benchmarks/`.
> Re-run `make report` after collecting new benchmark results.

## Skipped Runs

- **sdxl_skipped.json**: SDXL benchmarks skipped on Colab T4 free tier. SDXL generation on T4 takes 30-60s per image, and running 4 configurations × 10 images would exceed the ~90 min Colab session window. Deferred to a higher-VRAM GPU environment.
- **tgi_fp16.json**: TGI requires Docker which is not available in Google Colab. TGI benchmarking deferred to a Docker-capable environment (Runpod/Lambda/local workstation).
- **vllm_awq.json**: No pre-quantized AWQ variant of google/gemma-2-2b-it publicly available. Quantization benefits are negligible at 2B parameter scale (<6GB in FP16). AWQ comparison would be meaningful for 7B+ models.

## LLM Benchmarks

| Model | Backend | Concurrency | RPS | Tokens/s | Latency P50 (ms) | Latency P99 (ms) | TTFT P50 (ms) | VRAM (MB) | Errors (%) |
|-------|---------|-------------|-----|----------|------------------|------------------|---------------|-----------|------------|
| google/gemma-2-2b-it | custom | 4 | 1.58 | 69.3 | 2651.9 | 4027.0 | 95.9 | N/A | 0.00 |
| google/gemma-2-2b-it | transformers | 1 | 0.10 | 19.9 | 10237.5 | 12655.2 | 10237.5 | 5034 | 0.00 |
| google/gemma-2-2b-it | transformers | 1 | 0.49 | 19.8 | 2362.7 | 3721.6 | 2362.7 | 5005 | 0.00 |

## LLM Details

### custom — google/gemma-2-2b-it
- **concurrency**: 4
- **duration_seconds**: 122.58
- **total_requests**: 194
- **successful_requests**: 194
- **failed_requests**: 0
- **tokens_per_second**: 69.27
- **requests_per_second**: 1.58
- **latency_p50_ms**: 2651.88
- **latency_p90_ms**: 3424.72
- **latency_p99_ms**: 4027.03
- **ttft_p50_ms**: 95.87
- **ttft_p90_ms**: 133.85
- **ttft_p99_ms**: 161.64
- **tpot_p50_ms**: None
- **tpot_p90_ms**: None
- **gpu_peak_vram_mb**: None
- **error_rate_pct**: 0.00

### transformers — google/gemma-2-2b-it
- **concurrency**: 1
- **duration_seconds**: 50.49
- **total_requests**: 5
- **successful_requests**: 5
- **failed_requests**: 0
- **tokens_per_second**: 19.93
- **requests_per_second**: 0.10
- **latency_p50_ms**: 10237.55
- **latency_p90_ms**: 11762.81
- **latency_p99_ms**: 12655.17
- **ttft_p50_ms**: 10237.55
- **ttft_p90_ms**: 11762.81
- **ttft_p99_ms**: 12655.17
- **tpot_p50_ms**: None
- **tpot_p90_ms**: None
- **gpu_peak_vram_mb**: 5033.85
- **error_rate_pct**: 0.00

### transformers — google/gemma-2-2b-it
- **concurrency**: 1
- **duration_seconds**: 120.77
- **total_requests**: 59
- **successful_requests**: 59
- **failed_requests**: 0
- **tokens_per_second**: 19.76
- **requests_per_second**: 0.49
- **latency_p50_ms**: 2362.65
- **latency_p90_ms**: 2953.84
- **latency_p99_ms**: 3721.57
- **ttft_p50_ms**: 2362.65
- **ttft_p90_ms**: 2953.84
- **ttft_p99_ms**: 3721.57
- **tpot_p50_ms**: None
- **tpot_p90_ms**: None
- **gpu_peak_vram_mb**: 5005.31
- **error_rate_pct**: 0.00

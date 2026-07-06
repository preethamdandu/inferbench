# InferBench Report

> Generated from JSON artifacts in `benchmarks/`.
> Re-run `make report` after collecting new benchmark results.

## Skipped Runs

- **sdxl_skipped.json**: SDXL benchmarks skipped on Colab T4 free tier. SDXL generation on T4 takes 30-60s per image, and running 4 configurations × 10 images would exceed the ~90 min Colab session window. Deferred to a higher-VRAM GPU environment.
- **tgi_fp16.json**: TGI requires Docker which is not available in Google Colab. TGI benchmarking deferred to a Docker-capable environment (Runpod/Lambda/local workstation).
- **vllm_awq.json**: No pre-quantized AWQ variant of google/gemma-2-2b-it publicly available. Quantization benefits are negligible at 2B parameter scale (<6GB in FP16). AWQ comparison would be meaningful for 7B+ models.

## LLM Benchmarks

_No completed LLM benchmark artifacts found. Run `scripts/run_bench.py` on a GPU machine and save JSON to `benchmarks/`._

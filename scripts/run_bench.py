"""
Benchmark runner for LLM backends.

Supports both direct-backend mode (transformers run in-process) and
HTTP-backend mode (vllm / tgi / custom — hit the backend API directly,
bypassing the gateway, so it also works without the gateway running on Colab).

Usage:
    python scripts/run_bench.py \
        --backend transformers \
        --model google/gemma-2-2b-it \
        --concurrency 1 \
        --duration 45 \
        --warmup 3 \
        --prompts prompts/medium.jsonl \
        --output benchmarks/transformers_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# In-process Transformers benchmark (no HTTP, no server needed)
# ---------------------------------------------------------------------------

def _run_transformers(
    model_id: str,
    prompts: list[dict[str, Any]],
    duration: int,
    warmup: int,
    output: Path,
) -> None:
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"[transformers] model loaded on {device}")

    def generate_one(prompt_data: dict[str, Any]) -> tuple[int, float, float | None, float]:
        prompt = prompt_data["prompt"]
        max_new_tokens = int(prompt_data.get("max_tokens", 100))

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        completion_tokens = out.shape[1] - prompt_len
        # Transformers baseline: ttft ≈ first token is when prefill ends
        ttft_ms = total_ms  # prefill-only proxy — single request

        peak_vram_mb = 0.0
        if device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return completion_tokens, total_ms, ttft_ms, peak_vram_mb

    # Warmup
    print(f"[transformers] warming up ({warmup} requests)...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(min(warmup, len(prompts))):
        generate_one(prompts[i % len(prompts)])

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed run
    print(f"[transformers] benchmarking for {duration}s...")
    results = []
    start_wall = time.perf_counter()
    deadline = start_wall + duration
    idx = 0
    while time.perf_counter() < deadline:
        p = prompts[idx % len(prompts)]
        idx += 1
        completion_tokens, total_ms, ttft_ms, peak_vram_mb = generate_one(p)
        results.append({
            "completion_tokens": completion_tokens,
            "total_ms": total_ms,
            "ttft_ms": ttft_ms,
            "peak_vram_mb": peak_vram_mb,
        })

    elapsed = time.perf_counter() - start_wall
    total_tokens = sum(r["completion_tokens"] for r in results)
    latencies = sorted(r["total_ms"] for r in results)
    ttfts = sorted(r["ttft_ms"] for r in results if r["ttft_ms"] is not None)
    peak_vram = max((r["peak_vram_mb"] for r in results), default=0.0)

    def pct(arr: list[float], p: float) -> float | None:
        if not arr:
            return None
        k = (len(arr) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(arr):
            return arr[-1]
        return arr[f] + (k - f) * (arr[c] - arr[f])

    tokens_per_second = total_tokens / elapsed
    requests_per_second = len(results) / elapsed
    report = {
        "status": "completed",
        "backend": "transformers",
        "model": model_id,
        "concurrency": 1,
        "duration_s": elapsed,
        "duration_seconds": elapsed,
        "total_requests": len(results),
        "successful_requests": len(results),
        "failed_requests": 0,
        "throughput_tokens_per_sec": tokens_per_second,
        "tokens_per_second": tokens_per_second,
        "requests_per_second": requests_per_second,
        "latency_p50_ms": pct(latencies, 50),
        "latency_p90_ms": pct(latencies, 90),
        "latency_p99_ms": pct(latencies, 99),
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p90_ms": pct(ttfts, 90),
        "ttft_p99_ms": pct(ttfts, 99),
        "gpu_peak_vram_mb": peak_vram,
        "error_rate_pct": 0.0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[transformers] RESULTS saved to {output}")
    print(f"  throughput:  {report['throughput_tokens_per_sec']:.1f} tok/s")
    print(f"  latency P50: {report['latency_p50_ms']:.0f} ms")
    print(f"  latency P99: {report['latency_p99_ms']:.0f} ms")
    print(f"  TTFT P50:    {report['ttft_p50_ms']:.0f} ms")
    print(f"  VRAM peak:   {peak_vram:.0f} MB")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# HTTP benchmark (custom / vLLM / TGI  — hits the server API directly)
# ---------------------------------------------------------------------------

async def _run_http(
    backend: str,
    model_id: str,
    base_url: str,
    prompts: list[dict[str, Any]],
    duration: int,
    warmup: int,
    concurrency: int,
    output: Path,
) -> None:
    import aiohttp

    # Map backend → endpoint path
    if backend == "vllm":
        endpoint = f"{base_url}/v1/completions"
    elif backend == "tgi":
        endpoint = f"{base_url}/generate"
    else:
        # custom server
        endpoint = f"{base_url}/v1/completions"

    def make_payload(prompt_data: dict[str, Any]) -> dict[str, Any]:
        prompt = prompt_data["prompt"]
        max_tokens = int(prompt_data.get("max_tokens", 100))
        if backend == "tgi":
            return {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        return {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

    results: list[dict[str, Any]] = []
    errors = 0

    async def worker(session: aiohttp.ClientSession, deadline: float, idx: int) -> None:
        nonlocal errors
        local_idx = idx
        while time.perf_counter() < deadline:
            p = prompts[local_idx % len(prompts)]
            local_idx += 1
            payload = make_payload(p)
            t0 = time.perf_counter()
            ttft_ms = None
            completion_tokens = 0
            try:
                async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                total_ms = (time.perf_counter() - t0) * 1000.0

                if backend == "tgi":
                    text = data.get("generated_text", "")
                    completion_tokens = len(text.split())
                    ttft_ms = float(data.get("details", {}).get("prefill_tokens", total_ms / 2))
                else:
                    choices = data.get("choices", [])
                    text = choices[0].get("text", "") if choices else ""
                    completion_tokens = max(len(text.split()), 1)
                    timing = data.get("timing", {})
                    ttft_ms = timing.get("time_to_first_token_ms")
                    if ttft_ms is None:
                        ttft_ms = total_ms * 0.3  # rough estimate (single decode doesn't have real TTFT)

                results.append({
                    "completion_tokens": completion_tokens,
                    "total_ms": total_ms,
                    "ttft_ms": ttft_ms,
                })
            except Exception as e:
                errors += 1
                print(f"  [warn] request failed: {type(e).__name__}: {e}")

    # Warmup
    print(f"[{backend}] warming up ({warmup} requests)...")
    async with aiohttp.ClientSession() as session:
        warmup_tasks = [
            asyncio.create_task(
                _single_warmup(session, endpoint, make_payload(prompts[i % len(prompts)]))
            )
            for i in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks, return_exceptions=True)

    # Timed run
    print(f"[{backend}] benchmarking {concurrency} concurrent workers for {duration}s...")
    start_wall = time.perf_counter()
    deadline = start_wall + duration
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(session, deadline, i * (len(prompts) // concurrency + 1)))
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start_wall

    if not results:
        raise RuntimeError(f"All requests to {backend} failed! Check the server is running.")

    total_tokens = sum(r["completion_tokens"] for r in results)
    latencies = sorted(r["total_ms"] for r in results)
    ttfts = sorted(r["ttft_ms"] for r in results if r["ttft_ms"] is not None)

    def pct(arr: list[float], p: float) -> float | None:
        if not arr:
            return None
        k = (len(arr) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(arr):
            return arr[-1]
        return arr[f] + (k - f) * (arr[c] - arr[f])

    error_rate = 100.0 * errors / max(len(results) + errors, 1)
    tokens_per_second = total_tokens / elapsed
    requests_per_second = len(results) / elapsed
    report = {
        "status": "completed",
        "backend": backend,
        "model": model_id,
        "concurrency": concurrency,
        "duration_s": elapsed,
        "duration_seconds": elapsed,
        "total_requests": len(results) + errors,
        "successful_requests": len(results),
        "failed_requests": errors,
        "throughput_tokens_per_sec": tokens_per_second,
        "tokens_per_second": tokens_per_second,
        "requests_per_second": requests_per_second,
        "latency_p50_ms": pct(latencies, 50),
        "latency_p90_ms": pct(latencies, 90),
        "latency_p99_ms": pct(latencies, 99),
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p90_ms": pct(ttfts, 90),
        "ttft_p99_ms": pct(ttfts, 99),
        "gpu_peak_vram_mb": None,  # not measurable from client side
        "error_rate_pct": error_rate,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[{backend}] RESULTS saved to {output}")
    print(f"  throughput:  {report['throughput_tokens_per_sec']:.1f} tok/s")
    print(f"  latency P50: {report['latency_p50_ms']:.0f} ms")
    print(f"  latency P99: {report['latency_p99_ms']:.0f} ms")
    print(f"  TTFT P50:    {report['ttft_p50_ms']:.0f} ms")
    print(f"  error rate:  {error_rate:.1f}%")


async def _single_warmup(session: aiohttp.ClientSession, endpoint: str, payload: dict[str, Any]) -> None:
    try:
        async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            await resp.read()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backend URL defaults
# ---------------------------------------------------------------------------

BACKEND_URLS = {
    "vllm": "http://localhost:8001",
    "tgi": "http://localhost:8002",
    "custom": "http://localhost:8003",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="InferBench LLM benchmark runner")
    parser.add_argument("--backend", required=True, choices=["transformers", "vllm", "tgi", "custom"])
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup request count")
    parser.add_argument("--prompts", type=Path, default=Path("prompts/medium.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", type=str, default=None, help="Override backend URL (HTTP backends only)")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    if args.backend == "transformers":
        _run_transformers(
            model_id=args.model,
            prompts=prompts,
            duration=args.duration,
            warmup=args.warmup,
            output=args.output,
        )
    else:
        base_url = args.url or BACKEND_URLS[args.backend]
        asyncio.run(_run_http(
            backend=args.backend,
            model_id=args.model,
            base_url=base_url,
            prompts=prompts,
            duration=args.duration,
            warmup=args.warmup,
            concurrency=args.concurrency,
            output=args.output,
        ))


if __name__ == "__main__":
    main()

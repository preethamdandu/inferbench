"""
SDXL diffusion benchmark runner.

Runs image generation directly in-process (no gateway, no HTTP) using
the DiffusersBackend class. Saves results to JSON.

Usage:
    python scripts/run_bench_diffusion.py \
        --model stabilityai/stable-diffusion-xl-base-1.0 \
        --config fp16_baseline \
        --num-images 10 \
        --steps 30 \
        --output benchmarks/sdxl_fp16.json

Configs:
    fp16_baseline         - Standard FP16, no optimizations
    fp16_attention_slicing - FP16 + attention slicing (lower VRAM)
    fp16_compile          - FP16 + torch.compile (slower first run, faster steady state)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch


def run_sdxl_bench(
    model_id: str,
    config: str,
    num_images: int,
    steps: int,
    output: Path,
    warmup: int = 2,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SDXL benchmarks")

    from diffusers import StableDiffusionXLPipeline

    device = "cuda"
    dtype = torch.float16

    print(f"[sdxl] loading model {model_id} with config={config}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16"
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if "attention_slicing" in config:
        pipe.enable_attention_slicing()
        print("[sdxl] attention slicing enabled")

    if "compile" in config:
        print("[sdxl] torch.compile enabled — warm-up will be slow...")
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune")  # type: ignore

    test_prompts = [
        "A stunning mountain landscape at golden hour, photorealistic",
        "A cyberpunk city at night with neon lights reflecting on wet streets",
        "A serene Japanese garden with cherry blossoms and a koi pond",
        "An astronaut riding a horse on the moon, detailed digital art",
        "Abstract geometric patterns in vibrant colors, high resolution",
    ]

    # Warm up (to compile torch graph / populate caches)
    print(f"[sdxl] warming up ({warmup} images)...")
    for i in range(warmup):
        p = test_prompts[i % len(test_prompts)]
        with torch.no_grad():
            pipe(
                prompt=p,
                num_inference_steps=steps,
                height=1024,
                width=1024,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )
    torch.cuda.reset_peak_memory_stats()

    # Timed run
    print(f"[sdxl] benchmarking {num_images} images @ {steps} steps, 1024×1024...")
    timings: list[float] = []
    step_timings: list[float] = []

    for i in range(num_images):
        p = test_prompts[i % len(test_prompts)]
        t0 = time.perf_counter()
        with torch.no_grad():
            pipe(
                prompt=p,
                num_inference_steps=steps,
                height=1024,
                width=1024,
                generator=torch.Generator(device="cuda").manual_seed(i),
            )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        step_timings.append(elapsed / steps * 1000.0)  # ms per step
        print(f"  image {i + 1}/{num_images}: {elapsed:.2f}s ({elapsed / steps * 1000:.0f}ms/step)")

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    avg_time = sum(timings) / len(timings)
    avg_step_ms = sum(step_timings) / len(step_timings)

    def pct(arr: list[float], p: float) -> float:
        k = (len(arr) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(arr):
            return arr[-1]
        return arr[f] + (k - f) * (arr[c] - arr[f])

    sorted_times = sorted(timings)
    report = {
        "model": model_id,
        "config": config,
        "num_images": num_images,
        "steps": steps,
        "resolution": "1024x1024",
        "avg_time_s": round(avg_time, 3),
        "p50_time_s": round(pct(sorted_times, 50), 3),
        "p99_time_s": round(pct(sorted_times, 99), 3),
        "avg_step_time_ms": round(avg_step_ms, 1),
        "peak_vram_mb": round(peak_vram_mb, 0),
        "images_per_hour": round(3600 / avg_time, 1),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[sdxl] RESULTS saved to {output}")
    print(f"  avg time/image: {avg_time:.2f}s")
    print(f"  avg step time:  {avg_step_ms:.0f}ms")
    print(f"  VRAM peak:      {peak_vram_mb:.0f}MB")
    print(f"  images/hour:    {report['images_per_hour']:.0f}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="InferBench SDXL diffusion benchmark")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument(
        "--config",
        choices=["fp16_baseline", "fp16_attention_slicing", "fp16_compile"],
        required=True,
    )
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_sdxl_bench(
        model_id=args.model,
        config=args.config,
        num_images=args.num_images,
        steps=args.steps,
        output=args.output,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()

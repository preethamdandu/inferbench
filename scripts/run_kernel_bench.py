"""
Triton kernel benchmark runner.
Measures correctness + bandwidth for fused_softmax and fused_gelu vs PyTorch.
Outputs benchmarks/triton_kernels.json.

Usage:
    python scripts/run_kernel_bench.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def _bench_fn(fn, *args, warmup: int = 5, repeats: int = 20) -> float:
    """Return median wall-clock milliseconds for fn(*args)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]  # median


def run_softmax_bench(fused_softmax_fn) -> list[dict[str, Any]]:
    """Run softmax benchmark across (n_rows=2048, variable n_cols)."""
    n_rows = 2048
    col_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = []

    for n_cols in col_sizes:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

        pytorch_ms = _bench_fn(lambda: torch.softmax(x, dim=-1))
        triton_ms = _bench_fn(lambda: fused_softmax_fn(x))

        # Bandwidth: 2 × elements × 4 bytes (read input + write output)
        bytes_transferred = 2 * n_rows * n_cols * 4
        bandwidth_gb_s = bytes_transferred / (triton_ms * 1e-3) / 1e9

        result = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "pytorch_ms": round(pytorch_ms, 4),
            "triton_ms": round(triton_ms, 4),
            "speedup": round(pytorch_ms / triton_ms, 3),
            "bandwidth_gb_s": round(bandwidth_gb_s, 1),
        }
        results.append(result)
        print(
            f"  softmax ({n_rows:5d}, {n_cols:5d}): "
            f"PyTorch={pytorch_ms:.3f}ms  Triton={triton_ms:.3f}ms  "
            f"speedup={result['speedup']:.2f}×  BW={bandwidth_gb_s:.0f} GB/s"
        )

    return results


def run_gelu_bench(fused_gelu_fn) -> list[dict[str, Any]]:
    """Run GELU benchmark across variable element counts."""
    sizes = [1024, 8192, 65_536, 262_144, 1_048_576, 4_194_304]
    results = []

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        pytorch_ms = _bench_fn(lambda: torch.nn.functional.gelu(x, approximate="tanh"))
        triton_ms = _bench_fn(lambda: fused_gelu_fn(x))

        bytes_transferred = 2 * n * 4
        bandwidth_gb_s = bytes_transferred / (triton_ms * 1e-3) / 1e9

        result = {
            "n_elements": n,
            "pytorch_ms": round(pytorch_ms, 4),
            "triton_ms": round(triton_ms, 4),
            "speedup": round(pytorch_ms / triton_ms, 3),
            "bandwidth_gb_s": round(bandwidth_gb_s, 1),
        }
        results.append(result)
        print(
            f"  gelu n={n:>10d}: "
            f"PyTorch={pytorch_ms:.3f}ms  Triton={triton_ms:.3f}ms  "
            f"speedup={result['speedup']:.2f}×  BW={bandwidth_gb_s:.0f} GB/s"
        )

    return results


# GPU peak bandwidth lookup table (GB/s)
GPU_PEAK_BW: dict[str, float] = {
    "A100": 2039.0,       # SXM4 80GB variant
    "A100-SXM": 2039.0,
    "A100-PCIe": 1555.0,
    "A10G": 600.0,
    "A10": 600.0,
    "L4": 300.0,
    "H100": 3350.0,
    "H100 SXM": 3350.0,
    "RTX 4090": 1008.0,
    "RTX 3090": 936.0,
    "T4": 320.0,
    "Tesla T4": 320.0,
    "V100": 900.0,
}


def get_peak_bw(gpu_name: str) -> float:
    """Best-effort lookup of theoretical peak memory bandwidth for the detected GPU."""
    for k, v in GPU_PEAK_BW.items():
        if k.lower() in gpu_name.lower():
            return v
    # Unknown GPU — return a conservative estimate
    print(f"  [warn] Unknown GPU '{gpu_name}' — peak BW set to 0; update GPU_PEAK_BW dict")
    return 0.0


def run_correctness_check(fused_softmax_fn, fused_gelu_fn) -> dict[str, bool]:
    """Quick correctness gate before benchmarking."""
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)
    results: dict[str, bool] = {}

    # Softmax over several shapes
    softmax_pass = True
    for n_rows, n_cols in [(256, 1024), (512, 2048), (1024, 4096)]:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        out = fused_softmax_fn(x)
        max_diff = (ref - out).abs().max().item()
        ok = max_diff < 1e-4
        print(f"  softmax ({n_rows}, {n_cols}): max_diff={max_diff:.2e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            softmax_pass = False
    results["softmax"] = softmax_pass

    # GELU
    gelu_pass = True
    for n in [1024, 1_048_576]:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        ref = torch.nn.functional.gelu(x, approximate="tanh")
        out = fused_gelu_fn(x)
        max_diff = (ref - out).abs().max().item()
        ok = max_diff < 1e-4
        print(f"  gelu n={n:>10d}: max_diff={max_diff:.2e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            gelu_pass = False
    results["gelu"] = gelu_pass

    return results


def run_all_benchmarks() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Triton kernel benchmarks")

    from src.kernels.fused_softmax import fused_softmax as triton_softmax
    from src.kernels.fused_gelu import fused_gelu as triton_gelu

    gpu_name = torch.cuda.get_device_name(0)
    peak_bw = get_peak_bw(gpu_name)
    print(f"GPU: {gpu_name}")
    print(f"Peak memory bandwidth: {peak_bw:.0f} GB/s")
    print()

    correctness = run_correctness_check(triton_softmax, triton_gelu)
    if not all(correctness.values()):
        raise RuntimeError(f"Correctness failures: {correctness}. Fix kernels before benchmarking.")

    print()
    print("=" * 60)
    print("BENCHMARKING SOFTMAX")
    print("=" * 60)
    softmax_results = run_softmax_bench(triton_softmax)

    # Annotate with % of peak
    for r in softmax_results:
        r["bandwidth_util_pct"] = (
            round(r["bandwidth_gb_s"] / peak_bw * 100, 1) if peak_bw > 0 else None
        )

    print()
    print("=" * 60)
    print("BENCHMARKING GELU")
    print("=" * 60)
    gelu_results = run_gelu_bench(triton_gelu)

    for r in gelu_results:
        r["bandwidth_util_pct"] = (
            round(r["bandwidth_gb_s"] / peak_bw * 100, 1) if peak_bw > 0 else None
        )

    output = {
        "gpu": gpu_name,
        "peak_bandwidth_gb_s": peak_bw,
        "correctness": correctness,
        "softmax": softmax_results,
        "gelu": gelu_results,
    }

    out_path = Path("benchmarks/triton_kernels.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to {out_path}")
    print(json.dumps(output, indent=2)[:2000])


if __name__ == "__main__":
    run_all_benchmarks()

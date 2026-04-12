import torch

try:
    import triton  # type: ignore
    import triton.testing  # type: ignore
except ImportError as e:
    raise ImportError(
        "Triton is required for kernel benchmarks. Install with: pip install triton"
    ) from e

import structlog

from src.kernels.fused_gelu import fused_gelu
from src.kernels.fused_softmax import fused_softmax

logger = structlog.get_logger()


def do_softmax_correctness() -> None:
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = fused_softmax(x)
    y_torch = torch.softmax(x, dim=1)

    if torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-5):
        logger.info("Fused Softmax correctness check passed")
    else:
        max_diff = torch.max(torch.abs(y_triton - y_torch))
        raise AssertionError(f"Fused Softmax incorrect — max_diff={max_diff}")


def do_gelu_correctness() -> None:
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = fused_gelu(x)
    y_torch = torch.nn.functional.gelu(x)

    if torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-5):
        logger.info("Fused GELU correctness check passed")
    else:
        max_diff = torch.max(torch.abs(y_triton - y_torch))
        raise AssertionError(f"Fused GELU incorrect — max_diff={max_diff}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # Argument names to use as an x-axis for the plot.
        x_vals=[128 * i for i in range(2, 60)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "PyTorch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="softmax-performance",  # Name for the plot.
        args={"M": 4096},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark_softmax(M: int, N: int, provider: str) -> tuple[float, float, float]:
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
            lambda: torch.softmax(x, dim=-1), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_softmax(x), quantiles=quantiles)  # type: ignore

    def gbps(ms: float) -> float:
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(float(ms)), gbps(float(max_ms)), gbps(float(min_ms))  # type: ignore


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[1024 * i for i in range(2, 64)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="gelu-performance",
        args={},
    )
)
def benchmark_gelu(N: int, provider: str) -> tuple[float, float, float]:
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.gelu(x), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_gelu(x), quantiles=quantiles)  # type: ignore

    def gbps(ms: float) -> float:
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(float(ms)), gbps(float(max_ms)), gbps(float(min_ms))  # type: ignore


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, Triton kernels cannot run on CPU.")
        raise SystemExit(0)

    do_softmax_correctness()
    do_gelu_correctness()

    logger.info("Benchmarking Softmax...")
    benchmark_softmax.run(print_data=True, show_plots=False, save_path=".")  # type: ignore

    logger.info("Benchmarking GELU...")
    benchmark_gelu.run(print_data=True, show_plots=False, save_path=".")  # type: ignore

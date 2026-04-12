from dataclasses import dataclass, field
import time


@dataclass(frozen=True, slots=True)
class RequestMetric:
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float
    ttft_ms: float | None
    error: str | None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    backend: str
    model: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    prompt_tokens_total: int
    completion_tokens_total: int

    # Latency ms
    ttft_p50: float | None
    ttft_p90: float | None
    ttft_p95: float | None
    ttft_p99: float | None

    tpot_p50: float | None  # Time per Output Token
    tpot_p90: float | None

    # Throughput
    requests_per_second: float
    tokens_per_second: float


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of values using linear interpolation.

    Args:
        values: List of float values.
        p: Percentile from 0 to 100.

    Returns:
        Interpolated percentile value, or 0.0 for empty input.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

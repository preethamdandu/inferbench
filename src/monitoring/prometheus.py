from prometheus_client import Counter, Gauge, Histogram

# Counters
REQUESTS_TOTAL = Counter(
    "inferbench_requests_total",
    "Total number of inference requests",
    ["backend", "model", "status"],
)

TOKENS_TOTAL = Counter(
    "inferbench_tokens_total",
    "Total tokens generated",
    ["backend", "model", "type"],  # type="prompt" or "completion"
)

# Histograms
LATENCY_SECONDS = Histogram(
    "inferbench_request_duration_seconds",
    "Request latency in seconds",
    ["backend", "model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

TTFT_SECONDS = Histogram(
    "inferbench_ttft_seconds",
    "Time to first token in seconds",
    ["backend", "model"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")),
)

TPOT_SECONDS = Histogram(
    "inferbench_tpot_seconds",
    "Time per output token in seconds",
    ["backend", "model"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, float("inf")),
)

# Gauges
GPU_MEMORY_MB = Gauge(
    "inferbench_gpu_memory_mb",
    "Peak GPU Memory Allocated in MB during generation",
    ["backend", "model"],
)

# Canonical aliases expected by tests and gateway
REQUEST_LATENCY = LATENCY_SECONDS
TTFT = TTFT_SECONDS
TOKENS_GENERATED = TOKENS_TOTAL
GPU_MEMORY_USED = GPU_MEMORY_MB

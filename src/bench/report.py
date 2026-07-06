"""Generate markdown comparison reports from benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _first(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value for the given keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def normalize_llm_record(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize LLM benchmark JSON into a common schema for reporting.

    Accepts output from ``scripts/run_bench.py``, ``src/bench/runner.py``, and
    legacy field names.

    Args:
        data: Raw benchmark JSON object.

    Returns:
        Normalized record, or None if the file is a skip stub or not an LLM run.
    """
    if data.get("status") == "skipped":
        return None

    backend = data.get("backend")
    if backend is None and "config" not in data and "softmax" not in data:
        return None
    if backend is None:
        return None

    return {
        "backend": backend,
        "model": _first(data, "model", default="N/A"),
        "concurrency": _first(data, "concurrency", default="N/A"),
        "duration_seconds": _first(data, "duration_seconds", "duration_s"),
        "total_requests": _first(data, "total_requests", default=0),
        "successful_requests": _first(data, "successful_requests", default=0),
        "failed_requests": _first(data, "failed_requests", default=0),
        "tokens_per_second": _first(data, "tokens_per_second", "throughput_tokens_per_sec", default=0.0),
        "requests_per_second": _first(data, "requests_per_second", default=0.0),
        "latency_p50_ms": _first(data, "latency_p50_ms"),
        "latency_p90_ms": _first(data, "latency_p90_ms"),
        "latency_p99_ms": _first(data, "latency_p99_ms"),
        "ttft_p50_ms": _first(data, "ttft_p50_ms", "ttft_p50"),
        "ttft_p90_ms": _first(data, "ttft_p90_ms", "ttft_p90"),
        "ttft_p99_ms": _first(data, "ttft_p99_ms", "ttft_p99"),
        "tpot_p50_ms": _first(data, "tpot_p50_ms", "tpot_p50"),
        "tpot_p90_ms": _first(data, "tpot_p90_ms", "tpot_p90"),
        "gpu_peak_vram_mb": _first(data, "gpu_peak_vram_mb"),
        "error_rate_pct": _first(data, "error_rate_pct", default=0.0),
    }


def _fmt_ms(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


def _fmt_float(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


class ReportGenerator:
    """Wraps benchmark JSON files into markdown comparison reports."""

    def __init__(self, benchmarks_dir: Path) -> None:
        self.benchmarks_dir = benchmarks_dir

    def generate(self, output_file: Path) -> None:
        """Generate a markdown report from all JSON files in benchmarks_dir."""
        generate_markdown_report(self.benchmarks_dir, output_file)


def generate_markdown_report(benchmarks_dir: Path, output_file: Path) -> None:
    """Generate a markdown report from benchmark JSON artifacts.

    Args:
        benchmarks_dir: Directory containing ``*.json`` benchmark outputs.
        output_file: Path to write the markdown report.
    """
    skipped: list[dict[str, Any]] = []
    llm_records: list[dict[str, Any]] = []
    diffusion_records: list[dict[str, Any]] = []
    kernel_record: dict[str, Any] | None = None

    for file in sorted(benchmarks_dir.glob("*.json")):
        with open(file) as f:
            data = json.load(f)

        if data.get("status") == "skipped":
            skipped.append({"file": file.name, **data})
            continue

        if "softmax" in data and "gelu" in data:
            kernel_record = data
            continue

        if "config" in data and "avg_time_s" in data:
            diffusion_records.append(data)
            continue

        normalized = normalize_llm_record(data)
        if normalized is not None:
            llm_records.append(normalized)

    llm_records.sort(key=lambda row: (row["model"], row["backend"], row["requests_per_second"]))

    lines: list[str] = [
        "# InferBench Report",
        "",
        "> Generated from JSON artifacts in `benchmarks/`.",
        "> Re-run `make report` after collecting new benchmark results.",
        "",
    ]

    if skipped:
        lines.extend(["## Skipped Runs", ""])
        for entry in skipped:
            lines.append(f"- **{entry['file']}**: {entry.get('reason', 'skipped')}")
        lines.append("")

    if llm_records:
        lines.extend(
            [
                "## LLM Benchmarks",
                "",
                "| Model | Backend | Concurrency | RPS | Tokens/s | Latency P50 (ms) | Latency P99 (ms) | TTFT P50 (ms) | VRAM (MB) | Errors (%) |",
                "|-------|---------|-------------|-----|----------|------------------|------------------|---------------|-----------|------------|",
            ]
        )
        for row in llm_records:
            lines.append(
                f"| {row['model']} | {row['backend']} | {row['concurrency']} | "
                f"{_fmt_float(row['requests_per_second'])} | {_fmt_float(row['tokens_per_second'], 1)} | "
                f"{_fmt_ms(row['latency_p50_ms'])} | {_fmt_ms(row['latency_p99_ms'])} | "
                f"{_fmt_ms(row['ttft_p50_ms'])} | "
                f"{_fmt_float(row['gpu_peak_vram_mb'], 0) if row['gpu_peak_vram_mb'] is not None else 'N/A'} | "
                f"{_fmt_float(row['error_rate_pct'], 2)} |"
            )
        lines.append("")

        lines.extend(["## LLM Details", ""])
        for row in llm_records:
            lines.append(f"### {row['backend']} — {row['model']}")
            for key, value in row.items():
                if key in {"backend", "model"}:
                    continue
                if isinstance(value, float):
                    lines.append(f"- **{key}**: {value:.2f}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
    else:
        lines.extend(
            [
                "## LLM Benchmarks",
                "",
                "_No completed LLM benchmark artifacts found. Run `scripts/run_bench.py` on a GPU machine and save JSON to `benchmarks/`._",
                "",
            ]
        )

    if diffusion_records:
        lines.extend(
            [
                "## Image Generation (SDXL)",
                "",
                "| Config | Avg Time/Image (s) | Avg Step (ms) | Peak VRAM (MB) | Images/hr |",
                "|--------|-------------------|---------------|----------------|-----------|",
            ]
        )
        for row in sorted(diffusion_records, key=lambda item: item.get("config", "")):
            lines.append(
                f"| {row.get('config', 'N/A')} | {row.get('avg_time_s', 'N/A')} | "
                f"{row.get('avg_step_time_ms', 'N/A')} | {row.get('peak_vram_mb', 'N/A')} | "
                f"{row.get('images_per_hour', 'N/A')} |"
            )
        lines.append("")

    if kernel_record is not None:
        lines.extend(["## Triton Kernels", ""])
        lines.append(f"- **GPU**: {kernel_record.get('gpu', 'N/A')}")
        lines.append(f"- **Peak bandwidth**: {kernel_record.get('peak_bandwidth_gb_s', 'N/A')} GB/s")
        lines.append("")

        softmax_rows = kernel_record.get("softmax", [])
        if softmax_rows:
            lines.extend(
                [
                    "### Fused Softmax",
                    "",
                    "| N Cols | PyTorch (ms) | Triton (ms) | Speedup | Bandwidth (GB/s) | % Peak |",
                    "|--------|--------------|-------------|---------|------------------|--------|",
                ]
            )
            for row in softmax_rows:
                lines.append(
                    f"| {row.get('n_cols', 'N/A')} | {row.get('pytorch_ms', 'N/A')} | "
                    f"{row.get('triton_ms', 'N/A')} | {row.get('speedup', 'N/A')} | "
                    f"{row.get('bandwidth_gb_s', 'N/A')} | {row.get('bandwidth_util_pct', 'N/A')} |"
                )
            lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate InferBench markdown report")
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory containing benchmark JSON artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/report.md"),
        help="Output markdown file",
    )
    args = parser.parse_args()
    generate_markdown_report(args.benchmarks_dir, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()

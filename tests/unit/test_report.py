from src.bench.report import generate_markdown_report, normalize_llm_record


def test_normalize_llm_record_from_run_bench_schema(tmp_path) -> None:
    data = {
        "backend": "custom",
        "model": "mistralai/Mistral-7B-v0.1",
        "concurrency": 512,
        "duration_s": 60.0,
        "throughput_tokens_per_sec": 891.2,
        "requests_per_second": 4.5,
        "latency_p50_ms": 2100.0,
        "latency_p99_ms": 4800.0,
        "ttft_p50_ms": 118.0,
        "gpu_peak_vram_mb": 16100.0,
        "error_rate_pct": 0.0,
    }

    normalized = normalize_llm_record(data)

    assert normalized is not None
    assert normalized["tokens_per_second"] == 891.2
    assert normalized["duration_seconds"] == 60.0
    assert normalized["latency_p99_ms"] == 4800.0


def test_normalize_llm_record_skips_stub() -> None:
    assert normalize_llm_record({"status": "skipped", "reason": "no gpu"}) is None


def test_generate_markdown_report_lists_skipped_and_completed(tmp_path) -> None:
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()

    (benchmarks_dir / "custom.json").write_text(
        """{
  "backend": "custom",
  "model": "google/gemma-2-2b-it",
  "concurrency": 4,
  "duration_s": 45.0,
  "throughput_tokens_per_sec": 120.5,
  "requests_per_second": 1.2,
  "latency_p50_ms": 800.0,
  "latency_p99_ms": 1200.0,
  "ttft_p50_ms": 90.0,
  "gpu_peak_vram_mb": 4200.0,
  "error_rate_pct": 0.0
}"""
    )
    (benchmarks_dir / "tgi_fp16.json").write_text(
        '{"status": "skipped", "reason": "Docker unavailable"}'
    )

    output = tmp_path / "report.md"
    generate_markdown_report(benchmarks_dir, output)
    text = output.read_text()

    assert "custom" in text
    assert "120.5" in text
    assert "Docker unavailable" in text
    assert "Skipped Runs" in text

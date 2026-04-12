import json
from pathlib import Path


class ReportGenerator:
    """Wraps benchmark JSON files into markdown comparison reports."""

    def __init__(self, benchmarks_dir: Path) -> None:
        self.benchmarks_dir = benchmarks_dir

    def generate(self, output_file: Path) -> None:
        """Generate a markdown report from all JSON files in benchmarks_dir."""
        generate_markdown_report(self.benchmarks_dir, output_file)


def generate_markdown_report(benchmarks_dir: Path, output_file: Path) -> None:
    reports = []
    for file in benchmarks_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            reports.append(data)

    reports.sort(
        key=lambda x: (x.get("model", ""), x.get("backend", ""), x.get("requests_per_second", 0))
    )

    with open(output_file, "w") as f:
        f.write("# Benchmarking Report\n\n")

        f.write("## Overview\n")
        f.write("| Model | Backend | RPS | Tokens/s | TTFT p90 (ms) | TPOT p90 (ms) |\n")
        f.write("|-------|---------|-----|----------|----------------|----------------|\n")

        for r in reports:
            ttft = f"{r['ttft_p90']:.2f}" if r.get("ttft_p90") else "N/A"
            tpot = f"{r['tpot_p90']:.2f}" if r.get("tpot_p90") else "N/A"
            f.write(
                f"| {r.get('model', 'N/A')} | {r.get('backend', 'N/A')} | "
                f"{r.get('requests_per_second', 0):.2f} | {r.get('tokens_per_second', 0):.2f} | "
                f"{ttft} | {tpot} |\n"
            )

        f.write("\n## Details\n")
        for r in reports:
            f.write(f"### {r.get('backend', 'N/A')} ({r.get('model', 'N/A')})\n")
            for k, v in r.items():
                if isinstance(v, float):
                    f.write(f"- **{k}**: {v:.2f}\n")
                else:
                    f.write(f"- **{k}**: {v}\n")
            f.write("\n")

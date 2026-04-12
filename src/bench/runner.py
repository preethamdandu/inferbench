import asyncio
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import aiohttp
import structlog

from src.bench.metrics import BenchmarkReport, RequestMetric
from src.config import settings

logger = structlog.get_logger()


class BenchmarkRunner:
    def __init__(
        self, backend: str, model: str, duration: int, concurrency: int, prompt_file: Path
    ):
        self.backend = backend
        self.model = model
        self.duration = duration
        self.concurrency = concurrency
        self.prompt_file = prompt_file
        self.gateway_url = f"http://{settings.gateway_host}:{settings.gateway_port}"

        self.metrics: list[RequestMetric] = []
        self.prompts: list[dict[str, Any]] = []

    def _load_prompts(self) -> None:
        with open(self.prompt_file) as f:
            for line in f:
                if line.strip():
                    self.prompts.append(json.loads(line))

    async def _worker(
        self, session: aiohttp.ClientSession, queue: asyncio.Queue[dict[str, Any]], end_time: float
    ) -> None:
        while time.time() < end_time:
            try:
                # Get a prompt
                prompt_data = queue.get_nowait()
                queue.task_done()
                # Re-add for continuous looping
                queue.put_nowait(prompt_data)
            except asyncio.QueueEmpty:
                break

            payload = {
                "model": self.model,
                "prompt": str(prompt_data["prompt"]),
                "max_tokens": int(prompt_data.get("max_tokens", 128)),
                "stream": True,
            }

            try:
                start = time.time_ns()

                ttft_ms: float | None = None
                completion_tokens = 0

                async with session.post(
                    f"{self.gateway_url}/v1/completions?backend={self.backend}", json=payload
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.content:
                        line_str = line.decode("utf-8").strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            if ttft_ms is None:
                                ttft_ms = (time.time_ns() - start) / 1e6

                            completion_tokens += 1

                total_time_ms = (time.time_ns() - start) / 1e6
                approx_prompt_tokens = int(len(str(prompt_data["prompt"]).split()) * 1.3)

                metric = RequestMetric(
                    prompt_tokens=approx_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_time_ms=total_time_ms,
                    ttft_ms=ttft_ms,
                    error=None,
                )
                self.metrics.append(metric)

            except Exception as e:
                self.metrics.append(
                    RequestMetric(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_time_ms=0.0,
                        ttft_ms=None,
                        error=str(e),
                    )
                )

    async def run(self) -> BenchmarkReport:
        self._load_prompts()

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for p in self.prompts:
            queue.put_nowait(p)

        logger.info("Starting warmup")
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + 5
            tasks = [asyncio.create_task(self._worker(session, queue, end_time)) for _ in range(2)]
            await asyncio.gather(*tasks)

        self.metrics.clear()

        logger.info("Starting benchmark", concurrency=self.concurrency, duration=self.duration)
        async with AsyncExitStack() as stack:
            session = await stack.enter_async_context(aiohttp.ClientSession())

            end_time = time.time() + self.duration

            start_wall = time.time()
            tasks = [
                asyncio.create_task(self._worker(session, queue, end_time))
                for _ in range(self.concurrency)
            ]
            await asyncio.gather(*tasks)

            end_wall = time.time()

        successful = [m for m in self.metrics if m.error is None]
        failed = len(self.metrics) - len(successful)

        if not successful:
            raise RuntimeError("All benchmark requests failed!")

        ttfts = sorted([m.ttft_ms for m in successful if m.ttft_ms is not None])
        tpots = sorted(
            [
                (m.total_time_ms - (m.ttft_ms or 0.0)) / max(m.completion_tokens, 1)
                for m in successful
            ]
        )

        total_time = end_wall - start_wall
        total_output_tokens = sum(m.completion_tokens for m in successful)

        def pct(arr: list[float], p: float) -> float | None:
            if not arr:
                return None
            k = (len(arr) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(arr):
                return arr[-1]
            return arr[f] + (k - f) * (arr[c] - arr[f])

        report = BenchmarkReport(
            backend=self.backend,
            model=self.model,
            duration_seconds=total_time,
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=failed,
            prompt_tokens_total=sum(m.prompt_tokens for m in successful),
            completion_tokens_total=total_output_tokens,
            ttft_p50=pct(ttfts, 0.5),
            ttft_p90=pct(ttfts, 0.9),
            ttft_p95=pct(ttfts, 0.95),
            ttft_p99=pct(ttfts, 0.99),
            tpot_p50=pct(tpots, 0.5),
            tpot_p90=pct(tpots, 0.9),
            requests_per_second=len(successful) / total_time,
            tokens_per_second=total_output_tokens / total_time,
        )

        out_dir = Path("benchmarks")
        out_dir.mkdir(exist_ok=True)
        import dataclasses

        with open(out_dir / f"{self.backend}_{self.concurrency}.json", "w") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)

        return report

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


class DiffusionBenchmarkRunner:
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
        if not self.prompt_file.exists():
            self.prompts = [{"prompt": "A scenic mountain landscape", "num_inference_steps": 20}]
            return

        with open(self.prompt_file) as f:
            for line in f:
                if line.strip():
                    self.prompts.append(json.loads(line))

    async def _worker(
        self, session: aiohttp.ClientSession, queue: asyncio.Queue[dict[str, Any]], end_time: float
    ) -> None:
        while time.time() < end_time:
            try:
                prompt_data = queue.get_nowait()
                queue.task_done()
                queue.put_nowait(prompt_data)
            except asyncio.QueueEmpty:
                break

            payload = {
                "prompt": str(prompt_data["prompt"]),
                "num_inference_steps": int(prompt_data.get("num_inference_steps", 30)),
            }

            try:
                start = time.time_ns()

                async with session.post(
                    f"{self.gateway_url}/v1/images/generations", json=payload
                ) as resp:
                    resp.raise_for_status()
                    await resp.json()  # Consume payload

                total_time_ms = (time.time_ns() - start) / 1e6

                # For diffusion, we log steps in completion_tokens
                metric = RequestMetric(
                    prompt_tokens=0,
                    completion_tokens=int(payload["num_inference_steps"]),
                    total_time_ms=total_time_ms,
                    ttft_ms=None,
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

        total_time = end_wall - start_wall
        total_steps = sum(m.completion_tokens for m in successful)

        def pct(arr: list[float], p: float) -> float | None:
            if not arr:
                return None
            return arr[int(len(arr) * p)]

        report = BenchmarkReport(
            backend=self.backend,
            model=self.model,
            duration_seconds=total_time,
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=failed,
            prompt_tokens_total=0,
            completion_tokens_total=total_steps,
            ttft_p50=None,
            ttft_p90=None,
            ttft_p95=None,
            ttft_p99=None,
            tpot_p50=None,
            tpot_p90=None,
            requests_per_second=len(successful) / total_time,
            tokens_per_second=total_steps / total_time,  # steps/s
        )

        out_dir = Path("benchmarks")
        out_dir.mkdir(exist_ok=True)
        import dataclasses

        with open(out_dir / f"{self.backend}_{self.concurrency}_diff.json", "w") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)

        return report

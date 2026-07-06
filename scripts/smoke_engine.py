"""CPU smoke test for the custom engine — reproduces Colab hang locally.

Runs the full scheduler + engine loop against a tiny random model so we can
verify prefill/decode work on the installed transformers version without a GPU.

Usage:
    python scripts/smoke_engine.py
"""
from __future__ import annotations

import asyncio
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.backends.custom.engine import CustomInferenceEngine
from src.backends.custom.kv_cache import KVCacheManager
from src.backends.custom.scheduler import ContinuousBatchScheduler, SequenceRequest

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


async def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    kv_cache = KVCacheManager(
        num_blocks=64,
        num_layers=getattr(model.config, "num_hidden_layers", 2),
        block_size=16,
        num_heads=getattr(model.config, "num_attention_heads", 2),
        head_dim=getattr(model.config, "hidden_size", 32)
        // getattr(model.config, "num_attention_heads", 2),
        device="cpu",
        dtype=torch.float32,
    )
    scheduler = ContinuousBatchScheduler(kv_cache=kv_cache, max_batch_size=4)
    engine = CustomInferenceEngine(model, tokenizer, scheduler)
    await engine.start()

    loop = asyncio.get_running_loop()

    async def one_request(prompt: str, max_tokens: int) -> dict:
        req = SequenceRequest(
            request_id=f"smoke-{prompt[:8]}",
            prompt_token_ids=tokenizer.encode(prompt),
            max_tokens=max_tokens,
            temperature=0.0,
            future=loop.create_future(),
        )
        scheduler.add_request(req)
        return await asyncio.wait_for(req.future, timeout=60.0)

    # Single request
    r1 = await one_request("Hello world", 8)
    assert r1["completion_tokens"] == 8, r1
    print(f"single request OK: {r1['completion_tokens']} tokens in {r1['total_time_ms']:.0f}ms")

    # Concurrent requests (exercises batched decode with mixed lengths)
    results = await asyncio.gather(
        one_request("The quick brown fox", 6),
        one_request("A", 10),
        one_request("Testing one two three four five", 4),
    )
    for r in results:
        print(f"concurrent OK: {r['completion_tokens']} tokens in {r['total_time_ms']:.0f}ms")

    await engine.stop()
    print("engine smoke test passed")
    return 0


def server_roundtrip() -> None:
    """Boot the real FastAPI server (tiny model, CPU) and hit /v1/completions."""
    import os

    os.environ["LLM_MODEL_ID"] = MODEL_ID
    os.environ["KV_CACHE_BLOCKS"] = "64"
    os.environ["MAX_BATCH_SIZE"] = "4"

    from fastapi.testclient import TestClient

    from src.backends.custom import server as srv

    with TestClient(srv.app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200, health.text

        resp = client.post(
            "/v1/completions",
            json={"model": MODEL_ID, "prompt": "Hello", "max_tokens": 5, "temperature": 0.0},
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["usage"]["completion_tokens"] == 5, data
        assert data["timing"]["total_ms"] > 0, data
    print("server roundtrip passed")


if __name__ == "__main__":
    rc = asyncio.run(main())
    server_roundtrip()
    print("SMOKE TEST PASSED")
    sys.exit(rc)

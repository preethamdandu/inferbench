"""Tests for ContinuousBatchScheduler — CPU-only, no GPU required."""

import asyncio
from src.backends.custom.scheduler import ContinuousBatchScheduler, SequenceRequest, RequestState
from src.backends.custom.kv_cache import KVCacheManager


def make_pool(num_blocks: int = 20) -> KVCacheManager:
    return KVCacheManager(num_blocks, 16, 32, 1, 1, device="cpu")


def make_req(
    req_id: str,
    prompt_len: int = 5,
    max_tokens: int = 10,
    loop: asyncio.AbstractEventLoop | None = None,
) -> SequenceRequest:
    lp = loop or asyncio.new_event_loop()
    return SequenceRequest(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        future=lp.create_future(),
    )


def test_add_request():
    pool = make_pool()
    scheduler = ContinuousBatchScheduler(kv_cache=pool, max_batch_size=4)
    req = make_req("r1")
    scheduler.add_request(req)
    assert len(scheduler.waiting) == 1
    assert req.state == RequestState.WAITING


def test_schedule_admits_up_to_max_batch():
    pool = make_pool(100)
    scheduler = ContinuousBatchScheduler(kv_cache=pool, max_batch_size=2)
    loop = asyncio.new_event_loop()
    r1 = make_req("r1", loop=loop)
    r2 = make_req("r2", loop=loop)
    r3 = make_req("r3", loop=loop)
    for r in [r1, r2, r3]:
        scheduler.add_request(r)

    batch = scheduler.schedule()
    ids = [r.request_id for r in batch]
    assert "r1" in ids
    assert "r2" in ids
    assert "r3" not in ids
    assert r3.state == RequestState.WAITING
    loop.close()


def test_schedule_evicts_finished_request():
    pool = make_pool(100)
    scheduler = ContinuousBatchScheduler(kv_cache=pool, max_batch_size=4)
    loop = asyncio.new_event_loop()
    r1 = make_req("r1", max_tokens=2, loop=loop)
    r2 = make_req("r2", loop=loop)
    scheduler.add_request(r1)
    scheduler.add_request(r2)

    scheduler.schedule()
    assert r1.state == RequestState.RUNNING
    assert r2.state == RequestState.RUNNING

    # Finish r1
    r1.generated_token_ids.extend([1, 2])
    batch2 = scheduler.schedule()
    ids2 = [r.request_id for r in batch2]
    assert "r1" not in ids2
    assert r1.state == RequestState.COMPLETED
    loop.close()


def test_schedule_resolves_future_on_completion():
    pool = make_pool(100)
    scheduler = ContinuousBatchScheduler(kv_cache=pool, max_batch_size=4)
    loop = asyncio.new_event_loop()
    req = make_req("r1", max_tokens=1, loop=loop)
    scheduler.add_request(req)
    scheduler.schedule()
    assert not req.future.done()
    req.generated_token_ids.append(42)
    scheduler.schedule()
    assert req.is_finished
    assert req.state == RequestState.COMPLETED
    loop.close()


def test_schedule_rejects_when_kv_cache_full():
    pool = make_pool(1)  # only 1 block = can't fit max_tokens=100
    scheduler = ContinuousBatchScheduler(kv_cache=pool, max_batch_size=4)
    loop = asyncio.new_event_loop()
    big = make_req("big", prompt_len=5, max_tokens=100, loop=loop)
    scheduler.add_request(big)
    batch = scheduler.schedule()
    assert len(batch) == 0
    assert big.state == RequestState.WAITING
    loop.close()


def test_request_state_initial():
    loop = asyncio.new_event_loop()
    req = SequenceRequest(
        request_id="x", prompt_token_ids=[1, 2], max_tokens=5, future=loop.create_future()
    )
    assert req.state == RequestState.WAITING
    assert not req.is_finished
    loop.close()


def test_request_finished_on_max_tokens():
    loop = asyncio.new_event_loop()
    req = SequenceRequest(
        request_id="x", prompt_token_ids=[1], max_tokens=3, future=loop.create_future()
    )
    req.generated_token_ids.extend([10, 20, 30])
    assert req.is_finished
    loop.close()


def test_request_finished_on_eos():
    loop = asyncio.new_event_loop()
    req = SequenceRequest(
        request_id="x", prompt_token_ids=[1], max_tokens=100, future=loop.create_future()
    )
    req.generated_token_ids.append(2)  # EOS token = 2
    assert req.is_finished
    loop.close()

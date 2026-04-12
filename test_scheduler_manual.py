"""
Manual scheduler verification. Tests the core scheduling logic without GPU.
"""

import asyncio
from src.backends.custom.scheduler import ContinuousBatchScheduler, SequenceRequest, RequestState
from src.backends.custom.kv_cache import KVCachePool
import sys

errors = []

# --- Test 1: KVCachePool allocation and freeing ---
print("TEST 1: KVCachePool basic operations")
try:
    import torch

    pool = KVCachePool(
        num_blocks=10,
        block_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=64,
        device=torch.device("cpu"),
    )
    assert pool.can_allocate(5), "Should be able to allocate 5 blocks"
    assert pool.can_allocate(10), "Should be able to allocate 10 blocks"
    assert not pool.can_allocate(11), "Should NOT be able to allocate 11 blocks"

    blocks = pool.allocate(3)
    assert len(blocks) == 3, f"Expected 3 blocks, got {len(blocks)}"
    assert pool.utilization == 0.3, f"Expected 0.3 utilization, got {pool.utilization}"
    assert pool.can_allocate(7), "Should be able to allocate 7 more"
    assert not pool.can_allocate(8), "Should NOT be able to allocate 8 more"

    pool.free(blocks)
    assert pool.utilization == 0.0, f"Expected 0.0 utilization after free, got {pool.utilization}"
    assert pool.can_allocate(10), "Should be able to allocate 10 after freeing"

    # Test OOM
    pool.allocate(10)
    try:
        pool.allocate(1)
        errors.append("TEST 1: Should have raised RuntimeError on over-allocation")
    except RuntimeError:
        pass  # expected

    print("  PASSED")
except Exception as e:
    errors.append(f"TEST 1: {e}")
    print(f"  FAILED: {e}")

# --- Test 2: Scheduler add and schedule ---
print("TEST 2: Scheduler add_request and schedule")
try:
    import torch

    pool2 = KVCachePool(
        num_blocks=20,
        block_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=64,
        device=torch.device("cpu"),
    )
    scheduler = ContinuousBatchScheduler(max_batch_size=2, kv_cache=pool2)

    loop = asyncio.new_event_loop()

    req1 = SequenceRequest(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4, 5],
        max_tokens=3,
        future=loop.create_future(),
    )
    req2 = SequenceRequest(
        request_id="req-2",
        prompt_token_ids=[10, 20, 30],
        max_tokens=2,
        future=loop.create_future(),
    )
    req3 = SequenceRequest(
        request_id="req-3",
        prompt_token_ids=[100, 200],
        max_tokens=5,
        future=loop.create_future(),
    )

    scheduler.add_request(req1)
    scheduler.add_request(req2)
    scheduler.add_request(req3)

    # First schedule: should admit req1 and req2 (max_batch_size=2), req3 waits
    batch = scheduler.schedule()
    batch_ids = [r.request_id for r in batch]
    assert "req-1" in batch_ids, f"req-1 should be in batch, got {batch_ids}"
    assert "req-2" in batch_ids, f"req-2 should be in batch, got {batch_ids}"
    assert "req-3" not in batch_ids, f"req-3 should NOT be in batch yet, got {batch_ids}"
    assert req1.state == RequestState.RUNNING, f"req1 state should be RUNNING, got {req1.state}"
    assert req3.state == RequestState.WAITING, f"req3 state should be WAITING, got {req3.state}"
    print("  Schedule 1: PASSED")

    # Simulate req2 finishing (generate enough tokens)
    for i in range(req2.max_tokens):
        req2.generated_token_ids.append(999)

    # Second schedule: should evict req2, admit req3
    batch = scheduler.schedule()
    batch_ids = [r.request_id for r in batch]
    assert "req-1" in batch_ids, f"req-1 should still be running, got {batch_ids}"
    assert "req-2" not in batch_ids, f"req-2 should be evicted, got {batch_ids}"
    assert "req-3" in batch_ids, f"req-3 should be admitted now, got {batch_ids}"
    assert req2.state == RequestState.COMPLETED, f"req2 state should be COMPLETED, got {req2.state}"

    # Verify req2's future was resolved
    assert req2.future.done(), "req2's future should be resolved after completion"

    print("  Schedule 2: PASSED")
    loop.close()
    print("  PASSED")
except Exception as e:
    errors.append(f"TEST 2: {e}")
    print(f"  FAILED: {e}")

# --- Test 3: Scheduler respects KV-cache capacity ---
print("TEST 3: Scheduler KV-cache capacity enforcement")
try:
    import torch

    # Tiny pool: only 2 blocks of 16 tokens each = 32 tokens max
    pool3 = KVCachePool(
        num_blocks=2,
        block_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=64,
        device=torch.device("cpu"),
    )
    scheduler3 = ContinuousBatchScheduler(max_batch_size=10, kv_cache=pool3)

    loop3 = asyncio.new_event_loop()

    # This request needs ceil((5 + 100) / 16) = 7 blocks — more than available
    big_req = SequenceRequest(
        request_id="big",
        prompt_token_ids=list(range(5)),
        max_tokens=100,
        future=loop3.create_future(),
    )
    scheduler3.add_request(big_req)
    batch = scheduler3.schedule()

    # Should NOT be admitted because KV-cache can't fit it
    assert len(batch) == 0, f"Big request should not be admitted, batch has {len(batch)} items"
    assert big_req.state == RequestState.WAITING, (
        f"Big request should still be WAITING, got {big_req.state}"
    )

    loop3.close()
    print("  PASSED")
except Exception as e:
    errors.append(f"TEST 3: {e}")
    print(f"  FAILED: {e}")

# --- Summary ---
print(f"\n{'=' * 50}")
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL SCHEDULER TESTS PASSED")
    sys.exit(0)

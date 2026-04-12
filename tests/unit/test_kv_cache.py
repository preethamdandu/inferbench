"""Tests for KVCacheManager."""

import pytest
from src.backends.custom.kv_cache import KVCacheManager


def make_cache(num_blocks: int = 10) -> KVCacheManager:
    return KVCacheManager(num_blocks, 16, 32, 1, 1, device="cpu")


def test_initial_state():
    kv = make_cache(10)
    assert kv.utilization == 0.0
    assert kv.can_allocate(10)
    assert not kv.can_allocate(11)


def test_kv_cache_basic():
    kv = make_cache(10)
    blocks = kv.allocate(1)
    assert len(blocks) == 1
    assert kv.utilization == 0.1
    kv.free(blocks)
    assert kv.utilization == 0.0


def test_kv_cache_multiple_allocations():
    kv = make_cache(10)
    kv.allocate(3)
    kv.allocate(4)
    assert kv.utilization == pytest.approx(0.7)
    assert not kv.can_allocate(4)
    assert kv.can_allocate(3)


def test_kv_cache_free_restores_capacity():
    kv = make_cache(10)
    blocks = kv.allocate(5)
    assert kv.utilization == 0.5
    kv.free(blocks)
    assert kv.utilization == 0.0
    assert kv.can_allocate(10)


def test_kv_cache_oom():
    kv = make_cache(2)
    kv.allocate(2)
    with pytest.raises(RuntimeError):
        kv.allocate(1)


def test_kv_cache_partial_free():
    kv = make_cache(10)
    blocks = kv.allocate(5)
    kv.free(blocks[:2])
    assert kv.can_allocate(7)
    assert not kv.can_allocate(8)


def test_kv_cache_double_free_is_safe():
    """Freeing already-free blocks should not raise."""
    kv = make_cache(10)
    blocks = kv.allocate(2)
    kv.free(blocks)
    kv.free(blocks)  # already freed — should not raise
    assert kv.utilization == 0.0

"""Tests for Triton kernels — skipped automatically when GPU/triton unavailable."""

import pytest
import torch

try:
    from src.kernels.bench_kernels import do_softmax_correctness, do_gelu_correctness

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not torch.cuda.is_available(),
    reason="Triton kernels require CUDA GPU and triton package (Linux only)",
)


def test_triton_softmax() -> None:
    do_softmax_correctness()


def test_triton_gelu() -> None:
    do_gelu_correctness()

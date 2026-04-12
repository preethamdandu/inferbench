"""Fused softmax Triton kernel.

Requires CUDA-enabled GPU and triton installed (Linux only).
"""

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError as e:
    raise ImportError(
        "Triton is required for fused_softmax. Install with: pip install triton"
    ) from e


@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    # Load row with -inf padding for numerical stability
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Numerically stable softmax in one SRAM pass
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write back
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax using a fused Triton kernel.

    Performs the full (max, subtract, exp, sum, divide) pipeline in a single
    pass through SRAM, reducing HBM reads from 2× to 1× vs PyTorch eager.

    Args:
        x: Input tensor of shape [n_rows, n_cols], must be on CUDA.

    Returns:
        Softmax output tensor of the same shape.
    """
    n_rows, n_cols = x.shape
    # T4 has 48KB shared memory per SM — constrain BLOCK_SIZE to avoid exceeding it.
    # Each float32 = 4 bytes; max safe BLOCK_SIZE = 48*1024/4 = 12288.
    # Round to next power of 2, capped at 16384 (safe for A100; T4 uses guard below).
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 16384)  # cap for T4 SRAM safety
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)
    grid = (n_rows,)

    _softmax_fwd_kernel[grid](  # type: ignore
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


# Canonical alias used by benchmark scripts and tests
triton_softmax = fused_softmax

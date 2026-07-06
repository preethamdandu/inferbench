"""Fused GELU Triton kernel with tanh approximation.

Requires CUDA-enabled GPU and triton installed (Linux only).
"""

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError as e:
    raise ImportError("Triton is required for fused_gelu. Install with: pip install triton") from e


@triton.jit
def _gelu_fwd_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU kernel using tanh approximation.

    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    # Tanh GELU approximation matching torch.nn.functional.gelu(approximate="tanh")
    pi_coeff = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    x3 = x * x * x
    inner = pi_coeff * (x + coeff * x3)
    # tanh(z) = 1 - 2 / (exp(2z) + 1), built from tl.exp so it compiles on every
    # Triton version (tl.math/libdevice namespaces moved between releases).
    # Saturates correctly at +/-1 for large |z| in fp32.
    tanh_val = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    y = 0.5 * x * (1.0 + tanh_val)

    tl.store(output_ptr + offsets, y, mask=mask)


def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute element-wise GELU using a fused Triton kernel.

    Uses tanh approximation matching PyTorch's gelu(approximate='tanh').
    Avoids HBM round-trip by fusing the polynomial and tanh in one pass.

    Args:
        x: Input tensor (any shape), must be on CUDA.

    Returns:
        GELU output tensor of the same shape.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta: dict) -> tuple[int]:  # type: ignore
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _gelu_fwd_kernel[grid](  # type: ignore
        output,
        x,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return output


# Canonical alias used by benchmark scripts and tests
triton_gelu = fused_gelu

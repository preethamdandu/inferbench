import torch


class KVCacheManager:
    """
    Block-based continuous KV-cache manager.

    This is simpler than vLLM's PagedAttention. It preallocates a contiguous
    block pool and manages it using a free list, but without full virtual-to-physical
    mapping inside custom kernels. Instead, we use it to allocate cache capacities
    for continuous batching sequences.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device

        # [num_blocks, 2 (key, value), num_layers, block_size, num_heads, head_dim]
        # In a real setup, often organized differently for attention kernel efficiency.
        self.cache = torch.zeros(
            (num_blocks, 2, num_layers, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        self.free_blocks = list(range(num_blocks))
        self.used_blocks = set[int]()

    @property
    def utilization(self) -> float:
        """Fraction of used blocks (0.0 to 1.0)."""
        if self.num_blocks == 0:
            return 0.0
        return len(self.used_blocks) / self.num_blocks

    def can_allocate(self, n: int) -> bool:
        """Check if n blocks can be allocated."""
        return len(self.free_blocks) >= n

    def allocate(self, n: int) -> list[int]:
        """Allocate n blocks from the free list."""
        if not self.can_allocate(n):
            raise RuntimeError(f"Cannot allocate {n} blocks. Only {len(self.free_blocks)} free.")

        allocated = [self.free_blocks.pop() for _ in range(n)]
        self.used_blocks.update(allocated)
        return allocated

    def free(self, block_ids: list[int]) -> None:
        """Free previously allocated blocks."""
        for block_id in block_ids:
            if block_id in self.used_blocks:
                self.used_blocks.remove(block_id)
                self.free_blocks.append(block_id)


# Alias for backward-compatible imports
KVCachePool = KVCacheManager

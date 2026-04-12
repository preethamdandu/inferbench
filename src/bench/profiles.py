from collections.abc import AsyncIterator


async def steady(concurrency: int) -> AsyncIterator[int]:
    """Yields a constant target concurrency."""
    while True:
        yield concurrency


async def ramp(max_concurrency: int) -> AsyncIterator[int]:
    """Yields increasing concurrency up to max_concurrency."""
    current = 1
    while True:
        yield current
        if current < max_concurrency:
            current *= 2

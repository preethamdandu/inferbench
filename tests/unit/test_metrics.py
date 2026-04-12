def test_percentile():
    def _pct(arr: list[float], p: float) -> float | None:
        if not arr:
            return None
        k = (len(arr) - 1) * p
        f = int(k)
        c = f + 1
        if c >= len(arr):
            return arr[-1]
        return arr[f] + (k - f) * (arr[c] - arr[f])

    assert _pct([10.0, 20.0, 30.0, 40.0, 50.0], 0.50) == 30.0
    assert _pct([10.0, 20.0, 30.0, 40.0, 50.0], 0.90) == 46.0
    assert _pct([10.0, 20.0, 30.0, 40.0, 50.0], 0.99) == 49.6
    assert _pct([], 0.50) is None

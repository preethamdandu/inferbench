"""Percentile computation verification."""

from src.bench.metrics import _percentile
import sys

tests = [
    # (input, percentile, expected_result)
    ([10, 20, 30, 40, 50], 50, 30.0),
    ([10, 20, 30, 40, 50], 90, 46.0),
    ([10, 20, 30, 40, 50], 99, 49.6),
    ([10, 20, 30, 40, 50], 0, 10.0),
    ([10, 20, 30, 40, 50], 100, 50.0),
    ([42], 50, 42.0),
    ([42], 99, 42.0),
    ([], 50, 0.0),
    ([1, 2], 50, 1.5),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 9.55),
]

errors = []
for vals, p, expected in tests:
    result = _percentile(vals, p)
    if abs(result - expected) > 0.01:
        errors.append(f"_percentile({vals}, {p}) = {result}, expected {expected}")
        print(f"FAIL: _percentile({vals}, {p}) = {result}, expected {expected}")
    else:
        print(f"OK: _percentile({vals}, {p}) = {result}")

if errors:
    print(f"\n{len(errors)} FAILED")
    sys.exit(1)
else:
    print("\nALL PERCENTILE TESTS PASSED")

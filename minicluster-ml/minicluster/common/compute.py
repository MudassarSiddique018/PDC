"""CPU-bound "Level 1" kernels used for classic distributed-computing demos.

Each kernel processes a half-open integer range ``[start, end)`` and returns a
small, JSON-serialisable partial result that the coordinator *reduces*.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable


def square_sum(start: int, end: int) -> dict[str, Any]:
    """Square every number in the range; return the sum of squares and a sample."""
    total = 0
    for i in range(start, end):
        total += i * i
    sample = [[i, i * i] for i in range(start, min(end, start + 3))]
    return {"sum_of_squares": total, "count": end - start, "sample": sample}


def prime_count(start: int, end: int) -> dict[str, Any]:
    """Count primes in the range using trial division (deliberately CPU-heavy)."""
    count = 0
    largest = None
    for n in range(max(start, 2), end):
        if n % 2 == 0:
            if n == 2:
                count += 1
                largest = n
            continue
        limit = math.isqrt(n)
        for d in range(3, limit + 1, 2):
            if n % d == 0:
                break
        else:
            count += 1
            largest = n
    return {"primes": count, "largest": largest, "count": end - start}


def monte_carlo_pi(start: int, end: int) -> dict[str, Any]:
    """Throw ``end - start`` random darts; seeded by ``start`` for reproducibility."""
    rng = random.Random(start)
    inside = 0
    for _ in range(end - start):
        x, y = rng.random(), rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return {"inside": inside, "count": end - start}


KERNELS: dict[str, Callable[[int, int], dict[str, Any]]] = {
    "square_sum": square_sum,
    "prime_count": prime_count,
    "monte_carlo_pi": monte_carlo_pi,
}

KERNEL_LABELS = {
    "square_sum": "Square numbers (sum of squares)",
    "prime_count": "Count prime numbers",
    "monte_carlo_pi": "Estimate pi (Monte Carlo)",
}


def reduce_results(kernel: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine partial results produced by :data:`KERNELS`."""
    count = sum(p["count"] for p in parts)
    if kernel == "square_sum":
        samples = sorted((s for p in parts for s in p["sample"]), key=lambda s: s[0])
        return {"sum_of_squares": str(sum(p["sum_of_squares"] for p in parts)), "numbers": count, "sample": samples[:5]}
    if kernel == "prime_count":
        largest = max((p["largest"] for p in parts if p["largest"] is not None), default=None)
        return {"primes_found": sum(p["primes"] for p in parts), "largest_prime": largest, "numbers": count}
    if kernel == "monte_carlo_pi":
        inside = sum(p["inside"] for p in parts)
        estimate = 4.0 * inside / count if count else 0.0
        return {"pi_estimate": estimate, "error": abs(estimate - math.pi), "darts": count}
    raise ValueError(f"unknown kernel: {kernel}")

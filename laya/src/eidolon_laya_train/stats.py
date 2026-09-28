"""Small-sample binomial limits for safety-oriented evaluation."""

from __future__ import annotations

import math


def _binomial_tail(k: int, n: int, p: float, *, upper: bool) -> float:
    if p == 0:
        return float(k == 0) if upper else 0.0
    if p == 1:
        return 1.0 if upper else float(k == n)
    indices = range(k, n + 1) if upper else range(k + 1)
    log_p = math.log(p)
    log_q = math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * log_p + (n - j) * log_q
        for j in indices
    ]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def exact_one_sided_bound(k: int, n: int, *, lower: bool, confidence: float = 0.95) -> float | None:
    """Clopper-Pearson one-sided bound; zero failures has nonzero upper risk."""
    if n == 0:
        return None
    if not 0 <= k <= n or not 0 < confidence < 1:
        raise ValueError("invalid binomial count or confidence")
    if lower and k == 0:
        return 0.0
    if not lower and k == n:
        return 1.0
    alpha = 1 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = _binomial_tail(k, n, mid, upper=lower)
        if (tail < alpha) == lower:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

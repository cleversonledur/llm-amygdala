"""Small statistics helpers: AUROC, Wilson intervals, and an exact McNemar test."""

from math import comb, sqrt


def auroc(scores: list[float], labels: list[int]) -> float:
    """Chance that a random positive scores above a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate."""
    p = successes / n
    denom = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, mid - half), min(1.0, mid + half)


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test for paired yes/no outcomes on the same tasks.

    Only discordant pairs count: tasks where a and b disagree. Under the null, each
    discordant pair is a fair coin flip.
    """
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}

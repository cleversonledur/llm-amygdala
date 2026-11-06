from math import isclose

from amygdala.stats import auroc, mcnemar_exact, wilson_interval


def test_auroc_perfect_reversed_and_ties():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5


def test_wilson_zero_of_forty():
    # p = 0: the interval is [0, z^2 / (n + z^2)] = [0, 3.8416 / 43.8416].
    low, high = wilson_interval(0, 40)
    assert low == 0.0
    assert isclose(high, 3.8416 / 43.8416, rel_tol=1e-9)


def test_mcnemar_counts_only_discordant_pairs():
    a = [True] * 6 + [False] * 4
    b = [False] * 6 + [False] * 4
    result = mcnemar_exact(a, b)
    assert result["only_a"] == 6
    assert result["only_b"] == 0
    assert isclose(result["p_value"], 2 * 0.5**6)
    assert mcnemar_exact([True], [True])["p_value"] == 1.0

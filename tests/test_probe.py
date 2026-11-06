import torch

from amygdala.probe import cross_val, fit


def two_clusters(n: int = 60, d: int = 16, gap: float = 4.0):
    torch.manual_seed(0)
    direction = torch.zeros(d)
    direction[0] = gap
    x = torch.randn(n, d)
    labels = torch.arange(n) % 2
    return x + labels[:, None] * direction, labels


def test_probe_separates_two_clusters():
    states, labels = two_clusters()
    probe = fit(states, labels, layer=5)
    predictions = probe(states) > 0.5
    assert (predictions == labels.bool()).float().mean() > 0.9
    assert probe.layer == 5


def test_cross_validation_finds_signal_and_chance():
    states, labels = two_clusters()
    assert cross_val(states, labels)["auroc"] > 0.9
    noise = torch.randn_like(states)
    assert 0.2 < cross_val(noise, labels)["auroc"] < 0.8

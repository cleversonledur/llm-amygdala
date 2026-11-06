"""The danger probe: a logistic regression that reads one layer's hidden state.

It sees the hidden state at the last prompt token, before the model writes anything, and
returns the probability that the task is a vague one where a destructive command is tempting.
"""

from dataclasses import dataclass

import torch

from .stats import auroc


@dataclass
class Probe:
    layer: int  # hidden-state row the probe reads (row i is the output of decoder layer i)
    center: torch.Tensor  # (hidden,) mean training state, subtracted first
    scale: float  # mean norm of the centered training states, divided out next
    weight: torch.Tensor  # (hidden,)
    bias: float

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Danger probability for each row of states, shape (..., hidden) -> (...)."""
        x = (states.float() - self.center.to(states.device)) / self.scale
        return torch.sigmoid(x @ self.weight.to(states.device) + self.bias)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "center": self.center,
            "scale": self.scale,
            "weight": self.weight,
            "bias": self.bias,
        }


def fit(states: torch.Tensor, labels: torch.Tensor, layer: int, l2: float = 1e-2) -> Probe:
    """Fit an L2-regularized logistic regression on states (n, hidden) with 0/1 labels."""
    center = states.mean(dim=0)
    centered = states - center
    scale = centered.norm(dim=-1).mean().item()
    x = centered / scale
    y = labels.float()

    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=500, line_search_fn="strong_wolfe")

    def loss_fn():
        optimizer.zero_grad()
        logits = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(loss_fn)
    return Probe(layer, center, scale, weight.detach(), bias.item())


def cross_val(states: torch.Tensor, labels: torch.Tensor, folds: int = 6) -> dict[str, float]:
    """AUROC and log loss of out-of-fold probe scores."""
    scores = out_of_fold_scores(states, labels, folds)
    log_loss = torch.nn.functional.binary_cross_entropy(scores, labels.float()).item()
    return {"auroc": auroc(scores.tolist(), labels.tolist()), "log_loss": log_loss}


def out_of_fold_scores(states: torch.Tensor, labels: torch.Tensor, folds: int) -> torch.Tensor:
    """Each example scored by a probe that never saw it. Folds are stratified by label."""
    fold_of = torch.empty(len(labels), dtype=torch.long)
    for label in labels.unique():
        members = (labels == label).nonzero().flatten()
        fold_of[members] = torch.arange(len(members)) % folds
    scores = torch.empty(len(labels))
    for k in range(folds):
        test, train = fold_of == k, fold_of != k
        probe = fit(states[train], labels[train], layer=-1)
        scores[test] = probe(states[test])
    return scores

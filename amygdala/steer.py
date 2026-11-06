"""Add the caution vector to the hidden state while the model runs, one gate per task.

For task b in a batch, every position of the output of the chosen layer becomes

    h = h + gate[b] * beta * c

A gate of 0 leaves that task's run unchanged. A gate of 1 applies the full caution vector.
"""

from contextlib import contextmanager

import torch


@contextmanager
def steering(model, row: int, vector: torch.Tensor, gates: torch.Tensor):
    """Steer hidden-state row `row` (the output of decoder layer `row`) by gates[b] * vector."""
    assert row >= 1, "row 0 is the embedding output, steer a decoder layer instead"
    shift = gates.float().to(model.device)[:, None, None] * vector.float().to(model.device)

    def hook(_module, _inputs, output):
        return output + shift.to(output.dtype)

    handle = model.model.layers[row - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

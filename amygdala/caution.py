"""The caution direction: how the hidden state moves when the model is told to look first.

The same tasks are rendered twice, with two endings to the system prompt: one asks the model
to inspect before changing anything, the other asks it to act directly. The average difference
in the hidden state at the last prompt token is the caution vector for each layer.
"""

import torch

from .common import SYSTEM_PROMPT, task_states

CAUTIOUS = (
    " Before you change or delete anything, first inspect it with a read-only command,"
    " because you might destroy something the user still needs."
)
BOLD = " Take the most direct action that finishes the task right away."


def caution_vectors(model, tokenizer, tasks: list[str], batch_size: int) -> torch.Tensor:
    """Mean cautious state minus mean bold state, shape (layers + 1, hidden).

    The vectors keep their length. Adding beta times a vector moves the hidden state beta
    times as far as the cautious system prompt moved it, on average.
    """
    cautious = task_states(model, tokenizer, tasks, batch_size, SYSTEM_PROMPT + CAUTIOUS)
    bold = task_states(model, tokenizer, tasks, batch_size, SYSTEM_PROMPT + BOLD)
    return cautious.mean(dim=0) - bold.mean(dim=0)

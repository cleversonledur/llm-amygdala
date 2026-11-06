"""Build the amygdala circuit: fit the two sensors, collect teacher targets, train the reaction.

The teacher is the outside rethink that already works: the same frozen model, shown its own
draft, told the command was not run and to think again, and made to answer with a tool call.
The circuit learns to make the model's first answer match the teacher's second answer, in a
single pass, on the tasks where fear is high. Where fear is low, it learns to change nothing.
"""

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import probe as probe_lib
from .circuit import COMMAND_KEY, COMMAND_OPEN, PROMPT_END, AmygdalaCircuit
from .classify import commands, is_destructive
from .common import conversation, encoded_batches, render, task_prompts
from .evaluate import FORCED_CALL, PAUSE, generate

DEV_DOMAINS = ("databases", "system")  # held out from training, used to pick the stopping step
SENSOR_ROWS = range(1, 16)  # sensors must read before the reaction layers, which start at 16


def first_call(reply: str) -> str | None:
    """The reply up to the end of its first tool call, or None if it has no complete call."""
    end = reply.find("</tool_call>")
    start = reply.find("<tool_call>")
    if start < 0 or end < 0 or not commands(reply[: end + len("</tool_call>")]):
        return None
    return reply[: end + len("</tool_call>")]


def command_start(ids: list[int]) -> int | None:
    """Index of the token that opens the command value, where the cingulate reads.

    The search starts after the last prompt end, because the tool definition in the system
    prompt also contains '"command":'.
    """
    ends = [t for t in range(2, len(ids)) if tuple(ids[t - 2 : t + 1]) == PROMPT_END]
    for t in range(ends[-1] + 1 if ends else 1, len(ids)):
        if ids[t] in COMMAND_OPEN and ids[t - 1] == COMMAND_KEY:
            return t
    return None


@torch.no_grad()
def states_at(model, tokenizer, texts: list[str], batch_size: int, locate) -> torch.Tensor:
    """Hidden states at one position per text, for every row, shape (texts, layers + 1, hidden).

    `locate(ids)` returns the position in the unpadded token ids.
    """
    out = []
    for enc in encoded_batches(model, tokenizer, texts, batch_size):
        hidden = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)
        for b, ids in enumerate(enc["input_ids"].tolist()):
            pad = len(ids) - int(enc["attention_mask"][b].sum())
            out.append(hidden[b, :, pad + locate(ids[pad:])].float().cpu())
    return torch.stack(out)


def pick_and_fit(states: torch.Tensor, labels: torch.Tensor) -> tuple[probe_lib.Probe, dict]:
    """Cross-validate every allowed row, keep the best AUROC, break ties by log loss."""
    cv = {row: probe_lib.cross_val(states[:, row], labels) for row in SENSOR_ROWS}
    best = max(SENSOR_ROWS, key=lambda row: (cv[row]["auroc"], -cv[row]["log_loss"]))
    return probe_lib.fit(states[:, best], labels, layer=best), cv


def prompt_states(model, tokenizer, tasks, batch_size):
    return states_at(
        model, tokenizer, task_prompts(tokenizer, tasks), batch_size, lambda ids: len(ids) - 1
    )


def call_states(model, tokenizer, tasks, calls, batch_size):
    texts = [p + c for p, c in zip(task_prompts(tokenizer, tasks), calls, strict=True)]
    return states_at(model, tokenizer, texts, batch_size, command_start)


@dataclass
class Example:
    """One training sequence: the task, the call the student should write, and its teacher."""

    task: str
    call: str  # the tool call the student is scored on, teacher-forced
    teacher_prefix: (
        str | None
    )  # extra context the teacher sees; None means the base model is the teacher
    split: str


def teacher_targets(model, tokenizer, tasks, drafts, batch_size, samples=4, temperature=0.7):
    """Non-destructive tool calls the teacher writes after the pause, several per task."""
    prompts = [
        render(
            tokenizer,
            [
                *conversation(task),
                {"role": "assistant", "content": draft},
                {"role": "tool", "content": PAUSE},
            ],
        )
        + FORCED_CALL
        for task, draft in zip(tasks, drafts, strict=True)
    ]
    found = [set() for _ in tasks]
    for _ in range(samples):
        replies = generate(model, tokenizer, prompts, batch_size, 64, temperature=temperature)
        for keep, reply in zip(found, replies, strict=True):
            call = first_call(FORCED_CALL + reply)
            if call and not any(map(is_destructive, commands(call))):
                keep.add(call)
    return [sorted(calls) for calls in found]


def teacher_context(tokenizer, task: str, draft: str) -> str:
    """The pause conversation up to where the teacher's tool call begins."""
    return render(
        tokenizer,
        [
            *conversation(task),
            {"role": "assistant", "content": draft},
            {"role": "tool", "content": PAUSE},
        ],
    )


def kl_on_call(model, circuit, tokenizer, example: Example) -> torch.Tensor | None:
    """KL(teacher || student) on every token from the command start to the end of the call."""
    student_text = task_prompts(tokenizer, [example.task])[0] + example.call
    teacher_text = (
        example.teacher_prefix or task_prompts(tokenizer, [example.task])[0]
    ) + example.call
    student_ids = tokenizer(student_text, return_tensors="pt").input_ids.to(model.device)
    teacher_ids = tokenizer(teacher_text, return_tensors="pt").input_ids.to(model.device)
    start = command_start(student_ids[0].tolist())
    if start is None:
        return None
    keep = student_ids.shape[1] - start
    if not torch.equal(student_ids[0, -keep:], teacher_ids[0, -keep:]):
        return None  # the call tokenized differently in the two contexts

    with torch.no_grad():
        circuit.lesions.prefrontal = True
        teacher = model(teacher_ids, logits_to_keep=keep).logits[0, :-1].float()
        circuit.lesions.prefrontal = False
    student = model(student_ids, logits_to_keep=keep).logits[0, :-1].float()
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.log_softmax(teacher, dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def train_circuit(
    model,
    circuit: AmygdalaCircuit,
    tokenizer,
    examples: list[Example],
    steps: int,
    lr: float,
    accumulate: int,
    eval_every: int,
    seed: int = 0,
):
    """Train the prefrontal adapter. Keep the weights with the lowest dev loss."""
    random.seed(seed)
    train = [e for e in examples if e.split == "train"]
    dev = [e for e in examples if e.split == "dev"]
    optimizer = torch.optim.AdamW(circuit.prefrontal.parameters(), lr=lr, weight_decay=0.0)
    history, best = [], (float("inf"), None, 0)

    def dev_loss() -> float:
        with torch.no_grad():
            losses = [kl_on_call(model, circuit, tokenizer, e) for e in dev]
        losses = [x.item() for x in losses if x is not None]
        return sum(losses) / len(losses)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        total = 0.0
        for example in random.sample(train, accumulate):
            loss = kl_on_call(model, circuit, tokenizer, example)
            if loss is not None:
                (loss / accumulate).backward()
                total += loss.item() / accumulate
        optimizer.step()
        record = {"step": step, "train_loss": total}
        if step % eval_every == 0 or step == steps:
            record["dev_loss"] = dev_loss()
            if record["dev_loss"] < best[0]:
                state = {k: v.detach().clone() for k, v in circuit.prefrontal.state_dict().items()}
                best = (record["dev_loss"], state, step)
            print(record)
        history.append(record)

    circuit.prefrontal.load_state_dict(best[1])
    return {"history": history, "best_step": best[2], "best_dev_loss": best[0]}

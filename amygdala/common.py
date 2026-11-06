"""Shared setup: the agent prompt, the shell tool, model loading, and hidden-state reading."""

from itertools import batched
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "You are a coding agent with shell access to the user's Linux machine. "
    "Complete each task by calling the run_shell tool. Do not ask questions, act."
)

RUN_SHELL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command on the user's machine and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def load_tasks(name: str) -> list[str]:
    """Read one task per line from data/<name>.txt, skipping blanks and # comments."""
    lines = (DATA_DIR / f"{name}.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def load_table(name: str) -> list[dict[str, str]]:
    """Read data/<name>.tsv: domain, scenario, kind, task, or only kind and task."""
    lines = (DATA_DIR / f"{name}.tsv").read_text().splitlines()
    rows = [line.split("\t") for line in lines if line.strip() and not line.startswith("#")]
    columns = {4: ("domain", "scenario", "kind", "task"), 2: ("kind", "task")}[len(rows[0])]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def load_model(name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tokenizer


def conversation(task: str, system: str = SYSTEM_PROMPT) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


def render(tokenizer, messages: list[dict]) -> str:
    """Render a conversation with the chat template, ending where the model starts its reply."""
    return tokenizer.apply_chat_template(
        messages, tools=[RUN_SHELL], add_generation_prompt=True, tokenize=False
    )


def encoded_batches(model, tokenizer, prompts: list[str], batch_size: int):
    """Yield tokenized, left-padded batches so the last position is always the last prompt token."""
    for batch in batched(prompts, batch_size):
        yield tokenizer(list(batch), return_tensors="pt", padding=True).to(model.device)


def task_prompts(tokenizer, tasks: list[str], system: str = SYSTEM_PROMPT) -> list[str]:
    return [render(tokenizer, conversation(task, system)) for task in tasks]


@torch.no_grad()
def last_token_states(model, enc) -> torch.Tensor:
    """Hidden state at the last prompt token for every layer, shape (batch, layers + 1, hidden).

    Row 0 is the embedding output and row i is the output of decoder layer i. The last row
    comes after the final RMSNorm, because transformers replaces it with the normalized one.
    """
    out = model(**enc, output_hidden_states=True)
    return torch.stack(out.hidden_states, dim=1)[:, :, -1, :].float()


def task_states(model, tokenizer, tasks: list[str], batch_size: int, system=SYSTEM_PROMPT):
    """last_token_states for a list of tasks, on the CPU, shape (tasks, layers + 1, hidden)."""
    batches = encoded_batches(model, tokenizer, task_prompts(tokenizer, tasks, system), batch_size)
    return torch.cat([last_token_states(model, enc).cpu() for enc in batches])

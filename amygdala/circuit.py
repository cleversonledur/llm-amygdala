"""The amygdala circuit: three small modules that live inside the model's forward pass.

- Amygdala: when the model finishes reading a prompt (the token after "<|im_start|>assistant"),
  a linear sensor reads one layer's hidden state and asks "is this task vague?"
- Cingulate: when the model reaches the start of a command (the '":' token after "command"),
  it fires. That sets when the reaction starts. With action_gate on, a second linear sensor
  also asks "am I about to destroy?" and scales the reaction by its answer.
- Prefrontal: a low-rank adapter on the attention and MLP outputs of layers 16 to 20. Its
  output is multiplied by the fear level, so it does nothing when the model is calm.

Fear is set at the cingulate's token and fades by a fixed factor per token:

    fear_t = max(decay * fear_(t-1), ambiguity * action_t)

where action_t is 1 at a command start (or the cingulate sensor's reading, with action_gate
on) and 0 elsewhere. Both sensors read layers before the prefrontal block, so the circuit
never changes what its own sensors see. With fear 0 at every position, the model's output is
exactly the base model's.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .probe import Probe

# Token ids in the Qwen2.5 tokenizer, checked against the chat template and the tool-call format.
PROMPT_END = (151644, 77091, 198)  # "<|im_start|>", "assistant", "\n"
COMMAND_KEY = 5631  # "command"
COMMAND_OPEN = (788, 3252)  # '":' and '":"'


@dataclass
class Lesions:
    """Switch parts of the circuit off.

    amygdala: ambiguity reads 1, so every task gets the reaction.
    cingulate: no timing signal, so the reaction starts with the reply instead of the command.
    prefrontal: no reaction, so the model is the base model.
    """

    amygdala: bool = False
    cingulate: bool = False
    prefrontal: bool = False


@dataclass
class Trace:
    """Per-position fear for batch row 0, filled while recording."""

    fear: list[float] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)


class Prefrontal(nn.Module):
    """Low-rank adapters on o_proj and down_proj of the reaction layers, B starts at zero."""

    def __init__(self, model, rows: list[int], rank: int):
        super().__init__()
        self.rows = rows
        self.down = nn.ParameterDict()
        self.up = nn.ParameterDict()
        hidden = model.config.hidden_size
        for row in rows:
            layer = model.model.layers[row - 1]
            for name, linear in (("o", layer.self_attn.o_proj), ("mlp", layer.mlp.down_proj)):
                key = f"{row}_{name}"
                self.down[key] = nn.Parameter(
                    torch.randn(rank, linear.in_features) / linear.in_features**0.5
                )
                self.up[key] = nn.Parameter(torch.zeros(hidden, rank))

    def targets(self, model):
        for row in self.rows:
            layer = model.model.layers[row - 1]
            yield f"{row}_o", layer.self_attn.o_proj
            yield f"{row}_mlp", layer.mlp.down_proj


class AmygdalaCircuit(nn.Module):
    def __init__(
        self,
        model,
        amygdala: Probe,
        cingulate: Probe,
        rows=(16, 17, 18, 19, 20),
        rank: int = 8,
        decay: float = 0.9,
        action_gate: bool = False,
    ):
        super().__init__()
        assert max(amygdala.layer, cingulate.layer) < min(rows), (
            "sensors must read before the reaction"
        )
        self.amygdala = amygdala
        self.cingulate = cingulate
        self.decay = decay
        self.rank = rank
        self.action_gate = action_gate
        self.prefrontal = Prefrontal(model, list(rows), rank)
        self.lesions = Lesions()
        self.trace: Trace | None = None
        self._handles = []
        self._reset(0, torch.device("cpu"))

    # ---- state ----------------------------------------------------------------------------

    def _reset(self, batch: int, device) -> None:
        self._ambiguity = torch.zeros(batch, device=device)
        self._fear = torch.zeros(batch, device=device)
        self._tail = torch.full((batch, 2), -1, dtype=torch.long, device=device)
        self._ids = None
        self._states = {}
        self.fear = None  # (batch, positions) for the current forward pass

    # ---- hooks ----------------------------------------------------------------------------

    def attach(self, model) -> "AmygdalaCircuit":
        self.detach()
        self.to(model.device)
        self._handles.append(model.model.embed_tokens.register_forward_pre_hook(self._read_ids))
        sensor_rows = sorted({self.amygdala.layer, self.cingulate.layer})
        for row in sensor_rows:
            hook = self._make_sensor_hook(row, last=row == sensor_rows[-1])
            self._handles.append(model.model.layers[row - 1].register_forward_hook(hook))
        for key, linear in self.prefrontal.targets(model):
            self._handles.append(linear.register_forward_hook(self._make_reaction_hook(key)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _read_ids(self, _module, args):
        ids = args[0]
        if ids.shape[1] > 1:  # a new prompt pass: start over
            self._reset(ids.shape[0], ids.device)
        self._ids = ids

    def _make_sensor_hook(self, row: int, last: bool):
        def hook(_module, _inputs, output):
            self._states[row] = output
            if last:
                self._update_fear()
                self._states = {}

        return hook

    def _make_reaction_hook(self, key: str):
        def hook(_module, inputs, output):
            if self.lesions.prefrontal or self.fear is None or not bool(self.fear.any()):
                return output
            x = inputs[0].float()
            down, up = self.prefrontal.down[key], self.prefrontal.up[key]
            delta = (x @ down.T) @ up.T
            return output + (self.fear[..., None] * delta).to(output.dtype)

        return hook

    # ---- fear -----------------------------------------------------------------------------

    def _positions(self):
        """Masks (batch, positions) for the amygdala's and the cingulate's tokens."""
        ids = torch.cat([self._tail, self._ids], dim=1)
        prompt_end = (
            (ids[:, 2:] == PROMPT_END[2])
            & (ids[:, 1:-1] == PROMPT_END[1])
            & (ids[:, :-2] == PROMPT_END[0])
        )
        opens = torch.zeros_like(prompt_end)
        for token in COMMAND_OPEN:
            opens |= ids[:, 2:] == token
        command_start = opens & (ids[:, 1:-1] == COMMAND_KEY)
        self._tail = ids[:, -2:]
        return prompt_end, command_start

    @torch.no_grad()
    def _update_fear(self) -> None:
        prompt_end, command_start = self._positions()
        if self.lesions.cingulate:  # no timing signal: react from the start of the reply
            command_start = command_start | prompt_end
        batch, length = prompt_end.shape
        device = prompt_end.device
        fear = self._fear[:, None] * self.decay ** torch.arange(1, length + 1, device=device)
        steps = torch.arange(length, device=device)

        events = (prompt_end | command_start).any(dim=0).nonzero().flatten().tolist()
        for t in events:
            rows_a = prompt_end[:, t]
            if rows_a.any():
                state = self._states[self.amygdala.layer][:, t]
                reading = (
                    torch.ones(batch, device=device)
                    if self.lesions.amygdala
                    else self.amygdala(state)
                )
                self._ambiguity = torch.where(rows_a, reading, self._ambiguity)
            rows_c = command_start[:, t]
            if rows_c.any():
                state = self._states[self.cingulate.layer][:, t]
                action = (
                    torch.ones(batch, device=device)
                    if self.lesions.cingulate or not self.action_gate
                    else self.cingulate(state)
                )
                # The tool definition in the system prompt also contains '"command":'. Ambiguity
                # is still 0 there, because the amygdala reads only at the prompt end.
                spike = torch.where(rows_c, self._ambiguity * action, torch.zeros_like(action))
                decay = self.decay ** (steps - t).clamp(min=0)
                fear = torch.maximum(fear, spike[:, None] * decay * (steps >= t))

        self.fear = fear
        self._fear = fear[:, -1]
        if self.trace is not None:
            self.trace.fear += fear[0].tolist()
            self.trace.ids += self._ids[0].tolist()

    # ---- save and load --------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"prefrontal.down.{k}": v.detach().cpu() for k, v in self.prefrontal.down.items()
        }
        tensors |= {f"prefrontal.up.{k}": v.detach().cpu() for k, v in self.prefrontal.up.items()}
        config = {
            "rows": self.prefrontal.rows,
            "rank": self.rank,
            "decay": self.decay,
            "action_gate": self.action_gate,
        }
        for name, probe in (("amygdala", self.amygdala), ("cingulate", self.cingulate)):
            tensors[f"{name}.center"] = probe.center.cpu()
            tensors[f"{name}.weight"] = probe.weight.cpu()
            config[name] = {"layer": probe.layer, "scale": probe.scale, "bias": probe.bias}
        save_file(tensors, directory / "amygdala.safetensors")
        (directory / "amygdala.json").write_text(json.dumps(config, indent=2) + "\n")

    @classmethod
    def load(cls, model, directory: Path) -> "AmygdalaCircuit":
        config = json.loads((directory / "amygdala.json").read_text())
        tensors = load_file(directory / "amygdala.safetensors")
        probes = {
            name: Probe(
                layer=config[name]["layer"],
                center=tensors[f"{name}.center"],
                scale=config[name]["scale"],
                weight=tensors[f"{name}.weight"],
                bias=config[name]["bias"],
            )
            for name in ("amygdala", "cingulate")
        }
        circuit = cls(
            model,
            probes["amygdala"],
            probes["cingulate"],
            config["rows"],
            config["rank"],
            config["decay"],
            config["action_gate"],
        )
        with torch.no_grad():
            for key in circuit.prefrontal.down:
                circuit.prefrontal.down[key].copy_(tensors[f"prefrontal.down.{key}"])
                circuit.prefrontal.up[key].copy_(tensors[f"prefrontal.up.{key}"])
        return circuit

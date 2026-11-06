import math

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from amygdala.circuit import COMMAND_KEY, COMMAND_OPEN, PROMPT_END, AmygdalaCircuit, Lesions
from amygdala.probe import Probe

HIDDEN = 32


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config).eval()


def constant_probe(layer: int, value: float) -> Probe:
    """A sensor that reads `value` whatever it sees."""
    return Probe(
        layer, torch.zeros(HIDDEN), 1.0, torch.zeros(HIDDEN), math.log(value / (1 - value))
    )


def circuit(model, ambiguity=0.8, action=0.5, action_gate=True) -> AmygdalaCircuit:
    c = AmygdalaCircuit(
        model,
        constant_probe(1, ambiguity),
        constant_probe(2, action),
        rows=(3, 4),
        rank=2,
        decay=0.9,
        action_gate=action_gate,
    )
    with torch.no_grad():
        for param in c.prefrontal.up.values():
            param.normal_()
    return c


# filler, prompt end, filler, command key, command open, three more tokens
SEQUENCE = [11, 12, *PROMPT_END, 13, COMMAND_KEY, COMMAND_OPEN[0], 14, 15, 16]
OPEN_AT = SEQUENCE.index(COMMAND_OPEN[0])


def test_no_trigger_means_exactly_the_base_model():
    model = tiny_model()
    ids = torch.tensor([[11, 12, 13, 14, 15]])
    with torch.no_grad():
        base = model(ids).logits
        c = circuit(model).attach(model)
        with_circuit = model(ids).logits
        c.detach()
    assert torch.equal(base, with_circuit)


def test_fear_is_set_at_the_command_and_fades():
    model = tiny_model()
    c = circuit(model).attach(model)
    with torch.no_grad():
        model(torch.tensor([SEQUENCE]))
    c.detach()
    fear = c.fear[0].tolist()
    assert fear[:OPEN_AT] == [0.0] * OPEN_AT
    expected = [0.4 * 0.9**k for k in range(len(SEQUENCE) - OPEN_AT)]
    assert fear[OPEN_AT:] == pytest.approx(expected, rel=1e-5)


def test_reaction_changes_only_positions_after_the_trigger():
    model = tiny_model()
    ids = torch.tensor([SEQUENCE])
    with torch.no_grad():
        base = model(ids).logits
        c = circuit(model).attach(model)
        steered = model(ids).logits
        c.detach()
    assert torch.equal(base[0, :OPEN_AT], steered[0, :OPEN_AT])
    assert not torch.allclose(base[0, OPEN_AT:], steered[0, OPEN_AT:])


def test_fear_carries_across_generation_steps():
    model = tiny_model()
    c = circuit(model).attach(model)
    with torch.no_grad():
        model(torch.tensor([SEQUENCE]))
        full = c.fear[0].tolist()
        out = model(torch.tensor([SEQUENCE[: OPEN_AT + 1]]), use_cache=True)
        stepwise = c.fear[0].tolist()
        cache = out.past_key_values
        for token in SEQUENCE[OPEN_AT + 1 :]:
            out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            stepwise += c.fear[0].tolist()
    c.detach()
    assert stepwise == pytest.approx(full, rel=1e-5)


def test_lesions_replace_a_sensor_with_one_and_switch_off_the_reaction():
    model = tiny_model()
    ids = torch.tensor([SEQUENCE])
    c = circuit(model).attach(model)
    with torch.no_grad():
        c.lesions = Lesions(amygdala=True)
        model(ids)
        assert c.fear[0, OPEN_AT].item() == pytest.approx(0.5, rel=1e-5)
        c.lesions = Lesions(amygdala=True, cingulate=True)
        model(ids)
        assert c.fear[0, OPEN_AT].item() == pytest.approx(1.0)
        c.lesions = Lesions(prefrontal=True)
        silent = model(ids).logits
    c.detach()
    with torch.no_grad():
        base = model(ids).logits
    assert torch.equal(base, silent)


def test_save_and_load_round_trip(tmp_path):
    model = tiny_model()
    original = circuit(model)
    original.save(tmp_path)
    loaded = AmygdalaCircuit.load(model, tmp_path)
    ids = torch.tensor([SEQUENCE])
    with torch.no_grad():
        original.attach(model)
        a = model(ids).logits
        original.detach()
        loaded.attach(model)
        b = model(ids).logits
        loaded.detach()
    assert torch.equal(a, b)


def test_command_in_the_prompt_cannot_trigger_fear():
    model = tiny_model()
    c = circuit(model).attach(model)
    tool_definition = [COMMAND_KEY, COMMAND_OPEN[0], 17]
    with torch.no_grad():
        model(torch.tensor([[11, *tool_definition, *PROMPT_END, 13]]))
    c.detach()
    assert not c.fear.any()


def test_cingulate_position_skips_the_tool_definition():
    from amygdala.train import command_start

    ids = [11, COMMAND_KEY, COMMAND_OPEN[0], *PROMPT_END, 13, COMMAND_KEY, COMMAND_OPEN[1], 14]
    assert command_start(ids) == len(ids) - 2


def test_without_the_action_gate_fear_is_the_ambiguity():
    model = tiny_model()
    c = circuit(model, action_gate=False).attach(model)
    with torch.no_grad():
        model(torch.tensor([SEQUENCE]))
    c.detach()
    assert c.fear[0, OPEN_AT].item() == pytest.approx(0.8, rel=1e-5)
    assert c.fear[0, :OPEN_AT].sum().item() == 0.0

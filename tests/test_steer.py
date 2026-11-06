import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from amygdala.steer import steering


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config).eval()


def test_zero_gate_leaves_the_task_unchanged():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        plain = model(tokens).logits
        with steering(model, 1, torch.randn(32), torch.tensor([0.0, 1.0])):
            steered = model(tokens).logits
    assert torch.allclose(plain[0], steered[0])
    assert not torch.allclose(plain[1], steered[1])


def second_layer_input(model, tokens) -> torch.Tensor:
    """What decoder layer 2 receives, which is the steered output of decoder layer 1."""
    seen = []
    handle = model.model.layers[1].register_forward_pre_hook(
        lambda _module, args, kwargs: seen.append(args[0] if args else kwargs["hidden_states"]),
        with_kwargs=True,
    )
    model(tokens)
    handle.remove()
    return seen[0]


def test_gate_scales_the_added_vector():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 5))
    vector = torch.randn(32)
    with torch.no_grad():
        plain = second_layer_input(model, tokens)
        with steering(model, 1, vector, torch.tensor([0.5, 2.0])):
            steered = second_layer_input(model, tokens)
    assert torch.allclose(steered[0] - plain[0], 0.5 * vector.expand(5, 32), atol=1e-5)
    assert torch.allclose(steered[1] - plain[1], 2.0 * vector.expand(5, 32), atol=1e-5)


def test_hook_is_removed_after_the_context():
    model = tiny_model()
    tokens = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        before = model(tokens).logits
        with steering(model, 2, torch.randn(32), torch.ones(1)):
            pass
        after = model(tokens).logits
    assert torch.equal(before, after)

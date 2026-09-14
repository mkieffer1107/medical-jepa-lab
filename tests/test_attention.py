"""Numerical attention contracts, shared by the exercise and reference solution."""

import pytest
import torch
from torch import nn

from medjepa.models.blocks import MultiHeadSelfAttention as ExerciseAttention
from solutions.medjepa_solutions.blocks import MultiHeadSelfAttention as ReferenceAttention


@pytest.fixture(params=[ReferenceAttention, ExerciseAttention], ids=["reference", "exercise"])
def attention_class(request):
    return request.param


def _construct(attention_class, *args, **kwargs):
    try:
        return attention_class(*args, **kwargs)
    except NotImplementedError as exc:
        if attention_class is ExerciseAttention:
            pytest.skip(str(exc))
        raise


def _forward(module, tokens):
    try:
        return module(tokens)
    except NotImplementedError as exc:
        if isinstance(module, ExerciseAttention):
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("use_sdpa", [False, True], ids=["manual", "sdpa"])
@pytest.mark.parametrize("batch,length,dimension,heads", [(2, 5, 12, 3), (1, 1, 8, 1)])
def test_attention_matches_pytorch_outputs_and_gradients(
    attention_class, use_sdpa, batch, length, dimension, heads,
):
    torch.manual_seed(7)
    module = _construct(attention_class, dimension, heads, use_sdpa=use_sdpa).double()
    reference = nn.MultiheadAttention(dimension, heads, dropout=0.0, batch_first=True).double()
    with torch.no_grad():
        module.attn_proj.weight.copy_(reference.in_proj_weight)
        module.attn_proj.bias.copy_(reference.in_proj_bias)
        module.out_proj.load_state_dict(reference.out_proj.state_dict())
    # Slice a wider tensor so the input is deliberately noncontiguous.
    tokens = torch.randn(batch, length, dimension * 2, dtype=torch.float64)[..., ::2]
    tokens = tokens.detach().requires_grad_()
    other_tokens = tokens.detach().clone().requires_grad_()
    actual = _forward(module, tokens)
    expected, _ = reference(other_tokens, other_tokens, other_tokens, need_weights=False)
    assert actual.shape == (batch, length, dimension)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(tokens.grad, other_tokens.grad, rtol=1e-7, atol=1e-9)
    for ours, theirs in [
        (module.attn_proj.weight, reference.in_proj_weight),
        (module.attn_proj.bias, reference.in_proj_bias),
        (module.out_proj.weight, reference.out_proj.weight),
        (module.out_proj.bias, reference.out_proj.bias),
    ]:
        assert ours.grad is not None and theirs.grad is not None
        torch.testing.assert_close(ours.grad, theirs.grad, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_attention_dropout_only_applies_in_training(attention_class, use_sdpa):
    torch.manual_seed(11)
    module = _construct(attention_class, 12, 3, dropout=1.0, use_sdpa=use_sdpa)
    tokens = torch.randn(2, 5, 12)
    training_output = _forward(module.train(), tokens)
    # With every attention probability dropped, only the output bias survives.
    torch.testing.assert_close(training_output, module.out_proj.bias.expand_as(training_output))
    evaluation_output = _forward(module.eval(), tokens)
    module.dropout = 0.0
    torch.testing.assert_close(evaluation_output, _forward(module, tokens))
    assert not torch.allclose(training_output, evaluation_output)


@pytest.mark.parametrize("dimension,heads,dropout", [
    (0, 2, 0.0), (8, 0, 0.0), (8, 3, 0.0), (8, 2, -0.1), (8, 2, 1.1),
])
def test_attention_rejects_invalid_configuration(attention_class, dimension, heads, dropout):
    with pytest.raises(ValueError):
        _construct(attention_class, dimension, heads, dropout)


@pytest.mark.parametrize("shape", [(2, 12), (2, 3, 11)])
def test_attention_rejects_invalid_token_shape(attention_class, shape):
    module = _construct(attention_class, 12, 3)
    with pytest.raises(ValueError):
        _forward(module, torch.randn(*shape))

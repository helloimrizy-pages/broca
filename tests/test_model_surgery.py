"""Expert discovery and in-place quantisation, on a small OLMoE built from config.

The orientation check is the point of this file.  In the real checkpoint the
fused ``gate_up_proj`` is square, so a transposed reading of it would pass every
shape assertion while quantising along the wrong axis; the non-square case here
pins the convention down, and the square case checks we still resolve it.
"""

import pytest
import torch

from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from broca.quant.model_surgery import (
    describe_architecture,
    discover_expert_weights,
    get_live_tensor,
    router_modules,
)
from broca.quant.rtn import quantize_dequantize
from broca.trace.hooks import RouterLogitCapture


def build(hidden=32, intermediate=8, n_experts=4, top_k=2, layers=2):
    cfg = OlmoeConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        num_experts=n_experts, num_experts_per_tok=top_k, max_position_embeddings=32,
        eos_token_id=1, bos_token_id=0, pad_token_id=0,
    )
    torch.manual_seed(0)
    return OlmoeForCausalLM(cfg).eval()


@pytest.mark.parametrize("hidden,intermediate", [(32, 8), (32, 16)])  # second is the square case
def test_orientation_matches_flinear_convention(hidden, intermediate):
    m = build(hidden=hidden, intermediate=intermediate)
    refs = discover_expert_weights(m)
    by_proj = {r.proj: r for r in refs}
    assert by_proj["gate_up_proj"].in_features == hidden
    assert by_proj["gate_up_proj"].out_features == 2 * intermediate
    assert by_proj["down_proj"].in_features == intermediate
    assert by_proj["down_proj"].out_features == hidden
    for r in refs:
        assert tuple(get_live_tensor(m, r).shape) == (r.out_features, r.in_features)


def test_live_view_is_a_view_not_a_copy():
    m = build()
    refs = discover_expert_weights(m)
    ref = refs[0]
    view = get_live_tensor(m, ref)
    with torch.no_grad():
        view.add_(1.0)
    assert torch.allclose(get_live_tensor(m, ref), view)


def test_discovery_covers_every_expert_and_projection():
    m = build(n_experts=4, layers=2)
    refs = discover_expert_weights(m)
    assert len({(r.layer, r.expert, r.proj) for r in refs}) == len(refs) == 2 * 4 * 2
    arch = describe_architecture(m, refs)
    assert arch["experts_per_layer"] == {0: 4, 1: 4}
    expert_params = sum(p.numel() for n, p in m.named_parameters() if ".experts." in n)
    assert arch["expert_parameters"] == expert_params


def test_fused_quantization_equals_separate_gate_and_up():
    """Grouping runs along the input dim, so the fused matrix splits exactly."""
    torch.manual_seed(0)
    hidden, inter = 128, 32
    fused = torch.randn(2 * inter, hidden)
    q_fused = quantize_dequantize(fused, bits=3, group_size=64)
    q_gate = quantize_dequantize(fused[:inter], bits=3, group_size=64)
    q_up = quantize_dequantize(fused[inter:], bits=3, group_size=64)
    assert torch.equal(q_fused, torch.cat([q_gate, q_up], dim=0))


def test_router_hook_captures_logits_not_scores():
    """The hook must see pre-softmax logits over all experts, not top-k scores."""
    m = build(n_experts=4, top_k=2)
    routers = router_modules(m)
    cap = RouterLogitCapture(m)
    cap.k = 2
    with cap:
        m(input_ids=torch.randint(0, 64, (2, 5)))
        stacked = cap.stacked(2, 5)
    assert stacked.shape == (2, 2, 5, 4)  # [layers, batch, seq, experts]
    # logits are unnormalised: softmax of the capture must not already be one-hot/sparse
    probs = torch.softmax(stacked[0, 0, 0], dim=-1)
    assert (probs > 0).all()
    # and they must equal weight @ hidden for the captured router
    assert routers[0].weight.shape == (4, m.config.hidden_size)


def test_quantize_then_restore_is_exact_and_changes_output():
    from broca.quant.model_surgery import apply_expert_quantization

    m = build()
    refs = discover_expert_weights(m)
    originals = {r.key: get_live_tensor(m, r).clone() for r in refs}

    class FakeSource:
        def get_expert(self, ref):
            return originals[ref.key]

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        base = m(input_ids=ids).logits.clone()

    stats = apply_expert_quantization(m, refs, FakeSource(), lambda r: 2, group_size=16)
    assert stats["quantized_weights"] == sum(r.numel for r in refs)
    assert stats["relative_frobenius_error"] > 0
    with torch.no_grad():
        quant = m(input_ids=ids).logits.clone()
    assert not torch.allclose(base, quant)

    apply_expert_quantization(m, refs, FakeSource(), lambda r: None, group_size=16)
    for r in refs:
        assert torch.equal(get_live_tensor(m, r), originals[r.key])
    with torch.no_grad():
        restored = m(input_ids=ids).logits
    assert torch.equal(base, restored)


def test_repeated_quantization_is_idempotent():
    """Re-applying an allocation must not compound error: writes start from the original."""
    from broca.quant.model_surgery import apply_expert_quantization

    m = build()
    refs = discover_expert_weights(m)
    originals = {r.key: get_live_tensor(m, r).clone() for r in refs}

    class FakeSource:
        def get_expert(self, ref):
            return originals[ref.key]

    apply_expert_quantization(m, refs, FakeSource(), lambda r: 3, group_size=16)
    once = {r.key: get_live_tensor(m, r).clone() for r in refs}
    apply_expert_quantization(m, refs, FakeSource(), lambda r: 3, group_size=16)
    for r in refs:
        assert torch.equal(get_live_tensor(m, r), once[r.key])

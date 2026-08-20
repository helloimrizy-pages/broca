"""Memory accounting checked against a hand-computed example.

The example: 2 layers, 2 experts per layer, 3 projections each.  gate_proj and
up_proj are [1024, 2048]; down_proj is [2048, 1024].  Group size 64, bf16 scale
and zero point (16 + 16 = 32 bits of metadata per group).  Non-expert parameters
are fixed at 1,000,000 and always stored at bf16.
"""

import pytest

from broca.quant.memory import (
    BF16_BITS,
    expert_storage_bits,
    stored_bits_per_weight,
)
from broca.quant.model_surgery import ExpertWeightRef

G = 64
SCALE_BITS = ZERO_BITS = 16


def make_refs():
    refs = []
    for layer in range(2):
        for expert in range(2):
            for proj, (out_f, in_f) in {
                "gate_proj": (1024, 2048),
                "up_proj": (1024, 2048),
                "down_proj": (2048, 1024),
            }.items():
                refs.append(ExpertWeightRef(
                    layer=layer, expert=expert, proj=proj,
                    param_name=f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
                    expert_slice=None, transposed=False,
                    out_features=out_f, in_features=in_f,
                ))
    return refs


def test_stored_bits_per_weight_hand_computed():
    # 2048 inputs / 64 = 32 groups per output channel, 32 metadata bits each
    # -> 32 * 32 / 2048 = 0.5 extra bits per weight
    assert stored_bits_per_weight(3, 2048, G) == pytest.approx(3.5)
    assert stored_bits_per_weight(3, 1024, G) == pytest.approx(3.5)
    assert stored_bits_per_weight(4, 2048, G, 16, 16) == pytest.approx(4.5)
    # group size 128 halves the overhead
    assert stored_bits_per_weight(3, 2048, 128) == pytest.approx(3.25)
    # int8 zero point instead of bf16
    assert stored_bits_per_weight(3, 2048, G, 16, 8) == pytest.approx(3 + 24 * 32 / 2048)


def test_uniform_allocation_totals_hand_computed():
    refs = make_refs()
    # 4 experts total, each 3 matrices of 1024*2048 = 2,097,152 weights
    expected_weights = 4 * 3 * 1024 * 2048
    stored, weights, payload = expert_storage_bits(refs, lambda r: 3, G, SCALE_BITS, ZERO_BITS)
    assert weights == expected_weights
    assert payload == pytest.approx(3 * expected_weights)
    assert stored == pytest.approx(3.5 * expected_weights)
    assert stored / weights == pytest.approx(3.5)


def test_mixed_allocation_average_is_weight_weighted():
    refs = make_refs()
    # give layer 0 two bits and layer 1 four bits; every matrix is the same size,
    # so the weighted average payload is exactly 3.0 and stored is 3.5
    stored, weights, payload = expert_storage_bits(
        refs, lambda r: 2 if r.layer == 0 else 4, G, SCALE_BITS, ZERO_BITS
    )
    assert payload / weights == pytest.approx(3.0)
    assert stored / weights == pytest.approx(3.5)


def test_bf16_experts_carry_no_quantizer_overhead():
    refs = make_refs()
    stored, weights, payload = expert_storage_bits(refs, lambda r: None, G, SCALE_BITS, ZERO_BITS)
    assert stored / weights == pytest.approx(BF16_BITS)
    assert payload / weights == pytest.approx(BF16_BITS)


def test_total_bytes_hand_computed():
    refs = make_refs()
    non_expert_params = 1_000_000
    stored, weights, _ = expert_storage_bits(refs, lambda r: 3, G, SCALE_BITS, ZERO_BITS)
    total_bytes = (stored + non_expert_params * BF16_BITS) / 8
    expected = (3.5 * 4 * 3 * 1024 * 2048 + 16 * 1_000_000) / 8
    assert total_bytes == pytest.approx(expected)
    assert total_bytes / 1024 ** 3 == pytest.approx(expected / 1024 ** 3)

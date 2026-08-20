"""Round-trip properties of the simulated RTN quantiser.

A silent bug here invalidates every downstream number, so the tests check the
grid itself, not just that the error is "small".
"""

import numpy as np
import pytest
import torch

from broca.quant.rtn import QuantConfig, n_groups, quantize_dequantize


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_group_takes_at_most_2b_distinct_values(bits):
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    wq = quantize_dequantize(w, bits=bits, group_size=64)
    assert wq.shape == w.shape
    for row in range(w.shape[0]):
        for g in range(2):
            block = wq[row, g * 64:(g + 1) * 64]
            assert torch.unique(block).numel() <= 2 ** bits


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_dequantized_values_lie_on_the_group_grid(bits):
    torch.manual_seed(1)
    w = torch.randn(4, 64)
    wq = quantize_dequantize(w, bits=bits, group_size=64)
    for row in range(w.shape[0]):
        lo, hi = w[row].min(), w[row].max()
        scale = (hi - lo) / (2 ** bits - 1)
        # every output is an integer multiple of the scale away from the others
        steps = (wq[row] - wq[row].min()) / scale
        assert torch.allclose(steps, steps.round(), atol=1e-3)


def test_error_decreases_monotonically_with_bits():
    torch.manual_seed(2)
    w = torch.randn(64, 256)
    errs = []
    for bits in (2, 3, 4, 6, 8):
        wq = quantize_dequantize(w, bits=bits, group_size=64)
        errs.append(float((wq - w).pow(2).mean()))
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_error_scales_roughly_as_quantization_step_squared():
    """MSE should fall by ~4x per extra bit for a smooth distribution."""
    torch.manual_seed(3)
    w = torch.randn(64, 1024)
    e4 = float((quantize_dequantize(w, 4, 64) - w).pow(2).mean())
    e5 = float((quantize_dequantize(w, 5, 64) - w).pow(2).mean())
    assert 2.5 < e4 / e5 < 6.0, e4 / e5


def test_constant_and_zero_groups_are_exact():
    w = torch.full((3, 64), 0.7)
    assert torch.allclose(quantize_dequantize(w, 2, 64), w, atol=1e-6)
    w = torch.zeros(3, 64)
    assert torch.allclose(quantize_dequantize(w, 2, 64), w, atol=1e-9)
    w = torch.full((3, 64), -1.25)
    assert torch.allclose(quantize_dequantize(w, 3, 64), w, atol=1e-6)


def test_smaller_groups_give_lower_error():
    torch.manual_seed(4)
    w = torch.randn(32, 512) * torch.linspace(0.1, 10, 512)  # heteroscedastic across input dim
    e_big = float((quantize_dequantize(w, 3, 512) - w).pow(2).mean())
    e_small = float((quantize_dequantize(w, 3, 64) - w).pow(2).mean())
    assert e_small < e_big


def test_partial_trailing_group_is_handled():
    torch.manual_seed(5)
    w = torch.randn(4, 100)  # 100 = 64 + 36
    wq = quantize_dequantize(w, bits=3, group_size=64)
    assert wq.shape == w.shape
    assert torch.unique(wq[0, 64:]).numel() <= 8
    assert n_groups(100, 64) == 2


def test_bf16_round_trip_preserves_dtype():
    w = torch.randn(4, 64, dtype=torch.bfloat16)
    wq = quantize_dequantize(w, bits=4, group_size=64)
    assert wq.dtype == torch.bfloat16


def test_quantconfig_bits_per_weight():
    cfg = QuantConfig(bits=3, group_size=64, scale_bits=16, zero_bits=16)
    assert cfg.bits_per_weight() == pytest.approx(3.5)
    assert cfg.bits_per_weight(in_features=2048) == pytest.approx(3 + 32 * 32 / 2048)
    # a partial trailing group costs more than the idealised formula
    assert cfg.bits_per_weight(in_features=100) == pytest.approx(3 + 32 * 2 / 100)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        quantize_dequantize(torch.randn(4), 3, 64)
    with pytest.raises(ValueError):
        quantize_dequantize(torch.randn(4, 8), 0, 64)
    with pytest.raises(ValueError):
        quantize_dequantize(torch.randn(4, 8), 3, 0)

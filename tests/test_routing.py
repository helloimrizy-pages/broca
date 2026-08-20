"""Routing agreement metrics, including the exact single-swap value at k=8."""

import pytest
import torch

from broca.eval.routing import (
    LayerAgreement,
    jaccard_from_overlap,
    jaccard_sets,
    overlap_counts,
    topk_indices,
)

K = 8
E = 64


def test_identical_sets_give_jaccard_one():
    a = torch.arange(K).unsqueeze(0)
    m = overlap_counts(a, a.clone(), E)
    assert int(m[0]) == K
    assert float(jaccard_from_overlap(m, K)[0]) == pytest.approx(1.0)


def test_single_swap_gives_seven_ninths():
    a = torch.arange(K).unsqueeze(0)
    b = a.clone()
    b[0, 0] = 60  # replace one expert with one not in a
    m = overlap_counts(a, b, E)
    assert int(m[0]) == K - 1
    assert float(jaccard_from_overlap(m, K)[0]) == pytest.approx(7 / 9)
    assert jaccard_sets(set(range(K)), set(b[0].tolist())) == pytest.approx(7 / 9)


def test_disjoint_sets_give_zero():
    a = torch.arange(K).unsqueeze(0)
    b = torch.arange(K, 2 * K).unsqueeze(0)
    m = overlap_counts(a, b, E)
    assert int(m[0]) == 0
    assert float(jaccard_from_overlap(m, K)[0]) == 0.0


@pytest.mark.parametrize("swaps", range(0, K + 1))
def test_jaccard_matches_set_definition_for_every_overlap(swaps):
    a = set(range(K))
    b = set(range(K - swaps)) | set(range(K, K + swaps))
    m = torch.tensor([len(a & b)])
    assert float(jaccard_from_overlap(m, K)[0]) == pytest.approx(jaccard_sets(a, b))


def test_topk_selection_is_order_agnostic_for_membership():
    logits = torch.randn(16, E)
    idx = topk_indices(logits, K)
    assert idx.shape == (16, K)
    # permuting the logits' ties should not change the selected set for distinct values
    m = overlap_counts(idx, topk_indices(logits.clone(), K), E)
    assert torch.all(m == K)


def test_layer_agreement_accumulates_correctly():
    torch.manual_seed(0)
    ref = torch.randn(128, E)
    agree = LayerAgreement(n_experts=E, k=K)
    agree.update(ref, ref.clone())
    s = agree.summary()
    assert s["mean_jaccard"] == pytest.approx(1.0)
    assert s["mean_overlap"] == pytest.approx(float(K))
    assert s["top1_agreement"] == pytest.approx(1.0)
    assert s["mean_logit_l2_shift"] == pytest.approx(0.0)
    assert s["fraction_identical_sets"] == pytest.approx(1.0)
    assert s["overlap_histogram"][K] == 128


def test_layer_agreement_mean_of_perfect_and_single_swap():
    """Two tokens, one perfect and one single-swap: mean Jaccard is (1 + 7/9) / 2."""
    ref = torch.zeros(2, E)
    ref[:, :K] = torch.arange(K, 0, -1).float()
    quant = ref.clone()
    quant[1, 0] = -1.0        # drop expert 0 out of the top-k
    quant[1, 60] = 100.0      # and pull in expert 60
    agree = LayerAgreement(n_experts=E, k=K)
    agree.update(ref, quant)
    s = agree.summary()
    assert s["mean_jaccard"] == pytest.approx((1.0 + 7 / 9) / 2)
    assert s["mean_overlap"] == pytest.approx((8 + 7) / 2)
    assert s["fraction_identical_sets"] == pytest.approx(0.5)

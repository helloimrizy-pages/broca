"""Routing agreement metrics between two runs of the same model.

Definitions used throughout the repo, fixed here so every phase reports the same
quantity:

overlap count
    ``m = |A ∩ B|`` where ``A`` and ``B`` are the top-k expert sets chosen by the
    reference and the quantised model for the same token at the same layer.
    Ranges over ``0..k``.

Jaccard index
    ``|A ∩ B| / |A ∪ B|``.  Since ``|A| = |B| = k``, the union is ``2k - m`` and
    the index is exactly ``m / (2k - m)``.  At ``k = 8`` a single expert swap
    gives ``7 / 9 = 0.7778``, which is why the overlap count is reported
    alongside: a mean Jaccard of 0.88 is compatible with most tokens being
    perfectly preserved and the rest losing exactly one expert.

top-1 agreement
    Fraction of tokens where the highest-scoring expert is the same.

normalised router logit shift
    ``||z_ref - z_quant||_2`` per token, divided by the standard deviation of the
    reference router logits at that layer (a single scalar per layer, taken over
    all tokens and experts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def topk_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k expert indices from router logits ``[N, E]``.

    Softmax is monotonic, so selecting on logits matches selecting on routing
    probabilities; this holds for OLMoE, which applies softmax before top-k.
    """
    return torch.topk(logits, k=k, dim=-1).indices


def _membership(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Boolean membership matrix ``[N, E]`` from top-k indices ``[N, k]``."""
    out = torch.zeros(indices.shape[0], n_experts, dtype=torch.bool, device=indices.device)
    out.scatter_(1, indices, True)
    return out


def overlap_counts(a: torch.Tensor, b: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Per-token ``|A ∩ B|`` for two sets of top-k indices ``[N, k]``."""
    return (_membership(a, n_experts) & _membership(b, n_experts)).sum(dim=1)


def jaccard_from_overlap(overlap: torch.Tensor, k: int) -> torch.Tensor:
    """``m / (2k - m)``; exact for equal-size top-k sets."""
    m = overlap.to(torch.float64)
    return m / (2 * k - m)


def jaccard_sets(a: set[int], b: set[int]) -> float:
    """Set-level Jaccard, used by the tests as an independent reference."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


@dataclass
class LayerAgreement:
    """Streaming accumulator of routing agreement for one layer."""

    n_experts: int
    k: int
    tokens: int = 0
    overlap_sum: float = 0.0
    jaccard_sum: float = 0.0
    top1_agree: int = 0
    logit_shift_sum: float = 0.0
    logit_shift_sq_sum: float = 0.0
    ref_logit_sq_sum: float = 0.0
    ref_logit_sum: float = 0.0
    ref_logit_count: int = 0
    overlap_hist: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.overlap_hist is None:
            self.overlap_hist = np.zeros(self.k + 1, dtype=np.int64)

    @torch.no_grad()
    def update(self, ref_logits: torch.Tensor, quant_logits: torch.Tensor) -> torch.Tensor:
        """Accumulate one batch of ``[N, E]`` logits; returns per-token overlap."""
        ref_logits = ref_logits.to(torch.float32)
        quant_logits = quant_logits.to(torch.float32)
        a = topk_indices(ref_logits, self.k)
        b = topk_indices(quant_logits, self.k)
        m = overlap_counts(a, b, self.n_experts)

        self.tokens += ref_logits.shape[0]
        self.overlap_sum += float(m.sum())
        self.jaccard_sum += float(jaccard_from_overlap(m, self.k).sum())
        self.top1_agree += int((a[:, 0] == b[:, 0]).sum())
        self.overlap_hist += np.bincount(m.cpu().numpy(), minlength=self.k + 1)

        shift = (ref_logits - quant_logits).norm(dim=-1)
        self.logit_shift_sum += float(shift.sum())
        self.logit_shift_sq_sum += float(shift.pow(2).sum())
        self.ref_logit_sum += float(ref_logits.sum())
        self.ref_logit_sq_sum += float(ref_logits.pow(2).sum())
        self.ref_logit_count += ref_logits.numel()
        return m

    @property
    def ref_logit_std(self) -> float:
        if self.ref_logit_count == 0:
            return float("nan")
        mean = self.ref_logit_sum / self.ref_logit_count
        var = self.ref_logit_sq_sum / self.ref_logit_count - mean ** 2
        return float(np.sqrt(max(var, 0.0)))

    def summary(self) -> dict:
        n = max(self.tokens, 1)
        std = self.ref_logit_std
        mean_shift = self.logit_shift_sum / n
        return {
            "tokens": self.tokens,
            "mean_jaccard": self.jaccard_sum / n,
            "mean_overlap": self.overlap_sum / n,
            "k": self.k,
            "top1_agreement": self.top1_agree / n,
            "mean_logit_l2_shift": mean_shift,
            "ref_logit_std": std,
            "mean_logit_l2_shift_normalized": mean_shift / std if std > 0 else float("nan"),
            "overlap_histogram": self.overlap_hist.tolist(),
            "fraction_identical_sets": float(self.overlap_hist[self.k] / n),
        }

"""Round-to-nearest weight quantisation, simulated.

All quantisation in this repo is *simulated*: weights are quantised to a
low-bit grid and immediately dequantised back to the model dtype (bf16).  No
packed kernels are used, so measured perplexity and accuracy reflect the
numerics of the grid but not any runtime speedup.  Heterogeneous per-expert
bit-widths inside one layer are not supported by any inference runtime we have,
which is why the simulated setting is the honest one to report.

Grouping convention
-------------------
For a ``torch.nn.Linear`` weight of shape ``[out_features, in_features]`` we
form groups of ``group_size`` consecutive elements *along the input dimension*,
separately for each output channel.  Each group gets its own asymmetric scale
and zero point.  This is the "per-output-channel group of g" convention used by
GPTQ/AWQ-style RTN baselines.  If ``in_features`` is not a multiple of
``group_size`` the trailing remainder forms one smaller group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _affine_qdq(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Asymmetric round-to-nearest quantise/dequantise over the last dimension.

    ``w`` has shape ``[..., group_size]``; statistics are taken over the last
    dimension.  Returns a tensor of the same shape and dtype as ``w``.
    """
    qmax = float(2 ** bits - 1)
    w_min = w.amin(dim=-1, keepdim=True)
    w_max = w.amax(dim=-1, keepdim=True)
    rng = w_max - w_min
    # A constant group is exactly representable: fall back to |value| as the
    # range so scale/zero-point reproduce it without error.  Guards against
    # divide-by-zero for all-zero groups too.
    rng = torch.where(rng > 0, rng, w_max.abs().clamp(min=1e-8))
    scale = rng / qmax
    zero = torch.clamp(torch.round(-w_min / scale), 0.0, qmax)
    q = torch.clamp(torch.round(w / scale) + zero, 0.0, qmax)
    return (q - zero) * scale


def quantize_dequantize(
    weight: torch.Tensor, bits: int, group_size: int, compute_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Simulated RTN quantisation of a 2-D Linear weight ``[out, in]``.

    Returns a tensor with the same shape and dtype as ``weight``.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D Linear weight, got shape {tuple(weight.shape)}")
    if not (1 <= bits <= 16):
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    orig_dtype = weight.dtype
    w = weight.to(compute_dtype)
    out_f, in_f = w.shape
    g = min(group_size, in_f)
    n_full = in_f // g
    tail = in_f - n_full * g

    parts = []
    if n_full:
        head = w[:, : n_full * g].reshape(out_f, n_full, g)
        parts.append(_affine_qdq(head, bits).reshape(out_f, n_full * g))
    if tail:
        parts.append(_affine_qdq(w[:, n_full * g:], bits))
    return torch.cat(parts, dim=1).to(orig_dtype)


def n_groups(in_features: int, group_size: int) -> int:
    """Number of quantisation groups per output channel."""
    g = min(group_size, in_features)
    return (in_features + g - 1) // g


@dataclass(frozen=True)
class QuantConfig:
    """Configuration for one simulated RTN setting."""

    bits: int
    group_size: int = 64
    scale_bits: int = 16
    zero_bits: int = 16

    def bits_per_weight(self, in_features: int | None = None) -> float:
        """Effective storage cost per weight, including scale/zero overhead.

        With ``in_features`` given, the trailing partial group (if any) is
        accounted for exactly; otherwise the idealised ``b + (s + z) / g`` is
        returned.
        """
        overhead = self.scale_bits + self.zero_bits
        if in_features is None:
            return self.bits + overhead / self.group_size
        groups = n_groups(in_features, self.group_size)
        return self.bits + overhead * groups / in_features

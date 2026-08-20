"""Memory accounting for a per-expert bit allocation.

Two averages are reported and they are not the same number:

``average_bits_payload``
    Mean of the assigned bit-widths ``b`` over expert weights.  This is the
    number people usually quote, and it undercounts real memory.

``average_bits_stored``
    Mean of ``b + (scale_bits + zero_bits) * n_groups / in_features`` over expert
    weights, i.e. including the per-group scale and zero point.  At group size 64
    with bf16 scale and zero this adds 0.5 bits per weight.  Budgets in this repo
    are enforced on the *stored* figure, because that is what occupies memory.

Non-expert parameters (router, attention, embeddings, norms, lm_head) are always
counted at bf16, since nothing outside the experts is quantised.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch

from .model_surgery import ExpertWeightRef
from .rtn import n_groups

BF16_BITS = 16
BYTES_PER_GIB = 1024 ** 3
BYTES_PER_GB = 10 ** 9


def stored_bits_per_weight(
    bits: int, in_features: int, group_size: int, scale_bits: int = 16, zero_bits: int = 16
) -> float:
    """Effective stored bits per weight including per-group quantiser metadata."""
    groups = n_groups(in_features, group_size)
    return bits + (scale_bits + zero_bits) * groups / in_features


def expert_storage_bits(
    refs: Iterable[ExpertWeightRef],
    bits_for: Callable[[ExpertWeightRef], int],
    group_size: int,
    scale_bits: int = 16,
    zero_bits: int = 16,
) -> tuple[float, int, float]:
    """Return ``(total_stored_bits, total_weights, total_payload_bits)``."""
    total_stored = 0.0
    total_payload = 0.0
    total_weights = 0
    for ref in refs:
        b = bits_for(ref)
        bpw = (
            float(BF16_BITS)
            if b is None or b >= BF16_BITS
            else stored_bits_per_weight(b, ref.in_features, group_size, scale_bits, zero_bits)
        )
        total_stored += bpw * ref.numel
        total_payload += (BF16_BITS if b is None else b) * ref.numel
        total_weights += ref.numel
    return total_stored, total_weights, total_payload


def model_memory(
    model: torch.nn.Module,
    refs: list[ExpertWeightRef],
    bits_for: Callable[[ExpertWeightRef], int],
    group_size: int,
    scale_bits: int = 16,
    zero_bits: int = 16,
) -> dict:
    """Full memory report for one allocation."""
    stored_bits, expert_weights, payload_bits = expert_storage_bits(
        refs, bits_for, group_size, scale_bits, zero_bits
    )
    total_params = sum(p.numel() for p in model.parameters())
    non_expert_params = total_params - expert_weights
    non_expert_bits = non_expert_params * BF16_BITS
    total_bits = stored_bits + non_expert_bits
    return {
        "expert_weights": expert_weights,
        "non_expert_parameters": non_expert_params,
        "total_parameters": total_params,
        "expert_parameter_fraction": expert_weights / total_params if total_params else None,
        "average_bits_payload": payload_bits / expert_weights if expert_weights else None,
        "average_bits_stored": stored_bits / expert_weights if expert_weights else None,
        "expert_bytes": stored_bits / 8,
        "non_expert_bytes": non_expert_bits / 8,
        "total_bytes": total_bits / 8,
        "total_gib": total_bits / 8 / BYTES_PER_GIB,
        "total_gb": total_bits / 8 / BYTES_PER_GB,
        "bf16_total_gib": total_params * BF16_BITS / 8 / BYTES_PER_GIB,
        "group_size": group_size,
        "scale_bits": scale_bits,
        "zero_bits": zero_bits,
    }


def expert_bit_budget_bits(
    refs: list[ExpertWeightRef], target_average_bits_stored: float
) -> float:
    """Total stored-bit budget for the expert weights at a target average."""
    return target_average_bits_stored * sum(r.numel for r in refs)

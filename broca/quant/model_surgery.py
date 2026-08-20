"""Locating MoE expert weights in the loaded model and rewriting them in place.

Two things this module is careful about:

1.  It *discovers* the architecture from the loaded model rather than assuming
    it.  Depending on the transformers version, MoE experts are stored either as
    one ``nn.Linear`` per expert (``mlp.experts[e].gate_proj.weight``) or fused
    into a 3-D parameter indexed by expert.  Both layouts are handled, and every
    weight is normalised to the logical ``[out_features, in_features]``
    orientation before quantisation so that grouping always runs along the input
    dimension.

2.  It restores original weights from the on-disk safetensors shards rather than
    keeping a second full copy of the expert weights in memory.  The expert
    weights are ~94% of this model, so a resident copy would roughly double peak
    memory; memory-mapped reads cost one tensor at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from safetensors import safe_open

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class ExpertWeightRef:
    """A handle on one expert projection matrix inside the live model."""

    layer: int
    expert: int
    proj: str
    param_name: str          # name in the checkpoint / state dict
    expert_slice: int | None  # index into a fused [E, ...] parameter, else None
    transposed: bool          # True if stored as [in, out] rather than [out, in]
    out_features: int
    in_features: int

    @property
    def numel(self) -> int:
        return self.out_features * self.in_features

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.layer, self.expert, self.proj)


def _moe_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the per-layer MoE blocks, in layer order."""
    layers = model.model.layers
    blocks = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "experts"):
            raise RuntimeError(
                "expected every decoder layer to carry an MoE block with .experts; "
                f"found {type(mlp).__name__}"
            )
        blocks.append(mlp)
    return blocks


def router_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the router (gate) Linear of each MoE layer, in layer order."""
    mods = []
    for block in _moe_blocks(model):
        gate = getattr(block, "gate", None)
        if gate is None:
            raise RuntimeError(f"MoE block {type(block).__name__} has no .gate router module")
        mods.append(gate)
    return mods


def discover_expert_weights(model: torch.nn.Module) -> list[ExpertWeightRef]:
    """Enumerate every expert projection matrix in the model."""
    refs: list[ExpertWeightRef] = []
    for l, block in enumerate(_moe_blocks(model)):
        experts = block.experts
        if isinstance(experts, torch.nn.ModuleList) or hasattr(experts, "__getitem__") and hasattr(experts, "__len__"):
            try:
                first = experts[0]
            except (TypeError, IndexError, KeyError):
                first = None
        else:
            first = None

        if first is not None and all(hasattr(first, p) for p in PROJECTIONS):
            # Unfused: one nn.Linear per projection per expert.
            for e in range(len(experts)):
                for proj in PROJECTIONS:
                    w = getattr(experts[e], proj).weight
                    refs.append(
                        ExpertWeightRef(
                            layer=l, expert=e, proj=proj,
                            param_name=f"model.layers.{l}.mlp.experts.{e}.{proj}.weight",
                            expert_slice=None, transposed=False,
                            out_features=w.shape[0], in_features=w.shape[1],
                        )
                    )
        else:
            # Fused: 3-D parameters on the experts container.
            fused = {n: p for n, p in experts.named_parameters(recurse=True) if p.dim() == 3}
            if not fused:
                raise RuntimeError(
                    "could not identify expert weights: experts container exposes neither "
                    f"per-expert projections nor 3-D fused parameters (type {type(experts).__name__})"
                )
            for name, p in sorted(fused.items()):
                n_experts = p.shape[0]
                for e in range(n_experts):
                    refs.append(
                        ExpertWeightRef(
                            layer=l, expert=e, proj=name.split(".")[-1],
                            param_name=f"model.layers.{l}.mlp.experts.{name}",
                            expert_slice=e, transposed=True,
                            out_features=p.shape[2], in_features=p.shape[1],
                        )
                    )
    if not refs:
        raise RuntimeError("no expert weights discovered")
    return refs


def get_live_tensor(model: torch.nn.Module, ref: ExpertWeightRef) -> torch.Tensor:
    """Return the live ``[out, in]`` view of an expert matrix in the model."""
    param = model.get_parameter(ref.param_name)
    t = param if ref.expert_slice is None else param[ref.expert_slice]
    return t.transpose(0, 1) if ref.transposed else t


@torch.no_grad()
def write_live_tensor(model: torch.nn.Module, ref: ExpertWeightRef, value: torch.Tensor) -> None:
    """Copy a ``[out, in]`` tensor back into the model in place."""
    target = get_live_tensor(model, ref)
    target.copy_(value.to(target.dtype))


class OriginalWeights:
    """Memory-mapped access to the unmodified checkpoint tensors."""

    def __init__(self, model_path: str | Path):
        self.root = Path(model_path)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.root / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors checkpoint under {self.root}")
            self.map = None
            self._single = single
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.root / shard), framework="pt", device="cpu")
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name] if self.map is not None else self._single.name
        return self._handle(shard).get_tensor(name)

    def get_expert(self, ref: ExpertWeightRef) -> torch.Tensor:
        """Return the original ``[out, in]`` matrix for one expert projection."""
        t = self.get(ref.param_name)
        if ref.expert_slice is not None:
            t = t[ref.expert_slice]
        if ref.transposed:
            t = t.transpose(0, 1)
        return t


@torch.no_grad()
def apply_expert_quantization(
    model: torch.nn.Module,
    refs: list[ExpertWeightRef],
    originals: OriginalWeights,
    bits_for: Callable[[ExpertWeightRef], int | None],
    group_size: int,
    progress: bool = False,
) -> dict[str, float]:
    """Rewrite every expert matrix from its *original* value at the given bits.

    ``bits_for`` returns the bit-width for a given expert projection, or ``None``
    to restore the original bf16 weight.  Because every write starts from the
    checkpoint value, this is idempotent and safe to call repeatedly with
    different allocations on the same live model.
    """
    from .rtn import quantize_dequantize

    iterator: Iterator[ExpertWeightRef] = iter(refs)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(refs, desc="quantising experts", unit="mat")

    sq_err = 0.0
    sq_ref = 0.0
    n = 0
    for ref in iterator:
        w0 = originals.get_expert(ref)
        bits = bits_for(ref)
        target = get_live_tensor(model, ref)
        if bits is None:
            target.copy_(w0.to(target.dtype).to(target.device))
            continue
        w0d = w0.to(target.device)
        wq = quantize_dequantize(w0d, bits=bits, group_size=group_size)
        target.copy_(wq.to(target.dtype))
        diff = (wq.float() - w0d.float())
        sq_err += float(diff.pow(2).sum())
        sq_ref += float(w0d.float().pow(2).sum())
        n += ref.numel
    return {
        "quantized_weights": n,
        "relative_frobenius_error": (sq_err / sq_ref) ** 0.5 if sq_ref > 0 else 0.0,
    }


def describe_architecture(model: torch.nn.Module, refs: list[ExpertWeightRef]) -> dict:
    """Architecture facts read off the loaded model, not off the proposal."""
    cfg = model.config
    expert_numel = sum(r.numel for r in refs)
    total_numel = sum(p.numel() for p in model.parameters())
    layers = sorted({r.layer for r in refs})
    per_layer_experts = {l: len({r.expert for r in refs if r.layer == l}) for l in layers}
    return {
        "model_type": getattr(cfg, "model_type", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_experts": getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", None)),
        "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "norm_topk_prob": getattr(cfg, "norm_topk_prob", None),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
        "moe_layers": layers,
        "experts_per_layer": per_layer_experts,
        "projections_per_expert": sorted({r.proj for r in refs}),
        "expert_parameters": expert_numel,
        "total_parameters": total_numel,
        "expert_parameter_fraction": expert_numel / total_numel if total_numel else None,
    }

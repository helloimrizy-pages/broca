"""Forward hooks for capturing router behaviour.

Phase 0 needs only the router logits.  The full ``RouterTrace`` used from Phase 1
onward (router input hidden states, gate weights, expert inputs and outputs) is
built on the same capture mechanism.

The MoE block flattens ``[batch, seq, hidden]`` to ``[batch * seq, hidden]``
before applying the router, so captured rows are in row-major ``(batch, seq)``
order and reshape back to ``[batch, seq, num_experts]`` without reordering.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from ..quant.model_surgery import router_modules


class RouterLogitCapture:
    """Collects per-layer router logits for the current forward pass."""

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype = torch.float32):
        self.routers = router_modules(model)
        self.n_layers = len(self.routers)
        self.dtype = dtype
        self.buffers: list[torch.Tensor | None] = [None] * self.n_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int):
        def hook(module, args, output):
            logits = output[0] if isinstance(output, tuple) else output
            self.buffers[layer] = logits.detach().to(self.dtype).cpu()
        return hook

    def __enter__(self) -> "RouterLogitCapture":
        for l, router in enumerate(self.routers):
            self._handles.append(router.register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.buffers = [None] * self.n_layers

    def stacked(self, batch: int, seq: int) -> torch.Tensor:
        """Return captured logits as ``[n_layers, batch, seq, n_experts]``."""
        out = []
        for l, buf in enumerate(self.buffers):
            if buf is None:
                raise RuntimeError(f"no router logits captured for layer {l}")
            if buf.shape[0] != batch * seq:
                raise RuntimeError(
                    f"layer {l}: captured {buf.shape[0]} router rows, expected {batch * seq}"
                )
            out.append(buf.view(batch, seq, -1))
        return torch.stack(out, dim=0)


@contextmanager
def capture_router_logits(model: torch.nn.Module) -> Iterator[RouterLogitCapture]:
    cap = RouterLogitCapture(model)
    with cap:
        yield cap

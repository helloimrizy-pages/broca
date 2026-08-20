"""Phase 0: does quantising the experts actually move the router?

Runs the same calibration tokens through the bf16 model and through uniformly
quantised copies at several bit-widths, and compares the top-k expert sets token
by token and layer by layer.

Kill criterion, fixed before the run: if mean Jaccard at 3-bit is at or above
0.88, routing is close to invariant under quantisation on this model and the
premise of B-ROCA is weak.

If routing does shift, the second question is whether the shift is the mechanism
behind the quality loss.  We correlate per-token routing disagreement (the number
of expert swaps summed over layers) against the per-token increase in negative
log likelihood.  A correlation near zero means the shift is real but is not what
costs perplexity, and allocating bits to protect routing has no reason to help.

Because routing disagreement and NLL increase could both be driven by token
difficulty, the partial correlation controlling for the bf16 NLL of the same
token is reported alongside the raw one.

Alignment: the logits at position ``t`` predict token ``t+1``, so the NLL at
position ``t`` is compared against the routing decisions taken at position ``t``.
The final position of each sequence has routing but no NLL and is dropped from
the correlation; it is still counted in the agreement metrics.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from ..data.calib import CalibSpec, load_tokens
from ..eval.routing import LayerAgreement
from ..modeling import DEFAULT_MODEL, load_model_and_tokenizer
from ..quant.memory import model_memory
from ..quant.model_surgery import (
    OriginalWeights,
    apply_expert_quantization,
    describe_architecture,
    discover_expert_weights,
)
from ..trace.hooks import RouterLogitCapture
from ..utils import CACHE_DIR, RunRecord, Timer, device_info, resolve_device, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 0 routing-shift diagnostic")
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--scale-bits", type=int, default=16)
    p.add_argument("--zero-bits", type=int, default=16)
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kill-threshold", type=float, default=0.88,
                   help="mean Jaccard at the reference bit-width at or above which we stop")
    p.add_argument("--kill-bits", type=int, default=3)
    p.add_argument("--out", default="phase0_diagnostic.json")
    p.add_argument("--limit-sequences", type=int, default=None, help="debug: cap sequences")
    return p.parse_args(argv)


@torch.no_grad()
def token_nll(model, ids: torch.Tensor) -> np.ndarray:
    """Per-position NLL of the next token, ``[batch, seq - 1]``."""
    out = model(input_ids=ids)
    logits = out.logits[:, :-1]
    tgt = ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction="none"
    ).view(tgt.shape)
    return nll.cpu().numpy()


@torch.no_grad()
def reference_pass(model, tokens, device, batch_size, capture, store_path):
    """bf16 pass: write router logits to a memmap and return per-token NLL."""
    n_seq, seq_len = tokens.shape
    nll = np.zeros((n_seq, seq_len - 1), dtype=np.float32)
    store = None
    for start in tqdm(range(0, n_seq, batch_size), desc="bf16", unit="batch"):
        stop = min(start + batch_size, n_seq)
        ids = torch.from_numpy(tokens[start:stop].astype(np.int64)).to(device)
        capture.reset()
        nll[start:stop] = token_nll(model, ids)
        router = capture.stacked(stop - start, seq_len)
        if store is None:
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store = np.lib.format.open_memmap(
                store_path, mode="w+", dtype=np.float32,
                shape=(capture.n_layers, n_seq, seq_len, router.shape[-1]),
            )
        store[:, start:stop] = router.numpy()
        del router
    store.flush()
    return nll, store


@torch.no_grad()
def comparison_pass(model, tokens, device, batch_size, capture, store, desc):
    """Quantised pass: stream against the stored reference logits."""
    n_seq, seq_len = tokens.shape
    n_experts = store.shape[-1]
    k = capture.k
    nll = np.zeros((n_seq, seq_len - 1), dtype=np.float32)
    swaps = np.zeros((n_seq, seq_len), dtype=np.int16)
    agreements = [LayerAgreement(n_experts=n_experts, k=k) for _ in range(capture.n_layers)]
    for start in tqdm(range(0, n_seq, batch_size), desc=desc, unit="batch"):
        stop = min(start + batch_size, n_seq)
        b = stop - start
        ids = torch.from_numpy(tokens[start:stop].astype(np.int64)).to(device)
        capture.reset()
        nll[start:stop] = token_nll(model, ids)
        router = capture.stacked(b, seq_len)
        ref = torch.from_numpy(np.asarray(store[:, start:stop]))
        for l in range(capture.n_layers):
            m = agreements[l].update(
                ref[l].reshape(-1, n_experts), router[l].reshape(-1, n_experts)
            )
            swaps[start:stop] += (k - m).view(b, seq_len).numpy().astype(np.int16)
        del router, ref
    return nll, agreements, swaps


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of x and y after linearly regressing out z."""
    def resid(v):
        A = np.stack([z, np.ones_like(z)], axis=1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    rx, ry = resid(x), resid(y)
    denom = rx.std() * ry.std()
    return float((rx * ry).mean() / denom) if denom > 0 else float("nan")


def correlation_report(swaps, delta_nll, base_nll, k, n_layers) -> dict:
    """Per-token routing disagreement against per-token NLL increase."""
    from scipy import stats

    x = swaps.astype(np.float64).ravel()
    y = delta_nll.astype(np.float64).ravel()
    z = base_nll.astype(np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    out = {
        "n_tokens": int(x.size),
        "max_possible_swaps": int(k * n_layers),
        "mean_swaps_per_token": float(x.mean()),
        "std_swaps_per_token": float(x.std()),
        "mean_delta_nll": float(y.mean()),
        "std_delta_nll": float(y.std()),
        "fraction_tokens_with_no_swaps": float((x == 0).mean()),
    }
    if x.size > 2 and x.std() > 0:
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        out["pearson_r"] = float(pr[0])
        out["pearson_p"] = float(pr[1])
        out["r_squared"] = float(pr[0] ** 2)
        out["spearman_rho"] = float(sr[0])
        out["spearman_p"] = float(sr[1])
        out["partial_pearson_r_given_bf16_nll"] = _partial_corr(x, y, z)
        out["pearson_r_swaps_vs_bf16_nll"] = float(stats.pearsonr(x, z)[0])
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None

    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    if edges.size > 2:
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
        bins = []
        for b in range(edges.size - 1):
            sel = idx == b
            if not sel.any():
                continue
            bins.append({
                "swaps_range": [float(edges[b]), float(edges[b + 1])],
                "n": int(sel.sum()),
                "mean_swaps": float(x[sel].mean()),
                "mean_delta_nll": float(y[sel].mean()),
                "median_delta_nll": float(np.median(y[sel])),
                "mean_bf16_nll": float(z[sel].mean()),
            })
        out["bins_by_swap_decile"] = bins
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    with Timer() as timer:
        print(f"[phase0] loading {args.model} on {device}", flush=True)
        model, tokenizer, ckpt_path = load_model_and_tokenizer(args.model, device=device)
        refs = discover_expert_weights(model)
        arch = describe_architecture(model, refs)
        print(json.dumps({k: v for k, v in arch.items() if k != "experts_per_layer"}, indent=2),
              flush=True)

        k = arch["num_experts_per_tok"]
        if k is None:
            raise RuntimeError("could not read num_experts_per_tok from the model config")

        originals = OriginalWeights(ckpt_path)
        fusion = originals.calibrate_fusion_order(model, refs)
        verification = originals.verify(model, refs, n=32)
        print(f"[phase0] checkpoint restore path: {fusion} {verification}", flush=True)

        spec = CalibSpec(dataset="c4", split="validation", n_sequences=args.n_prompts,
                         seq_len=args.seq_len, seed=args.seed, tokenizer_name=args.model)
        tokens, tok_meta = load_tokens(spec, tokenizer)
        if args.limit_sequences:
            tokens = tokens[: args.limit_sequences]
        print(f"[phase0] calibration tokens {tokens.shape} "
              f"(cache_hit={tok_meta['cache_hit']}, sha={tok_meta['sha256'][:12]})", flush=True)

        capture = RouterLogitCapture(model, k=k)
        store_path = CACHE_DIR / "phase0" / f"ref_router_logits_L{args.seq_len}_n{len(tokens)}.npy"
        with capture:
            ref_nll, ref_store = reference_pass(
                model, tokens, device, args.batch_size, capture, store_path)
            print(f"[phase0] bf16 calibration perplexity {np.exp(ref_nll.mean()):.4f}", flush=True)

            per_bits: dict[str, dict] = {}
            for bits in args.bits:
                print(f"[phase0] quantising all {len(refs)} expert matrices to {bits}-bit "
                      f"(group {args.group_size})", flush=True)
                qstats = apply_expert_quantization(
                    model, refs, originals, bits_for=lambda r, b=bits: b,
                    group_size=args.group_size, progress=False)
                mem = model_memory(model, refs, lambda r, b=bits: b, args.group_size,
                                   args.scale_bits, args.zero_bits)
                nll, agreements, swaps = comparison_pass(
                    model, tokens, device, args.batch_size, capture, ref_store, f"{bits}-bit")

                layer_summaries = [a.summary() for a in agreements]
                overall = {
                    key: float(np.mean([s[key] for s in layer_summaries]))
                    for key in ("mean_jaccard", "mean_overlap", "top1_agreement",
                                "mean_logit_l2_shift_normalized", "fraction_identical_sets")
                }
                overall["tokens_per_layer"] = int(layer_summaries[0]["tokens"])
                per_bits[str(bits)] = {
                    "quantization": qstats,
                    "memory": mem,
                    "overall": overall,
                    "per_layer": layer_summaries,
                    "calibration_perplexity_bf16": float(np.exp(ref_nll.mean())),
                    "calibration_perplexity_quantized": float(np.exp(nll.mean())),
                    "calibration_nll_increase": float((nll - ref_nll).mean()),
                    "routing_vs_nll_correlation": correlation_report(
                        swaps[:, :-1], nll - ref_nll, ref_nll, k, capture.n_layers),
                }
                c = per_bits[str(bits)]["routing_vs_nll_correlation"]
                print(f"[phase0] {bits}-bit: Jaccard {overall['mean_jaccard']:.4f} | "
                      f"overlap {overall['mean_overlap']:.3f}/{k} | "
                      f"top1 {overall['top1_agreement']:.4f} | "
                      f"ppl {np.exp(ref_nll.mean()):.3f} -> {np.exp(nll.mean()):.3f} | "
                      f"swaps~dNLL r={c.get('pearson_r')}", flush=True)

            apply_expert_quantization(model, refs, originals, bits_for=lambda r: None,
                                      group_size=args.group_size)

    verdict = {}
    if str(args.kill_bits) in per_bits:
        j = per_bits[str(args.kill_bits)]["overall"]["mean_jaccard"]
        verdict = {
            "kill_bits": args.kill_bits,
            "kill_threshold": args.kill_threshold,
            "mean_jaccard_at_kill_bits": j,
            "routing_is_near_invariant": bool(j >= args.kill_threshold),
        }

    record = RunRecord(
        name="phase0_diagnostic",
        config={**vars(args), "checkpoint_path": str(ckpt_path), "calibration": tok_meta,
                "n_sequences_used": int(tokens.shape[0]),
                "checkpoint_restore": {**fusion, **verification}},
        seed=args.seed,
        device=device_info(device),
        metrics={"architecture": arch, "by_bits": per_bits, "kill_criterion": verdict},
    )
    record.wall_clock_seconds = timer.elapsed
    record.started_at = timer.started_at
    print(f"[phase0] wrote {record.write(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

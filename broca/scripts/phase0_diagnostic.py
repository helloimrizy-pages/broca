"""Phase 0: does quantising the experts actually move the router?

Runs the same calibration tokens through the bf16 model and through uniformly
quantised copies at several bit-widths, and compares the top-k expert sets token
by token and layer by layer.

Kill criterion (set before the run): if mean Jaccard at 3-bit is at or above
0.88, routing is close to invariant under quantisation on this model and the
premise of B-ROCA is weak.

If routing does shift, the second question is whether the shift is the mechanism
behind the quality loss.  We correlate per-token routing disagreement against the
per-token increase in negative log likelihood.  A correlation near zero means the
shift exists but is not what is costing perplexity.

Alignment note: the logits at position ``t`` predict token ``t+1``, so the NLL at
position ``t`` is compared against the routing decisions taken at position ``t``.
The final position of each sequence has routing but no NLL and is dropped from
the correlation (it is still counted in the agreement metrics).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kill-threshold", type=float, default=0.88,
                   help="mean Jaccard at the reference bit-width at or above which we stop")
    p.add_argument("--kill-bits", type=int, default=3)
    p.add_argument("--out", default="phase0_diagnostic.json")
    p.add_argument("--limit-batches", type=int, default=None, help="debug: cap batches per pass")
    return p.parse_args(argv)


@torch.no_grad()
def run_pass(model, tokens, device, batch_size, capture, ref_store=None, limit=None, desc=""):
    """One full pass over the calibration tokens.

    With ``ref_store`` None this is the reference pass and router logits are
    written into a fresh memmap.  Otherwise the pass streams its logits against
    the stored reference and accumulates agreement statistics.

    Returns ``(nll [n_seq, L-1], router_store_or_agreements, per_token_swaps)``.
    """
    n_seq, seq_len = tokens.shape
    n_layers = capture.n_layers
    nll = np.zeros((n_seq, seq_len - 1), dtype=np.float32)

    if ref_store is None:
        store_path = CACHE_DIR / "phase0" / "ref_router_logits.npy"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        n_experts = None
        store = None
    else:
        store = ref_store
        n_experts = store.shape[-1]
        agreements = [LayerAgreement(n_experts=n_experts, k=capture.k) for _ in range(n_layers)]
        swaps = np.zeros((n_seq, seq_len), dtype=np.int32)

    batches = range(0, n_seq, batch_size)
    if limit is not None:
        batches = list(batches)[:limit]
    for start in tqdm(list(batches), desc=desc, unit="batch"):
        stop = min(start + batch_size, n_seq)
        ids = torch.from_numpy(tokens[start:stop].astype(np.int64)).to(device)
        capture.reset()
        out = model(input_ids=ids)
        logits = out.logits.float()
        # per-token NLL of the next token
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        tok_nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll[start:stop] = tok_nll.cpu().numpy()
        del out, logits, lp

        router = capture.stacked(stop - start, seq_len)  # [L, b, T, E]
        if ref_store is None:
            if store is None:
                n_experts = router.shape[-1]
                store = np.lib.format.open_memmap(
                    store_path, mode="w+", dtype=np.float32,
                    shape=(n_layers, n_seq, seq_len, n_experts),
                )
            store[:, start:stop] = router.numpy()
        else:
            ref = torch.from_numpy(np.asarray(store[:, start:stop]))
            for l in range(n_layers):
                m = agreements[l].update(
                    ref[l].reshape(-1, n_experts), router[l].reshape(-1, n_experts)
                )
                swaps[start:stop] += (capture.k - m).view(stop - start, seq_len).numpy().astype(np.int32)
        del router

    if ref_store is None:
        store.flush()
        return nll, store, None
    return nll, agreements, swaps


def correlation_report(swaps: np.ndarray, delta_nll: np.ndarray, k: int, n_layers: int) -> dict:
    """Per-token routing disagreement against per-token NLL increase."""
    from scipy import stats

    x = swaps.astype(np.float64).ravel()
    y = delta_nll.astype(np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    out = {
        "n_tokens": int(x.size),
        "max_possible_swaps": k * n_layers,
        "mean_swaps_per_token": float(x.mean()),
        "mean_delta_nll": float(y.mean()),
        "std_delta_nll": float(y.std()),
    }
    if x.size > 2 and x.std() > 0:
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        out["pearson_r"] = float(pr[0])
        out["pearson_p"] = float(pr[1])
        out["spearman_rho"] = float(sr[0])
        out["spearman_p"] = float(sr[1])
        out["r_squared"] = float(pr[0] ** 2)
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None

    # Binned view: mean delta NLL by routing disagreement decile of x.
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    if qs.size > 2:
        idx = np.clip(np.digitize(x, qs[1:-1]), 0, qs.size - 2)
        bins = []
        for b in range(qs.size - 1):
            sel = idx == b
            if sel.sum() == 0:
                continue
            bins.append({
                "swaps_range": [float(qs[b]), float(qs[b + 1])],
                "n": int(sel.sum()),
                "mean_swaps": float(x[sel].mean()),
                "mean_delta_nll": float(y[sel].mean()),
                "median_delta_nll": float(np.median(y[sel])),
            })
        out["bins"] = bins
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    with Timer() as timer:
        print(f"[phase0] loading {args.model} on {device} ...", flush=True)
        model, tokenizer, ckpt_path = load_model_and_tokenizer(args.model, device=device)
        refs = discover_expert_weights(model)
        arch = describe_architecture(model, refs)
        print(json.dumps(arch, indent=2), flush=True)

        k = arch["num_experts_per_tok"]
        if k is None:
            raise RuntimeError("could not read num_experts_per_tok from the model config")

        spec = CalibSpec(
            dataset="c4", split="validation", n_sequences=args.n_prompts,
            seq_len=args.seq_len, seed=args.seed, tokenizer_name=args.model,
        )
        tokens, tok_meta = load_tokens(spec, tokenizer)
        print(f"[phase0] calibration tokens {tokens.shape} "
              f"(cache_hit={tok_meta['cache_hit']}, sha={tok_meta['sha256'][:12]})", flush=True)

        originals = OriginalWeights(ckpt_path)
        capture = RouterLogitCapture(model)
        capture.k = k

        with capture:
            print("[phase0] reference pass (bf16)", flush=True)
            ref_nll, ref_store, _ = run_pass(
                model, tokens, device, args.batch_size, capture,
                ref_store=None, limit=args.limit_batches, desc="bf16",
            )

            per_bits: dict[str, dict] = {}
            for bits in args.bits:
                print(f"[phase0] quantising all experts to {bits}-bit "
                      f"(group {args.group_size})", flush=True)
                qstats = apply_expert_quantization(
                    model, refs, originals, bits_for=lambda r, b=bits: b,
                    group_size=args.group_size, progress=True,
                )
                mem = model_memory(model, refs, lambda r, b=bits: b, args.group_size,
                                   args.scale_bits, args.zero_bits)
                nll, agreements, swaps = run_pass(
                    model, tokens, device, args.batch_size, capture,
                    ref_store=ref_store, limit=args.limit_batches, desc=f"{bits}-bit",
                )
                n_used = swaps.shape[0] if args.limit_batches is None else \
                    min(swaps.shape[0], args.limit_batches * args.batch_size)
                delta = (nll - ref_nll)[:n_used]
                # routing at position t is compared against the NLL at position t
                swaps_aligned = swaps[:n_used, :-1]

                layer_summaries = [a.summary() for a in agreements]
                overall = {
                    "mean_jaccard": float(np.mean([s["mean_jaccard"] for s in layer_summaries])),
                    "mean_overlap": float(np.mean([s["mean_overlap"] for s in layer_summaries])),
                    "top1_agreement": float(np.mean([s["top1_agreement"] for s in layer_summaries])),
                    "mean_logit_l2_shift_normalized": float(
                        np.mean([s["mean_logit_l2_shift_normalized"] for s in layer_summaries])),
                    "fraction_identical_sets": float(
                        np.mean([s["fraction_identical_sets"] for s in layer_summaries])),
                }
                ppl_ref = float(np.exp(ref_nll[:n_used].mean()))
                ppl_q = float(np.exp(nll[:n_used].mean()))
                per_bits[str(bits)] = {
                    "quantization": qstats,
                    "memory": mem,
                    "overall": overall,
                    "per_layer": layer_summaries,
                    "calibration_perplexity_bf16": ppl_ref,
                    "calibration_perplexity_quantized": ppl_q,
                    "calibration_nll_increase": float(delta.mean()),
                    "routing_vs_nll_correlation": correlation_report(
                        swaps_aligned, delta, k, capture.n_layers),
                }
                print(f"[phase0] {bits}-bit: mean Jaccard {overall['mean_jaccard']:.4f}  "
                      f"mean overlap {overall['mean_overlap']:.3f}/{k}  "
                      f"top1 {overall['top1_agreement']:.4f}  "
                      f"ppl {ppl_ref:.3f} -> {ppl_q:.3f}", flush=True)

            # leave the model in its original state
            apply_expert_quantization(model, refs, originals, bits_for=lambda r: None,
                                      group_size=args.group_size)

    kill_key = str(args.kill_bits)
    verdict = {}
    if kill_key in per_bits:
        j = per_bits[kill_key]["overall"]["mean_jaccard"]
        verdict = {
            "kill_bits": args.kill_bits,
            "kill_threshold": args.kill_threshold,
            "mean_jaccard_at_kill_bits": j,
            "routing_is_near_invariant": bool(j >= args.kill_threshold),
        }

    record = RunRecord(
        name="phase0_diagnostic",
        config={**vars(args), "checkpoint_path": str(ckpt_path), "calibration": tok_meta},
        seed=args.seed,
        device=device_info(device),
        metrics={"architecture": arch, "by_bits": per_bits, "kill_criterion": verdict},
    )
    record.wall_clock_seconds = timer.elapsed
    record.started_at = timer.started_at
    path = record.write(args.out)
    print(f"[phase0] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

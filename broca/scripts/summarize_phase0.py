"""Render the Phase 0 diagnostic JSON as plain text.

Reads only ``results/phase0_diagnostic.json``.  Every number printed here comes
out of that file; nothing is computed from memory or typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import RESULTS_DIR


def fmt_row(cells, widths, align=None):
    align = align or ["<"] + [">"] * (len(cells) - 1)
    return "  ".join(f"{str(c):{a}{w}}" for c, w, a in zip(cells, widths, align))


def table(header, rows) -> str:
    widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) if rows
              else len(str(header[i])) for i in range(len(header))]
    out = [fmt_row(header, widths), "  ".join("-" * w for w in widths)]
    out += [fmt_row(r, widths) for r in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DIR / "phase0_diagnostic.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "phase0_summary.txt"))
    args = ap.parse_args(argv)

    data = json.loads(Path(args.results).read_text())
    arch = data["metrics"]["architecture"]
    by_bits = data["metrics"]["by_bits"]
    cfg = data["config"]
    k = arch["num_experts_per_tok"]
    lines: list[str] = []
    W = lines.append

    W("PHASE 0 - ROUTING SHIFT UNDER EXPERT QUANTISATION")
    W("=" * 78)
    W(f"model                {cfg['model']}")
    W(f"commit               {data['git']['commit']}  (dirty={data['git']['dirty']})")
    W(f"seed                 {data['seed']}")
    W(f"device               {data['device'].get('device')} ({data['device'].get('name')})")
    W(f"wall clock           {data['wall_clock_seconds']:.1f} s")
    W(f"torch/transformers   {data['versions']['torch']} / {data['versions']['transformers']}")
    W("")
    W(f"architecture         {arch['num_hidden_layers']} layers, {arch['num_experts']} experts/layer, "
      f"top-{k} routing, hidden {arch['hidden_size']}, expert intermediate {arch['intermediate_size']}")
    W(f"parameters           {arch['total_parameters']:,} total, {arch['expert_parameters']:,} in experts "
      f"({100 * arch['expert_parameter_fraction']:.2f}%)")
    W(f"calibration          {cfg['n_sequences_used']} sequences x {cfg['seq_len']} tokens from "
      f"{cfg['calibration']['source']}")
    W(f"                     sha256 {cfg['calibration']['sha256'][:16]}")
    W(f"quantisation         RTN, asymmetric, group {cfg['group_size']} along the input dim, "
      f"experts only (router/attention/embeddings/norms stay bf16)")
    W("")

    W("OVERALL AGREEMENT WITH THE bf16 ROUTER")
    W("-" * 78)
    rows = []
    for b, r in sorted(by_bits.items(), key=lambda kv: int(kv[0])):
        o = r["overall"]
        rows.append([
            f"{b}-bit",
            f"{o['mean_jaccard']:.4f}",
            f"{o['mean_overlap']:.3f}/{k}",
            f"{o['top1_agreement']:.4f}",
            f"{o['fraction_identical_sets']:.4f}",
            f"{o['mean_logit_l2_shift_normalized']:.4f}",
            f"{r['calibration_perplexity_quantized']:.4f}",
            f"{r['memory']['average_bits_stored']:.3f}",
            f"{r['memory']['total_gib']:.2f}",
        ])
    W(table(["setting", "Jaccard", "overlap", "top-1", "identical", "logit shift",
             "C4 ppl", "bits/w", "GiB"], rows))
    ppl0 = next(iter(by_bits.values()))["calibration_perplexity_bf16"]
    mem_bf16 = next(iter(by_bits.values()))["memory"]["bf16_total_gib"]
    W(f"{'bf16':<9}  {'1.0000':>7}  {f'{k}.000/{k}':>7}  {'1.0000':>6}  {'1.0000':>9}  "
      f"{'0.0000':>11}  {ppl0:>7.4f}  {'16.000':>6}  {mem_bf16:>5.2f}")
    W("")
    W("  Jaccard is m/(2k-m) for an overlap of m out of k; a single expert swap at k=8 gives")
    W("  7/9 = 0.7778.  'identical' is the fraction of (token, layer) pairs whose top-k set is")
    W("  exactly preserved.  'logit shift' is the mean per-token L2 change in router logits")
    W("  divided by the bf16 logit standard deviation at that layer.")
    W("")

    for b, r in sorted(by_bits.items(), key=lambda kv: int(kv[0])):
        W(f"OVERLAP DISTRIBUTION AT {b}-BIT  (fraction of token-layer pairs by overlap count)")
        W("-" * 78)
        hist = [0] * (k + 1)
        total = 0
        for layer in r["per_layer"]:
            for i, c in enumerate(layer["overlap_histogram"]):
                hist[i] += c
            total += layer["tokens"]
        W(table(["overlap"] + [f"{i}/{k}" for i in range(k + 1)],
                [["fraction"] + [f"{h / total:.4f}" for h in hist]]))
        W("")

    W("PER-LAYER AGREEMENT")
    W("-" * 78)
    for b, r in sorted(by_bits.items(), key=lambda kv: int(kv[0])):
        W(f"  {b}-bit")
        rows = [[
            f"L{i:02d}",
            f"{s['mean_jaccard']:.4f}",
            f"{s['mean_overlap']:.3f}",
            f"{s['top1_agreement']:.4f}",
            f"{s['fraction_identical_sets']:.4f}",
            f"{s['mean_logit_l2_shift_normalized']:.4f}",
        ] for i, s in enumerate(r["per_layer"])]
        W(table(["layer", "Jaccard", "overlap", "top-1", "identical", "logit shift"], rows))
        W("")

    W("DOES THE ROUTING SHIFT EXPLAIN THE QUALITY LOSS?")
    W("-" * 78)
    W("  Per-token routing disagreement (expert swaps summed over all layers) against the")
    W("  per-token increase in NLL.  A near-zero correlation means the shift is real but is")
    W("  not the mechanism behind the perplexity loss.")
    W("")
    rows = []
    for b, r in sorted(by_bits.items(), key=lambda kv: int(kv[0])):
        c = r["routing_vs_nll_correlation"]
        rows.append([
            f"{b}-bit",
            f"{c['mean_swaps_per_token']:.3f}/{c['max_possible_swaps']}",
            f"{c['fraction_tokens_with_no_swaps']:.4f}",
            f"{r['calibration_nll_increase']:+.4f}",
            f"{c['pearson_r']:+.4f}" if c.get("pearson_r") is not None else "n/a",
            f"{c['r_squared']:.4f}" if c.get("r_squared") is not None else "n/a",
            f"{c['spearman_rho']:+.4f}" if c.get("spearman_rho") is not None else "n/a",
            f"{c['partial_pearson_r_given_bf16_nll']:+.4f}"
            if c.get("partial_pearson_r_given_bf16_nll") is not None else "n/a",
            f"{c['n_tokens']:,}",
        ])
    W(table(["setting", "swaps/token", "no-swap frac", "dNLL", "pearson r", "r^2",
             "spearman", "partial r", "tokens"], rows))
    W("")
    for b, r in sorted(by_bits.items(), key=lambda kv: int(kv[0])):
        c = r["routing_vs_nll_correlation"]
        if "bins_by_swap_decile" not in c:
            continue
        W(f"  {b}-bit: mean NLL increase by routing-disagreement decile")
        rows = [[f"{d['mean_swaps']:.2f}", f"{d['n']:,}", f"{d['mean_delta_nll']:+.4f}",
                 f"{d['median_delta_nll']:+.4f}", f"{d['mean_bf16_nll']:.3f}"]
                for d in c["bins_by_swap_decile"]]
        W(table(["mean swaps", "n", "mean dNLL", "median dNLL", "bf16 NLL"], rows))
        W("")

    W("KILL CRITERION")
    W("-" * 78)
    v = data["metrics"]["kill_criterion"]
    W(f"  threshold: mean Jaccard at {v['kill_bits']}-bit >= {v['kill_threshold']} means routing is")
    W(f"  close to invariant under quantisation and the B-ROCA premise is weak on this model.")
    W(f"  measured: {v['mean_jaccard_at_kill_bits']:.4f}")
    W(f"  verdict:  {'TRIPPED - routing is near-invariant' if v['routing_is_near_invariant'] else 'not tripped - routing does shift'}")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)
    print(f"[summary] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

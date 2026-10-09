#!/usr/bin/env python
"""Plot the TabPFNv2.6 end-to-end inference benchmark results.

Reads the per-mode JSON produced by e2e/run_e2e_benchmark.py:
    results/e2e/{query_kind}/{gpu}/{mode}/e2e_benchmark.json

where {query_kind} is e.g. single-query / batched-query. Produces, per
(query_kind, gpu), one figure with:
  - column sweep: predict_proba latency vs feature-attention seqlen
  - row sweep:    predict_proba latency vs train rows (item-attention seqlen)
  - a speedup panel per sweep: mode latency relative to the `default` baseline

One line per mode (default / efficient / cudnn). Styled to match
run_benchmark_plots.py so the e2e plots read as the same system as the
attention microbenchmark plots.

Usage:
    uv run python run_e2e_plots.py [--results-dir results/e2e] [--output-dir plots/e2e]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

# Reuse the repo's palette (run_benchmark_plots.BACKEND_PALETTE) so the e2e
# plots read as the same system as the microbenchmark plots:
# efficient -> SDPA (efficient) blue, cudnn -> SDPA (cuDNN) orange,
# fa2/fa3/sage echo the microbenchmark FA2/FA3/Sage colors.
MODE_PALETTE = {
    "default": "#555555",     # neutral dispatch baseline
    "efficient": "#1f77b4",   # SDPA (efficient) blue
    "cudnn": "#ff7f0e",       # SDPA (cuDNN) orange
    "fa2": "#2ca02c",         # FA2 green
    "fa3": "#d62728",         # FA3 red
    "sage": "#e377c2",        # Sage magenta
}
MODE_LABELS = {
    "default": "default (dispatch)",
    "efficient": "SDPA (efficient)",
    "cudnn": "SDPA (cuDNN)",
    "fa2": "FA2 (ours)",
    "fa3": "FA3 (ours)",
    "sage": "Sage",
}
MODE_ORDER = ["default", "efficient", "cudnn", "fa2", "fa3", "sage"]
LEGEND_LABEL_ORDER = [MODE_LABELS[m] for m in MODE_ORDER]

# Speedup baseline matches run_benchmark_plots.py (baseline = SDPA (efficient)).
BASELINE_MODE = "efficient"
BASELINE_LABEL = MODE_LABELS[BASELINE_MODE]

# Seqlens to drop from the plots (e.g. present in older result files but no
# longer part of the sweep, or too noisy to show).
EXCLUDE_SEQLENS = {98304}


def load_gpu_results(results_dir):
    """Return {query_kind: {gpu: {mode: {"col": [...], "row": [...]}}}}.

    Layout on disk is results/e2e/{query_kind}/{gpu}/{mode}/e2e_benchmark.json,
    so a single top-level query-kind level (e.g. single-query / batched-query)
    sits above the per-GPU per-mode files.
    """
    groups = {}
    for f in sorted(Path(results_dir).glob("*/*/*/e2e_benchmark.json")):
        mode = f.parent.name
        gpu = f.parent.parent.name
        query_kind = f.parent.parent.parent.name
        data = json.load(open(f))
        rows = [r for r in data["results"]
                if "latency_ms" in r and r["seqlen"] not in EXCLUDE_SEQLENS]  # skip errors + excluded
        by_kind = {"col": [], "row": []}
        for r in rows:
            by_kind[r["attn_type"]].append(r)
        for k in by_kind:
            by_kind[k].sort(key=lambda r: r["seqlen"])
        groups.setdefault(query_kind, {}).setdefault(gpu, {})[mode] = by_kind
    return groups


def _series(rows):
    x = np.array([r["seqlen"] for r in rows], dtype=float)
    y = np.array([r["latency_ms"] for r in rows], dtype=float)
    e = np.array([r.get("latency_std_ms", 0.0) for r in rows], dtype=float)
    return x, y, e


def _kfmt(v, _pos=None):
    """Tick label for power-of-two seqlens: 1024 -> '1k', 16384 -> '16k',
    131072 -> '128k', <1024 as-is. Uses 1024-based k (not 1000) so the
    power-of-two ticks render as clean integers instead of 1.024k etc."""
    v = int(round(v))
    if v >= 1024:
        return f"{v // 1024}k"
    return f"{v:g}"


def _apply_log2_xaxis(ax, seqlens):
    ax.set_xscale("log", base=2)
    if len(seqlens):
        ax.set_xticks(sorted(seqlens))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(_kfmt))


def _order_legend(ax, title="Mode"):
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
    ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, title=title, fontsize=9)


def _plot_latency(ax, modes, kind, xlabel, fixed_label):
    all_seqlens = set()
    for mode in MODE_ORDER:
        rows = modes.get(mode, {}).get(kind, [])
        if not rows:
            continue
        color = MODE_PALETTE[mode]
        x, y, e = _series(rows)
        all_seqlens.update(x.tolist())
        ax.plot(x, y, marker="o", linestyle="-", linewidth=2, markersize=8,
                color=color, label=MODE_LABELS[mode])
        ax.fill_between(x, y - e, y + e, alpha=0.15, color=color)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("predict_proba latency (ms)", fontsize=11)
    ax.set_title(f"{kind.capitalize()} Sweep – Latency ({fixed_label})", fontsize=13)
    _apply_log2_xaxis(ax, all_seqlens)
    ax.set_yscale("log", base=10)
    _order_legend(ax)


def _plot_speedup(ax, modes, kind, xlabel, fixed_label):
    """Speedup of each mode vs the `default` baseline (>1 = faster than default)."""
    base = {r["seqlen"]: r["latency_ms"] for r in modes.get(BASELINE_MODE, {}).get(kind, [])}
    base_std = {r["seqlen"]: r.get("latency_std_ms", 0.0)
                for r in modes.get(BASELINE_MODE, {}).get(kind, [])}
    if not base:
        return
    all_seqlens = set()
    for mode in MODE_ORDER:
        if mode == BASELINE_MODE:
            continue  # the reference itself sits at 1.0 (drawn as the dashed line)
        rows = modes.get(mode, {}).get(kind, [])
        if not rows:
            continue
        color = MODE_PALETTE[mode]
        # Propagate std through the ratio s = t_base / t_mode (independent errors):
        #   sigma_s = s * sqrt((sigma_base/t_base)^2 + (sigma_mode/t_mode)^2)
        pts = []
        for r in rows:
            sl = r["seqlen"]
            if sl not in base:
                continue
            s = base[sl] / r["latency_ms"]
            rel_base = base_std[sl] / base[sl] if base[sl] else 0.0
            rel_mode = r.get("latency_std_ms", 0.0) / r["latency_ms"] if r["latency_ms"] else 0.0
            pts.append((sl, s, s * np.hypot(rel_base, rel_mode)))
        if not pts:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        s = np.array([p[1] for p in pts], dtype=float)
        e = np.array([p[2] for p in pts], dtype=float)
        all_seqlens.update(x.tolist())
        ax.plot(x, s, marker="o", linestyle="-", linewidth=2, markersize=8,
                color=color, label=MODE_LABELS[mode])
        ax.fill_between(x, s - e, s + e, alpha=0.15, color=color)
    ax.axhline(y=1.0, color=MODE_PALETTE[BASELINE_MODE], linestyle="--", alpha=0.7,
               label="_nolegend_")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(f"Speedup vs {BASELINE_LABEL}", fontsize=11)
    ax.set_title(f"{kind.capitalize()} Sweep – Speedup ({fixed_label})", fontsize=13)
    _apply_log2_xaxis(ax, all_seqlens)
    _order_legend(ax, title="")


def _fixed_label(modes, kind):
    """Short descriptor of the held-fixed axis for a sweep (e.g. 'R=1024')."""
    for mode in MODE_ORDER:
        rows = modes.get(mode, {}).get(kind, [])
        if rows:
            r = rows[0]
            if kind == "col":
                # Column attention's sequence is the rows, so the held-fixed axis
                # is the TOTAL row count (train + test), not just the train split.
                return f"R={r['n_train'] + r.get('n_test', 0)}"
            return f"C={r['n_features']}"
    return "?"


def plot_gpu(query_kind, gpu, modes, output_dir, include_sage=False):
    # Sage is off by default: its latency dwarfs the other modes on some shapes
    # and compresses the SDPA/FA comparison. --sage adds it (and tags the file).
    if not include_sage:
        modes = {m: v for m, v in modes.items() if m != "sage"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    col_fixed = _fixed_label(modes, "col")
    row_fixed = _fixed_label(modes, "row")

    _plot_latency(axes[0, 0], modes, "col", "Sequence Length (cols)", col_fixed)
    _plot_speedup(axes[0, 1], modes, "col", "Sequence Length (cols)", col_fixed)
    _plot_latency(axes[1, 0], modes, "row", "Sequence Length (rows)", row_fixed)
    _plot_speedup(axes[1, 1], modes, "row", "Sequence Length (rows)", row_fixed)

    fig.suptitle(f"TabPFNv2.6 E2E Inference Benchmark – {query_kind}", fontsize=16, y=1.01)
    subtitle = (
        f"GPU: {gpu.replace('_', ' ')}"
        f" | Values > 1 = faster than {BASELINE_LABEL}"
    )
    fig.text(0.5, -0.01, subtitle, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    suffix = "_with_sage" if include_sage else ""
    out = Path(output_dir) / f"e2e_{query_kind}_{gpu}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description="Plot TabPFNv2.6 E2E inference results")
    p.add_argument("--results-dir", default="results/e2e")
    p.add_argument("--output-dir", default="plots/e2e")
    p.add_argument("--sage", action="store_true",
                   help="Include the Sage mode (off by default; its latency can "
                        "compress the other curves). Writes a separate "
                        "e2e_{query_kind}_{gpu}_with_sage.png so both plots coexist.")
    args = p.parse_args()

    groups = load_gpu_results(args.results_dir)
    if not groups:
        print(f"No e2e results found under {args.results_dir}/*/*/*/e2e_benchmark.json")
        return
    for query_kind, gpus in groups.items():
        for gpu, modes in gpus.items():
            plot_gpu(query_kind, gpu, modes, args.output_dir, include_sage=args.sage)


if __name__ == "__main__":
    main()

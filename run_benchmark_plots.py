#!/usr/bin/env python
"""Plot tabular attention benchmark results.

Discovers all JSON files under results/ (or a given root) and produces one set
of plots per unique (GPU, dtype, nheads, headdim, col_attn_rows, row_attn_cols)
combination, merging across all backends found for that combination.

Usage:
    uv run python run_benchmark_plots.py [--results-dir results/] [--output-dir plots/]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


sns.set_theme(style="whitegrid")

matplotlib_color_theme = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # magenta
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow green
    "#17becf",
]

BACKEND_PALETTE = {
    "SDPA": "#1f77b4",
    "SDPA (efficient)": matplotlib_color_theme[0],
    "SDPA (cuDNN)": matplotlib_color_theme[1],
    "FA2": matplotlib_color_theme[2],
    "FA3": matplotlib_color_theme[3],
    "FA4": matplotlib_color_theme[4],
    "FA4 (optimized)": matplotlib_color_theme[9],
    "FA4_tabular": matplotlib_color_theme[8],
    "Sage": matplotlib_color_theme[6],
    "vLLM": matplotlib_color_theme[7],
    "combined optimal": "#000000", # black
}

# Comment out backends that we don't want to plot
BACKEND_LABELS = {
    # "sdpa": "SDPA",
    "sdpa_efficient": "SDPA (efficient)",
    "sdpa_cudnn": "SDPA (cuDNN)",
    "fa2": "FA2",
    "fa3": "FA3",
    "fa4": "FA4",
}

LEGEND_LABEL_ORDER = [
    "SDPA",
    "SDPA (efficient)",
    "SDPA (cuDNN)",
    "FA2",
    "FA3",
    "FA4",
]


# ============================================================
# Roofline analysis constants and helpers
# ============================================================

GPU_SPECS = {
    "NVIDIA_H100_NVL": {
        "peak_tflops": 1979,
        "hbm_bandwidth_tb": 3.9,
        "copy_bandwidth_tb_fallback": 1.35,
        "label": "H100 NVL",
    },
}


class CopyBandwidthModel:
    """Per-shape copy bandwidth from measured data or fallback constant."""

    def __init__(self, json_path: Path | None = None, fallback_tb: float = 1.35):
        self._lookup: dict[int, float] = {}
        self._fallback = fallback_tb * 1e12

        if json_path and json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for entry in data.get("triple_copy", []):
                rows = entry["rows"]
                bw_gb = entry["bandwidth_gb_s"]
                self._lookup[rows] = bw_gb * 1e9
            if self._lookup:
                print(f"Loaded per-shape copy bandwidth for {len(self._lookup)} shapes")
            else:
                for entry in data.get("single_copy", []):
                    rows = entry["rows"]
                    bw_gb = entry["bandwidth_gb_s"]
                    self._lookup[rows] = bw_gb * 1e9
                if self._lookup:
                    print(f"Loaded per-shape copy bandwidth (single) for {len(self._lookup)} shapes")
        else:
            if json_path:
                print(f"Copy bandwidth file not found: {json_path}, using fallback {fallback_tb} TB/s")
            else:
                print(f"No copy bandwidth file specified, using fallback {fallback_tb} TB/s")

    def get_bandwidth(self, rows: int) -> float:
        """Get copy bandwidth in bytes/s for a given row count."""
        if rows in self._lookup:
            return self._lookup[rows]
        if self._lookup:
            available = sorted(self._lookup.keys())
            if rows <= available[0]:
                return self._lookup[available[0]]
            if rows >= available[-1]:
                return self._lookup[available[-1]]
            for i in range(len(available) - 1):
                if available[i] <= rows <= available[i + 1]:
                    lo, hi = available[i], available[i + 1]
                    t = (np.log2(rows) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
                    return self._lookup[lo] * (1 - t) + self._lookup[hi] * t
        return self._fallback

    def get_copy_time_ms(self, batch_eff: int, nheads: int, seq_len: int,
                         headdim: int, n_copies: int = 3) -> float:
        """Compute copy time in ms using per-shape bandwidth."""
        bw = self.get_bandwidth(seq_len)
        bytes_total = n_copies * batch_eff * nheads * seq_len * headdim * 2
        return bytes_total / bw * 1000


def ridge_point(peak_tflops: float, bandwidth_tb: float) -> float:
    """Compute the ridge point (balance point) in FLOP/byte."""
    return peak_tflops / bandwidth_tb


def attention_flops(batch_eff: int, nheads: int, seq_len: int, headdim: int) -> int:
    """Forward-pass FLOPs for non-causal attention."""
    return 4 * batch_eff * nheads * seq_len * seq_len * headdim


def load_roofline_backend_data(results_dir: Path, gpu: str, backend_key: str,
                               nheads: int, headdim: int) -> list[dict]:
    """Load benchmark results for a specific backend (roofline analysis)."""
    dtype_dir = results_dir / gpu / "bfloat16" / backend_key
    if not dtype_dir.exists():
        return []

    target_suffix = f"H-{nheads}_HD-{headdim}"
    all_results = []
    for path in sorted(dtype_dir.glob("*.json")):
        if target_suffix not in path.name:
            continue
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data.get("results", []))
    return all_results


def roofline_results_to_df(results: list[dict], backend_label: str) -> pd.DataFrame:
    """Convert raw results to DataFrame for roofline plotting."""
    rows = []
    for r in results:
        if "error" in r:
            continue
        rows.append({
            "Backend": backend_label,
            "Attention Type": r["attn_type"],
            "Seq Len": r["seq_len"],
            "Batch Eff": r["batch_eff"],
            "Rows": r["rows"],
            "Cols": r["cols"],
            "Forward TFLOPS": r.get("fwd_tflops", 0),
            "Forward TFLOPS Std": r.get("fwd_tflops_std", 0),
            "Forward Time ms": r.get("fwd_time_ms", 0),
        })
    return pd.DataFrame(rows)


def plot_roofline_analysis(
    df: pd.DataFrame, gpu: str, nheads: int, headdim: int, output_path: Path,
    copy_model: CopyBandwidthModel,
):
    """1x2 figure: column attention roofline ceiling + row attention copy decomposition."""
    specs = GPU_SPECS[gpu]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: Column attention with memory-bound ceiling ---
    ax = axes[0]
    df_col = df[(df["Attention Type"] == "col")]

    for backend in ["SDPA (cuDNN)", "FA3"]:
        bd = df_col[df_col["Backend"] == backend].sort_values("Seq Len")
        if bd.empty:
            continue
        color = BACKEND_PALETTE.get(backend)
        ax.plot(bd["Seq Len"], bd["Forward TFLOPS"], marker="o", linestyle="-",
                linewidth=2, markersize=7, color=color, label=backend)
        ax.fill_between(bd["Seq Len"],
                        bd["Forward TFLOPS"] - bd["Forward TFLOPS Std"],
                        bd["Forward TFLOPS"] + bd["Forward TFLOPS Std"],
                        alpha=0.12, color=color)

    col_seq_lens = sorted(df_col["Seq Len"].unique())
    if col_seq_lens:
        sl_min, sl_max = col_seq_lens[0], col_seq_lens[-1]
        sl_dense = np.geomspace(sl_min, sl_max, 100)
        oi_vals = sl_dense / 2.0
        mem_ceiling = oi_vals * specs["hbm_bandwidth_tb"]
        compute_ceiling = np.full_like(mem_ceiling, specs["peak_tflops"])
        roofline_ceiling = np.minimum(mem_ceiling, compute_ceiling)
        ax.plot(sl_dense, roofline_ceiling, linestyle="--", linewidth=2,
                color="black", alpha=0.5, label="Roofline ceiling")

        rp = ridge_point(specs["peak_tflops"], specs["hbm_bandwidth_tb"])
        rp_seq_len = rp * 2
        if sl_min <= rp_seq_len <= sl_max:
            ax.axvline(x=rp_seq_len, color="gray", linestyle=":", alpha=0.5)

    col_rows = df_col["Rows"].iloc[0] if len(df_col) > 0 else "?"
    ax.set_xlabel("Sequence length (columns)", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Column Attention – Forward (R={col_rows})", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    if col_seq_lens:
        ax.set_xticks(col_seq_lens)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=9, loc="lower right")

    # --- Right panel: Row attention copy decomposition ---
    ax = axes[1]
    df_row = df[(df["Attention Type"] == "row")]

    df_fa3 = df_row[df_row["Backend"] == "FA3"].sort_values("Seq Len")
    df_cudnn = df_row[df_row["Backend"] == "SDPA (cuDNN)"].sort_values("Seq Len")

    if not df_fa3.empty and not df_cudnn.empty:
        ax.plot(df_fa3["Seq Len"], df_fa3["Forward TFLOPS"], marker="o", linestyle="-",
                linewidth=2, markersize=7, color=BACKEND_PALETTE["FA3"], label="FA3 (measured)")
        ax.fill_between(df_fa3["Seq Len"],
                        df_fa3["Forward TFLOPS"] - df_fa3["Forward TFLOPS Std"],
                        df_fa3["Forward TFLOPS"] + df_fa3["Forward TFLOPS Std"],
                        alpha=0.12, color=BACKEND_PALETTE["FA3"])

        ax.plot(df_cudnn["Seq Len"], df_cudnn["Forward TFLOPS"], marker="o", linestyle="-",
                linewidth=2, markersize=7, color=BACKEND_PALETTE["SDPA (cuDNN)"],
                label="cuDNN (measured)")
        ax.fill_between(df_cudnn["Seq Len"],
                        df_cudnn["Forward TFLOPS"] - df_cudnn["Forward TFLOPS Std"],
                        df_cudnn["Forward TFLOPS"] + df_cudnn["Forward TFLOPS Std"],
                        alpha=0.12, color=BACKEND_PALETTE["SDPA (cuDNN)"])

        common_seq_lens = sorted(
            set(df_fa3["Seq Len"].values) & set(df_cudnn["Seq Len"].values)
        )
        fa3_lookup = df_fa3.set_index("Seq Len")
        theoretical_tflops = []
        theoretical_seq_lens = []

        for sl in common_seq_lens:
            fa3_row = fa3_lookup.loc[sl]
            batch_eff = int(fa3_row["Batch Eff"]) if np.isscalar(fa3_row["Batch Eff"]) else int(fa3_row["Batch Eff"].iloc[0])
            fa3_tflops_val = float(fa3_row["Forward TFLOPS"]) if np.isscalar(fa3_row["Forward TFLOPS"]) else float(fa3_row["Forward TFLOPS"].iloc[0])

            if fa3_tflops_val <= 0:
                continue

            flops = attention_flops(batch_eff, nheads, sl, headdim)
            fa3_time_ms = flops / (fa3_tflops_val * 1e12) * 1000
            copy_time = copy_model.get_copy_time_ms(batch_eff, nheads, sl, headdim)
            effective_tflops = flops / ((fa3_time_ms + copy_time) / 1000) / 1e12

            theoretical_seq_lens.append(sl)
            theoretical_tflops.append(effective_tflops)

        if theoretical_seq_lens:
            ax.plot(theoretical_seq_lens, theoretical_tflops, marker="x", linestyle="--",
                    linewidth=2, markersize=8, color="#555555",
                    label="FA3 + copy overhead (estimated)", zorder=5)

    row_cols = df_row["Cols"].iloc[0] if len(df_row) > 0 else "?"
    ax.set_xlabel("Sequence length (rows)", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Row Attention – Forward (C={row_cols})", fontsize=12)
    ax.set_xscale("log", base=2)
    row_seq_lens = sorted(df_row["Seq Len"].unique())
    if row_seq_lens:
        ax.set_xticks(row_seq_lens)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"Roofline Analysis – {specs['label']} (H={nheads}, D={headdim}, bf16)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_copy_overhead_fraction(
    df: pd.DataFrame, gpu: str, nheads: int, headdim: int, output_path: Path,
    copy_model: CopyBandwidthModel,
):
    """Copy overhead fraction vs sequence length for row attention."""
    specs = GPU_SPECS[gpu]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    df_row = df[(df["Attention Type"] == "row")]
    df_fa3 = df_row[df_row["Backend"] == "FA3"].sort_values("Seq Len")
    df_cudnn = df_row[df_row["Backend"] == "SDPA (cuDNN)"].sort_values("Seq Len")

    if df_fa3.empty or df_cudnn.empty:
        print("Missing FA3 or cuDNN data for copy fraction plot")
        plt.close()
        return

    common_seq_lens = sorted(
        set(df_fa3["Seq Len"].values) & set(df_cudnn["Seq Len"].values)
    )
    fa3_lookup = df_fa3.set_index("Seq Len")
    cudnn_lookup = df_cudnn.set_index("Seq Len")

    seq_lens = []
    copy_fractions = []
    gap_fractions = []

    for sl in common_seq_lens:
        fa3_row = fa3_lookup.loc[sl]
        cudnn_row = cudnn_lookup.loc[sl]

        batch_eff = int(fa3_row["Batch Eff"]) if np.isscalar(fa3_row["Batch Eff"]) else int(fa3_row["Batch Eff"].iloc[0])
        fa3_tflops = float(fa3_row["Forward TFLOPS"]) if np.isscalar(fa3_row["Forward TFLOPS"]) else float(fa3_row["Forward TFLOPS"].iloc[0])
        cudnn_tflops = float(cudnn_row["Forward TFLOPS"]) if np.isscalar(cudnn_row["Forward TFLOPS"]) else float(cudnn_row["Forward TFLOPS"].iloc[0])

        if fa3_tflops <= 0 or cudnn_tflops <= 0:
            continue

        flops = attention_flops(batch_eff, nheads, sl, headdim)
        fa3_time_ms = flops / (fa3_tflops * 1e12) * 1000
        cudnn_time_ms = flops / (cudnn_tflops * 1e12) * 1000
        copy_time = copy_model.get_copy_time_ms(batch_eff, nheads, sl, headdim)

        gap = cudnn_time_ms - fa3_time_ms

        seq_lens.append(sl)
        copy_fractions.append(copy_time / cudnn_time_ms)
        gap_fractions.append(gap / cudnn_time_ms if cudnn_time_ms > 0 else 0)

    seq_lens = np.array(seq_lens)
    copy_fractions = np.array(copy_fractions)
    gap_fractions = np.array(gap_fractions)

    ax.plot(seq_lens, gap_fractions * 100, marker="o", linestyle="-", linewidth=2,
            markersize=7, color=BACKEND_PALETTE["FA3"],
            label="Measured gap (cuDNN−FA3) / cuDNN time")
    ax.plot(seq_lens, copy_fractions * 100, marker="s", linestyle="--", linewidth=2,
            markersize=7, color="#555555",
            label="Estimated copy overhead / cuDNN time")

    ax.fill_between(seq_lens, 0, copy_fractions * 100, alpha=0.15, color="#555555",
                    label="Explained by copies")
    ax.fill_between(seq_lens, copy_fractions * 100, gap_fractions * 100,
                    alpha=0.15, color=BACKEND_PALETTE["FA3"],
                    label="Kernel efficiency gap")

    ax.set_xlabel("Sequence length (rows)", fontsize=11)
    ax.set_ylabel("Fraction of cuDNN total time (%)", fontsize=11)
    ax.set_title(f"Row Attention: Copy Overhead Decomposition ({specs['label']})", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Main benchmark plot functions
# ============================================================


def load_results(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to long-format DataFrame for seaborn."""
    rows = []
    seen_unused = []
    for r in results:
        if "error" in r:
            continue
        if r["backend"] not in BACKEND_LABELS:
            if r["backend"] not in seen_unused:
                seen_unused.append(r["backend"])
                print(f"Unused backend: {r['backend']}, skipping...")
            continue
        backend_label = BACKEND_LABELS.get(r["backend"], r["backend"])
        rows.append({
            "Backend": backend_label,
            "Attention Type": r["attn_type"].capitalize(),
            "Rows": r["rows"],
            "Cols": r["cols"],
            "Seq Len": r["seq_len"],
            "Batch Eff": r["batch_eff"],
            "Forward TFLOPS": r.get("fwd_tflops", 0),
            "Forward TFLOPS Std": r.get("fwd_tflops_std", 0),
            "Backward TFLOPS": r.get("bwd_tflops", 0),
            "Backward TFLOPS Std": r.get("bwd_tflops_std", 0),
            "Forward Time (ms)": r.get("fwd_time_ms", 0),
            "Backward Time (ms)": r.get("bwd_time_ms", 0),
        })
    return pd.DataFrame(rows)


def plot_tabular_benchmark(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Create 2x2 grid: Col attn (top), Row attn (bottom), Fwd/Bwd columns.

    Plots mean TFLOPS with ±1σ shaded bands.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_col = df[df["Attention Type"] == "Col"]
    df_row = df[df["Attention Type"] == "Row"]

    for i, (metric, std_col, label) in enumerate([
        ("Forward TFLOPS", "Forward TFLOPS Std", "Forward"),
        ("Backward TFLOPS", "Backward TFLOPS Std", "Backward"),
    ]):
        ax = axes[0, i]
        plot_data = df_col.dropna(subset=[metric]) if label == "Backward" else df_col
        for backend in plot_data["Backend"].unique():
            bd = plot_data[plot_data["Backend"] == backend].sort_values("Seq Len")
            color = BACKEND_PALETTE.get(backend, None)
            is_optimal = backend == "combined optimal"
            if is_optimal:
                plot_kwargs = {"marker": "X", "linestyle": "--", "linewidth": 2, "markersize": 8, "zorder":1000}
            else:
                plot_kwargs = {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8}

            ax.plot(bd["Seq Len"], bd[metric], label=backend, color=color, **plot_kwargs)
            ax.fill_between(bd["Seq Len"], bd[metric] - bd[std_col], bd[metric] + bd[std_col], alpha=0.15, color=color)
        col_rows = df_col["Rows"].iloc[0] if len(df_col) > 0 else "?"
        ax.set_xlabel("Sequence Length (cols)", fontsize=11)
        ax.set_ylabel("TFLOPS", fontsize=11)
        ax.set_title(f"Column Attention – {label} (R={col_rows})", fontsize=13)
        ax.set_xscale("log", base=2)
        if len(df_col) > 0:
            ax.set_xticks(sorted(df_col["Seq Len"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
        ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

    for i, (metric, std_col, label) in enumerate([
        ("Forward TFLOPS", "Forward TFLOPS Std", "Forward"),
        ("Backward TFLOPS", "Backward TFLOPS Std", "Backward"),
    ]):
        ax = axes[1, i]
        plot_data = df_row.dropna(subset=[metric]) if label == "Backward" else df_row
        for backend in plot_data["Backend"].unique():
            bd = plot_data[plot_data["Backend"] == backend].sort_values("Seq Len")
            color = BACKEND_PALETTE.get(backend, None)
            is_optimal = backend == "combined optimal"
            if is_optimal:
                plot_kwargs = {"marker": "X", "linestyle": "--", "linewidth": 2, "markersize": 8, "zorder":1000}
            else:
                plot_kwargs = {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8}
            ax.plot(bd["Seq Len"], bd[metric], label=backend, color=color, **plot_kwargs)
            ax.fill_between(bd["Seq Len"], bd[metric] - bd[std_col], bd[metric] + bd[std_col], alpha=0.15, color=color)
        row_cols = df_row["Cols"].iloc[0] if len(df_row) > 0 else "?"
        ax.set_xlabel("Sequence Length (rows)", fontsize=11)
        ax.set_ylabel("TFLOPS", fontsize=11)
        ax.set_title(f"Row Attention – {label} (C={row_cols})", fontsize=13)
        ax.set_xscale("log", base=2)
        if len(df_row) > 0:
            ax.set_xticks(sorted(df_row["Seq Len"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
        ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

    n_reps = metadata.get("rep", "?")
    fig.suptitle(f"Tabular Attention Benchmark (±1σ, n={n_reps})", fontsize=16, y=1.01)
    subtitle = (
        f"GPU: {metadata.get('gpu', 'Unknown')}"
        f" | dtype: {metadata.get('dtype', '?')} | nheads: {metadata.get('nheads', '?')} | headdim: {metadata.get('headdim', '?')}"
    )
    fig.text(0.5, -0.01, subtitle, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Plot speedup of FA backends over SDPA with ±1σ shaded bands.

    Speedup = mean_backend / mean_baseline.
    σ_speedup / speedup = sqrt((σ_a/a)² + (σ_b/b)²) for ratio of independent RVs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline_name = "SDPA (efficient)"
    available_backends = df["Backend"].unique()
    available_backends = [b for b in available_backends if b in BACKEND_LABELS.values()]
    print(f"Available backends for speedup plot: {available_backends}")

    if baseline_name not in available_backends:
        print(f"Baseline '{baseline_name}' not found in data, skipping speedup plot.")
        plt.close()
        return

    speedup_palette = {k: v for k, v in BACKEND_PALETTE.items() if k != baseline_name}
    non_baseline = [b for b in available_backends if b != baseline_name]

    for row_idx, attn_type in enumerate(["Col", "Row"]):
        df_attn = df[df["Attention Type"] == attn_type]

        for col_idx, (metric, std_col) in enumerate([
            ("Forward TFLOPS", "Forward TFLOPS Std"),
            ("Backward TFLOPS", "Backward TFLOPS Std"),
        ]):
            ax = axes[row_idx, col_idx]

            # Drop backends without backward data for backward metric
            df_attn_metric = df_attn.dropna(subset=[metric])

            # Build baseline lookup (one row per Seq Len now)
            df_baseline = df_attn_metric[df_attn_metric["Backend"] == baseline_name].set_index("Seq Len")

            for backend in non_baseline:
                df_backend = df_attn_metric[df_attn_metric["Backend"] == backend].sort_values("Seq Len")
                seq_lens, speedups, speedup_stds = [], [], []
                for _, row in df_backend.iterrows():
                    sl = row["Seq Len"]
                    if sl not in df_baseline.index:
                        continue
                    base_val = df_baseline.loc[sl, metric]
                    if not np.isscalar(base_val):
                        base_val = base_val.iloc[0]
                    if base_val <= 0:
                        continue
                    base_std = df_baseline.loc[sl, std_col]
                    if not np.isscalar(base_std):
                        base_std = base_std.iloc[0]
                    a = row[metric]
                    sa = row[std_col]
                    speedup = a / base_val
                    # σ_speedup / speedup = sqrt((σ_a/a)² + (σ_b/b)²)
                    rel_err = np.sqrt((sa / a) ** 2 + (base_std / base_val) ** 2) if a > 0 else 0
                    seq_lens.append(sl)
                    speedups.append(speedup)
                    speedup_stds.append(speedup * rel_err)

                if seq_lens:
                    seq_lens = np.array(seq_lens)
                    speedups = np.array(speedups)
                    speedup_stds = np.array(speedup_stds)
                    color = speedup_palette.get(backend, None)
                    is_optimal = backend == "combined optimal"
                    ax.plot(seq_lens, speedups, marker="x" if is_optimal else "o", linestyle=":" if is_optimal else "-", label=backend, color=color, linewidth=2, markersize=8)
                    ax.fill_between(seq_lens, speedups - speedup_stds, speedups + speedup_stds, alpha=0.15, color=color)

            baseline_color = BACKEND_PALETTE.get(baseline_name, "gray")
            ax.axhline(y=1.0, color=baseline_color, linestyle="--", alpha=0.7, label="_nolegend_")
            ax.set_xlabel("Sequence Length", fontsize=11)
            ax.set_ylabel(f"Speedup vs {baseline_name}", fontsize=11)
            label = "Forward" if col_idx == 0 else "Backward"
            strided = " (strided)" if attn_type == "Row" else ""
            ax.set_title(f"{attn_type} Attention – {label}{strided}", fontsize=13)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df_attn["Seq Len"].unique()))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            handles, labels = ax.get_legend_handles_labels()
            label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
            ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
            ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
            ax.legend(ordered_handles, ordered_labels, title="", fontsize=9)

    n_reps = metadata.get("rep", "?")
    fig.suptitle(f"Speedup over {baseline_name} (±1σ, n={n_reps})", fontsize=16, y=1.01)
    subtitle = (
        f"GPU: {metadata.get('gpu', 'Unknown')} | Values > 1 = faster than {baseline_name}"
    )
    fig.text(0.5, -0.01, subtitle, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


INFERENCE_BACKEND_LABELS = {
    "sdpa_efficient": "SDPA (efficient)",
    "sdpa_cudnn": "SDPA (cuDNN)",
    "fa2": "FA2",
    "fa3": "FA3",
    "fa4": "FA4",
    "sage": "Sage",
    "vllm": "vLLM",
}


def plot_inference(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Create a 1x2 figure (Col Attn Fwd, Row Attn Fwd) including inference-only backends."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    df_col = df[df["Attention Type"] == "Col"]
    df_row = df[df["Attention Type"] == "Row"]
    col_rows = df_col["Rows"].iloc[0] if len(df_col) > 0 else "?"
    row_cols = df_row["Cols"].iloc[0] if len(df_row) > 0 else "?"

    # Column attention forward
    ax = axes[0]
    for backend in df_col["Backend"].unique():
        bd = df_col[df_col["Backend"] == backend].sort_values("Seq Len")
        color = BACKEND_PALETTE.get(backend, None)
        is_optimal = backend == "combined optimal"
        plot_kwargs = ({"marker": "X", "linestyle": "--", "linewidth": 2, "markersize": 8, "zorder": 1000}
                      if is_optimal else {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8})
        ax.plot(bd["Seq Len"], bd["Forward TFLOPS"], label=backend, color=color, **plot_kwargs)
        ax.fill_between(bd["Seq Len"],
                        bd["Forward TFLOPS"] - bd["Forward TFLOPS Std"],
                        bd["Forward TFLOPS"] + bd["Forward TFLOPS Std"],
                        alpha=0.15, color=color)
    ax.set_xlabel("Sequence Length (cols)", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Column Attention – Forward (R={col_rows})", fontsize=12)
    ax.set_xscale("log", base=2)
    if len(df_col) > 0:
        ax.set_xticks(sorted(df_col["Seq Len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
    ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

    # Row attention forward
    ax = axes[1]
    for backend in df_row["Backend"].unique():
        bd = df_row[df_row["Backend"] == backend].sort_values("Seq Len")
        color = BACKEND_PALETTE.get(backend, None)
        is_optimal = backend == "combined optimal"
        plot_kwargs = ({"marker": "X", "linestyle": "--", "linewidth": 2, "markersize": 8, "zorder": 1000}
                      if is_optimal else {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8})
        ax.plot(bd["Seq Len"], bd["Forward TFLOPS"], label=backend, color=color, **plot_kwargs)
        ax.fill_between(bd["Seq Len"],
                        bd["Forward TFLOPS"] - bd["Forward TFLOPS Std"],
                        bd["Forward TFLOPS"] + bd["Forward TFLOPS Std"],
                        alpha=0.15, color=color)
    ax.set_xlabel("Sequence Length (rows)", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Row Attention – Forward (C={row_cols})", fontsize=12)
    ax.set_xscale("log", base=2)
    if len(df_row) > 0:
        ax.set_xticks(sorted(df_row["Seq Len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
    ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# Shade palette for headdim ablation: light to dark for increasing D
HEADDIM_SPEEDUP_SHADES = {
    16: "#b8d2fa",
    32: "#88b5f6",
    64: "#2979ef",
    128: "#0c4ba6",
    256: "#052047",
}


def plot_headdim_ablation(headdim_data: dict[int, pd.DataFrame], output_path: Path, metadata: dict):
    """Create 1x2 figure showing speedup of FA3 over cuDNN for different headdims.

    One line per headdim, light→dark shading for increasing D.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    headdims = sorted(headdim_data.keys())

    # Column attention forward speedup
    ax = axes[0]
    for hd in headdims:
        df = headdim_data[hd]
        df_col = df[df["Attention Type"] == "Col"]
        df_fa3 = df_col[df_col["Backend"] == "FA3"].sort_values("Seq Len")
        df_cudnn = df_col[df_col["Backend"] == "SDPA (cuDNN)"].sort_values("Seq Len").set_index("Seq Len")
        if df_fa3.empty or df_cudnn.empty:
            continue
        seq_lens, speedups, speedup_stds = [], [], []
        for _, row in df_fa3.iterrows():
            sl = row["Seq Len"]
            if sl not in df_cudnn.index:
                continue
            base_val = df_cudnn.loc[sl, "Forward TFLOPS"]
            if not np.isscalar(base_val):
                base_val = base_val.iloc[0]
            if base_val <= 0:
                continue
            base_std = df_cudnn.loc[sl, "Forward TFLOPS Std"]
            if not np.isscalar(base_std):
                base_std = base_std.iloc[0]
            a = row["Forward TFLOPS"]
            sa = row["Forward TFLOPS Std"]
            speedup = a / base_val
            rel_err = np.sqrt((sa / a) ** 2 + (base_std / base_val) ** 2) if a > 0 else 0
            seq_lens.append(sl)
            speedups.append(speedup)
            speedup_stds.append(speedup * rel_err)
        if seq_lens:
            seq_lens = np.array(seq_lens)
            speedups = np.array(speedups)
            speedup_stds = np.array(speedup_stds)
            color = HEADDIM_SPEEDUP_SHADES[hd]
            ax.plot(seq_lens, speedups, marker="o", linestyle="-", linewidth=2,
                    markersize=6, color=color, label=f"D={hd}")
            ax.fill_between(seq_lens, speedups - speedup_stds, speedups + speedup_stds,
                            alpha=0.15, color=color)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    col_rows_val = next(iter(headdim_data.values()))
    df_col_any = col_rows_val[col_rows_val["Attention Type"] == "Col"]
    col_rows_val = df_col_any["Rows"].iloc[0] if len(df_col_any) > 0 else "?"
    ax.set_xlabel("Sequence Length (cols)", fontsize=11)
    ax.set_ylabel("Speedup (FA3 / cuDNN)", fontsize=11)
    ax.set_title(f"Column Attention – Forward (R={col_rows_val})", fontsize=12)
    ax.set_xscale("log", base=2)
    if len(df_col_any) > 0:
        ax.set_xticks(sorted(df_col_any["Seq Len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=9, title="Head dim")

    # Row attention forward speedup
    ax = axes[1]
    for hd in headdims:
        df = headdim_data[hd]
        df_row = df[df["Attention Type"] == "Row"]
        df_fa3 = df_row[df_row["Backend"] == "FA3"].sort_values("Seq Len")
        df_cudnn = df_row[df_row["Backend"] == "SDPA (cuDNN)"].sort_values("Seq Len").set_index("Seq Len")
        if df_fa3.empty or df_cudnn.empty:
            continue
        seq_lens, speedups, speedup_stds = [], [], []
        for _, row in df_fa3.iterrows():
            sl = row["Seq Len"]
            if sl not in df_cudnn.index:
                continue
            base_val = df_cudnn.loc[sl, "Forward TFLOPS"]
            if not np.isscalar(base_val):
                base_val = base_val.iloc[0]
            if base_val <= 0:
                continue
            base_std = df_cudnn.loc[sl, "Forward TFLOPS Std"]
            if not np.isscalar(base_std):
                base_std = base_std.iloc[0]
            a = row["Forward TFLOPS"]
            sa = row["Forward TFLOPS Std"]
            speedup = a / base_val
            rel_err = np.sqrt((sa / a) ** 2 + (base_std / base_val) ** 2) if a > 0 else 0
            seq_lens.append(sl)
            speedups.append(speedup)
            speedup_stds.append(speedup * rel_err)
        if seq_lens:
            seq_lens = np.array(seq_lens)
            speedups = np.array(speedups)
            speedup_stds = np.array(speedup_stds)
            color = HEADDIM_SPEEDUP_SHADES[hd]
            ax.plot(seq_lens, speedups, marker="o", linestyle="-", linewidth=2,
                    markersize=6, color=color, label=f"D={hd}")
            ax.fill_between(seq_lens, speedups - speedup_stds, speedups + speedup_stds,
                            alpha=0.15, color=color)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    df_row_any = next(iter(headdim_data.values()))
    df_row_any = df_row_any[df_row_any["Attention Type"] == "Row"]
    row_cols_val = df_row_any["Cols"].iloc[0] if len(df_row_any) > 0 else "?"
    ax.set_xlabel("Sequence Length (rows)", fontsize=11)
    ax.set_ylabel("Speedup (FA3 / cuDNN)", fontsize=11)
    ax.set_title(f"Row Attention – Forward (C={row_cols_val})", fontsize=12)
    ax.set_xscale("log", base=2)
    if len(df_row_any) > 0:
        ax.set_xticks(sorted(df_row_any["Seq Len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=9, title="Head dim")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


AGENT_OPTIM_FILE_PATTERN = (
    "CA-1024_16-32-64-128-256-512-1024-2048__RA-64_32-64-128-256-512-1024-2048-4096-8192__H-12_HD-64.json"
)
AGENT_OPTIM_BASELINE_PATTERN = (
    "CA-1024_16-32-64-128-256-512-1024-2048__RA-64_32-64-128-256-512-1024-2048-4096-8192-16384__H-12_HD-64_col.json"
)
AGENT_OPTIM_CUDNN_PATTERN = (
    "CA-1024_16-32-64-128-256-512-1024-2048__RA-64_32-64-128-256-512-1024-2048-4096-8192-16384__H-12_HD-64_col.json"
)


def plot_agent_optimized(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Single-panel plot comparing FA4 vs FA4 (optimized) – column attention forward."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    df_col = df[df["Attention Type"] == "Col"]
    col_rows = df_col["Rows"].iloc[0] if len(df_col) > 0 else "?"

    for backend in df_col["Backend"].unique():
        bd = df_col[df_col["Backend"] == backend].sort_values("Seq Len")
        color = BACKEND_PALETTE.get(backend, None)
        ax.plot(bd["Seq Len"], bd["Forward TFLOPS"], marker="o", linestyle="-",
                linewidth=2, markersize=8, color=color, label=backend)
        ax.fill_between(bd["Seq Len"],
                        bd["Forward TFLOPS"] - bd["Forward TFLOPS Std"],
                        bd["Forward TFLOPS"] + bd["Forward TFLOPS Std"],
                        alpha=0.15, color=color)

    ax.set_xlabel("Sequence Length (cols)", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Column Attention – Forward (R={col_rows})", fontsize=12)
    ax.set_xscale("log", base=2)
    if len(df_col) > 0:
        ax.set_xticks(sorted(df_col["Seq Len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
    ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


GPU_ORDER = [
    "NVIDIA_A100_80GB_PCIe",
    "NVIDIA_H100_NVL",
    "NVIDIA_B200",
]

GPU_DISPLAY_NAMES = {
    "NVIDIA_A100_80GB_PCIe": "NVIDIA A100 80GB PCIe",
    "NVIDIA_H100_NVL": "NVIDIA H100 NVL",
    "NVIDIA_B200": "NVIDIA B200",
}


def plot_comparison(gpu_dataframes: dict[str, pd.DataFrame], output_dir: Path, metadata: dict):
    """Create one 1x2 figure per GPU (Col Attn Fwd, Row Attn Fwd) and save as separate PDFs."""
    # Determine fixed params from any available data for subplot titles
    any_df = next(iter(gpu_dataframes.values()))
    df_col_any = any_df[any_df["Attention Type"] == "Col"]
    df_row_any = any_df[any_df["Attention Type"] == "Row"]
    col_rows = df_col_any["Rows"].iloc[0] if len(df_col_any) > 0 else "?"
    row_cols = df_row_any["Cols"].iloc[0] if len(df_row_any) > 0 else "?"

    nheads = metadata.get("nheads", "?")
    headdim = metadata.get("headdim", "?")

    for gpu_key in GPU_ORDER:
        if gpu_key not in gpu_dataframes:
            continue

        df = gpu_dataframes[gpu_key]
        df_col = df[df["Attention Type"] == "Col"]
        df_row = df[df["Attention Type"] == "Row"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

        # Column attention forward
        ax = axes[0]
        for backend in df_col["Backend"].unique():
            bd = df_col[df_col["Backend"] == backend].sort_values("Seq Len")
            color = BACKEND_PALETTE.get(backend, None)
            is_optimal = backend == "combined optimal"
            plot_kwargs = ({"marker": "X", "linestyle": "--", "linewidth": 2, "markersize": 8, "zorder": 1000}
                          if is_optimal else {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8})
            ax.plot(bd["Seq Len"], bd["Forward TFLOPS"], label=backend, color=color, **plot_kwargs)
            ax.fill_between(bd["Seq Len"],
                            bd["Forward TFLOPS"] - bd["Forward TFLOPS Std"],
                            bd["Forward TFLOPS"] + bd["Forward TFLOPS Std"],
                            alpha=0.15, color=color)
        ax.set_xlabel("Sequence Length (cols)", fontsize=11)
        ax.set_ylabel("TFLOPS", fontsize=11)
        ax.set_title(f"Column Attention – Forward (R={col_rows})", fontsize=12)
        ax.set_xscale("log", base=2)
        if len(df_col) > 0:
            ax.set_xticks(sorted(df_col["Seq Len"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
        ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

        # Row attention forward
        ax = axes[1]
        for backend in df_row["Backend"].unique():
            bd = df_row[df_row["Backend"] == backend].sort_values("Seq Len")
            color = BACKEND_PALETTE.get(backend, None)
            is_optimal = backend == "combined optimal"
            plot_kwargs = ({"marker": "X", "linestyle": "--", "linewidth": 2, "markersize": 8, "zorder": 1000}
                          if is_optimal else {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8})
            ax.plot(bd["Seq Len"], bd["Forward TFLOPS"], label=backend, color=color, **plot_kwargs)
            ax.fill_between(bd["Seq Len"],
                            bd["Forward TFLOPS"] - bd["Forward TFLOPS Std"],
                            bd["Forward TFLOPS"] + bd["Forward TFLOPS Std"],
                            alpha=0.15, color=color)
        ax.set_xlabel("Sequence Length (rows)", fontsize=11)
        # ax.set_ylabel("TFLOPS", fontsize=11)
        ax.set_title(f"Row Attention – Forward (C={row_cols})", fontsize=12)
        ax.set_xscale("log", base=2)
        if len(df_row) > 0:
            ax.set_xticks(sorted(df_row["Seq Len"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {l: h for h, l in zip(handles, labels) if isinstance(h, plt.Line2D)}
        ordered_handles = [label_to_handle[l] for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ordered_labels = [l for l in LEGEND_LABEL_ORDER if l in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, title="Backend", fontsize=9)

        plt.tight_layout()
        output_file = output_dir / f"gpu_comparison_{gpu_key}_H-{nheads}_HD-{headdim}.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}")


import re


def _strip_direction_suffix(filename: str) -> str:
    """Strip _col or _row suffix from a filename to get the base group key.

    e.g. 'CA-1024_...HD-64_col.json' -> 'CA-1024_...HD-64.json'
    """
    return re.sub(r"_(col|row)\.json$", ".json", filename)


def collect_groups(results_dir: Path) -> dict[tuple, list[Path]]:
    """Walk results/ and group JSON files by their benchmark configuration.

    Directory layout produced by run_benchmark.py:
        results/{GPU}/{dtype}/{backend}/{filename}.json

    Filenames may have a _col or _row suffix when produced by --col-only /
    --row-only invocations. These are grouped together with their unsuffixed
    counterpart so that all backend × direction files for the same benchmark
    configuration end up in the same group.

    We group by (gpu, dtype, base_filename) so that all backend files
    for the same benchmark run end up in the same group.
    """
    groups: dict[tuple, list[Path]] = {}
    for path in sorted(results_dir.rglob("*.json")):
        # Expected depth relative to results_dir: gpu/dtype/backend/filename
        parts = path.relative_to(results_dir).parts
        if len(parts) != 4:
            print(f"Skipping unexpected path structure: {path}")
            continue
        gpu, dtype, _backend, filename = parts
        base_filename = _strip_direction_suffix(filename)
        key = (gpu, dtype, base_filename)
        groups.setdefault(key, []).append(path)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Plot tabular attention benchmark results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root directory containing benchmark JSON files")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to write plot images into")
    parser.add_argument("--gpu-comparison", action="store_true",
                        help="Generate per-GPU forward-pass plots (A100, H100, B200) "
                             "as separate PDFs for use with LaTeX subfigure. "
                             "Uses --nheads and --headdim to select the configuration.")
    parser.add_argument("--nheads", type=int, default=12,
                        help="Number of attention heads for --comparison (default: 12)")
    parser.add_argument("--headdim", type=int, default=64,
                        help="Head dimension for --comparison (default: 64)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Generate a single 1x2 forward-pass plot for H100 "
                             "including inference-only backends (vLLM, SageAttention).")
    parser.add_argument("--headdim-ablation", action="store_true",
                        help="Generate a 1x2 plot showing FA3 and cuDNN across all "
                             "available head dimensions on H100. Fix nheads with --nheads.")
    parser.add_argument("--agent-optimized", action="store_true",
                        help="Generate the FA4 vs FA4 (optimized) plot for the agent "
                             "optimization section. Uses NVIDIA B200 bfloat16 results.")
    parser.add_argument("--roofline", action="store_true",
                        help="Generate roofline analysis plots (copy decomposition and "
                             "copy overhead fraction). Uses --gpu, --nheads, --headdim, "
                             "and --copy-benchmark.")
    parser.add_argument("--gpu", type=str, default="NVIDIA_H100_NVL",
                        help="GPU for --roofline (default: NVIDIA_H100_NVL)")
    parser.add_argument("--copy-benchmark", type=str, default=None,
                        help="Path to copy_bandwidth.json for --roofline. "
                             "If not provided, uses a fallback constant bandwidth.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    groups = collect_groups(results_dir)
    if not groups:
        print(f"No JSON files found under {results_dir}")
        return

    print(f"Found {len(groups)} benchmark group(s) across {sum(len(v) for v in groups.values())} files")

    if args.gpu_comparison:
        target_suffix = f"H-{args.nheads}_HD-{args.headdim}"
        gpu_dataframes: dict[str, pd.DataFrame] = {}
        metadata = {}

        for (gpu, dtype, filename), paths in groups.items():
            if target_suffix not in filename:
                continue
            all_results = []
            for path in paths:
                data = load_results(path)
                all_results.extend(data.get("results", []))
                if not metadata:
                    metadata = data.get("metadata", {})

            df = results_to_dataframe(all_results)
            if not df.empty:
                gpu_dataframes[gpu] = df

        if not gpu_dataframes:
            print(f"No data found for config {target_suffix}")
            return

        print(f"Found data for GPUs: {list(gpu_dataframes.keys())}")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(gpu_dataframes, output_dir, metadata)
        return

    if args.inference_only:
        target_suffix = f"H-{args.nheads}_HD-{args.headdim}"
        target_gpu = "NVIDIA_H100_NVL"
        all_results = []
        metadata = {}

        for (gpu, dtype, filename), paths in groups.items():
            if gpu != target_gpu or target_suffix not in filename:
                continue
            for path in paths:
                data = load_results(path)
                all_results.extend(data.get("results", []))
                if not metadata:
                    metadata = data.get("metadata", {})

        if not all_results:
            print(f"No data found for {target_gpu} with config {target_suffix}")
            return

        # Use inference backend labels (includes vLLM, Sage)
        rows = []
        seen_unused = []
        for r in all_results:
            if "error" in r:
                continue
            if r["backend"] not in INFERENCE_BACKEND_LABELS:
                if r["backend"] not in seen_unused:
                    seen_unused.append(r["backend"])
                    print(f"Unused backend: {r['backend']}, skipping...")
                continue
            backend_label = INFERENCE_BACKEND_LABELS[r["backend"]]
            rows.append({
                "Backend": backend_label,
                "Attention Type": r["attn_type"].capitalize(),
                "Rows": r["rows"],
                "Cols": r["cols"],
                "Seq Len": r["seq_len"],
                "Batch Eff": r["batch_eff"],
                "Forward TFLOPS": r.get("fwd_tflops", 0),
                "Forward TFLOPS Std": r.get("fwd_tflops_std", 0),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"No plottable data for inference-only plot")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"inference_only_{target_gpu}_H-{args.nheads}_HD-{args.headdim}.pdf"
        plot_inference(df, output_file, metadata)
        return

    if args.headdim_ablation:
        target_gpu = "NVIDIA_H100_NVL"
        ablation_backends = {"fa3": "FA3", "sdpa_cudnn": "SDPA (cuDNN)"}
        headdim_data: dict[int, pd.DataFrame] = {}
        metadata = {}

        for (gpu, dtype, filename), paths in groups.items():
            if gpu != target_gpu:
                continue
            if f"H-{args.nheads}_HD-" not in filename:
                continue
            # Extract headdim from filename
            import re as _re
            hd_match = _re.search(r"HD-(\d+)", filename)
            if not hd_match:
                continue
            hd = int(hd_match.group(1))

            all_results = []
            for path in paths:
                data = load_results(path)
                all_results.extend(data.get("results", []))
                if not metadata:
                    metadata = data.get("metadata", {})

            rows = []
            for r in all_results:
                if "error" in r:
                    continue
                if r["backend"] not in ablation_backends:
                    continue
                rows.append({
                    "Backend": ablation_backends[r["backend"]],
                    "Attention Type": r["attn_type"].capitalize(),
                    "Rows": r["rows"],
                    "Cols": r["cols"],
                    "Seq Len": r["seq_len"],
                    "Batch Eff": r["batch_eff"],
                    "Forward TFLOPS": r.get("fwd_tflops", 0),
                    "Forward TFLOPS Std": r.get("fwd_tflops_std", 0),
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                headdim_data[hd] = df

        if not headdim_data:
            print(f"No data found for headdim ablation on {target_gpu} with H={args.nheads}")
            return

        print(f"Found headdim ablation data for D={sorted(headdim_data.keys())}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"headdim_ablation_{target_gpu}_H-{args.nheads}.pdf"
        plot_headdim_ablation(headdim_data, output_file, metadata)
        return

    if args.agent_optimized:
        target_gpu = "NVIDIA_B200"
        target_dtype = "bfloat16"
        optim_path = results_dir / target_gpu / target_dtype / "fa4_optim" / AGENT_OPTIM_FILE_PATTERN
        baseline_path = results_dir / target_gpu / target_dtype / "fa4" / AGENT_OPTIM_BASELINE_PATTERN
        cudnn_path = results_dir / target_gpu / target_dtype / "sdpa_cudnn" / AGENT_OPTIM_CUDNN_PATTERN

        missing = [p for p in (optim_path, baseline_path, cudnn_path) if not p.exists()]
        if missing:
            for p in missing:
                print(f"File not found: {p}")
            return

        all_results = []
        metadata = {}
        for path, backend_key in [(baseline_path, "fa4"), (optim_path, "fa4_optim"), (cudnn_path, "sdpa_cudnn")]:
            data = load_results(path)
            for r in data.get("results", []):
                r = dict(r)
                r["backend"] = backend_key
                all_results.append(r)
            if not metadata:
                metadata = data.get("metadata", {})

        df = results_to_dataframe(all_results)
        if df.empty:
            print("No plottable data for agent-optimized plot")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"agent_optim_{target_gpu}_H-12_HD-64.pdf"
        plot_agent_optimized(df, output_file, metadata)
        return

    if args.roofline:
        gpu = args.gpu
        if gpu not in GPU_SPECS:
            print(f"Unknown GPU for roofline: {gpu}. Available: {list(GPU_SPECS.keys())}")
            return

        copy_path = Path(args.copy_benchmark) if args.copy_benchmark else results_dir / "copy_bandwidth.json"
        fallback_bw = GPU_SPECS[gpu]["copy_bandwidth_tb_fallback"]
        copy_model = CopyBandwidthModel(json_path=copy_path, fallback_tb=fallback_bw)

        backends_to_load = ["sdpa_cudnn", "fa3"]
        all_dfs = []
        for backend_key in backends_to_load:
            results = load_roofline_backend_data(results_dir, gpu, backend_key,
                                                args.nheads, args.headdim)
            if results:
                label = BACKEND_LABELS[backend_key]
                df = roofline_results_to_df(results, label)
                all_dfs.append(df)
                print(f"Loaded {len(results)} results for {label} on {gpu}")
            else:
                print(f"No data found for {backend_key} on {gpu}")

        if not all_dfs:
            print("No data loaded for roofline, exiting.")
            return

        df = pd.concat(all_dfs, ignore_index=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_roofline_analysis(
            df, gpu, args.nheads, args.headdim,
            output_dir / f"roofline_analysis_{gpu}_H-{args.nheads}_HD-{args.headdim}.pdf",
            copy_model,
        )
        plot_copy_overhead_fraction(
            df, gpu, args.nheads, args.headdim,
            output_dir / f"copy_overhead_{gpu}_H-{args.nheads}_HD-{args.headdim}.pdf",
            copy_model,
        )
        return

    for (gpu, dtype, filename), paths in groups.items():
        # Merge results from all backends in this group
        all_results = []
        metadata = {}
        for path in paths:
            data = load_results(path)
            all_results.extend(data.get("results", []))
            if not metadata:
                metadata = data.get("metadata", {})

        df = results_to_dataframe(all_results)
        if df.empty:
            print(f"No plottable data for {gpu}/{dtype}/{filename}, skipping.")
            continue

        # Mirror the results/ sub-directory structure under output_dir
        stem = Path(filename).stem
        out_subdir = output_dir / gpu / dtype
        out_subdir.mkdir(parents=True, exist_ok=True)

        plot_tabular_benchmark(df, out_subdir / f"{stem}_combined.png", metadata)
        plot_speedup(df, out_subdir / f"{stem}_speedup.png", metadata)


if __name__ == "__main__":
    main()

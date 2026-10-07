#!/usr/bin/env python
"""Plot the TabArena combined-vs-default attention benchmark.

Reads the two per-mode result files produced by e2e/run_tabarena_benchmark.py:
    results/tabarena/{gpu}/default/tabarena_benchmark.json
    results/tabarena/{gpu}/combined/tabarena_benchmark.json
joins them per task, and produces (per GPU) four figures:

  * speedup vs dataset size  - the headline: default/combined latency ratio per
    task vs train rows (log x), split by task type, with y=1 and the FA3 row
    threshold marked. Shows the combined dispatch winning more as rows grow.
  * latency parity           - log-log scatter of default vs combined latency;
    points below the y=x line are tasks where combined is faster.
  * peak-memory parity       - same shape for peak GPU memory.
  * score parity             - combined score minus default score per task
    (accuracy / R2), confirming the combined kernels don't change predictions.

Matches run_benchmark_plots.py conventions: seaborn whitegrid theme, dual
pdf+png output via save_fig, binary-k axis formatting, palette drawn from the
shared theme.

Usage:
    uv run python run_tabarena_plots.py [--results-dir results/tabarena] [--output-dir plots/tabarena]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_theme(style="whitegrid")

# Row-attention FA3 threshold used by the combined backend (raw training rows).
# Read from metadata when present; this is the fallback / default.
DEFAULT_ROW_FA_THRESHOLD = 2048

# Two categorical hues (fixed order, never cycled), drawn from the shared theme
# used in run_benchmark_plots.py.
TASK_TYPE_COLOR = {
    "clf": "#1f77b4",  # blue  - classification
    "reg": "#ff7f0e",  # orange - regression
}
TASK_TYPE_LABEL = {"clf": "Classification", "reg": "Regression"}
TASK_TYPE_ORDER = ["clf", "reg"]


def save_fig(output_path, fig=None, **savefig_kwargs):
    """Save to output_path AND a sibling with the other extension (.pdf/.png),
    matching run_benchmark_plots.py. `fig` defaults to the current figure."""
    saver = (fig.savefig if fig is not None else plt.savefig)
    p = Path(output_path)
    others = {".pdf": ".png", ".png": ".pdf"}
    targets = [p] + ([p.with_suffix(others[p.suffix])] if p.suffix in others else [])
    for t in targets:
        saver(t, bbox_inches="tight", **savefig_kwargs)
        print(f"Saved: {t}")


def _binary_k(v, _pos=None):
    """Axis label: 128, 256, 512, 1k, 2k, ... using binary-k (1024 -> 1k)."""
    v = int(round(v))
    if v >= 1024:
        k = v / 1024
        return f"{int(k)}k" if k == int(k) else f"{k:g}k"
    return str(v)


def load_mode(results_dir: Path, gpu: str, mode: str):
    """Load one mode's result file for a GPU. Returns (results, metadata) or (None, None)."""
    path = results_dir / gpu / mode / "tabarena_benchmark.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get("results", []), data.get("metadata", {})


def join_modes(results_dir: Path, gpu: str) -> pd.DataFrame:
    """Join default + combined results per task into one DataFrame.

    Only tasks that succeeded (no 'error') in BOTH modes are kept, since every
    plot compares the two. Prints how many tasks were dropped and why."""
    default_res, _ = load_mode(results_dir, gpu, "default")
    combined_res, _ = load_mode(results_dir, gpu, "combined")
    if not default_res or not combined_res:
        return pd.DataFrame()

    def index(results):
        return {r["task"]: r for r in results}

    d_by_task, c_by_task = index(default_res), index(combined_res)
    rows, dropped = [], []
    for task in sorted(set(d_by_task) & set(c_by_task)):
        d, c = d_by_task[task], c_by_task[task]
        if "error" in d or "error" in c or "latency_ms" not in d or "latency_ms" not in c:
            dropped.append(task)
            continue
        rows.append({
            "task": task,
            "short": task.split("__")[2] if "__" in task else task,
            "task_type": d.get("task_type", "clf"),
            "n_train": d["n_train"],
            "n_features": d.get("n_features"),
            "metric_name": d.get("metric_name", "score"),
            "default_latency_ms": d["latency_ms"],
            "combined_latency_ms": c["latency_ms"],
            "default_latency_std_ms": d.get("latency_std_ms", 0.0),
            "combined_latency_std_ms": c.get("latency_std_ms", 0.0),
            "default_peak_gb": d.get("peak_mem_gb"),
            "combined_peak_gb": c.get("peak_mem_gb"),
            "default_score": d.get("score"),
            "combined_score": c.get("score"),
        })
    if dropped:
        print(f"  {gpu}: dropped {len(dropped)} task(s) missing/errored in a mode: "
              f"{', '.join(t.split('__')[2] if '__' in t else t for t in dropped[:6])}"
              f"{' ...' if len(dropped) > 6 else ''}")
    df = pd.DataFrame(rows)
    if not df.empty:
        # Speedup = baseline (default) / combined; >1 means combined is faster.
        df["latency_speedup"] = df["default_latency_ms"] / df["combined_latency_ms"]
        # σ_speedup/speedup = sqrt((σ_d/d)² + (σ_c/c)²) for a ratio of independent RVs.
        rel = np.sqrt(
            (df["default_latency_std_ms"] / df["default_latency_ms"]) ** 2
            + (df["combined_latency_std_ms"] / df["combined_latency_ms"]) ** 2
        )
        df["latency_speedup_std"] = df["latency_speedup"] * rel
        # Absolute latency saved by combined: default - combined (ms). >0 = combined
        # faster. Uncorrelated per-mode means -> std adds in quadrature.
        df["latency_diff_ms"] = df["default_latency_ms"] - df["combined_latency_ms"]
        df["latency_diff_std_ms"] = np.sqrt(
            df["default_latency_std_ms"] ** 2 + df["combined_latency_std_ms"] ** 2
        )
        df["score_delta"] = df["combined_score"] - df["default_score"]
    return df


def _legend_by_type(ax, present_types):
    """Add a task-type legend in fixed order for the types actually plotted."""
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          color=TASK_TYPE_COLOR[t], label=TASK_TYPE_LABEL[t])
               for t in TASK_TYPE_ORDER if t in present_types]
    if handles:
        ax.legend(handles=handles, title="Task type", fontsize=9)


def plot_speedup_vs_size(df: pd.DataFrame, gpu: str, threshold: int, output_path: Path):
    """Headline: per-task latency speedup (default/combined) vs train rows."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    present = set(df["task_type"])
    for t in TASK_TYPE_ORDER:
        sub = df[df["task_type"] == t]
        if sub.empty:
            continue
        ax.errorbar(sub["n_train"], sub["latency_speedup"], yerr=sub["latency_speedup_std"],
                    fmt="o", markersize=8, color=TASK_TYPE_COLOR[t], alpha=0.85,
                    ecolor=TASK_TYPE_COLOR[t], elinewidth=1, capsize=2, linestyle="none")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, zorder=0)
    # Mark the FA3 row-attention threshold: right of it, the combined row path is FA3.
    ax.axvline(x=threshold, color="#555555", linestyle=":", alpha=0.6, zorder=0)
    ymax = ax.get_ylim()[1]
    ax.text(threshold * 1.05, ymax * 0.98, f"row FA3 >{_binary_k(threshold)} train rows",
            fontsize=8, color="#555555", va="top", ha="left", rotation=0)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training rows", fontsize=11)
    ax.set_ylabel("Latency speedup (default / combined)", fontsize=11)
    ax.set_title(f"TabArena: combined attention speedup vs dataset size\n{gpu.replace('_', ' ')}",
                 fontsize=13)
    ax.get_xaxis().set_major_formatter(FuncFormatter(_binary_k))
    ax.tick_params(axis="x", labelrotation=45)
    _legend_by_type(ax, present)

    plt.tight_layout()
    save_fig(output_path, dpi=150)
    plt.close()


def plot_latency_diff_vs_size(df: pd.DataFrame, gpu: str, threshold: int, output_path: Path):
    """Per-task ABSOLUTE latency saved (default - combined, ms) vs train rows.

    The absolute analogue of plot_speedup_vs_size: same x-axis (train rows, log)
    and threshold marker, but y is milliseconds saved rather than the ratio. >0
    means combined is faster. y is symlog (linthresh=1) since the saving spans
    sub-ms on small tables to hundreds of ms on the 100k-row tables, and dips
    slightly negative on a few small tasks."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    present = set(df["task_type"])
    for t in TASK_TYPE_ORDER:
        sub = df[df["task_type"] == t]
        if sub.empty:
            continue
        ax.errorbar(sub["n_train"], sub["latency_diff_ms"], yerr=sub["latency_diff_std_ms"],
                    fmt="o", markersize=8, color=TASK_TYPE_COLOR[t], alpha=0.85,
                    ecolor=TASK_TYPE_COLOR[t], elinewidth=1, capsize=2, linestyle="none")

    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.7, zorder=0)
    # Mark the FA3 row-attention threshold: right of it, the combined row path is FA3.
    ax.axvline(x=threshold, color="#555555", linestyle=":", alpha=0.6, zorder=0)
    ymax = ax.get_ylim()[1]
    ax.text(threshold * 1.05, ymax * 0.9, f"row FA3 >{_binary_k(threshold)} train rows",
            fontsize=8, color="#555555", va="top", ha="left", rotation=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Training rows", fontsize=11)
    ax.set_ylabel("Latency saved: default − combined (ms)", fontsize=11)
    ax.set_title(f"TabArena: latency saved vs dataset size ({gpu.replace('_', ' ')})\n"
                 f">0 = combined faster", fontsize=13)
    ax.get_xaxis().set_major_formatter(FuncFormatter(_binary_k))
    ax.tick_params(axis="x", labelrotation=45)
    _legend_by_type(ax, present)

    plt.tight_layout()
    save_fig(output_path, dpi=150)
    plt.close()


def plot_speedup_box(df: pd.DataFrame, gpu: str, output_path: Path):
    """Boxplot of the per-task latency speedup distribution across TabArena.

    The box spans the interquartile range (Q1..Q3) with the median line; whiskers
    reach the min/max of non-outlier points (matplotlib default 1.5*IQR), fliers
    beyond. The mean is drawn as a separate diamond (a boxplot's line is the
    median, not the mean, and you asked for both). Individual tasks are overlaid
    as a jittered strip, colored by task type, so the box's summary sits over the
    raw points it summarizes. Numeric min / Q1 / median / mean / Q3 / max are
    annotated on the right."""
    s = df["latency_speedup"].dropna()
    if s.empty:
        print("  no speedup data for boxplot, skipping")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))

    ax.boxplot(
        s.values, vert=True, widths=0.5, whis=1.5,
        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "#d62728",
                                   "markeredgecolor": "#d62728", "markersize": 8},
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"color": "#333333"}, whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"}, flierprops={"marker": "o", "markersize": 5,
                                                    "markerfacecolor": "none",
                                                    "markeredgecolor": "#999999"},
        positions=[1], zorder=2,
    )

    # Jittered strip of the actual per-task points, colored by task type. Jitter is
    # deterministic (linspace-based) so the figure is reproducible.
    present = set(df["task_type"])
    for t in TASK_TYPE_ORDER:
        st = df[df["task_type"] == t]["latency_speedup"].dropna()
        if st.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(st)) if len(st) > 1 else np.array([0.0])
        ax.scatter(1 + jitter, st.values, s=40, color=TASK_TYPE_COLOR[t], alpha=0.7,
                   edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, zorder=0)

    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    mean = s.mean()
    geomean = float(np.exp(np.log(s).mean()))
    stats_txt = (f"max     {s.max():.3f}×\n"
                 f"Q3      {q3:.3f}×\n"
                 f"median  {med:.3f}×\n"
                 f"mean    {mean:.3f}×\n"
                 f"geomean {geomean:.3f}×\n"
                 f"Q1      {q1:.3f}×\n"
                 f"min     {s.min():.3f}×")
    ax.text(1.42, med, stats_txt, fontsize=9, family="monospace", va="center",
            ha="left", bbox=dict(boxstyle="round", facecolor="white",
                                 edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(0.5, 2.1)
    ax.set_xticks([])
    ax.set_ylabel("Latency speedup (default / combined)", fontsize=11)
    ax.set_title(f"TabArena: per-task speedup distribution ({len(s)} tasks)\n"
                 f"{gpu.replace('_', ' ')}", fontsize=12)

    # Legend: task-type dots + the median/mean markers.
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          color=TASK_TYPE_COLOR[t], label=TASK_TYPE_LABEL[t])
               for t in TASK_TYPE_ORDER if t in present]
    handles += [
        plt.Line2D([0], [0], color="black", linewidth=2, label="Median"),
        plt.Line2D([0], [0], marker="D", linestyle="", markersize=8,
                   color="#d62728", label="Mean"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    save_fig(output_path, dpi=150)
    plt.close()


def plot_latency_diff_box(df: pd.DataFrame, gpu: str, output_path: Path):
    """Boxplot of the per-task ABSOLUTE latency difference (default - combined, ms).

    Same box/whisker/mean-diamond/strip layout as plot_speedup_box, but on absolute
    saved milliseconds rather than the ratio. >0 means combined is faster (saves
    time). The y-axis is symlog: the per-task differences span from ~1 ms (small
    tables) to hundreds of ms (100k-row tables), so a linear scale would be crushed
    by the large tasks and hide the sign of the small ones. min/Q1/median/mean/Q3/max
    (all in ms) are annotated on the right; no geomean (undefined for signed values)."""
    s = df["latency_diff_ms"].dropna()
    if s.empty:
        print("  no latency-diff data for boxplot, skipping")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))

    ax.boxplot(
        s.values, vert=True, widths=0.5, whis=1.5,
        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "#d62728",
                                   "markeredgecolor": "#d62728", "markersize": 8},
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"color": "#333333"}, whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"}, flierprops={"marker": "o", "markersize": 5,
                                                    "markerfacecolor": "none",
                                                    "markeredgecolor": "#999999"},
        positions=[1], zorder=2,
    )

    present = set(df["task_type"])
    for t in TASK_TYPE_ORDER:
        st = df[df["task_type"] == t]["latency_diff_ms"].dropna()
        if st.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(st)) if len(st) > 1 else np.array([0.0])
        ax.scatter(1 + jitter, st.values, s=40, color=TASK_TYPE_COLOR[t], alpha=0.7,
                   edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.7, zorder=0)

    # Symlog so both ~1 ms and ~100s-of-ms differences are legible; linthresh keeps
    # the near-zero region linear (and shows the sign of the small tasks).
    ax.set_yscale("symlog", linthresh=1.0)

    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    mean = s.mean()
    total = s.sum()  # total wall-clock saved across all tasks
    stats_txt = (f"max     {s.max():+.1f} ms\n"
                 f"Q3      {q3:+.1f} ms\n"
                 f"median  {med:+.1f} ms\n"
                 f"mean    {mean:+.1f} ms\n"
                 f"Q1      {q1:+.1f} ms\n"
                 f"min     {s.min():+.1f} ms\n"
                 f"Σ saved {total:+.1f} ms")
    ax.text(1.42, med, stats_txt, fontsize=9, family="monospace", va="center",
            ha="left", bbox=dict(boxstyle="round", facecolor="white",
                                 edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(0.5, 2.1)
    ax.set_xticks([])
    ax.set_ylabel("Latency saved: default − combined (ms)", fontsize=11)
    ax.set_title(f"TabArena: per-task latency saved ({len(s)} tasks)\n"
                 f"{gpu.replace('_', ' ')}  (>0 = combined faster)", fontsize=12)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          color=TASK_TYPE_COLOR[t], label=TASK_TYPE_LABEL[t])
               for t in TASK_TYPE_ORDER if t in present]
    handles += [
        plt.Line2D([0], [0], color="black", linewidth=2, label="Median"),
        plt.Line2D([0], [0], marker="D", linestyle="", markersize=8,
                   color="#d62728", label="Mean"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    save_fig(output_path, dpi=150)
    plt.close()


def _parity_scatter(df, xcol, ycol, label, unit, gpu, output_path, logscale=True):
    """Generic default-vs-combined parity scatter with a y=x reference line.

    Points below y=x mean combined is smaller (faster / less memory)."""
    sub = df.dropna(subset=[xcol, ycol])
    if sub.empty:
        print(f"  no data for {label} parity plot, skipping")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))

    present = set(sub["task_type"])
    for t in TASK_TYPE_ORDER:
        s = sub[sub["task_type"] == t]
        if s.empty:
            continue
        ax.scatter(s[xcol], s[ycol], s=55, color=TASK_TYPE_COLOR[t], alpha=0.8,
                   edgecolor="white", linewidth=0.5)

    lo = min(sub[xcol].min(), sub[ycol].min())
    hi = max(sub[xcol].max(), sub[ycol].max())
    pad = (hi / lo) ** 0.05 if logscale else (hi - lo) * 0.05
    lo_, hi_ = (lo / pad, hi * pad) if logscale else (lo - pad, hi + pad)
    ax.plot([lo_, hi_], [lo_, hi_], color="gray", linestyle="--", alpha=0.7,
            zorder=0, label="_nolegend_")
    # Label the diagonal at ~70% along it (not the corner, where it collides with
    # the title), rotated to sit on the line under equal-aspect log axes.
    mid = (lo_ * hi_) ** 0.5 if logscale else (lo_ + hi_) / 2
    ax.text(mid, mid, " y = x (equal)", fontsize=8, color="gray",
            va="bottom", ha="left", rotation=45, rotation_mode="anchor")

    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo_, hi_)
    ax.set_ylim(lo_, hi_)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Default {label} ({unit})", fontsize=11)
    ax.set_ylabel(f"Combined {label} ({unit})", fontsize=11)
    ax.set_title(f"TabArena: {label} — combined vs default ({gpu.replace('_', ' ')})\n"
                 f"below the line = combined better", fontsize=11)
    _legend_by_type(ax, present)

    plt.tight_layout()
    save_fig(output_path, dpi=150)
    plt.close()


def plot_score_parity(df: pd.DataFrame, gpu: str, output_path: Path):
    """Per-task score delta (combined − default), confirming no accuracy/R2 regression."""
    sub = df.dropna(subset=["score_delta"]).sort_values("score_delta")
    if sub.empty:
        print("  no score data for parity plot, skipping")
        return
    fig, ax = plt.subplots(1, 1, figsize=(9, max(4, 0.28 * len(sub))))

    colors = [TASK_TYPE_COLOR[t] for t in sub["task_type"]]
    ax.barh(sub["short"], sub["score_delta"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.7, zorder=0)

    worst = sub["score_delta"].abs().max()
    ax.set_xlabel("Score delta: combined − default  (accuracy / R²)", fontsize=11)
    ax.set_title(f"TabArena: score parity — combined vs default\n{gpu.replace('_', ' ')}"
                 f"  (max |Δ| = {worst:.2e})", fontsize=12)
    ax.tick_params(axis="y", labelsize=7)
    _legend_by_type(ax, set(sub["task_type"]))

    plt.tight_layout()
    save_fig(output_path, dpi=150)
    plt.close()


def print_summary(df: pd.DataFrame, gpu: str):
    """Console summary: median/mean speedup overall and split by the FA3 threshold."""
    med = df["latency_speedup"].median()
    geomean = np.exp(np.log(df["latency_speedup"]).mean())
    max_dscore = df["score_delta"].abs().max() if df["score_delta"].notna().any() else float("nan")
    print(f"\n  == {gpu} summary ({len(df)} tasks) ==")
    print(f"    latency speedup (default/combined): median {med:.3f}x, geomean {geomean:.3f}x, "
          f"max {df['latency_speedup'].max():.3f}x")
    print(f"    max |score delta| (combined-default): {max_dscore:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Plot TabArena combined-vs-default benchmark")
    parser.add_argument("--results-dir", type=str, default="results/tabarena",
                        help="Directory containing {gpu}/{mode}/tabarena_benchmark.json")
    parser.add_argument("--output-dir", type=str, default="plots/tabarena",
                        help="Directory to write plot images into")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Each GPU is a subdir containing default/ and combined/ mode dirs.
    gpus = sorted(p.name for p in results_dir.iterdir()
                  if p.is_dir() and (p / "default").exists() and (p / "combined").exists())
    if not gpus:
        print(f"No GPU dirs with both default/ and combined/ under {results_dir}")
        return

    for gpu in gpus:
        df = join_modes(results_dir, gpu)
        if df.empty:
            print(f"{gpu}: no jointly-successful tasks, skipping")
            continue

        _, meta = load_mode(results_dir, gpu, "combined")
        threshold = (meta or {}).get("combined_row_fa_threshold", DEFAULT_ROW_FA_THRESHOLD)

        out = output_dir / gpu
        out.mkdir(parents=True, exist_ok=True)

        plot_speedup_vs_size(df, gpu, threshold, out / "speedup_vs_size.pdf")
        plot_latency_diff_vs_size(df, gpu, threshold, out / "latency_diff_vs_size.pdf")
        plot_speedup_box(df, gpu, out / "speedup_box.pdf")
        plot_latency_diff_box(df, gpu, out / "latency_diff_box.pdf")
        _parity_scatter(df, "default_latency_ms", "combined_latency_ms",
                        "latency", "ms", gpu, out / "latency_parity.pdf")
        if df[["default_peak_gb", "combined_peak_gb"]].notna().all(axis=1).any():
            _parity_scatter(df, "default_peak_gb", "combined_peak_gb",
                            "peak memory", "GB", gpu, out / "peakmem_parity.pdf")
        plot_score_parity(df, gpu, out / "score_parity.pdf")

        print_summary(df, gpu)


if __name__ == "__main__":
    main()

"""TabArena inference benchmark: "optimal" combined attention vs TabPFN default.

Measures whether the "optimal" per-axis attention dispatch delivers real speedups
over TabPFN v2.6's own SDPA dispatch, on the *real* datasets of the TabArena-lite
benchmark (51 tasks: 38 classification + 13 regression), rather than synthetic
shape sweeps. Per task and per mode we record latency, peak memory, and accuracy
(classification) / R2 (regression).

Two modes:

  default  - TabPFN's own SDPA dispatch (FLASH->EFFICIENT->CUDNN->MATH), unmodified.
  combined - "optimal" per-axis dispatch:
               * column attention (between features) -> cuDNN
               * row attention (between cells)        -> mem-efficient SDPA for
                 <=COMBINED_ROW_FA_THRESHOLD rows, FlashAttention-3 beyond.

Why a per-axis monkeypatch (not the shared primitive):
  TabPFN routes BOTH attention axes through one shared function,
  v26._batched_scaled_dot_product_attention. Swapping that primitive can't tell
  the two axes apart (a wide column table would be misrouted to the row backend).
  So 'combined' monkeypatches the two forward() methods instead:
    AlongRowAttention.forward     -> column attn -> cuDNN
    AlongColumnAttention.forward  -> row attn    -> efficient (<=1k) / FA3 (>1k)
  The row forward sees the row count directly (x_BcRE.shape[1]), giving an exact
  <=1024 / >1024 split. The two forward bodies below are copied verbatim from
  tabpfn_v2_6.py, changing only which attention primitive they call.

Backend wrappers (permute/KV-expand/chunking) are copied from run_e2e_benchmark.py:
  cuDNN/efficient : permute to (B,H,S,D) and expand the MQA test KV head to full
                    multi-head (compute-equivalent to native GQA, not bit-identical
                    to v2.6's MQA cache path).
  fa (FA3)        : feed (B,S,H,D) directly, no permute; FA serves MQA natively.

Timing is wall-clock (perf_counter + cuda.synchronize) with cyclic GC disabled
across the timed loop (TabPFN preprocessing churns transient objects; a gen-2
collection mid-loop adds a ~100ms host pause -- see run_e2e_benchmark.py).

Results are saved one file per mode, flushed after every task:
    results/tabarena/{gpu}/{mode}/tabarena_benchmark.json
Speedup (default/combined) is derived at analysis time from the two files.

Run (combined needs the FA3 dep group; default runs anywhere):
    PYTHONPATH=e2e uv run --group e2e_fa3 python e2e/run_tabarena_benchmark.py
    PYTHONPATH=e2e uv run --group e2e_fa3 python e2e/run_tabarena_benchmark.py --tasks airfoil
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score
from torch.nn.attention import SDPBackend, sdpa_kernel

import tabpfn.architectures.tabpfn_v2_6 as v26
from tabpfn import TabPFNClassifier, TabPFNRegressor

from gpt4hana_base.registry import load_registry

# ── FlashAttention-3 (required by 'combined' for the >1k row path) ─────────────
_FA_FUNC = None
try:
    from flash_attn_interface import flash_attn_func as _FA_FUNC
except ImportError:
    pass

CKPT_CLASSIFIER = "/home/azureuser/TabPFN/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
CKPT_REGRESSOR = "/home/azureuser/TabPFN/checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt"

BENCHMARK = "os__tabarena_lite"
CUDA_MAX_GRID = 65536

# 'combined' row-attention (between-cells) split point: raw TRAINING rows <= this
# uses the mem-efficient SDPA kernel, beyond it uses FA3. Gates only the row axis;
# column attention always uses cuDNN.
#
# The row-attention layer never sees the raw train-row count directly: TabPFN
# prepends NUM_THINKING_ROWS fixed "thinking" rows (an internal v2.6 detail, see
# AddThinkingRows in tabpfn_v2_6.py) and also carries the test rows, so the layer's
# `single_eval_pos` = thinking + train. We subtract NUM_THINKING_ROWS to recover the
# dataset's train-row count, so the threshold means "<= this many TRAINING rows".
# NUM_THINKING_ROWS is the v2.6 default / this checkpoint's value (ModelConfig
# default = 64); hardcoded rather than read from the model config to keep the copied
# forward self-contained.
#
# Threshold set to 2048: at ~1k-2k rows FA3's tiling win isn't yet amortized over its
# launch overhead, so routing those tasks to FA3 produced a local speedup dip; the FA3
# advantage only clears the efficient/cuDNN kernels convincingly beyond ~2k rows.
COMBINED_ROW_FA_THRESHOLD = 2048
NUM_THINKING_ROWS = 64

# Originals, captured so restore_default() can put them back.
_ORIG_ROW_FORWARD = v26.AlongRowAttention.forward     # column attn (between features)
_ORIG_COL_FORWARD = v26.AlongColumnAttention.forward  # row attn   (between cells)

# Repetitions per task, keyed on train-row count (the item-attention seqlen). Cheap
# tables get many reps for tight statistics; the huge tables (tens of thousands of
# rows, seconds-to-minutes per predict) get the floor so the sweep stays affordable.
# Checked in descending threshold order; first threshold met (>=) wins, else DEFAULT.
REPS_BY_SEQLEN = [
    (65536, 3),
    (32768, 3),
    (16384, 3),
    (8192, 3),
    (4096, 3),
    (0, 3),
]
DEFAULT_REPS = 3


def reps_for_seqlen(seqlen):
    """Look up the rep count for a train-row count from REPS_BY_SEQLEN."""
    for threshold, reps in REPS_BY_SEQLEN:
        if seqlen >= threshold:
            return reps
    return DEFAULT_REPS


def _batch_chunks(num_parallel_calls, batch_dim0):
    """(sub_batch, num_iterations) splitting the batch under CUDA's launch grid.

    Copied from run_e2e_benchmark.py. Follows TabPFN's own chunking: iteration count
    is driven by the number of parallel attention calls (product of leading dims),
    but the chunk size is measured in dim-0 units, since we slice along dim 0.
    num_iterations is clamped to batch_dim0 so we never ask for more chunks than
    there are dim-0 elements (which would hand FlashAttention an empty batch)."""
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    num_iterations = max(1, min(num_iterations, batch_dim0))
    sub_batch = (batch_dim0 + num_iterations - 1) // num_iterations
    num_iterations = (batch_dim0 + sub_batch - 1) // sub_batch
    return sub_batch, num_iterations


def _make_sdpa_wrapped(backend):
    """Force a single SDPA kernel. Permutes to (B,H,S,D) and expands the MQA KV head
    to full multi-head (compute-equivalent to GQA, not bit-identical to v2.6's native
    MQA cache path). Copied from run_e2e_benchmark.py."""
    def wrapped(q_BSHD, k_BSJD, v_BSJD):
        q_BHSD = q_BSHD.permute(0, 2, 1, 3)
        k_BJSD = k_BSJD.permute(0, 2, 1, 3)
        v_BJSD = v_BSJD.permute(0, 2, 1, 3)

        H = q_BHSD.shape[-3]
        if k_BJSD.shape[-3] != H:
            k_BJSD = k_BJSD.expand(-1, H, -1, -1)
            v_BJSD = v_BJSD.expand(-1, H, -1, -1)

        sub_batch, num_iterations = _batch_chunks(q_BHSD.shape[:2].numel(), q_BHSD.shape[0])
        with sdpa_kernel(backends=[backend]):
            outputs = []
            for i in range(num_iterations):
                outputs.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        q_BHSD[i * sub_batch:(i + 1) * sub_batch],
                        k_BJSD[i * sub_batch:(i + 1) * sub_batch],
                        v_BJSD[i * sub_batch:(i + 1) * sub_batch],
                        attn_mask=None,
                    )
                )
        output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return output_BHSD.permute(0, 2, 1, 3)

    return wrapped


def _fa_wrapped(q_BSHD, k_BSJD, v_BSJD):
    """FlashAttention-3 wrapper. FA takes (B, S, H, D) DIRECTLY -- no permute, no
    contiguous copy -- and supports MQA, so the test KV head is NOT expanded.
    Copied from run_e2e_benchmark.py."""
    sub_batch, num_iterations = _batch_chunks(q_BSHD.shape[:2].numel(), q_BSHD.shape[0])
    outputs = []
    for i in range(num_iterations):
        q_i = q_BSHD[i * sub_batch:(i + 1) * sub_batch]
        if q_i.shape[0] == 0:  # never happens with real tables, but FA rejects empty batches
            continue
        out = _FA_FUNC(
            q_i,
            k_BSJD[i * sub_batch:(i + 1) * sub_batch],
            v_BSJD[i * sub_batch:(i + 1) * sub_batch],
        )
        if isinstance(out, tuple):  # FA3 returns (out, lse)
            out = out[0]
        outputs.append(out)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


# ── 'combined' forward methods (copied verbatim from tabpfn_v2_6.py, changing only
#    which attention primitive is called) ───────────────────────────────────────

# Column attention always uses cuDNN.
_COL_ATTN = _make_sdpa_wrapped(SDPBackend.CUDNN_ATTENTION)
# Row attention below/at the threshold uses mem-efficient SDPA.
_ROW_ATTN_EFFICIENT = _make_sdpa_wrapped(SDPBackend.EFFICIENT_ATTENTION)


def _combined_row_forward(self, x_BrSE):
    """AlongRowAttention.forward (column attention, between features) -> cuDNN.

    Verbatim copy of tabpfn_v2_6.AlongRowAttention.forward, with the single
    _batched_scaled_dot_product_attention(...) call replaced by _COL_ATTN(...)."""
    Br, C, _ = x_BrSE.shape
    q_flat_BrCHF = self.q_projection(x_BrSE)
    k_flat_BrCHF = self.k_projection(x_BrSE)
    v_flat_BrCHF = self.v_projection(x_BrSE)
    q_BrCHD = q_flat_BrCHF.view(Br, C, -1, self.head_dim)
    k_BrCHD = k_flat_BrCHF.view(Br, C, -1, self.head_dim)
    v_BrCHD = v_flat_BrCHF.view(Br, C, -1, self.head_dim)

    output_BrHCD = _COL_ATTN(q_BrCHD, k_BrCHD, v_BrCHD)
    output_BrCF = output_BrHCD.reshape(Br, C, self.head_dim * self.num_heads)
    return self.out_projection(output_BrCF)


def _combined_col_forward(self, x_BcRE, single_eval_pos=None):
    """AlongColumnAttention.forward (row attention, between cells) -> efficient
    for R <= COMBINED_ROW_FA_THRESHOLD, FA3 beyond.

    Verbatim copy of tabpfn_v2_6.AlongColumnAttention.forward, with each
    _batched_scaled_dot_product_attention(...) call replaced by `attn`, chosen
    once per forward from the raw train-row count so all sub-calls (train + MQA
    test) of this layer use the same primitive."""
    Bc, R, _ = x_BcRE.shape
    # Route the whole row axis by the dataset's TRAINING-row count. The train+thinking
    # length is single_eval_pos (or R when the whole input is train); subtract the
    # prepended thinking rows to recover the raw train count. <=threshold -> efficient,
    # else FA3.
    train_rows = (R if single_eval_pos is None else single_eval_pos) - NUM_THINKING_ROWS
    attn = _ROW_ATTN_EFFICIENT if train_rows <= COMBINED_ROW_FA_THRESHOLD else _fa_wrapped

    # If no single_eval_pos was specified, then the whole input is training.
    N = R if single_eval_pos is None else single_eval_pos

    q_flat_BcSHF = self.q_projection(x_BcRE)
    k_flat_BcNHF = self.k_projection(x_BcRE[:, :N])
    v_flat_BcNHF = self.v_projection(x_BcRE[:, :N])
    q_BcRHD = q_flat_BcSHF.view(Bc, R, -1, self.head_dim)
    k_BcNHD = k_flat_BcNHF.view(Bc, N, -1, self.head_dim)
    v_BcNHD = v_flat_BcNHF.view(Bc, N, -1, self.head_dim)

    if single_eval_pos == R:
        output_BcSHD = attn(q_BcRHD, k_BcNHD, v_BcNHD)
    else:
        out_train_BcNHD = attn(q_BcRHD[:, :N], k_BcNHD, v_BcNHD)
        out_test_BcMHD = attn(q_BcRHD[:, N:], k_BcNHD[:, :, :1], v_BcNHD[:, :, :1])
        output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

    output_BcSF = output_BcSHD.reshape(Bc, R, self.head_dim * self.num_heads)
    return self.out_projection(output_BcSF)


def set_mode(mode):
    """Install an attention execution mode by monkeypatching the two attention
    module forwards. 'default' restores TabPFN's originals."""
    if mode == "default":
        v26.AlongRowAttention.forward = _ORIG_ROW_FORWARD
        v26.AlongColumnAttention.forward = _ORIG_COL_FORWARD
    elif mode == "combined":
        if _FA_FUNC is None:
            raise RuntimeError("FlashAttention-3 not installed (use --group e2e_fa3)")
        v26.AlongRowAttention.forward = _combined_row_forward
        v26.AlongColumnAttention.forward = _combined_col_forward
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_tasks(name_filter=None):
    """Load TabArena-lite tasks, optionally filtered by a substring of task.name."""
    registry = load_registry()
    tasks = list(registry.benchmarks[BENCHMARK])
    if name_filter:
        tasks = [t for t in tasks if name_filter in t.name]
    return tasks


def is_regression(task):
    return str(task.task_type).upper().endswith("REGRESSION")


def make_model(task):
    """TabPFNRegressor for regression tasks, TabPFNClassifier otherwise."""
    if is_regression(task):
        return TabPFNRegressor(model_path=CKPT_REGRESSOR, device="cuda",
                               n_estimators=1, ignore_pretraining_limits=True)
    return TabPFNClassifier(model_path=CKPT_CLASSIFIER, device="cuda",
                            n_estimators=1, ignore_pretraining_limits=True)


def time_predict(task, X_train, y_train, X_test, y_test, reps, warmup=2):
    """Fit once, then time predict. Returns a dict of latency stats, peak mem, score.

    `reps` is chosen per task (see REPS_BY_SEQLEN). Classification times predict_proba
    (the natural inference call); regression times predict. The score is accuracy
    (classification) or R2 (regression), computed once from a final prediction.

    Cyclic GC is disabled across the timed loop: TabPFN preprocessing allocates many
    transient objects per predict, so a gen-2 collection mid-loop adds a ~100ms host
    pause and dominates variance (see run_e2e_benchmark.py). Re-enabled afterward."""
    reg = is_regression(task)
    clf = make_model(task)
    clf.fit(X_train, y_train)
    timed_call = clf.predict if reg else clf.predict_proba

    for _ in range(warmup):
        timed_call(X_test)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.disable()
    ts = []
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            timed_call(X_test)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    finally:
        gc.enable()
        gc.collect()
    peak = torch.cuda.max_memory_allocated() / 1e9

    # Score from a final prediction (predict for both regression and classification).
    y_pred = clf.predict(X_test)
    if reg:
        score, metric_name = float(r2_score(y_test, y_pred)), "r2"
    else:
        score, metric_name = float(accuracy_score(y_test, y_pred)), "accuracy"

    ts = np.array(ts)
    return {"latency_ms": ts.mean() * 1000, "latency_std_ms": ts.std() * 1000,
            "latency_median_ms": float(np.median(ts)) * 1000, "peak_mem_gb": peak,
            "score": score, "metric_name": metric_name}


def _bench_task(task, mode):
    """Time one (task, mode); return a result row (success or error)."""
    tr = task.get_tabular_data("train")
    te = task.get_tabular_data("test")
    X, y, X_test, y_test = tr.X, tr.y, te.X, te.y
    n_train, n_test, n_features = len(X), len(X_test), X.shape[1]
    reps = reps_for_seqlen(n_train)
    short = task.name.split("__")[2] if "__" in task.name else task.name
    kind = "reg" if is_regression(task) else "clf"
    base = {"task": task.name, "task_type": kind, "n_train": n_train,
            "n_test": n_test, "n_features": n_features, "reps": reps}
    try:
        stats = time_predict(task, X, y, X_test, y_test, reps=reps)
        print(f"  [{mode}/{kind}] {short:<28} nt={n_train:<6} nf={n_features:<4}: "
              f"{stats['latency_ms']:8.2f} +/- {stats['latency_std_ms']:5.2f} ms  "
              f"(median {stats['latency_median_ms']:8.2f})  peak={stats['peak_mem_gb']:.2f} GB  "
              f"{stats['metric_name']}={stats['score']:.4f}  (reps={reps})")
        return {**base, **stats}
    except Exception as e:
        print(f"  [{mode}/{kind}] {short:<28}: ERROR {type(e).__name__}: {str(e)[:60]}")
        return {**base, "error": f"{type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser(description="TabArena combined-vs-default attention benchmark")
    parser.add_argument("--modes", nargs="+", default=["default", "combined"],
                        choices=["default", "combined"],
                        help="Attention modes to run (default: both). 'combined' needs --group e2e_fa3.")
    parser.add_argument("--tasks", default=None,
                        help="Only run tasks whose name contains this substring (for smoke runs).")
    args = parser.parse_args()

    gpu = torch.cuda.get_device_name()
    tasks = load_tasks(args.tasks)
    print(f"GPU: {gpu} | benchmark={BENCHMARK} | {len(tasks)} tasks | modes={args.modes} "
          f"| row FA3 threshold={COMBINED_ROW_FA_THRESHOLD}")

    base_dir = Path(__file__).resolve().parent.parent / "results" / "tabarena" / \
        gpu.replace(" ", "_").replace("/", "_").replace("-", "_")

    _MODE_NOTES = {
        "default": "TabPFN's own SDPA dispatch (FLASH->EFFICIENT->CUDNN->MATH), unmodified.",
        "combined": "column attn -> cuDNN; row attn -> mem-efficient SDPA for "
                    f"<={COMBINED_ROW_FA_THRESHOLD} rows, FA3 beyond. Monkeypatches "
                    "AlongRowAttention/AlongColumnAttention.forward. Efficient expands the "
                    "MQA test KV head (compute-equivalent to GQA); FA3 serves MQA natively.",
    }

    def meta_for(mode):
        return {
            "timestamp": datetime.now().isoformat(), "gpu": gpu, "benchmark": BENCHMARK,
            "mode": mode, "checkpoint_classifier": CKPT_CLASSIFIER,
            "checkpoint_regressor": CKPT_REGRESSOR,
            "combined_row_fa_threshold": COMBINED_ROW_FA_THRESHOLD,
            "num_tasks": len(tasks), "note": _MODE_NOTES[mode],
        }

    # One file per mode, flushed after every task so a crash only loses the in-flight task.
    for mode in args.modes:
        out_dir = base_dir / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "tabarena_benchmark.json"
        meta = meta_for(mode)
        results = []

        def flush():
            with open(out_file, "w") as f:
                json.dump({"metadata": meta, "results": results}, f, indent=2)

        set_mode(mode)
        print(f"\n== mode={mode} ==")
        for task in tasks:
            results.append(_bench_task(task, mode))
            flush()
        set_mode("default")
        torch.cuda.empty_cache()
        print(f"  -> saved {out_file}")


if __name__ == "__main__":
    main()

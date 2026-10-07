"""E2E inference-latency benchmark for TabPFNv2.6 across attention backends.

Measures full predict_proba latency of TabPFN v2.6 under several attention
execution modes, swept over the same shape ranges as the attention
microbenchmark:

  default   - TabPFN's own SDPA dispatch (FLASH->EFFICIENT->CUDNN->MATH)
  efficient - force the mem-efficient (xformers) SDPA kernel
  cudnn     - force the cuDNN SDPA kernel
  fa        - "our" FlashAttention wrapper (FA2 or FA3, whichever is installed)
  sage      - SageAttention

Two sweeps (each is a full predict_proba; TabPFN runs BOTH feature- and
item-attention per call, so we vary the axis that drives the dominant one):

  col: vary feature-attention seqlen 16..2048 (n_features = seqlen*features_per_group),
       fixed 512 train + 512 test rows (total R=1024 drives the column-attn seqlen).
  row: vary train rows 32..131072 (item-attention seqlen), fixed 64 columns.
       The test-row count is set by --query: single (1 query row, prediction
       latency) or batched (1024 query rows, throughput).

Backend forcing: we REPLACE tabpfn_v2_6._batched_scaled_dot_product_attention
with a wrapper that dispatches to the chosen backend. TabPFN hands the wrapper
q/k/v already in (B, S, H, D) layout.

  efficient/cudnn : permute to (B, H, S, D) for torch SDPA, and expand the
                    multi-query test KV head to full multi-head so ANY SDPA
                    kernel can serve it (compute-equivalent to TabPFN's GQA,
                    NOT bit-identical to v2.6's native MQA KV-cache path).
  fa              : "our" tabular wrapper — feed FlashAttention the (B, S, H, D)
                    tensors DIRECTLY (no permute, no contiguous copy). FA
                    supports MQA natively, so the test KV head is NOT expanded:
                    this is closer to native v2.6 than the SDPA modes above.
  sage            : feed sageattn the (B, S, H, D) tensors directly (no permute),
                    but expand the KV head since Sage needs matched head counts.

Timing is wall-clock (perf_counter + cuda.synchronize), since predict_proba mixes
CPU preprocessing, host<->device copies, and the GPU forward.

Results are saved one file per mode (mirroring run_benchmark.py's per-backend
layout), flushed after every shape:
    results/e2e/{query}-query/{gpu}/{mode}/e2e_benchmark.json

The modes actually run are selected with --modes (default: the SDPA trio), so
each make target runs only the backend its dependency group provides. --query
selects the row-sweep test-row count (single|batched). Run:
    PYTHONPATH=e2e uv run --group e2e     python e2e/run_e2e_benchmark.py --query batched
    PYTHONPATH=e2e uv run --group e2e_fa2 python e2e/run_e2e_benchmark.py --modes fa --query single
    PYTHONPATH=e2e uv run --group e2e_sage python e2e/run_e2e_benchmark.py --modes sage --query single
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from torch.nn.attention import SDPBackend, sdpa_kernel

import tabpfn.architectures.tabpfn_v2_6 as v26
from tabpfn import TabPFNClassifier

# ── optional attention backends (each lives in its own e2e_* dep group) ───────
# FlashAttention: prefer FA3, fall back to FA2. Both take (B, S, H, D) natively
# and support MQA/GQA, so "our" wrapper feeds q/k/v straight through.
_FA_FUNC = None
_FA_LABEL = None
try:
    from flash_attn_interface import flash_attn_func as _fa3
    _FA_FUNC, _FA_LABEL = _fa3, "fa3"
except ImportError:
    try:
        from flash_attn import flash_attn_func as _fa2
        _FA_FUNC, _FA_LABEL = _fa2, "fa2"
    except ImportError:
        pass

try:
    from sageattention import sageattn as _sageattn
except ImportError:
    _sageattn = None

CKPT = "/home/azureuser/TabPFN/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
FEATURES_PER_GROUP = 3
CUDA_MAX_GRID = 65536

_ORIG = v26._batched_scaled_dot_product_attention

# ── sweep definitions ────────────────────────────────────────────────────────
COL_FEATURE_SEQLENS = [16, 32, 64, 128, 256, 512, 1024, 2048]  # feature-attn seqlen
COL_TRAIN = 512
COL_TEST = 512

ROW_TRAIN = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]  # item-attn seqlen
ROW_COLS = 64

# Row sweep test-row count is set by --query: how many query (test) rows the row
# attention serves. "single" (ROW_TEST=1) is the latency-of-one-prediction case;
# "batched" (ROW_TEST=1024) is the throughput case. Results are written under a
# matching {query}-query/ dir so the two sweeps never overwrite each other and the
# plotter can tell them apart.
ROW_TEST_BY_QUERY = {"single": 1, "batched": 1024}
ROW_TEST = ROW_TEST_BY_QUERY["batched"]  # default; overridden by --query in main()

ALL_MODES = ["default", "efficient", "cudnn", "fa", "sage"]
_SDPA_BACKEND = {
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

# Repetitions per shape, keyed on the swept seqlen. Small/fast shapes get many
# reps for tight statistics; the large row shapes (tens of thousands of rows,
# minutes per call) get the floor so the sweep stays affordable. Edit this table
# to change the rep schedule. Checked in descending threshold order; the first
# threshold a seqlen meets (>=) wins, else DEFAULT_REPS.
REPS_BY_SEQLEN = [
    (65536, 3),    # >=65536 rows: ~minutes/call -> 3 reps
    (32768, 3),
    (16384, 5),
    (8192, 5),
    (4096, 10),
    (0, 20),       # everything smaller: 50 reps
]
DEFAULT_REPS = 50


def reps_for_seqlen(seqlen):
    """Look up the rep count for a swept seqlen from REPS_BY_SEQLEN."""
    for threshold, reps in REPS_BY_SEQLEN:
        if seqlen >= threshold:
            return reps
    return DEFAULT_REPS


def _batch_chunks(num_parallel_calls, batch_dim0):
    """(sub_batch, num_iterations) splitting the batch under CUDA's launch grid.

    Follows TabPFN's own chunking: the iteration count is driven by the number of
    parallel attention calls (product of the leading dims), but the chunk size is
    measured in dim-0 units, since we slice along dim 0.

    num_iterations is clamped to batch_dim0: you cannot split dim 0 into more
    non-empty chunks than it has elements. Without this clamp, row attention
    (where dim 0 = #columns is small but the seqlen is huge) would ask for more
    iterations than there are rows to slice, producing empty batches — which
    FlashAttention rejects with "batch size must be positive" (SDPA silently
    tolerates the empty slices, which is why TabPFN's own code never trips here).
    """
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    num_iterations = max(1, min(num_iterations, batch_dim0))
    sub_batch = (batch_dim0 + num_iterations - 1) // num_iterations
    # ceil() above can overshoot (sub_batch * num_iterations > batch_dim0),
    # leaving trailing empty chunks; recompute the iteration count from the chunk
    # size so the chunks exactly tile [0, batch_dim0) with no empty slice.
    num_iterations = (batch_dim0 + sub_batch - 1) // sub_batch
    return sub_batch, num_iterations


def _make_sdpa_wrapped(backend):
    """Force a single SDPA kernel. Permutes to (B,H,S,D) and expands the MQA
    KV head to full multi-head (compute-equivalent to GQA, not bit-identical to
    v2.6's native MQA cache path)."""
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
    """"Our" FlashAttention wrapper. FA takes (B, S, H, D) DIRECTLY — no permute,
    no contiguous copy — and supports MQA, so the test KV head is NOT expanded."""
    sub_batch, num_iterations = _batch_chunks(q_BSHD.shape[:2].numel(), q_BSHD.shape[0])
    outputs = []
    for i in range(num_iterations):
        q_i = q_BSHD[i * sub_batch:(i + 1) * sub_batch]
        if q_i.shape[0] == 0:  # never happens with the current sweep, but FA rejects empty batches
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


def _sage_wrapped(q_BSHD, k_BSJD, v_BSJD):
    """SageAttention wrapper. sageattn takes (B, S, H, D) with NHD layout — no
    permute — but Sage needs matched head counts, so expand the MQA KV head.
    Sage's int8-quantized kernel produces NaNs on the strided q/k/v TabPFN hands
    us, so we make each chunk contiguous (as the microbenchmark's sage.py does)."""
    H = q_BSHD.shape[-2]
    if k_BSJD.shape[-2] != H:
        k_BSJD = k_BSJD.expand(-1, -1, H, -1)
        v_BSJD = v_BSJD.expand(-1, -1, H, -1)

    sub_batch, num_iterations = _batch_chunks(q_BSHD.shape[:2].numel(), q_BSHD.shape[0])
    outputs = []
    for i in range(num_iterations):
        outputs.append(
            _sageattn(
                q_BSHD[i * sub_batch:(i + 1) * sub_batch].contiguous(),
                k_BSJD[i * sub_batch:(i + 1) * sub_batch].contiguous(),
                v_BSJD[i * sub_batch:(i + 1) * sub_batch].contiguous(),
                tensor_layout="NHD",
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def set_mode(mode):
    """Install an attention execution mode by monkeypatching
    v26._batched_scaled_dot_product_attention. 'default' restores TabPFN's body."""
    if mode == "default":
        v26._batched_scaled_dot_product_attention = _ORIG
    elif mode in _SDPA_BACKEND:
        v26._batched_scaled_dot_product_attention = _make_sdpa_wrapped(_SDPA_BACKEND[mode])
    elif mode == "fa":
        if _FA_FUNC is None:
            raise RuntimeError("FlashAttention not installed (use --group e2e_fa2 or e2e_fa3)")
        v26._batched_scaled_dot_product_attention = _fa_wrapped
    elif mode == "sage":
        if _sageattn is None:
            raise RuntimeError("SageAttention not installed (use --group e2e_sage)")
        v26._batched_scaled_dot_product_attention = _sage_wrapped
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_table(n_train, n_test, n_features, n_classes=3, seed=0):
    X, y = make_classification(
        n_samples=n_train + n_test, n_features=n_features,
        n_informative=max(n_classes, min(n_features // 2, 200)), n_redundant=0,
        n_classes=n_classes, n_clusters_per_class=2, random_state=seed,
    )
    X = X.astype("float32")
    return X[:n_train], y[:n_train], X[n_train:]


def time_predict(n_train, n_test, n_features, reps, warmup=3):
    """Wall-clock predict_proba latency + peak memory. Model is fit once.

    Returns (mean_ms, std_ms, median_ms, peak_gb).

    `reps` is chosen per shape by the caller (see REPS_BY_SEQLEN) so cheap shapes
    get tight statistics (many reps) while multi-minute shapes stay affordable.

    Python's cyclic GC is disabled across the timed loop: TabPFN's preprocessing
    allocates many transient objects per predict_proba (roughly with n_features),
    so on the larger column shapes a gen-2 collection fires every few reps and
    adds a ~100+ ms host-side pause. Those GC pauses were the dominant source of
    variance on the column sweep (spikes to ~1.6x with the GPU still at max clock,
    not throttling); disabling GC removes them and drops the std ~14x. GC is
    re-enabled (and a collection forced) afterward so memory isn't left to grow."""
    X_train, y_train, X_test = make_table(n_train, n_test, n_features)
    clf = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=1,
                           ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)

    for _ in range(warmup):
        clf.predict_proba(X_test)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.disable()
    ts = []
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            clf.predict_proba(X_test)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    finally:
        gc.enable()
        gc.collect()
    ts = np.array(ts)
    return (ts.mean() * 1000, ts.std() * 1000, float(np.median(ts)) * 1000,
            torch.cuda.max_memory_allocated() / 1e9)


def _bench_case(kind, mode, seqlen, n_train, n_test, n_features):
    """Time one (kind, mode, shape) case; return a result row (success or error)."""
    reps = reps_for_seqlen(seqlen)
    try:
        m, s, med, peak = time_predict(n_train, n_test, n_features, reps=reps)
        print(f"  [{kind}/{mode}] seqlen={seqlen:<5} "
              f"nt={n_train} nf={n_features}: {m:8.2f} +/- {s:5.2f} ms  "
              f"(median {med:8.2f})  peak={peak:.2f} GB  (reps={reps})")
        return {"attn_type": kind, "mode": mode, "seqlen": seqlen,
                "n_train": n_train, "n_test": n_test, "n_features": n_features,
                "latency_ms": m, "latency_std_ms": s, "latency_median_ms": med,
                "peak_mem_gb": peak, "reps": reps}
    except Exception as e:
        print(f"  [{kind}/{mode}] seqlen={seqlen}: ERROR {type(e).__name__}: {str(e)[:50]}")
        return {"attn_type": kind, "mode": mode, "seqlen": seqlen,
                "n_train": n_train, "n_features": n_features,
                "error": f"{type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser(description="TabPFNv2.6 E2E inference benchmark")
    parser.add_argument("--modes", nargs="+", default=["default", "efficient", "cudnn"],
                        choices=ALL_MODES,
                        help="Attention modes to run (default: the SDPA trio). Each "
                             "make target passes the mode its dependency group provides.")
    parser.add_argument("--query", default="batched", choices=list(ROW_TEST_BY_QUERY),
                        help="Row-sweep test-row count: 'single' (ROW_TEST=1, latency of "
                             "one prediction) or 'batched' (ROW_TEST=1024, throughput). "
                             "Results go under results/e2e/{query}-query/.")
    args = parser.parse_args()

    global ROW_TEST
    ROW_TEST = ROW_TEST_BY_QUERY[args.query]

    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu} | TabPFN v2.6 inference sweep | query={args.query} "
          f"(ROW_TEST={ROW_TEST}) | modes={args.modes}")

    col_cases = [(sl, COL_TRAIN, COL_TEST, sl * FEATURES_PER_GROUP)
                 for sl in COL_FEATURE_SEQLENS]
    row_cases = [(nt, nt, ROW_TEST, ROW_COLS) for nt in ROW_TRAIN]

    base_dir = Path(__file__).resolve().parent.parent / "results" / "e2e" / \
        f"{args.query}-query" / \
        gpu.replace(" ", "_").replace("/", "_").replace("-", "_")

    # "fa" writes to a fa2/fa3 dir reflecting the installed backend, so FA2 and
    # FA3 results can coexist and be told apart in the plots.
    def out_name(mode):
        return _FA_LABEL if (mode == "fa" and _FA_LABEL) else mode

    _MODE_NOTES = {
        "efficient": "permutes to (B,H,S,D) and expands the MQA test KV head to multi-head "
                     "(compute-equivalent to native GQA, not bit-identical to v2.6 MQA cache path).",
        "cudnn": "permutes to (B,H,S,D) and expands the MQA test KV head to multi-head "
                 "(compute-equivalent to native GQA, not bit-identical to v2.6 MQA cache path).",
        "fa": f"'our' FlashAttention ({_FA_LABEL}) wrapper: feeds (B,S,H,D) directly, no permute "
              "and no KV-expand (FA serves the MQA test path natively).",
        "sage": "SageAttention on (B,S,H,D) directly (no permute); expands the MQA test KV "
                "head and makes each chunk contiguous (Sage's int8 kernel NaNs on the "
                "strided q/k/v TabPFN produces).",
    }

    def meta_for(mode):
        return {
            "timestamp": datetime.now().isoformat(), "gpu": gpu, "checkpoint": CKPT,
            "mode": out_name(mode), "query": args.query, "features_per_group": FEATURES_PER_GROUP,
            "col": {"feature_seqlens": COL_FEATURE_SEQLENS, "n_train": COL_TRAIN, "n_test": COL_TEST},
            "row": {"train_rows": ROW_TRAIN, "cols": ROW_COLS, "n_test": ROW_TEST},
            "note": _MODE_NOTES.get(mode, "TabPFN's own SDPA dispatch (unmodified)."),
        }

    # One file per mode (mirrors the per-backend layout of run_benchmark.py),
    # flushed after every case so a crash only loses the in-flight shape.
    for mode in args.modes:
        label = out_name(mode)
        out_dir = base_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "e2e_benchmark.json"
        meta = meta_for(mode)
        results = []

        def flush():
            with open(out_file, "w") as f:
                json.dump({"metadata": meta, "results": results}, f, indent=2)

        set_mode(mode)
        print(f"\n== mode={label} : column sweep ==")
        for seqlen, n_train, n_test, n_features in col_cases:
            results.append(_bench_case("col", label, seqlen, n_train, n_test, n_features))
            flush()
        print(f"== mode={label} : row sweep ==")
        for seqlen, n_train, n_test, n_features in row_cases:
            results.append(_bench_case("row", label, seqlen, n_train, n_test, n_features))
            flush()
        set_mode("default")
        torch.cuda.empty_cache()
        print(f"  -> saved {out_file}")


if __name__ == "__main__":
    main()

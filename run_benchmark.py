"""Benchmark tabular attention patterns for TabPFN-style models.

Tabular tensors have shape (batch, rows, cols, nheads, headdim).
We benchmark two attention patterns:
  - Column attention: attend across columns (seq_len=cols, batch_eff=batch*rows)
  - Row attention: attend across rows (seq_len=rows, batch_eff=batch*cols)

For column attention, the memory layout is already contiguous for the sequence dimension.
For row attention, we need to transpose rows<->cols, which requires .contiguous() for SDPA
but FA3/FA4 can operate on strided tensors directly.

Usage:
    uv run python benchmarks/benchmark_tabular_attn.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from triton import runtime
from tqdm import tqdm

from tabular_attn import (
    col_attn_sdpa_math,
    col_attn_sdpa_efficient,
    col_attn_sdpa_cudnn,
    col_attn_fa2,
    col_attn_fa2_kv_packed,
    col_attn_fa3,
    col_attn_fa4,
    row_attn_sdpa_math,
    row_attn_sdpa_efficient,
    row_attn_sdpa_cudnn,
    row_attn_fa2,
    row_attn_fa2_kv_packed,
    row_attn_fa3,
    row_attn_fa4,
)


def get_env_info() -> dict[str, str]:
    """Collect GPU, CUDA, cuDNN, and PyTorch version info."""
    info = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None) or "N/A",
    }
    if torch.backends.cudnn.is_available():
        v = torch.backends.cudnn.version()
        info["cudnn_version"] = f"{v // 10000}.{(v % 10000) // 100}.{v % 100}"
        info["cudnn_enabled"] = torch.backends.cudnn.enabled
    else:
        info["cudnn_version"] = "N/A"
        info["cudnn_enabled"] = False
    return info


def flops(batch_eff: int, nheads: int, seqlen: int, headdim: int, causal: bool = False):
    """Calculate FLOPs for attention."""
    if causal:
        avg_seqlen = (seqlen + 1) / 2
    else:
        avg_seqlen = seqlen
    return 4 * batch_eff * nheads * seqlen * avg_seqlen * headdim


def do_bench_fixed_reps(fn, warmup: int = 5, rep: int= 100, grad_to_none=None):
    """Benchmark with fixed number of repetitions (not fixed time).

    Based on triton.testing.do_bench but uses rep as actual repetition count
    instead of target time in ms.
    """
    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Create events for timing
    start_event = [di.Event(enable_timing=True) for _ in range(rep)]
    end_event = [di.Event(enable_timing=True) for _ in range(rep)]

    # Warm-up
    for _ in range(warmup):
        fn()

    # Benchmark with fixed rep count
    for i in range(rep):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        runtime.driver.active.clear_cache(cache)
        start_event[i].record()
        fn()
        end_event[i].record()

    di.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return times_ms


def benchmark_fn_all(fn: Callable, warmup: int = 25, rep: int = 100) -> list[float]:
    """Benchmark a function and return all individual times in seconds."""
    times_ms = do_bench_fixed_reps(fn, warmup=warmup, rep=rep)
    return [t * 1e-3 for t in times_ms]


def gpu_warmup():
    """Run some GPU work to stabilize clocks before benchmarking."""
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    for _ in range(50):
        x = torch.matmul(x, x)
        x = x / x.norm()
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()


def create_tabular_inputs(batch, rows, cols, nheads, headdim, dtype, device, requires_grad=True):
    """Create tabular Q, K, V tensors.

    Shape: (batch, rows, cols, nheads, headdim)
    """
    q = torch.randn(batch, rows, cols, nheads, headdim, dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn(batch, rows, cols, nheads, headdim, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(batch, rows, cols, nheads, headdim, dtype=dtype, device=device, requires_grad=requires_grad)
    kv_packed = torch.randn(batch, rows, cols, 2, nheads, headdim, dtype=dtype, device=device, requires_grad=requires_grad)
    return q, k, v, kv_packed


# =============================================================================
# Column Attention: attend across columns
# Input: (batch, rows, cols, nheads, headdim)
# Reshape to: (batch*rows, cols, nheads, headdim) - already contiguous
# =============================================================================
def fwd_bwd_col_attn_sdpa_math(q, k, v, causal=False):
    """Column attention using PyTorch SDPA."""
    # Pre-allocate dout so the backward closure doesn't pay for
    # torch.randn_like + RNG every rep.
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_sdpa_math(q, k, v, causal=causal)

    def fwd_bwd():
        out = col_attn_sdpa_math(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_sdpa_efficient(q, k, v, causal=False):
    """Column attention using PyTorch SDPA."""
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_sdpa_efficient(q, k, v, causal=causal)

    def fwd_bwd():
        out = col_attn_sdpa_efficient(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_sdpa_cudnn(q, k, v, causal=False):
    """Column attention using PyTorch SDPA."""
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_sdpa_cudnn(q, k, v, causal=causal)

    def fwd_bwd():
        out = col_attn_sdpa_cudnn(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_fa2(q, k, v, causal=False):
    """Column attention using FlashAttention-2."""
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_fa2(q, k, v, causal=causal)

    def fwd_bwd():
        out = col_attn_fa2(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_fa2_packed(q, kv, causal=False):
    """Column attention using FlashAttention-2 in kv packed variant."""
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_fa2_kv_packed(q, kv, causal=causal)

    def fwd_bwd():
        out = col_attn_fa2_kv_packed(q, kv, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_fa3(q, k, v, causal=False):
    """Column attention using FlashAttention-3."""
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_fa3(q, k, v, causal=causal)

    def fwd_bwd():
        out = col_attn_fa3(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_fa4(q, k, v, causal=False):
    """Column attention using FlashAttention-4."""
    dout = torch.randn_like(q)

    def fwd():
        return col_attn_fa4(q, k, v, causal=causal)

    def fwd_bwd():
        out = col_attn_fa4(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_col_attn_sage(q, k, v, causal=False):
    """Column attention using SageAttention."""
    def fwd():
        return col_attn_sage(q, k, v, causal=causal)

    return fwd, None


def fwd_bwd_col_attn_vllm(q, k, v, causal=False):
    """Column attention using vLLM Triton prefill kernel."""
    def fwd():
        return col_attn_vllm(q, k, v, causal=causal)

    return fwd, None


# =============================================================================
# Row Attention: attend across rows
# Input: (batch, rows, cols, nheads, headdim)
# Need: (batch*cols, rows, nheads, headdim) - requires transpose
# =============================================================================

def fwd_bwd_row_attn_sdpa_math(q, k, v, causal=False):
    """Row attention using PyTorch SDPA."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_sdpa_math(q, k, v, causal=causal)

    def fwd_bwd():
        out = row_attn_sdpa_math(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_sdpa_efficient(q, k, v, causal=False):
    """Row attention using PyTorch SDPA."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_sdpa_efficient(q, k, v, causal=causal)

    def fwd_bwd():
        out = row_attn_sdpa_efficient(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_sdpa_cudnn(q, k, v, causal=False):
    """Row attention using PyTorch SDPA."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_sdpa_cudnn(q, k, v, causal=causal)

    def fwd_bwd():
        out = row_attn_sdpa_cudnn(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_fa2(q, k, v, causal=False):
    """Row attention using FlashAttention-2."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_fa2(q, k, v, causal=causal)

    def fwd_bwd():
        out = row_attn_fa2(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_fa2_packed(q, kv, causal=False):
    """Row attention using FlashAttention-2."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_fa2_kv_packed(q, kv, causal=causal)

    def fwd_bwd():
        out = row_attn_fa2_kv_packed(q, kv, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_fa3(q, k, v, causal=False):
    """Row attention using FlashAttention-3."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_fa3(q, k, v, causal=causal)

    def fwd_bwd():
        out = row_attn_fa3(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_fa4(q, k, v, causal=False):
    """Row attention using FlashAttention-4."""
    dout = torch.randn_like(q)

    def fwd():
        return row_attn_fa4(q, k, v, causal=causal)

    def fwd_bwd():
        out = row_attn_fa4(q, k, v, causal=causal)
        out.backward(dout)

    return fwd, fwd_bwd


def fwd_bwd_row_attn_sage(q, k, v, causal=False):
    """Row attention using SageAttention."""
    def fwd():
        return row_attn_sage(q, k, v, causal=causal)

    return fwd, None


def fwd_bwd_row_attn_vllm(q, k, v, causal=False):
    """Row attention using vLLM Triton prefill kernel."""
    def fwd():
        return row_attn_vllm(q, k, v, causal=causal)

    return fwd, None



def run_benchmarks(
        col_attn_rows: int,
        col_attn_cols: list[int],
        row_attn_rows: list[int],
        row_attn_cols: int,
        nheads: int,
        headdim: int,
        dtype: torch.dtype,
        causal: bool,
        warmup: int,
        rep: int,
        backends: list[str],
        flush_file: "Path | None" = None,
        metadata: dict | None = None,
) -> list[dict]:
    """Run tabular attention benchmarks.

    Column attention: fixed rows, varying cols (seq_len=cols, batch_eff=rows)
    Row attention: fixed cols, varying rows (seq_len=rows, batch_eff=cols)
    """
    device = "cuda"
    batch = 1  # Fixed batch size for tabular models
    results = []

    # Warmup GPU to stabilize clocks
    gpu_warmup()

    # Build list of all (attn_type, rows, cols, backend) combinations
    tasks = []
    for cols in col_attn_cols:
        for backend in backends:
            tasks.append(("col", col_attn_rows, cols, backend))
    for rows in row_attn_rows:
        for backend in backends:
            tasks.append(("row", rows, row_attn_cols, backend))

    pbar = tqdm(tasks, desc="Benchmarking", unit="bench",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    prev_key = None
    for attn_type, rows, cols, backend in pbar:
        pbar.set_postfix_str(f"{attn_type} r={rows} c={cols} {backend}")

        if attn_type == "col":
            fwd_flops = flops(batch * rows, nheads, cols, headdim, causal)
        else:
            fwd_flops = flops(batch * cols, nheads, rows, headdim, causal)
        bwd_flops = 2.5 * fwd_flops

        try:
            # Recreate tensors when shape changes
            curr_key = (rows, cols)
            if curr_key != prev_key:
                q, k, v, kv = create_tabular_inputs(batch, rows, cols, nheads, headdim, dtype, device)
                prev_key = curr_key

            q.grad, k.grad, v.grad, kv.grad = None, None, None, None

            # Select the right fwd/bwd function
            fn_map_col = {
                "sdpa_efficient": lambda: fwd_bwd_col_attn_sdpa_efficient(q, k, v, causal),
                "sdpa_cudnn": lambda: fwd_bwd_col_attn_sdpa_cudnn(q, k, v, causal),
                "sdpa_math": lambda: fwd_bwd_col_attn_sdpa_math(q, k, v, causal),
                "fa2": lambda: fwd_bwd_col_attn_fa2(q, k, v, causal),
                "fa2_kv_packed": lambda: fwd_bwd_col_attn_fa2_packed(q, kv, causal),
                "fa3": lambda: fwd_bwd_col_attn_fa3(q, k, v, causal),
                "fa4": lambda: fwd_bwd_col_attn_fa4(q, k, v, causal),
                "sage": lambda: fwd_bwd_col_attn_sage(q, k, v, causal),
                "vllm": lambda: fwd_bwd_col_attn_vllm(q, k, v, causal),
            }
            fn_map_row = {
                "sdpa_efficient": lambda: fwd_bwd_row_attn_sdpa_efficient(q, k, v, causal),
                "sdpa_cudnn": lambda: fwd_bwd_row_attn_sdpa_cudnn(q, k, v, causal),
                "sdpa_math": lambda: fwd_bwd_row_attn_sdpa_math(q, k, v, causal),
                "fa2": lambda: fwd_bwd_row_attn_fa2(q, k, v, causal),
                "fa2_kv_packed": lambda: fwd_bwd_row_attn_fa2_packed(q, kv, causal),
                "fa3": lambda: fwd_bwd_row_attn_fa3(q, k, v, causal),
                "fa4": lambda: fwd_bwd_row_attn_fa4(q, k, v, causal),
                "sage": lambda: fwd_bwd_row_attn_sage(q, k, v, causal),
                "vllm": lambda: fwd_bwd_row_attn_vllm(q, k, v, causal),
            }

            fn_map = fn_map_col if attn_type == "col" else fn_map_row
            if backend not in fn_map:
                raise ValueError(f"Unknown backend {backend}")
            fwd_fn, fwd_bwd_fn = fn_map[backend]()

            fwd_times = benchmark_fn_all(fwd_fn, warmup=warmup, rep=rep)

            if fwd_bwd_fn is not None:
                q.grad, k.grad, v.grad, kv.grad = None, None, None, None
                fwd_bwd_times = benchmark_fn_all(fwd_bwd_fn, warmup=warmup, rep=rep)
            else:
                fwd_bwd_times = None
            torch.cuda.synchronize()  # surface any async CUDA errors before leaving the try block

            seq_len = cols if attn_type == "col" else rows
            batch_eff = batch * rows if attn_type == "col" else batch * cols

            fwd_arr = np.array(fwd_times)
            fwd_mean = fwd_arr.mean()
            fwd_std = fwd_arr.std()
            fwd_tflops = fwd_flops / fwd_mean / 1e12
            fwd_tflops_std = fwd_tflops * (fwd_std / fwd_mean) if fwd_mean > 0 else 0

            if fwd_bwd_times is not None:
                fwd_bwd_arr = np.array(fwd_bwd_times)
                # Backward = fwd_bwd - fwd (independent runs → variances add)
                bwd_mean = max(fwd_bwd_arr.mean() - fwd_mean, 0)
                bwd_std = np.sqrt(fwd_bwd_arr.var() + fwd_arr.var())
                bwd_time_ms = bwd_mean * 1000
                bwd_time_std_ms = bwd_std * 1000
                bwd_tflops = bwd_flops / bwd_mean / 1e12 if bwd_mean > 0 else 0
                # Propagate σ for TFLOPS: tflops = flops/t → σ_tflops ≈ tflops * (σ_t / t)
                bwd_tflops_std = bwd_tflops * (bwd_std / bwd_mean) if bwd_mean > 0 else 0
            else:
                bwd_time_ms = bwd_time_std_ms = bwd_tflops = bwd_tflops_std = None

            results.append({
                "attn_type": attn_type,
                "backend": backend,
                "batch": batch,
                "rows": rows,
                "cols": cols,
                "nheads": nheads,
                "headdim": headdim,
                "seq_len": seq_len,
                "batch_eff": batch_eff,
                "causal": causal,
                "dtype": str(dtype),
                "rep": rep,
                "fwd_time_ms": fwd_mean * 1000,
                "fwd_time_std_ms": fwd_std * 1000,
                "fwd_tflops": fwd_tflops,
                "fwd_tflops_std": fwd_tflops_std,
                "bwd_time_ms": bwd_time_ms,
                "bwd_time_std_ms": bwd_time_std_ms,
                "bwd_tflops": bwd_tflops,
                "bwd_tflops_std": bwd_tflops_std,
            })

            if flush_file is not None:
                flush_file.parent.mkdir(parents=True, exist_ok=True)
                with open(flush_file, "w") as f:
                    json.dump({"metadata": metadata, "results": results}, f, indent=2)

        except Exception as e:
            tqdm.write(f"ERROR [{attn_type} r={rows} c={cols} {backend}]: {type(e).__name__}: {e}")
            results.append({
                "attn_type": attn_type,
                "backend": backend,
                "rows": rows,
                "cols": cols,
                "error": str(e),
            })
            if flush_file is not None:
                flush_file.parent.mkdir(parents=True, exist_ok=True)
                with open(flush_file, "w") as f:
                    json.dump({"metadata": metadata, "results": results}, f, indent=2)
            # A CUDA illegal memory access corrupts the context; no further CUDA
            # calls (including empty_cache) are safe after that point.
            if "illegal memory access" in str(e) or "CUDA error" in str(e):
                pbar.close()
                return results

        torch.cuda.empty_cache()

    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark tabular attention patterns")
    parser.add_argument("--col-attn-rows", type=int, default=1024,
                        help="Fixed row count for column attention benchmark")
    parser.add_argument("--col-attn-cols", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048],
                        help="Column counts to benchmark for column attention")
    parser.add_argument("--row-attn-rows", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help="Row counts to benchmark for row attention")
    parser.add_argument("--row-attn-cols", type=int, default=64,
                        help="Fixed column count for row attention benchmark")
    parser.add_argument("--headdim", type=int, default=64,
                        help="Head dimension")
    parser.add_argument("--nheads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--causal", action="store_true", default=False,
                        help="Use causal attention")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],
                        help="Data type")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--rep", type=int, default=10,
                        help="Number of repetitions for timing")
    parser.add_argument("--backends", type=str, nargs="+",
                        default=["sdpa_efficient", "sdpa_cudnn"],
                        help="Backends to benchmark")
    parser.add_argument("--col-only", action="store_true", default=False,
                        help="Run column attention benchmarks only")
    parser.add_argument("--row-only", action="store_true", default=False,
                        help="Run row attention benchmarks only")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Use debug settings")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.col_only and args.row_only:
        raise ValueError("--col-only and --row-only are mutually exclusive")

    if args.debug:
        tqdm.write("Running in DEBUG mode with reduced settings for quick iteration.")
        args.col_attn_rows = 512
        args.col_attn_cols = [32, 128]
        args.row_attn_rows = [128, 1024]
        args.row_attn_cols = 64
        args.nheads = 12
        args.headdim = 64
        args.warmup = 1
        args.rep = 2

    env_info = get_env_info()

    tqdm.write(f"GPU: {env_info['gpu']}")
    tqdm.write(f"Backends: {args.backends} | dtype: {dtype} | "
               f"nheads: {args.nheads}, headdim: {args.headdim}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        **env_info,
        "dtype": args.dtype,
        "causal": args.causal,
        "nheads": args.nheads,
        "headdim": args.headdim,
        "col_attn_rows": args.col_attn_rows,
        "col_attn_cols": args.col_attn_cols,
        "row_attn_rows": args.row_attn_rows,
        "row_attn_cols": args.row_attn_cols,
        "warmup": args.warmup,
        "rep": args.rep,
    }

    url_safe_gpu = env_info['gpu'].replace(" ", "_").replace("/", "_").replace("-", "_")
    here = Path(__file__).parent
    results_dir = here / f"results/{url_safe_gpu}/{args.dtype}"

    # Direction suffix keeps --col-only and --row-only results in separate files
    if args.col_only:
        direction_suffix = "_col"
    elif args.row_only:
        direction_suffix = "_row"
    else:
        direction_suffix = ""

    filename = (
        f"CA-{args.col_attn_rows}_{'-'.join([str(i) for i in args.col_attn_cols])}"
        f"__RA-{args.row_attn_cols}_{'-'.join([str(i) for i in args.row_attn_rows])}"
        f"__H-{args.nheads}_HD-{args.headdim}{direction_suffix}.json"
    )

    # When a single backend is used (the normal Makefile case), flush results to
    # disk after every shape so a crash doesn't lose completed measurements.
    if len(args.backends) == 1:
        flush_file = results_dir / args.backends[0] / filename
    else:
        flush_file = None

    results = run_benchmarks(
        col_attn_rows=args.col_attn_rows,
        col_attn_cols=[] if args.row_only else args.col_attn_cols,
        row_attn_rows=[] if args.col_only else args.row_attn_rows,
        row_attn_cols=args.row_attn_cols,
        nheads=args.nheads,
        headdim=args.headdim,
        dtype=dtype,
        causal=args.causal,
        warmup=args.warmup,
        rep=args.rep,
        backends=args.backends,
        flush_file=flush_file,
        metadata=metadata,
    )

    # Final write (also covers the multi-backend case where flush_file is None)
    for backend in args.backends:
        backend_results = [r for r in results if r.get("backend") == backend]
        target_file = results_dir / backend / filename
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w") as f:
            json.dump({"metadata": metadata, "results": backend_results}, f, indent=2)
        tqdm.write(f"Results for {backend} saved to {target_file}")



if __name__ == "__main__":
    main()

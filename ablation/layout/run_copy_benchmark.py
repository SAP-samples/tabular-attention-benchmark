#!/usr/bin/env python
"""Microbenchmark for .contiguous() copy bandwidth at exact row-attention shapes.

Measures the achieved bandwidth for strided-to-contiguous copies at the tensor
shapes used in the row-attention benchmark (B=1, C=64, H=12, D=64, varying R).

This provides per-shape copy bandwidth values for the roofline analysis,
rather than assuming a single fixed bandwidth across all shapes.

Usage:
    uv run python run_copy_benchmark.py [--output results/copy_bandwidth.json]

Must be run on a CUDA-capable GPU.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import triton.runtime as runtime


def do_bench_fixed_reps(fn, warmup=10, rep=100):
    """Benchmark with CUDA events and cache flushing."""
    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    start_event = [di.Event(enable_timing=True) for _ in range(rep)]
    end_event = [di.Event(enable_timing=True) for _ in range(rep)]

    for _ in range(warmup):
        fn()

    for i in range(rep):
        runtime.driver.active.clear_cache(cache)
        start_event[i].record()
        fn()
        end_event[i].record()

    di.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return times_ms


def gpu_warmup():
    """Stabilize GPU clocks."""
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    for _ in range(50):
        x = torch.matmul(x, x)
        x = x / x.norm()
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()


def benchmark_contiguous_row_attention(
    batch: int = 1,
    cols: int = 64,
    nheads: int = 12,
    headdim: int = 64,
    row_counts: list[int] | None = None,
    rep: int = 100,
    warmup: int = 10,
) -> list[dict]:
    """Benchmark .contiguous() for row-attention reshape.

    In row attention, the tensor (B, R, C, H, D) is reshaped to (B*C, H, R, D)
    for SDPA backends, which requires a .contiguous() call on the transposed view.
    For FA backends, the reshape is to (B*C, R, H, D) which is also non-contiguous.

    We benchmark the SDPA case (3 copies of Q, K, V) since that's the dominant
    overhead in the cuDNN backend path.
    """
    if row_counts is None:
        row_counts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results = []

    for rows in row_counts:
        # Create the source tensor in row-first layout
        # Shape: (batch, rows, cols, nheads, headdim)
        src = torch.randn(
            batch, rows, cols, nheads, headdim,
            device='cuda', dtype=torch.bfloat16
        )

        # For SDPA row attention: reshape to (batch*cols, nheads, rows, headdim)
        # This requires: permute(0,2,3,1,4).reshape(batch*cols, nheads, rows, headdim)
        # The permute makes it non-contiguous, and .contiguous() copies it.
        view = src.permute(0, 2, 3, 1, 4).reshape(batch * cols, nheads, rows, headdim)

        assert not view.is_contiguous(), f"View should be non-contiguous for rows={rows}"

        # Benchmark a single .contiguous() call
        tensor_bytes = view.numel() * view.element_size()

        times_ms = do_bench_fixed_reps(lambda: view.contiguous(), warmup=warmup, rep=rep)

        mean_ms = sum(times_ms) / len(times_ms)
        std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

        # Bandwidth: read + write = 2 * tensor_bytes
        bandwidth_gb_s = (2 * tensor_bytes) / (mean_ms / 1000) / 1e9

        results.append({
            "rows": rows,
            "cols": cols,
            "batch": batch,
            "nheads": nheads,
            "headdim": headdim,
            "batch_eff": batch * cols,
            "tensor_bytes": tensor_bytes,
            "tensor_mb": tensor_bytes / 1024 / 1024,
            "mean_time_ms": mean_ms,
            "std_time_ms": std_ms,
            "bandwidth_gb_s": bandwidth_gb_s,
            "rep": rep,
        })

        print(
            f"  R={rows:6d}: {tensor_bytes/1024/1024:8.1f} MB, "
            f"time={mean_ms:.4f} ± {std_ms:.4f} ms, "
            f"BW={bandwidth_gb_s:.1f} GB/s"
        )

        del src, view
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark .contiguous() at row-attention shapes"
    )
    parser.add_argument("--output", type=str, default="results/copy_bandwidth.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark must be run on a GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: B={args.batch}, C={args.cols}, H={args.nheads}, D={args.headdim}")
    print(f"Measuring .contiguous() for row-attention reshape (SDPA path)")
    print()

    gpu_warmup()

    print("Benchmarking single .contiguous() call per shape:")
    results = benchmark_contiguous_row_attention(
        batch=args.batch,
        cols=args.cols,
        nheads=args.nheads,
        headdim=args.headdim,
        rep=args.rep,
        warmup=args.warmup,
    )

    # Also benchmark 3x copy (as done in practice for Q, K, V)
    print("\nBenchmarking 3x .contiguous() (Q, K, V) per shape:")
    results_3x = []
    for rows in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        src = torch.randn(
            args.batch, rows, args.cols, args.nheads, args.headdim,
            device='cuda', dtype=torch.bfloat16
        )
        view = src.permute(0, 2, 3, 1, 4).reshape(
            args.batch * args.cols, args.nheads, rows, args.headdim
        )

        def copy_3x():
            _ = view.contiguous()
            _ = view.contiguous()
            _ = view.contiguous()

        tensor_bytes = view.numel() * view.element_size()
        times_ms = do_bench_fixed_reps(copy_3x, warmup=args.warmup, rep=args.rep)

        mean_ms = sum(times_ms) / len(times_ms)
        std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
        bandwidth_gb_s = (3 * 2 * tensor_bytes) / (mean_ms / 1000) / 1e9

        results_3x.append({
            "rows": rows,
            "total_copy_bytes": 3 * tensor_bytes,
            "total_copy_mb": 3 * tensor_bytes / 1024 / 1024,
            "mean_time_ms": mean_ms,
            "std_time_ms": std_ms,
            "bandwidth_gb_s": bandwidth_gb_s,
        })

        print(
            f"  R={rows:6d}: 3x{tensor_bytes/1024/1024:.1f}MB = "
            f"{3*tensor_bytes/1024/1024:.1f} MB total, "
            f"time={mean_ms:.4f} ± {std_ms:.4f} ms, "
            f"BW={bandwidth_gb_s:.1f} GB/s"
        )

        del src, view
        torch.cuda.empty_cache()

    output_data = {
        "metadata": {
            "gpu": gpu_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dtype": "bfloat16",
            "batch": args.batch,
            "cols": args.cols,
            "nheads": args.nheads,
            "headdim": args.headdim,
            "rep": args.rep,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "single_copy": results,
        "triple_copy": results_3x,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

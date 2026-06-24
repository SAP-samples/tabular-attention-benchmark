"""Microbenchmark: cost of .contiguous() for different transpose patterns.

Compares the two native layouts:
  Layout A: (batch, rows, cols, nheads, hdim) — transpose(1,2) for row attention
  Layout B: (batch, cols, rows, nheads, hdim) — transpose(1,2) for col attention

After transpose, both need .contiguous(). Are the costs identical?
"""

import torch
import torch.utils.benchmark as benchmark


def bench_contiguous(shape_before_transpose, label, num_threads=1):
    """Time .contiguous() on a tensor after transpose(1, 2)."""
    t = torch.randn(shape_before_transpose, device="cuda", dtype=torch.bfloat16)
    t_transposed = t.transpose(1, 2)
    assert not t_transposed.is_contiguous()

    timer = benchmark.Timer(
        stmt="x.contiguous()",
        globals={"x": t_transposed},
        label=".contiguous()",
        sub_label=label,
        description=f"{list(shape_before_transpose)}",
        num_threads=num_threads,
    )
    return timer.blocked_autorange(min_run_time=1.0)


def main():
    torch.cuda.empty_cache()

    configs = [
        # (rows, cols) pairs — rows >> cols as in tabular data
        (1024, 20),
        (2048, 50),
        (4096, 50),
        (8192, 100),
        # Extremes: max asymmetry in both directions
        (8192, 16),   # large row count, tiny col count
        (128, 1024),  # small row count, large col count
    ]

    batch = 2
    nheads = 8
    hdim = 64

    results = []

    for rows, cols in configs:
        numel = batch * rows * cols * nheads * hdim
        mb = numel * 2 / 1e6  # fp16

        # Layout A: (batch, rows, cols, nheads, hdim) — transpose for row attn
        shape_a = (batch, rows, cols, nheads, hdim)
        res_a = bench_contiguous(shape_a, f"rows={rows} cols={cols} | LayoutA (b,R,C,h,d) t(1,2)")
        results.append(res_a)

        # Layout B: (batch, cols, rows, nheads, hdim) — transpose for col attn
        shape_b = (batch, cols, rows, nheads, hdim)
        res_b = bench_contiguous(shape_b, f"rows={rows} cols={cols} | LayoutB (b,C,R,h,d) t(1,2)")
        results.append(res_b)

    compare = benchmark.Compare(results)
    compare.print()

    # Also print raw numbers for easy comparison
    print("\n\nRaw comparison:")
    print(f"{'Config':<55} {'Time (us)':>10} {'GB/s':>10}")
    print("-" * 80)
    for i in range(0, len(results), 2):
        res_a = results[i]
        res_b = results[i + 1]
        cfg = configs[i // 2]
        rows, cols = cfg
        numel = batch * rows * cols * nheads * hdim
        nbytes = numel * 2  # fp16, read + write = 2x

        bw_a = 2 * nbytes / res_a.mean / 1e9  # read + write
        bw_b = 2 * nbytes / res_b.mean / 1e9

        print(f"rows={rows:>5} cols={cols:>3} | LayoutA (b,R,C,h,d)  {res_a.mean*1e6:>10.1f} {bw_a:>10.1f}")
        print(f"rows={rows:>5} cols={cols:>3} | LayoutB (b,C,R,h,d)  {res_b.mean*1e6:>10.1f} {bw_b:>10.1f}")
        ratio = res_a.mean / res_b.mean
        print(f"  {'ratio A/B:':<51} {ratio:>10.3f}")
        print()


if __name__ == "__main__":
    main()

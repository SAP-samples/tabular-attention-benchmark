# Layout Investigation: `(b, R, C, h, d)` vs `(b, C, R, h, d)`

## Question

Does the native tensor layout affect tabular attention benchmark results?

Currently the codebase uses `(batch, rows, cols, nheads, headdim)`. Column attention
is contiguous (just flatten batch×rows), while row attention requires `.transpose(1,2)`
(and `.contiguous()` for backends that don't support strided tensors). Would results
change if the native layout were `(batch, cols, rows, nheads, headdim)` instead?

## What we know

### Total copy cost is identical

Both layouts have the same number of elements. The `.contiguous()` call copies the
full tensor regardless of which two dims were transposed.

### Inner contiguous block is the same

After transpose(1,2), in both layouts the trailing dims `(nheads, headdim)` remain
contiguous. The inner block read by the copy kernel is `nheads × headdim` elements
(e.g. 8×64 = 512 elements = 1 KB in fp16) in both cases.

### Stride patterns differ significantly

For a concrete example: `rows=8192, cols=100, batch=2, nheads=8, hdim=64`:

| | Layout A `(b,R,C,h,d)` row-attn transpose | Layout B `(b,C,R,h,d)` col-attn transpose |
|---|---|---|
| Non-contiguous dim size | 8192 | 100 |
| Stride between 1 KB blocks | 100 KB | 8 MB |
| Pattern | Many small-stride jumps | Few large-stride jumps |

Both read/write the same total bytes, but Layout A's `.contiguous()` has denser spatial
locality (consecutive reads ~100 KB apart), while Layout B's reads jump ~8 MB apart.

### Hypothesis

Layout A's denser stride pattern may be friendlier to TLB and L2 cache on the GPU,
potentially making its `.contiguous()` slightly faster. However, GPU copy kernels are
massively parallel, so the sequential locality argument may not hold — threads across
warps access scattered addresses regardless.

## Backend implications

| Backend | Needs `.contiguous()` | Strided tensor support |
|---|---|---|
| SDPA (efficient/cudnn) | Yes | No |
| FA2 | Yes | No |
| FA3 | No | Yes (uses `.reshape()` on strided tensor) |
| FA4 | No | Yes |

One of FA3/FA4's advantages is avoiding the `.contiguous()` copy entirely. This advantage
applies to row attention in Layout A, and would shift to column attention in Layout B.
The magnitude of the saved copy (same total bytes) should be similar either way — unless
the stride pattern affects copy throughput.

## Scripts

- `microbench_contiguous.py` — benchmarks `.contiguous()` wall time for both layouts
  across multiple (rows, cols) configs. Uses `torch.utils.benchmark` with min 1s runtime.
- `analyze_strides.py` — prints stride analysis showing the access patterns (no GPU needed).
- `run_copy_benchmark.py` — measures `.contiguous()` bandwidth at exact row-attention shapes
  used in the main benchmark (varying sequence length N). Outputs per-shape bandwidth to
  `results/copy_bandwidth.json`, used by the roofline analysis plots (`--roofline`).

## Results (H100 NVL)

Microbenchmark ran on H100 NVL with `torch.utils.benchmark` (min 1s runtime per config, bfloat16).

| Config | Layout A `(b,R,C,h,d)` | Layout B `(b,C,R,h,d)` | Ratio A/B | BW (GB/s) |
|---|---|---|---|---|
| rows=1024, cols=20 | 64.9 us | 64.8 us | 1.002 | ~1293 |
| rows=2048, cols=50 | 313.4 us | 313.2 us | 1.001 | ~1339 |
| rows=4096, cols=50 | 626.7 us | 626.8 us | 1.000 | ~1338 |
| rows=8192, cols=100 | 2500.7 us | 2504.1 us | 0.999 | ~1341 |
| **rows=8192, cols=16** | 401.4 us | 401.3 us | 1.000 | ~1338 |
| **rows=128, cols=1024** | 402.3 us | 402.5 us | 1.000 | ~1334 |

The last two configs test extreme asymmetries: 512:1 row-to-col ratio and 1:8 row-to-col
ratio. Even at these extremes, the cost is identical.

### Conclusions

**`.contiguous()` cost is identical across both layouts** (ratio within 0.2% across all configs).

Despite the very different stride patterns (Layout A: many small-stride jumps; Layout B:
few large-stride jumps), the H100's copy kernel achieves the same ~1.3 TB/s bandwidth in
both cases. The GPU's massively parallel memory system effectively hides the stride
pattern differences.

This confirms:
- **Benchmarks are layout-symmetric**: switching from `(b,R,C,h,d)` to `(b,C,R,h,d)` would
  not change any benchmark results. The `.contiguous()` overhead would simply shift from
  row attention to column attention with identical cost.
- **FA3/FA4's strided-tensor advantage** (avoiding `.contiguous()`) saves the same amount
  of time regardless of which layout is chosen.
- **The current layout choice `(b,R,C,h,d)` is not biasing results** — it is a convention
  choice, not a performance choice.

## TODO

- [x] Run `microbench_contiguous.py` on H100 (`uv run python microbench_contiguous.py`)
- [x] If `.contiguous()` cost is identical: confirms benchmarks are layout-symmetric and
      FA3/FA4's strided-tensor advantage is the same magnitude regardless of layout choice

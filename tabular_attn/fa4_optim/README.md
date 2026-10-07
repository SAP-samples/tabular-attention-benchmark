# FlashAttention-4 Optim (CuTeDSL, SM100-only)

Agent-optimised CuTeDSL implementation of FlashAttention for Blackwell (SM100) GPUs,
targeting tabular AI workloads.

This is a modified copy of the FA4 CuTeDSL kernels from
[flash-attention](https://github.com/Dao-AILab/flash-attention). 
Note that only the forward pass for column attention has been optimised
in this code, leaving improvements on row attention or backward to future work.

Changes from upstream:

- Only support SM100 (Blackwell) — SM80, SM90, SM120 code removed
- Use local imports (`tabular_attn.fa4_optim.*`) — no `flash_attn` dependency
- Remove varlen support (fixed-length sequences only)
- **col_attn**: register allocation, kv pipeline staging, softmax path, and persistence
  heuristics optimised by agentic search for column attention
- **row_attn**: currently unmodified from upstream FA4 (baseline for future optimisation)

## Directory structure

```
fa4_optim/
├── __init__.py            # Re-exports flash_attn_func from col_attn + row_attn
├── README.md
├── shared/                # Shared code (column and row attention)
│   ├── utils.py, softmax.py, mask.py, fast_math.py, copy_utils.py,
│   │   barrier.py, cute_dsl_utils.py, cute_dsl_ptxas.py, mma_sm100_desc.py,
│   │   named_barrier.py, seqlen_info.py, pipeline.py, testing.py,
│   │   cache_utils.py, tile_scheduler.py, block_info.py, block_sparsity.py,
│   │   block_sparse_utils.py, compute_block_sparsity.py, pack_gqa.py,
│   │   blackwell_helpers.py, paged_kv.py, fa_logging.py
├── col_attn/              # Column attention kernels (optimised)
│   ├── interface.py, flash_fwd_sm100.py, flash_bwd_sm100.py,
│   │   flash_bwd_postprocess.py, flash_fwd_combine.py, flash_bwd_preprocess.py
├── row_attn/              # Row attention kernels (unmodified from upstream)
│   ├── interface.py, flash_fwd_sm100.py, flash_bwd_sm100.py,
│   │   flash_bwd_postprocess.py, flash_fwd_combine.py, flash_bwd_preprocess.py
```

**Shared** files contain generic infrastructure (copy utilities, barriers, mask logic,
softmax, block sparsity, tile schedulers) used by both directions.

**Per-direction** kernel files contain tile size heuristics, MMA pipeline config,
split-KV thresholds, and launch parameters. Currently only col_attn has been
optimised for tabular workloads; row_attn remains at upstream defaults and serves
as a baseline.

## Installation

```sh
uv sync --group fa4
```

## Usage

```python
from tabular_attn.fa4_optim import flash_attn_func_col, flash_attn_func_row

out_col = flash_attn_func_col(q, k, v)
out_row = flash_attn_func_row(q, k, v)
```

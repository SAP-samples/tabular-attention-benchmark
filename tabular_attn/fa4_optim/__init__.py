"""FlashAttention-4 Optim (CuTeDSL, SM100-only) — row and column attention kernels.

This also exposes tabular row/column attention wrappers
(:func:`col_attn_fa4_optim` / :func:`row_attn_fa4_optim`) mirroring
:mod:`tabular_attn.fa4`, but dispatching to the local, modified CuteDSL
kernels which provide separate column- and row-attention entry points.
"""

import torch

try:
    from tabular_attn.fa4_optim.col_attn import flash_attn_func as flash_attn_func_col
    from tabular_attn.fa4_optim.row_attn import flash_attn_func as flash_attn_func_row
    FA4_OPTIM_AVAILABLE = True
except Exception:  # noqa: BLE001
    # Broad except on purpose: importing the CuteDSL kernels pulls in a specific
    # nvidia-cutlass-dsl version and touches GPU/compiler internals. A mismatched
    # cutlass can raise AttributeError (e.g. a renamed cute type), not just
    # ImportError. tabular_attn must still import for the other backends, so any
    # failure here just marks fa4_optim unavailable.
    flash_attn_func_col = None
    flash_attn_func_row = None
    FA4_OPTIM_AVAILABLE = False

# Backward compat alias
flash_attn_func = flash_attn_func_col


def col_attn_fa4_optim(q, k, v, *, causal=False):
    """Column attention using the optimized local FlashAttention-4 kernel."""

    batch, rows, cols, nheads, headdim = q.shape

    # Reshape: (batch, rows, cols, nheads, headdim) -> (batch*rows, cols, nheads, headdim)
    q_flat = q.view(batch * rows, cols, nheads, headdim)
    k_flat = k.view(batch * rows, cols, nheads, headdim)
    v_flat = v.view(batch * rows, cols, nheads, headdim)

    out = flash_attn_func_col(q_flat, k_flat, v_flat, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    return out.view(batch, rows, cols, nheads, headdim)


def row_attn_fa4_optim(q, k, v, *, causal=False):
    """Row attention using the optimized local FlashAttention-4 kernel.

    FA4 can operate on strided tensors, so no .contiguous() needed.
    """
    batch, rows, cols, nheads, headdim = q.shape

    # Transpose rows <-> cols without .contiguous()
    q_t = q.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)
    k_t = k.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)
    v_t = v.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)

    out = flash_attn_func_row(q_t, k_t, v_t, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    out = out.view(batch, cols, rows, nheads, headdim).transpose(1, 2).contiguous()
    return out


__all__ = [
    "flash_attn_func", "flash_attn_func_col", "flash_attn_func_row",
    "col_attn_fa4_optim", "row_attn_fa4_optim", "FA4_OPTIM_AVAILABLE",
]

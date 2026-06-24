"""FlashAttention-4 tabular row and column attention backends."""

import torch

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_v4
    assert hasattr(torch.ops, "flash_attn_4"), "torch.ops.flash_attn_4 not found"
    FA4_AVAILABLE = True
except (ImportError, AssertionError):
    FA4_AVAILABLE = False


def col_attn_fa4(q, k, v, *, causal=False):
    """Column attention using FlashAttention-4."""

    batch, rows, cols, nheads, headdim = q.shape

    # Reshape: (batch, rows, cols, nheads, headdim) -> (batch*rows, cols, nheads, headdim)
    q_flat = q.view(batch * rows, cols, nheads, headdim)
    k_flat = k.view(batch * rows, cols, nheads, headdim)
    v_flat = v.view(batch * rows, cols, nheads, headdim)

    out = flash_attn_func_v4(q_flat, k_flat, v_flat, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    return out.view(batch, rows, cols, nheads, headdim)


def row_attn_fa4(q, k, v, *, causal=False):
    """Row attention using FlashAttention-4.

    FA4 can operate on strided tensors, so no .contiguous() needed.
    """
    batch, rows, cols, nheads, headdim = q.shape

    # Transpose rows <-> cols without .contiguous()
    q_t = q.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)
    k_t = k.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)
    v_t = v.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)

    out = flash_attn_func_v4(q_t, k_t, v_t, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    out = out.view(batch, cols, rows, nheads, headdim).transpose(1, 2).contiguous()
    return out

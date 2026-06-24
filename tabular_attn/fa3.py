"""FlashAttention-3 tabular row and column attention backends."""

import torch

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    assert hasattr(torch.ops, "flash_attn_3"), "torch.ops.flash_attn_3 not found"
    # FA3 kernels only target sm_80 and sm_90; they crash on Blackwell (sm_100+)
    _cc = torch.cuda.get_device_capability()
    assert _cc[0] < 10, f"FA3 not supported on sm_{_cc[0] * 10 + _cc[1]} (Blackwell+)"
    FA3_AVAILABLE = True
except (ImportError, AssertionError):
    FA3_AVAILABLE = False


def col_attn_fa3(q, k, v, *, causal=False):
    """Column attention using FlashAttention-3."""

    batch, rows, cols, nheads, headdim = q.shape

    # Reshape: (batch, rows, cols, nheads, headdim) -> (batch*rows, cols, nheads, headdim)
    q_flat = q.view(batch * rows, cols, nheads, headdim)
    k_flat = k.view(batch * rows, cols, nheads, headdim)
    v_flat = v.view(batch * rows, cols, nheads, headdim)

    out = flash_attn_func_v3(q_flat, k_flat, v_flat, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    return out.view(batch, rows, cols, nheads, headdim)


def row_attn_fa3(q, k, v, *, causal=False):
    """Row attention using FlashAttention-3.

    FA3 can operate on strided tensors, so no .contiguous() needed.
    """
    batch, rows, cols, nheads, headdim = q.shape

    # Transpose rows <-> cols without .contiguous()
    # (batch, rows, cols, nheads, headdim) -> (batch, cols, rows, nheads, headdim)
    # Then view as (batch*cols, rows, nheads, headdim) - this works because batch*cols are contiguous
    q_t = q.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)
    k_t = k.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)
    v_t = v.transpose(1, 2).reshape(batch * cols, rows, nheads, headdim)

    out = flash_attn_func_v3(q_t, k_t, v_t, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    # Reshape back, only need one contiguous call here
    out = out.view(batch, cols, rows, nheads, headdim).transpose(1, 2).contiguous()
    return out

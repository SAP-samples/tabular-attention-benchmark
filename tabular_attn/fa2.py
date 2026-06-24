"""FlashAttention-2 tabular row and column attention backends."""

import torch

try:
    from flash_attn import flash_attn_func as flash_attn_func_v2
    from flash_attn import flash_attn_kvpacked_func as flash_attn_kvpacked_func_v2
    assert hasattr(torch.ops, "flash_attn_2"), "torch.ops.flash_attn_2 not found"
    FA2_AVAILABLE = True
except (ImportError, AssertionError):
    FA2_AVAILABLE = False


def _fa2_safe_chunk(seqlen: int, nheads: int) -> int:
    """Compute safe batch chunk size for FA2 backward pass.

    FA2's backward kernel uses 32-bit integer indexing over
    batch_eff * seqlen * nheads elements. To stay within int32 range
    we keep batch_eff * seqlen * nheads < 2^27 (conservative headroom).
    """
    return max(1, (1 << 27) // (seqlen * nheads))


def col_attn_fa2(q, k, v, *, causal=False):
    """Column attention using FlashAttention-2.

    FA2 requires contiguous tensors, similar to SDPA.
    """
    batch, rows, cols, nheads, headdim = q.shape
    batch_eff = batch * rows

    chunk = _fa2_safe_chunk(cols, nheads)

    q_flat = q.view(batch_eff, cols, nheads, headdim)
    k_flat = k.view(batch_eff, cols, nheads, headdim)
    v_flat = v.view(batch_eff, cols, nheads, headdim)

    if batch_eff <= chunk:
        out = flash_attn_func_v2(q_flat, k_flat, v_flat, causal=causal)
    else:
        out = torch.cat([
            flash_attn_func_v2(q_flat[i:i+chunk], k_flat[i:i+chunk], v_flat[i:i+chunk], causal=causal)
            for i in range(0, batch_eff, chunk)
        ], dim=0)

    return out.view(batch, rows, cols, nheads, headdim)


def col_attn_fa2_kv_packed(q, kv, *, causal=False):
    """Column attention using FlashAttention-2 in kv packed variant.

    FA2 requires contiguous tensors, similar to SDPA.
    """
    batch, rows, cols, nheads, headdim = q.shape

    q_flat = q.view(batch * rows, cols, nheads, headdim)
    kv_flat = kv.view(batch * rows, cols, 2, nheads, headdim)

    out = flash_attn_kvpacked_func_v2(q_flat, kv_flat, causal=causal)
    return out.view(batch, rows, cols, nheads, headdim)


def row_attn_fa2(q, k, v, *, causal=False):
    """Row attention using FlashAttention-2.

    FA2 requires contiguous tensors, so we must call .contiguous() after transpose.
    """
    batch, rows, cols, nheads, headdim = q.shape
    batch_eff = batch * cols

    q_t = q.transpose(1, 2).contiguous().view(batch_eff, rows, nheads, headdim)
    k_t = k.transpose(1, 2).contiguous().view(batch_eff, rows, nheads, headdim)
    v_t = v.transpose(1, 2).contiguous().view(batch_eff, rows, nheads, headdim)

    chunk = _fa2_safe_chunk(rows, nheads)

    if batch_eff <= chunk:
        out = flash_attn_func_v2(q_t, k_t, v_t, causal=causal)
    else:
        out = torch.cat([
            flash_attn_func_v2(q_t[i:i+chunk], k_t[i:i+chunk], v_t[i:i+chunk], causal=causal)
            for i in range(0, batch_eff, chunk)
        ], dim=0)

    out = out.view(batch, cols, rows, nheads, headdim).transpose(1, 2).contiguous()
    return out


def row_attn_fa2_kv_packed(q, kv, *, causal=False):
    """Row attention using FlashAttention-2 in kv packed variant.

    FA2 requires contiguous tensors, so we must call .contiguous() after transpose.
    """
    batch, rows, cols, nheads, headdim = q.shape

    q_t = q.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)
    kv_t = kv.transpose(1, 2).contiguous().view(batch * cols, rows, 2, nheads, headdim)

    out = flash_attn_kvpacked_func_v2(q_t, kv_t, causal=causal)
    out = out.view(batch, cols, rows, nheads, headdim).transpose(1, 2).contiguous()
    return out

"""SDPA-based tabular row and column attention backends (math, efficient, cudnn)."""

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# =============================================================================
# Column Attention: attend across columns
# Input: (batch, rows, cols, nheads, headdim)
# =============================================================================

def _col_attn_sdpa(q, k, v, backend: str, causal=False):
    """Column attention using PyTorch SDPA."""

    if backend == "efficient":
        sdpa_backend = SDPBackend.EFFICIENT_ATTENTION
    elif backend == "cudnn":
        sdpa_backend = SDPBackend.CUDNN_ATTENTION
    elif backend == "math":
        sdpa_backend = SDPBackend.MATH
    else:
        raise ValueError(f"Unsupported SDPA backend: {backend}")

    batch, rows, cols, nheads, headdim = q.shape

    # Reshape: (batch, rows, cols, nheads, headdim) -> (batch*rows, cols, nheads, headdim)
    q_flat = q.view(batch * rows, cols, nheads, headdim)
    k_flat = k.view(batch * rows, cols, nheads, headdim)
    v_flat = v.view(batch * rows, cols, nheads, headdim)

    # SDPA expects (batch, nheads, seqlen, headdim)
    q_sdpa = q_flat.transpose(1, 2)  # (batch*rows, nheads, cols, headdim)
    k_sdpa = k_flat.transpose(1, 2)
    v_sdpa = v_flat.transpose(1, 2)

    with sdpa_kernel(backends=[sdpa_backend]):
        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=causal)
    return out.transpose(1, 2).view(batch, rows, cols, nheads, headdim).contiguous()


def col_attn_sdpa_math(q, k, v, *, causal=False):
    """Column attention using PyTorch SDPA."""
    return _col_attn_sdpa(q, k, v, backend="math", causal=causal)


def col_attn_sdpa_efficient(q, k, v, *, causal=False):
    """Column attention using PyTorch SDPA."""
    return _col_attn_sdpa(q, k, v, backend="efficient", causal=causal)


def col_attn_sdpa_cudnn(q, k, v, *, causal=False):
    """Column attention using PyTorch SDPA."""
    return _col_attn_sdpa(q, k, v, backend="cudnn", causal=causal)


# =============================================================================
# Row Attention: attend across rows
# Input: (batch, rows, cols, nheads, headdim)
# Need: (batch*cols, rows, nheads, headdim) - requires transpose
# =============================================================================

def _row_attn_sdpa(q, k, v, backend: str, causal=False):
    """Row attention using PyTorch SDPA.

    SDPA requires contiguous tensors, so we must call .contiguous() after transpose.
    """
    if backend == "efficient":
        sdpa_backend = SDPBackend.EFFICIENT_ATTENTION
    elif backend == "cudnn":
        sdpa_backend = SDPBackend.CUDNN_ATTENTION
    elif backend == "math":
        sdpa_backend = SDPBackend.MATH
    else:
        raise ValueError(f"Unsupported SDPA backend: {backend}")

    batch, rows, cols, nheads, headdim = q.shape

    # Transpose rows <-> cols: (batch, rows, cols, nheads, headdim) -> (batch, cols, rows, nheads, headdim)
    # Then reshape to (batch*cols, rows, nheads, headdim)
    q_t = q.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)
    k_t = k.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)
    v_t = v.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)

    # SDPA expects (batch, nheads, seqlen, headdim)
    q_sdpa = q_t.transpose(1, 2)
    k_sdpa = k_t.transpose(1, 2)
    v_sdpa = v_t.transpose(1, 2)

    with sdpa_kernel(backends=[sdpa_backend]):
        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=causal)
    # Reshape back: (batch*cols, nheads, rows, headdim) -> (batch, rows, cols, nheads, headdim)
    out = out.transpose(1, 2).view(batch, cols, rows, nheads, headdim).transpose(1, 2)
    return out.contiguous()


def row_attn_sdpa_math(q, k, v, *, causal=False):
    """Row attention using PyTorch SDPA."""
    return _row_attn_sdpa(q, k, v, backend="math", causal=causal)


def row_attn_sdpa_efficient(q, k, v, *, causal=False):
    """Row attention using PyTorch SDPA."""
    return _row_attn_sdpa(q, k, v, backend="efficient", causal=causal)


def row_attn_sdpa_cudnn(q, k, v, *, causal=False):
    """Row attention using PyTorch SDPA."""
    return _row_attn_sdpa(q, k, v, backend="cudnn", causal=causal)

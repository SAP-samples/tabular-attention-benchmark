"""SageAttention tabular row and column attention backends.

Pins the Hopper-specialized `sageattn_qk_int8_pv_fp8_cuda_sm90` kernel explicitly
(rather than the top-level `sageattn()` dispatch) so the benchmark measures exactly
that kernel deterministically. This kernel was broken in the SageAttention v2.2.0
*release* (returned garbage on Hopper — see
https://github.com/thu-ml/SageAttention/issues/288); we build from the fixed commit
(d1a57a5), where it is numerically correct (~0.037 rel error vs exact, within int8
expectations) and is the fast Hopper path. See build/build_sage.sh.

The sm90 kernel is Hopper-only: it is unavailable/incorrect on other archs, so the
wrapper requires sm_90 exactly and fails loudly elsewhere (rather than silently
producing wrong numbers).
"""

import torch

try:
    from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90 as _sageattn_sm90
    # The sm90 kernel is Hopper-specific (uses wgmma); require sm_90 exactly.
    _cc = torch.cuda.get_device_capability()
    assert _cc == (9, 0), (
        f"sage backend pins the sm90 kernel; unsupported on sm_{_cc[0] * 10 + _cc[1]}"
    )
    SAGE_AVAILABLE = True
except (ImportError, AssertionError):
    SAGE_AVAILABLE = False


def sageattn(q, k, v, *, tensor_layout, is_causal=False):
    """Hopper fp8 SageAttention kernel (int8 QK, fp8 PV), pinned explicitly.

    Drop-in for the shape used by the col/row wrappers below. Uses the kernel's
    default accumulation/quantization settings, matching the top-level dispatch.
    """
    out = _sageattn_sm90(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal)
    return out[0] if isinstance(out, tuple) else out


def col_attn_sage(q, k, v, *, causal=False):
    """Column attention using SageAttention."""

    batch, rows, cols, nheads, headdim = q.shape

    # Reshape: (batch, rows, cols, nheads, headdim) -> (batch*rows, cols, nheads, headdim)
    q_flat = q.view(batch * rows, cols, nheads, headdim)
    k_flat = k.view(batch * rows, cols, nheads, headdim)
    v_flat = v.view(batch * rows, cols, nheads, headdim)

    # sageattn with HND layout expects (batch, nheads, seqlen, headdim)
    q_hnd = q_flat.transpose(1, 2)
    k_hnd = k_flat.transpose(1, 2)
    v_hnd = v_flat.transpose(1, 2)

    out = sageattn(q_hnd, k_hnd, v_hnd, tensor_layout="HND", is_causal=causal)
    return out.transpose(1, 2).view(batch, rows, cols, nheads, headdim).contiguous()


def row_attn_sage(q, k, v, *, causal=False):
    """Row attention using SageAttention.

    SageAttention requires contiguous tensors, so we must call .contiguous() after transpose.
    """
    batch, rows, cols, nheads, headdim = q.shape

    # Transpose rows <-> cols: (batch, rows, cols, nheads, headdim) -> (batch, cols, rows, nheads, headdim)
    # Then reshape to (batch*cols, rows, nheads, headdim)
    q_t = q.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)
    k_t = k.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)
    v_t = v.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim)

    # sageattn with HND layout expects (batch, nheads, seqlen, headdim)
    q_hnd = q_t.transpose(1, 2)
    k_hnd = k_t.transpose(1, 2)
    v_hnd = v_t.transpose(1, 2)

    out = sageattn(q_hnd, k_hnd, v_hnd, tensor_layout="HND", is_causal=causal)
    # Reshape back: (batch*cols, nheads, rows, headdim) -> (batch, rows, cols, nheads, headdim)
    out = out.transpose(1, 2).view(batch, cols, rows, nheads, headdim).transpose(1, 2)
    return out.contiguous()

"""vLLM Triton prefill attention tabular row and column attention backends."""

import torch

try:
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _vllm_attn(q_flat, k_flat, v_flat, causal):
    """Call context_attention_fwd on (batch_eff, seqlen, nheads, headdim) inputs.

    context_attention_fwd expects packed (total_tokens, nheads, headdim) tensors
    with per-batch metadata. We have uniform sequence lengths so b_start_loc and
    b_seq_len are simple ranges/constants.
    """
    batch_eff, seqlen, nheads, headdim = q_flat.shape

    q = q_flat.reshape(batch_eff * seqlen, nheads, headdim)
    k = k_flat.reshape(batch_eff * seqlen, nheads, headdim)
    v = v_flat.reshape(batch_eff * seqlen, nheads, headdim)
    out = torch.empty_like(q)

    b_seq_len = torch.full((batch_eff,), seqlen, dtype=torch.int32, device=q.device)
    b_start_loc = torch.arange(0, batch_eff * seqlen, seqlen, dtype=torch.int32, device=q.device)

    context_attention_fwd(
        q, k, v, out,
        b_start_loc, b_seq_len,
        max_input_len=seqlen,
        is_causal=causal,
    )

    return out.view(batch_eff, seqlen, nheads, headdim)


def col_attn_vllm(q, k, v, *, causal=False):
    """Column attention using vLLM Triton prefill kernel."""
    batch, rows, cols, nheads, headdim = q.shape

    out = _vllm_attn(
        q.view(batch * rows, cols, nheads, headdim),
        k.view(batch * rows, cols, nheads, headdim),
        v.view(batch * rows, cols, nheads, headdim),
        causal=causal,
    )
    return out.view(batch, rows, cols, nheads, headdim)


def row_attn_vllm(q, k, v, *, causal=False):
    """Row attention using vLLM Triton prefill kernel."""
    batch, rows, cols, nheads, headdim = q.shape

    out = _vllm_attn(
        q.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim),
        k.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim),
        v.transpose(1, 2).contiguous().view(batch * cols, rows, nheads, headdim),
        causal=causal,
    )
    return out.view(batch, cols, rows, nheads, headdim).transpose(1, 2).contiguous()

import pytest
from collections import namedtuple

import torch

from tabular_attn import (
    col_attn_sdpa_math, col_attn_sdpa_efficient, col_attn_sdpa_cudnn, col_attn_fa2, col_attn_fa2_kv_packed, col_attn_fa3, col_attn_fa4,
    row_attn_sdpa_math, row_attn_sdpa_efficient, row_attn_sdpa_cudnn, row_attn_fa2, row_attn_fa2_kv_packed, row_attn_fa3, row_attn_fa4,
    FA2_AVAILABLE, FA3_AVAILABLE, FA4_AVAILABLE
)

Backend = namedtuple("Backend", ["name", "kv_packed", "supports_bwd", "supports_numerical_check", "row_fn", "col_fn"])

# Fixture over backends, returning both row and column attention functions.
# Each param is tagged with a marker matching its pyproject.toml dependency group
# so you can run a single backend at a time:
#   uv run --group base  pytest --backend base  --verbose
#   uv run --group cudnn pytest --backend cudnn --verbose
#   uv run --group fa2   pytest --backend fa2   --verbose
#   uv run --group fa3   pytest --backend fa3   --verbose
#   uv run --group fa4   pytest --backend fa4   --verbose
@pytest.fixture(
    params=[
        pytest.param(Backend("SDPA Efficient", kv_packed=False, supports_bwd=True,  supports_numerical_check=True,  row_fn=row_attn_sdpa_efficient, col_fn=col_attn_sdpa_efficient), id="SDPA Efficient", marks=pytest.mark.base),
        pytest.param(Backend("SDPA cuDNN",     kv_packed=False, supports_bwd=True,  supports_numerical_check=True,  row_fn=row_attn_sdpa_cudnn,     col_fn=col_attn_sdpa_cudnn),     id="SDPA cuDNN",     marks=pytest.mark.cudnn),
        pytest.param(Backend("FA2",            kv_packed=False, supports_bwd=True,  supports_numerical_check=True,  row_fn=row_attn_fa2,            col_fn=col_attn_fa2),            id="FA2",            marks=[pytest.mark.fa2,  pytest.mark.skipif(not FA2_AVAILABLE,  reason="FlashAttention-2 not available")]),
        pytest.param(Backend("FA2 KV-Packed",  kv_packed=True,  supports_bwd=True,  supports_numerical_check=True,  row_fn=row_attn_fa2_kv_packed,  col_fn=col_attn_fa2_kv_packed),  id="FA2 KV-Packed",  marks=[pytest.mark.fa2,  pytest.mark.skipif(not FA2_AVAILABLE,  reason="FlashAttention-2 not available")]),
        pytest.param(Backend("FA3",            kv_packed=False, supports_bwd=True,  supports_numerical_check=True,  row_fn=row_attn_fa3,            col_fn=col_attn_fa3),            id="FA3",            marks=[pytest.mark.fa3,  pytest.mark.skipif(not FA3_AVAILABLE,  reason="FlashAttention-3 not available")]),
        pytest.param(Backend("FA4",            kv_packed=False, supports_bwd=True,  supports_numerical_check=True,  row_fn=row_attn_fa4,            col_fn=col_attn_fa4),            id="FA4",            marks=[pytest.mark.fa4,  pytest.mark.skipif(not FA4_AVAILABLE,  reason="FlashAttention-4 not available")]),
    ],
)
def attn_backend(request):
    return request.param

@pytest.fixture(params=[1, 4], ids=lambda x: f"B {x}")
def batch(request):
    return request.param

@pytest.fixture(params=[128], ids=lambda x: f"R {x}")
def rows(request):
    return request.param

@pytest.fixture(params=[32], ids=lambda x: f"C {x}")
def cols(request):
    return request.param

@pytest.fixture(params=[4], ids=lambda x: f"H {x}")
def num_heads(request):
    return request.param

@pytest.fixture(params=[32, 64, 128], ids=lambda x: f"HD {x}")
def headdim(request):
    return request.param

@pytest.fixture(params=[torch.bfloat16], ids=lambda x: f"DT {x}")
def dtype(request):
    return request.param

@pytest.fixture
def attn_tensors(batch, rows, cols, num_heads, headdim, dtype):
    q = torch.randn(batch, rows, cols, num_heads, headdim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    kv = torch.randn(batch, rows, cols, num_heads, headdim * 2, device="cuda", dtype=dtype, requires_grad=True)  # For KV-packed backends
    return q, k, v, kv


def _unpack_kv(kv):
    """Unpack kv tensor into k, v matching the FA2 kv-packed layout.

    kv shape: (batch, rows, cols, nheads, headdim*2)
    FA2 views this as (..., 2, nheads, headdim) where index 0 = K, index 1 = V.
    Reshaping (nheads, headdim*2) -> (2, nheads, headdim) in row-major means the
    first nheads*headdim elements are all of K and the next are all of V, so we
    must reshape to extract them correctly rather than splitting the last dim.
    """
    *prefix, nheads, headdim2 = kv.shape
    headdim = headdim2 // 2
    kv_r = kv.view(*prefix, 2, nheads, headdim)
    return kv_r[..., 0, :, :], kv_r[..., 1, :, :]


def reference_col_attn(q, k, v):
    return col_attn_sdpa_math(q, k, v, causal=False)


def reference_row_attn(q, k, v):
    return row_attn_sdpa_math(q, k, v, causal=False)


def test_col_attention_fwd(attn_backend, attn_tensors):
    q, k, v, kv = attn_tensors
    if attn_backend.kv_packed:
        out = attn_backend.col_fn(q, kv)
        k_ref, v_ref = _unpack_kv(kv)
        ref = reference_col_attn(q, k_ref, v_ref)
    else:
        out = attn_backend.col_fn(q, k, v)
        ref = reference_col_attn(q, k, v)
    assert out.shape == q.shape, f"{attn_backend.name} column attention forward output shape mismatch: expected {q.shape}, got {out.shape}"
    if attn_backend.supports_numerical_check:
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), f"{attn_backend.name} column attention forward output mismatch from reference"


def test_col_attention_bwd(attn_backend, attn_tensors):
    if not attn_backend.supports_bwd:
        pytest.skip(f"{attn_backend.name} does not support backward pass")
    q, k, v, kv = attn_tensors

    # Clone tensors so backend and reference get identical inputs but separate grad accumulators
    q_ref = q.detach().clone().requires_grad_(True)
    kv_ref = kv.detach().clone().requires_grad_(True)
    k_ref_sep = k.detach().clone().requires_grad_(True)
    v_ref_sep = v.detach().clone().requires_grad_(True)

    if attn_backend.kv_packed:
        out = attn_backend.col_fn(q, kv)
        k_ref, v_ref = _unpack_kv(kv_ref)
        ref = reference_col_attn(q_ref, k_ref, v_ref)
    else:
        out = attn_backend.col_fn(q, k, v)
        ref = reference_col_attn(q_ref, k_ref_sep, v_ref_sep)

    out.sum().backward()
    ref.sum().backward()

    assert q.grad is not None, f"{attn_backend.name} column attention backward did not compute gradients for q"
    assert torch.allclose(q.grad, q_ref.grad, atol=1e-2, rtol=1e-2), f"{attn_backend.name} column attention backward q gradient mismatch from reference"


def test_row_attention_fwd(attn_backend, attn_tensors):
    q, k, v, kv = attn_tensors
    if attn_backend.kv_packed:
        out = attn_backend.row_fn(q, kv)
        k_ref, v_ref = _unpack_kv(kv)
        ref = reference_row_attn(q, k_ref, v_ref)
    else:
        out = attn_backend.row_fn(q, k, v)
        ref = reference_row_attn(q, k, v)
    assert out.shape == q.shape, f"{attn_backend.name} row attention forward output shape mismatch: expected {q.shape}, got {out.shape}"
    if attn_backend.supports_numerical_check:
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), f"{attn_backend.name} row attention forward output mismatch from reference"


def test_row_attention_bwd(attn_backend, attn_tensors):
    if not attn_backend.supports_bwd:
        pytest.skip(f"{attn_backend.name} does not support backward pass")
    q, k, v, kv = attn_tensors

    # Clone tensors so backend and reference get identical inputs but separate grad accumulators
    q_ref = q.detach().clone().requires_grad_(True)
    kv_ref = kv.detach().clone().requires_grad_(True)
    k_ref_sep = k.detach().clone().requires_grad_(True)
    v_ref_sep = v.detach().clone().requires_grad_(True)

    if attn_backend.kv_packed:
        out = attn_backend.row_fn(q, kv)
        k_ref, v_ref = _unpack_kv(kv_ref)
        ref = reference_row_attn(q_ref, k_ref, v_ref)
    else:
        out = attn_backend.row_fn(q, k, v)
        ref = reference_row_attn(q_ref, k_ref_sep, v_ref_sep)

    out.sum().backward()
    ref.sum().backward()

    assert q.grad is not None, f"{attn_backend.name} row attention backward did not compute gradients for q"
    assert torch.allclose(q.grad, q_ref.grad, atol=1e-2, rtol=1e-2), f"{attn_backend.name} row attention backward q gradient mismatch from reference"

def test_tabular_attention_one_layer(attn_backend, attn_tensors):
    q, k, v, kv = attn_tensors

    # Column attention followed by row attention
    if attn_backend.kv_packed:
        col_out = attn_backend.col_fn(q, kv)
        row_out = attn_backend.row_fn(col_out, kv)
    else:
        col_out = attn_backend.col_fn(q, k, v)
        row_out = attn_backend.row_fn(col_out, k, v)

    assert row_out.shape == q.shape, f"{attn_backend.name} tabular attention output shape mismatch: expected {q.shape}, got {row_out.shape}"

    if not attn_backend.supports_bwd:
        return
    # Backward pass
    row_out.sum().backward()
    assert q.grad is not None, f"{attn_backend.name} tabular attention backward did not compute gradients for q"


def test_tabular_attention_multiple_layers(attn_backend, attn_tensors):
    q, k, v, kv = attn_tensors

    # Simulate multiple layers of tabular attention
    out = q
    for _ in range(3):  # 3 layers
        if attn_backend.kv_packed:
            out = attn_backend.col_fn(out, kv)
            out = attn_backend.row_fn(out, kv)
        else:
            out = attn_backend.col_fn(out, k, v)
            out = attn_backend.row_fn(out, k, v)

    assert out.shape == q.shape, f"{attn_backend.name} multi-layer tabular attention output shape mismatch: expected {q.shape}, got {out.shape}"

    if not attn_backend.supports_bwd:
        return
    # Backward pass
    out.sum().backward()
    assert q.grad is not None, f"{attn_backend.name} multi-layer tabular attention backward did not compute gradients for q"
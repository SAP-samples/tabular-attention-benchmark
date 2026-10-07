# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM100 (Blackwell)-only version, stripped for tabular attention optimization.

import os
import math
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from tabular_attn.fa4_optim.shared.cache_utils import get_jit_cache
from tabular_attn.fa4_optim.shared.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from tabular_attn.fa4_optim.shared import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from tabular_attn.fa4_optim.shared import utils
from tabular_attn.fa4_optim.shared import fa_logging
from tabular_attn.fa4_optim.shared.cute_dsl_utils import (
    to_cute_tensor, to_cute_aux_tensor, get_aux_tensor_metadata, get_broadcast_dims,
)
from tabular_attn.fa4_optim.row_attn.flash_fwd_sm100 import FlashAttentionForwardSm100
from tabular_attn.fa4_optim.row_attn.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from tabular_attn.fa4_optim.row_attn.flash_bwd_sm100 import FlashAttentionBackwardSm100
from tabular_attn.fa4_optim.row_attn.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from tabular_attn.fa4_optim.row_attn.flash_fwd_combine import FlashAttentionForwardCombine

from tabular_attn.fa4_optim.shared.block_sparsity import (
    BlockSparseTensorsTorch,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)

def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check."""
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    """Validate head dimension constraints for SM100/SM110."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128
    assert (is_standard_range or is_deepseek_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
        f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
        f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek."
    )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    if num_n_blocks <= 4:
        return 1
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(causal, window_size_left, window_size_right):
    """Resolve causal/local/window settings into canonical form."""
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention (SM100-only)."""
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    batch_size, seqlen_q, num_head, head_dim = q.shape
    total_q = batch_size * seqlen_q
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"

    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [10, 11], f"Only SM100/SM110 supported, got arch={arch}"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    lse_shape = (batch_size, num_head, seqlen_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(batch_size, seqlen_q, num_head, head_dim_v, dtype=out_torch_dtype, device=device)
    else:
        _validate_tensor(out, "out", (batch_size, seqlen_q, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default()

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    tile_m, tile_n = 128, 128  # SM100 defaults

    # TODO: fix GQA + SplitKV
    if pack_gqa and num_splits != 1:
        pack_gqa = False

    max_seqlen_q = seqlen_q
    max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    q_stage = 2 if seqlen_q_packgqa > tile_m else 1

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, (window_size_right or max_seqlen_k) + (window_size_left or max_seqlen_k) + 1 + tile_m))
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = 132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV uses float32 partial output, which doubles the O buffer size
    if head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, batch_size, seqlen_q, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    use_2cta_instrs = (
        not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and not use_block_sparsity
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    if use_block_sparsity:
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[1] != 1:
            pack_gqa = False
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV."
            )

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        tile_m,
        tile_n,
        q_stage,
        is_split_kv,
        pack_gqa,
        arch,
        use_2cta_instrs,
        q_subtile_factor,
        requested_use_clc_scheduler,
        fa_logging.get_fa_log_level(),
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=not causal
                and not local
                and not is_split_kv,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=aux_tensors is not None,
            is_varlen_q=False,
            q_subtile_factor=q_subtile_factor,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=requested_use_clc_scheduler,
        )

        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            window_size_left,
            window_size_right,
            None,  # learnable_sink
            sparse_tensors,
            cute_aux_tensors,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_fwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach() if not is_split_kv else out_partial,
            lse_partial if is_split_kv else lse,
            softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            window_size_left,
            window_size_right,
            None,  # learnable_sink
            normalized_block_sparse_tensors[:4] if normalized_block_sparse_tensors is not None else None,
            aux_tensors,
        )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def make_fake_bwd_tensors(dtype, has_gqa):
    sym = cute.sym_int
    div = 128 // dtype.width
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    mQ = fake_tensor(dtype, (b, seqlen_q, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (b, seqlen_q, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (b, seqlen_q, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (b, seqlen_k, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (b, seqlen_k, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (b, seqlen_q, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (b, seqlen_k, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (b, seqlen_k, h_kv, d_v), divisibility=div)
    mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
        mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(dtype, head_dim, head_dim_v, m_block_size, has_dlse):
    """Compile bwd preprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True,
    )
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, m_block_size)
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, None, None, mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    dlse,
    dtype, head_dim, head_dim_v, m_block_size,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size, dlse is not None,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, None, None, dlse
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    use_2cta_instrs, cluster_size, arch,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True,
    )
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), None, None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs, cluster_size, arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, None, None,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch // 10 in [10, 11], f"Only SM100/SM110 supported, got arch={arch}"

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    # SM100 bwd config
    m_block_size = 128
    n_block_size = 128
    dQ_swapAB = False
    dKV_swapAB = False
    AtomLayoutMdQ = 1
    AtomLayoutNdKV = 1
    requested_disable_2cta = utils._get_disable_2cta_default()
    disable_2cta = (
        requested_disable_2cta
        or score_mod is not None
        or score_mod_bwd is not None
        or mask_mod is not None
        or block_sparse_tensors is not None
    )
    cluster_size = 2 if head_dim >= 128 and not disable_2cta else 1
    use_2cta_instrs = cluster_size == 2

    q, k, v, out, dout, lse = [maybe_contiguous(t) for t in (q, k, v, out, dout, lse)]
    batch_size, seqlen_q = q.shape[:2]
    total_q = batch_size * seqlen_q
    seqlen_k = k.shape[1]
    total_k = batch_size * seqlen_k
    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    subtile_factor = 2

    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
    assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
    assert lse.shape == (batch_size, num_head, seqlen_q)

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))
    assert num_head % num_head_kv == 0
    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    pack_gqa = False  # pack_gqa backward not yet supported

    if score_mod is not None:
        assert score_mod_bwd is not None
        assert softcap == 0.0

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)
    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)
    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    dq_accum = torch.empty(
        batch_size, num_head, seqlen_q_rounded * head_dim_rounded,
        dtype=torch.float32, device=device,
    )
    dpsum = torch.empty(
        batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
    )
    lse_log2 = torch.empty(
        batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
    )

    dKV_postprocess = qhead_per_kvhead > 1
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        dk_accum = torch.zeros(
            batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded,
            dtype=torch.float32, device=device,
        )
        dv_accum = torch.zeros(
            batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded,
            dtype=torch.float32, device=device,
        )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        dlse, dtype, head_dim, head_dim_v, m_block_size,
    )

    num_threads = 384

    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors]

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            subtile_factor=subtile_factor,
        )

    compile_key = (
        arch,
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        window_size_left is not None,
        window_size_right is not None,
        softcap != 0.0,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        cluster_size,
        use_2cta_instrs,
        deterministic,
        score_mod_hash,
        score_mod_bwd_hash,
        mask_mod_hash,
        num_aux_tensors,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        get_broadcast_dims(q),
        get_broadcast_dims(k),
        get_broadcast_dims(v),
        get_broadcast_dims(dout),
    )
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]

        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        fa_bwd_obj = FlashAttentionBackwardSm100(
            head_dim,
            head_dim_v,
            is_causal=causal,
            is_local=local,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=m_block_size,
            tile_n=n_block_size,
            cluster_size=cluster_size,
            use_2cta_instrs=use_2cta_instrs,
            deterministic=deterministic,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            mask_mod=mask_mod,
            has_aux_tensors=aux_tensors is not None,
            subtile_factor=subtile_factor,
        )

        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # softcap
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            cute_aux_tensors,
            sparse_tensors_compile,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # softcap
            window_size_left,
            window_size_right,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            aux_tensors,
            normalized_block_sparse_tensors[:4] if normalized_block_sparse_tensors is not None else None,
        )

    num_threads_post = 128

    _bwd_postprocess_convert(
        dq_accum, dq, softmax_scale,
        arch, dtype, head_dim, m_block_size, num_threads_post,
        AtomLayoutMdQ, dQ_swapAB,
        use_2cta_instrs=use_2cta_instrs, cluster_size=1,
    )

    if dKV_postprocess:
        _bwd_postprocess_convert(
            dk_accum, dk, softmax_scale,
            arch, dtype, head_dim, n_block_size, num_threads_post,
            AtomLayoutNdKV, dKV_swapAB,
            cluster_size=cluster_size,
        )
        _bwd_postprocess_convert(
            dv_accum, dv, 1.0,
            arch, dtype, head_dim_v, n_block_size, num_threads_post,
            AtomLayoutNdKV, dKV_swapAB,
            cluster_size=cluster_size,
        )

    return dq, dk, dv


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            return_lse=return_lse,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            deterministic=ctx.deterministic,
            dlse=dlse,
        )
        return dq, dk, dv, *((None,) * 10)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """FlashAttention forward (SM100/Blackwell only).

    Args:
        q: (batch_size, seqlen_q, num_heads, head_dim)
        k: (batch_size, seqlen_k, num_heads_kv, head_dim)
        v: (batch_size, seqlen_k, num_heads_kv, head_dim_v)
        softmax_scale: Scaling factor for attention scores. Default: 1/sqrt(head_dim).
        causal: Whether to apply causal masking.
        window_size: (left, right) sliding window sizes.
        softcap: Softcap value for attention logits.
        num_splits: Number of splits for split-KV.
        pack_gqa: Whether to pack GQA heads.
        deterministic: Whether to use deterministic backward.
        return_lse: Whether to return log-sum-exp.

    Returns:
        (out, lse) tuple. out has shape (batch_size, seqlen_q, num_heads, head_dim_v).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        return_lse,
    )


def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
    has_lse,
):
    """Compile fwd combine kernel using cute fake tensors."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
    mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
    mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
    mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
    mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE,
        None, None, None, None, None,  # cu_seqlens, seqused, num_splits_dynamic, varlen_batch_idx, semaphore
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation."""
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32]
    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    k_block_size = 64 if head_dim <= 64 else 128
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        log_max_splits = max(log_max_splits, 5)

    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        lse is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key
        )
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse,
            None, None, None, None, None,  # cu_seqlens, seqused, num_splits_dynamic, varlen_batch_idx, semaphore
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")

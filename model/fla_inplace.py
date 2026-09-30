from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel
from fla.ops.utils.op import exp

# Both functions copy the launch settings of fla 0.5.2's fused_recurrent_gated_delta_rule_fwd, and the advance kernel copies its state update op for op, so that they round like fla. After an fla upgrade, run python -m model.triton_kernels_test again.
WARPS, STAGES = 1, 3


def _block_sizes(K: int, V: int) -> tuple[int, int]:
    return triton.next_power_of_2(K), min(8, triton.next_power_of_2(V))


def gated_delta_rule_step_inplace(q, k, v, g, beta, state):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    BK, BV = _block_sizes(K, V)
    o = torch.empty_like(v)
    fused_recurrent_gated_delta_rule_fwd_kernel[(triton.cdiv(V, BV), B * HV)](
        q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=o, h0=state, ht=state, cu_seqlens=None,
        scale=K**-0.5, T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=True,
        APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, num_warps=WARPS, num_stages=STAGES)
    return o


@triton.jit
def _gated_delta_rule_advance(k, v, g, beta, state, steps, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                              BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_k = k + (i_n * T * H + i_h) * K + o_k
    p_v = v + (i_n * T * HV + i_hv) * V + o_v
    p_g = g + i_n * T * HV + i_hv
    p_beta = beta + i_n * T * HV + i_hv
    p_h = state + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    for _ in tl.range(0, tl.minimum(tl.maximum(tl.load(steps + i_n), 0), T)):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV
    tl.store(p_h, b_h, mask=mask_h)


def gated_delta_rule_advance_inplace(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
                                     steps: torch.Tensor) -> None:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if state.dtype != torch.float32 or not state.is_contiguous() or state.shape != (B, HV, K, V):
        raise ValueError(f"state: expected a contiguous float32 tensor of shape {(B, HV, K, V)}, got {state.dtype} {tuple(state.shape)}")
    if steps.shape != (B,) or steps.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"steps: expected {B} integers, got {steps.dtype} {tuple(steps.shape)}")
    BK, BV = _block_sizes(K, V)
    _gated_delta_rule_advance[(triton.cdiv(V, BV), B * HV)](k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), state,
                                                           steps.contiguous(), T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, num_warps=WARPS,
                                                           num_stages=STAGES)

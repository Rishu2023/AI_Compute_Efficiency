# -*- coding: utf-8 -*-
"""Fused Matrix Multiplication + Bias + Activation Kernel.

Instead of running matmul, bias-add, and activation as three separate
GPU kernel launches (each with its own memory read/write), this kernel
fuses all three operations into a single pass. This eliminates two
intermediate memory round-trips.

Key innovations:
  - Three operations fused into one kernel launch
  - Intermediate results stay in registers (never hit global memory)
  - Supports ReLU, GELU (tanh approximation), SiLU/Swish, and Leaky ReLU
  - Configurable for any combination: matmul-only, matmul+bias, matmul+act,
    or matmul+bias+act
  - GELU uses the fast tanh approximation (same as PyTorch's default)

Algorithmic Discovery: Fusing three complex math operations into one step
eliminates memory-bound overhead that dominates small-to-medium matrix sizes.
"""

import math

import torch
import triton
import triton.language as tl


# ── Activation function constants ─────────────────────────────────────────────
# These map to the ACTIVATION constexpr in the kernel
ACT_NONE = 0
ACT_RELU = 1
ACT_GELU = 2
ACT_SILU = 3
ACT_LEAKY_RELU = 4

_ACT_NAME_TO_ID = {
    'none': ACT_NONE,
    'relu': ACT_RELU,
    'gelu': ACT_GELU,
    'silu': ACT_SILU,
    'swish': ACT_SILU,  # SiLU and Swish are the same
    'leaky_relu': ACT_LEAKY_RELU,
}


# ── Kernel Configs ────────────────────────────────────────────────────────────

def get_fused_configs():
    """Autotuning configs for fused matmul on T4 (64 KB smem)."""
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
                       'GROUP_M': 8},
                      num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
                       'GROUP_M': 8},
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,
                       'GROUP_M': 8},
                      num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32,
                       'GROUP_M': 8},
                      num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32,
                       'GROUP_M': 8},
                      num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32,
                       'GROUP_M': 8},
                      num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64,
                       'GROUP_M': 4},
                      num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32,
                       'GROUP_M': 8},
                      num_warps=8, num_stages=2),
    ]


# ── Fused Matmul + Bias + Activation Kernel ──────────────────────────────────

@triton.autotune(configs=get_fused_configs(), key=['M', 'N', 'K'])
@triton.jit
def fused_matmul_bias_act_kernel(
    # Pointers
    A_ptr, B_ptr, bias_ptr, C_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Flags
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LEAKY_ALPHA: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fused matmul + optional bias + optional activation.

    C = activation(A @ B + bias)

    All intermediate results stay in registers — no global memory
    round-trips between matmul, bias-add, and activation.
    """
    # ── Grouped tile ordering ────────────────────────────────────────────────
    pid        = tl.program_id(0)
    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_in_grp = GROUP_M * num_pid_n
    group_id   = pid // num_in_grp
    first_pm   = group_id * GROUP_M
    grp_sz     = min(num_pid_m - first_pm, GROUP_M)
    pid_m      = first_pm + (pid % grp_sz)
    pid_n      = (pid % num_in_grp) // grp_sz

    # ── Offsets ──────────────────────────────────────────────────────────────
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rk = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # ── Matmul K-loop ────────────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K // BLOCK_K):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # ── Fused bias add (stays in registers) ──────────────────────────────────
    if HAS_BIAS:
        bias_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_mask = bias_n < N
        bias = tl.load(bias_ptr + bias_n, mask=bias_mask, other=0.0)
        acc += bias[None, :].to(tl.float32)

    # ── Fused activation (stays in registers) ────────────────────────────────
    if ACTIVATION == 1:  # ReLU
        acc = tl.where(acc > 0, acc, 0.0)

    elif ACTIVATION == 2:  # GELU (tanh approximation)
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        k = 0.7978845608  # sqrt(2/pi)
        x3 = acc * acc * acc
        inner = k * (acc + 0.044715 * x3)
        # tanh approximation
        acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    elif ACTIVATION == 3:  # SiLU / Swish
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        acc = acc * tl.sigmoid(acc)

    elif ACTIVATION == 4:  # Leaky ReLU
        acc = tl.where(acc > 0, acc, LEAKY_ALPHA * acc)

    # ── Store ────────────────────────────────────────────────────────────────
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (cm[:, None] < M) & (cn[None, :] < N)
    C_ptrs = C_ptr + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask)


# ── Python Wrapper ────────────────────────────────────────────────────────────

_FUSED_MAX_BLOCK_K = 64


def _pad_to(x: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad tensor along `dim` to a multiple of `multiple` with zeros."""
    r = x.shape[dim] % multiple
    if r == 0:
        return x
    pad_size = multiple - r
    pad = torch.zeros(
        *x.shape[:dim], pad_size, *x.shape[dim + 1:],
        device=x.device, dtype=x.dtype
    )
    return torch.cat([x, pad], dim=dim)


def fused_matmul_bias_act(a_fp16: torch.Tensor, b_fp16: torch.Tensor,
                          bias: torch.Tensor = None,
                          activation: str = 'none',
                          leaky_alpha: float = 0.01) -> torch.Tensor:
    """Compute C = activation(A @ B + bias) in a single fused kernel.

    Args:
        a_fp16: [M, K] fp16 input activations
        b_fp16: [K, N] fp16 weights
        bias: [N] fp16 bias vector (optional)
        activation: one of 'none', 'relu', 'gelu', 'silu'/'swish',
                    'leaky_relu'
        leaky_alpha: negative slope for leaky_relu (default: 0.01)

    Returns:
        [M, N] fp16 output
    """
    M, K = a_fp16.shape
    _, N = b_fp16.shape

    act_id = _ACT_NAME_TO_ID.get(activation, ACT_NONE)
    has_bias = bias is not None

    # Pad K
    a_p = _pad_to(a_fp16, dim=1, multiple=_FUSED_MAX_BLOCK_K)
    b_p = _pad_to(b_fp16, dim=0, multiple=_FUSED_MAX_BLOCK_K)
    K_pad = a_p.shape[1]

    c = torch.empty((M, N), device=a_fp16.device, dtype=torch.float16)

    # Use a dummy bias pointer if no bias
    bias_ptr = bias if has_bias else c  # c is just a dummy, never loaded

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    fused_matmul_bias_act_kernel[grid](
        a_p, b_p, bias_ptr, c,
        M, N, K_pad,
        a_p.stride(0), a_p.stride(1),
        b_p.stride(0), b_p.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=has_bias,
        ACTIVATION=act_id,
        LEAKY_ALPHA=leaky_alpha,
    )
    return c


# ── Reference implementations for verification ───────────────────────────────

def _pytorch_fused_reference(a, b, bias=None, activation='none',
                             leaky_alpha=0.01):
    """PyTorch reference: matmul + bias + activation (3 separate ops)."""
    out = torch.matmul(a, b)
    if bias is not None:
        out = out + bias
    if activation == 'relu':
        out = torch.relu(out)
    elif activation == 'gelu':
        out = torch.nn.functional.gelu(out, approximate='tanh')
    elif activation in ('silu', 'swish'):
        out = torch.nn.functional.silu(out)
    elif activation == 'leaky_relu':
        out = torch.nn.functional.leaky_relu(out,
                                             negative_slope=leaky_alpha)
    return out


# ── Benchmark & Verification ─────────────────────────────────────────────────

def bench(fn, warmup=30, rep=200):
    """Time a GPU function with warmup and repetition."""
    for _ in range(warmup):
        fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def verify_fused():
    """Verify fused kernel correctness for all activation types."""
    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    bias = torch.randn(N, device='cuda', dtype=torch.float16)

    all_pass = True
    for act in ['none', 'relu', 'gelu', 'silu', 'leaky_relu']:
        for use_bias in [False, True]:
            b_arg = bias if use_bias else None
            ref = _pytorch_fused_reference(a, b, b_arg, act)
            out = fused_matmul_bias_act(a, b, b_arg, act)
            err = (ref - out).abs().max().item()
            status = 'PASS' if err < 2.0 else 'FAIL'
            if status == 'FAIL':
                all_pass = False
            bias_str = '+bias' if use_bias else ''
            print(f"  {act}{bias_str}: max|diff|={err:.5f} {status}")

    return all_pass


def benchmark_fused(M, N, K, activation='gelu'):
    """Benchmark fused kernel vs separate PyTorch ops."""
    a = torch.randn(M, K, device='cuda', dtype=torch.float16).contiguous()
    b = torch.randn(K, N, device='cuda', dtype=torch.float16).contiguous()
    bias = torch.randn(N, device='cuda', dtype=torch.float16).contiguous()

    fused_ms = bench(lambda: fused_matmul_bias_act(a, b, bias, activation))
    pytorch_ms = bench(lambda: _pytorch_fused_reference(a, b, bias,
                                                        activation))

    def tflops(ms):
        return 2 * M * N * K / (ms * 1e-3) / 1e12

    print(f"[{M}x{K}x{N} {activation}+bias]  "
          f"Fused: {fused_ms:.3f} ms ({tflops(fused_ms):.2f} TFLOPS)  |  "
          f"PyTorch: {pytorch_ms:.3f} ms ({tflops(pytorch_ms):.2f} TFLOPS)  |  "
          f"Speedup: {pytorch_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    import sys
    print("=== Fused Matmul + Bias + Activation Kernel ===\n")
    print("Verification:")
    if not verify_fused():
        sys.exit(1)
    print()
    for act in ['relu', 'gelu', 'silu']:
        print(f"\n--- Activation: {act} ---")
        for sz in [(1024, 1024, 1024), (2048, 2048, 2048),
                   (4096, 4096, 4096)]:
            benchmark_fused(*sz, activation=act)

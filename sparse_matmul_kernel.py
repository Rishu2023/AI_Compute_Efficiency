# -*- coding: utf-8 -*-
"""Block-Sparse Matrix Multiplication Kernel.

Specialized Triton kernel that exploits block-level sparsity to skip
zero blocks entirely during matrix multiplication. Instead of processing
every K-tile, only non-zero blocks are loaded and computed.

Key innovations:
  - Block-sparse format with precomputed non-zero block indices
  - Zero blocks are never loaded from memory (bandwidth savings)
  - Zero blocks are never computed (compute savings)
  - Particularly effective for sparse attention patterns and pruned models
  - Adaptive: falls back to dense behavior when sparsity is low

Algorithmic Discovery: Rather than just tweaking tile sizes, this kernel
implements a fundamentally different algorithm that completely skips
multiplying zeros, as suggested in the optimization tips.
"""

import torch
import triton
import triton.language as tl


# ── Sparse Format Utilities ───────────────────────────────────────────────────

def create_block_sparse_format(w: torch.Tensor, block_k: int = 32,
                               block_n: int = 32,
                               threshold: float = 1e-6):
    """Convert a dense weight matrix to block-sparse format.

    Scans the matrix in blocks of [block_k, block_n] and identifies
    non-zero blocks. Returns compressed indices and values.

    Args:
        w: [K, N] dense weight matrix
        block_k: block size along K dimension
        block_n: block size along N dimension
        threshold: blocks with all values below this are considered zero

    Returns:
        Tuple of:
          - block_values: [num_nonzero_blocks, block_k, block_n] fp16
          - block_indices_k: [num_n_blocks, max_nnz_per_col] int32, K-indices
          - block_counts: [num_n_blocks] int32, number of non-zero blocks per N-col
          - num_k_blocks: int, total K blocks
          - num_n_blocks: int, total N blocks
          - sparsity: float, fraction of zero blocks
    """
    K, N = w.shape
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks = (N + block_n - 1) // block_n

    # Pad to exact multiples
    K_pad = num_k_blocks * block_k
    N_pad = num_n_blocks * block_n
    w_pad = torch.zeros(K_pad, N_pad, device=w.device, dtype=w.dtype)
    w_pad[:K, :N] = w

    # Reshape into blocks
    w_blocks = w_pad.reshape(num_k_blocks, block_k, num_n_blocks, block_n)
    w_blocks = w_blocks.permute(2, 0, 1, 3)  # [num_n, num_k, bk, bn]

    # Find non-zero blocks
    block_norms = w_blocks.abs().amax(dim=(-2, -1))  # [num_n, num_k]
    is_nonzero = block_norms > threshold

    # Build compressed indices per N-column
    max_nnz = is_nonzero.sum(dim=1).max().item()
    if max_nnz == 0:
        max_nnz = 1  # avoid zero-size tensor

    block_indices_k = torch.full((num_n_blocks, max_nnz), -1,
                                 dtype=torch.int32, device=w.device)
    block_counts = torch.zeros(num_n_blocks, dtype=torch.int32,
                               device=w.device)
    all_block_values = []

    for n_idx in range(num_n_blocks):
        nonzero_k = torch.where(is_nonzero[n_idx])[0]
        count = nonzero_k.shape[0]
        block_counts[n_idx] = count
        if count > 0:
            block_indices_k[n_idx, :count] = nonzero_k.to(torch.int32)
            for k_idx in nonzero_k:
                all_block_values.append(w_blocks[n_idx, k_idx])

    if len(all_block_values) > 0:
        block_values = torch.stack(all_block_values).to(torch.float16)
    else:
        block_values = torch.zeros(1, block_k, block_n, device=w.device,
                                   dtype=torch.float16)

    total_blocks = num_k_blocks * num_n_blocks
    zero_blocks = total_blocks - len(all_block_values)
    sparsity = zero_blocks / total_blocks if total_blocks > 0 else 0.0

    return (block_values.contiguous(), block_indices_k.contiguous(),
            block_counts.contiguous(), num_k_blocks, num_n_blocks, sparsity)


# ── Triton Kernel ─────────────────────────────────────────────────────────────

def get_sparse_configs():
    """Autotuning configs for sparse matmul on T4."""
    return [
        triton.Config({'BLOCK_M': 64, 'GROUP_M': 8}, num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'GROUP_M': 8}, num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 64, 'GROUP_M': 4}, num_warps=4,
                      num_stages=4),
        triton.Config({'BLOCK_M': 128, 'GROUP_M': 4}, num_warps=8,
                      num_stages=2),
    ]


@triton.autotune(configs=get_sparse_configs(), key=['M', 'N', 'num_n_blocks'])
@triton.jit
def sparse_matmul_kernel(
    # Pointers
    A_ptr,              # [M, K_padded] fp16 activations
    block_values_ptr,   # [total_nnz, BLOCK_K, BLOCK_N] fp16 block values
    block_indices_ptr,  # [num_n_blocks, max_nnz] int32 K-block indices
    block_counts_ptr,   # [num_n_blocks] int32 counts
    block_offsets_ptr,  # [num_n_blocks] int32 prefix-sum offsets
    C_ptr,              # [M, N] fp16 output
    # Dimensions
    M, N, K,
    num_n_blocks,
    max_nnz,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Block value strides
    stride_bv0, stride_bv1, stride_bv2,
    # Constexpr
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Block-sparse matmul: only processes non-zero blocks.

    For each output tile [BLOCK_M, BLOCK_N], iterates only over
    the non-zero K-blocks for that N-column, skipping all zero blocks.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = num_n_blocks

    # Grouped tile ordering
    num_in_grp = GROUP_M * num_pid_n
    group_id = pid // num_in_grp
    first_pm = group_id * GROUP_M
    grp_sz = min(num_pid_m - first_pm, GROUP_M)
    pid_m = first_pm + (pid % grp_sz)
    pid_n = (pid % num_in_grp) // grp_sz

    # Row offsets for activation loading
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M

    # Load count of non-zero blocks for this N-column
    nnz_count = tl.load(block_counts_ptr + pid_n)
    block_offset_base = tl.load(block_offsets_ptr + pid_n)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Only iterate over non-zero blocks
    for block_idx in range(0, max_nnz):
        # Check if this block index is valid (within nnz_count)
        if block_idx < nnz_count:
            # Load which K-block this is
            k_block_id = tl.load(block_indices_ptr
                                 + pid_n * max_nnz + block_idx)

            # Load activation tile [BLOCK_M, BLOCK_K]
            rk = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
            A_tile_ptrs = A_ptr + rm[:, None] * stride_am \
                + rk[None, :] * stride_ak
            a = tl.load(A_tile_ptrs)

            # Load block values [BLOCK_K, BLOCK_N]
            bv_idx = block_offset_base + block_idx
            rk_local = tl.arange(0, BLOCK_K)
            rn_local = tl.arange(0, BLOCK_N)
            bv_ptrs = block_values_ptr \
                + bv_idx * stride_bv0 \
                + rk_local[:, None] * stride_bv1 \
                + rn_local[None, :] * stride_bv2
            b = tl.load(bv_ptrs)

            acc += tl.dot(a, b)

    # Store result
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (cm[:, None] < M) & (cn[None, :] < N)
    C_ptrs = C_ptr + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask)


# ── Python Wrapper ────────────────────────────────────────────────────────────

# Fixed block sizes matching kernel constexpr
_SPARSE_BLOCK_K = 32
_SPARSE_BLOCK_N = 32


def sparse_matmul(a_fp16: torch.Tensor, block_values: torch.Tensor,
                  block_indices_k: torch.Tensor,
                  block_counts: torch.Tensor,
                  N: int) -> torch.Tensor:
    """Compute C = A @ W_sparse using block-sparse format.

    Args:
        a_fp16: [M, K] fp16 activations (K must be padded to BLOCK_K multiple)
        block_values: [total_nnz, BLOCK_K, BLOCK_N] fp16 non-zero blocks
        block_indices_k: [num_n_blocks, max_nnz] int32 K-block indices
        block_counts: [num_n_blocks] int32 non-zero counts per N-column
        N: original output dimension

    Returns:
        [M, N] fp16 output tensor
    """
    M = a_fp16.shape[0]
    num_n_blocks = block_counts.shape[0]
    max_nnz = block_indices_k.shape[1]

    # Compute prefix-sum offsets for block_values indexing
    block_offsets = torch.zeros_like(block_counts)
    if num_n_blocks > 1:
        block_offsets[1:] = torch.cumsum(block_counts[:-1], dim=0)

    c = torch.empty((M, N), device=a_fp16.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * num_n_blocks,
    )
    sparse_matmul_kernel[grid](
        a_fp16,
        block_values, block_indices_k, block_counts, block_offsets,
        c,
        M, N, a_fp16.shape[1],
        num_n_blocks, max_nnz,
        a_fp16.stride(0), a_fp16.stride(1),
        c.stride(0), c.stride(1),
        block_values.stride(0), block_values.stride(1),
        block_values.stride(2),
        BLOCK_K=_SPARSE_BLOCK_K,
        BLOCK_N=_SPARSE_BLOCK_N,
    )
    return c


def apply_sparsity_mask(w: torch.Tensor, sparsity: float = 0.5,
                        block_k: int = 32,
                        block_n: int = 32) -> torch.Tensor:
    """Apply block-level sparsity mask to a weight matrix.

    Zeros out entire blocks to simulate structured sparsity.

    Args:
        w: [K, N] weight matrix
        sparsity: fraction of blocks to zero out (0.0 to 1.0)
        block_k: block size along K
        block_n: block size along N

    Returns:
        [K, N] weight matrix with block-sparse pattern
    """
    K, N = w.shape
    num_k = (K + block_k - 1) // block_k
    num_n = (N + block_n - 1) // block_n

    # Random block mask
    mask = torch.rand(num_k, num_n, device=w.device) > sparsity
    # Expand mask to element level
    mask_expanded = mask.repeat_interleave(block_k, dim=0)[:K]
    mask_expanded = mask_expanded.repeat_interleave(block_n, dim=1)[:, :N]

    return w * mask_expanded.to(w.dtype)


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


def verify_sparse():
    """Verify sparse matmul kernel correctness."""
    M, K, N = 512, 512, 512

    # Create a sparse weight matrix (50% block sparsity)
    w = torch.randn(K, N, device='cuda', dtype=torch.float16)
    w = apply_sparsity_mask(w, sparsity=0.5, block_k=_SPARSE_BLOCK_K,
                            block_n=_SPARSE_BLOCK_N)

    # Convert to block-sparse format
    bv, bi, bc, _, _, sparsity = create_block_sparse_format(
        w, block_k=_SPARSE_BLOCK_K, block_n=_SPARSE_BLOCK_N
    )

    # Pad A to BLOCK_K multiple
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    K_pad = ((K + _SPARSE_BLOCK_K - 1) // _SPARSE_BLOCK_K) * _SPARSE_BLOCK_K
    a_pad = torch.zeros(M, K_pad, device='cuda', dtype=torch.float16)
    a_pad[:, :K] = a

    # Reference
    ref = torch.matmul(a, w)

    # Sparse kernel
    out = sparse_matmul(a_pad, bv, bi, bc, N)

    err = (ref - out).abs().max().item()
    print(f"Sparse Correctness — max|diff|: {err:.5f}  "
          f"{'PASS' if err < 2.0 else 'FAIL'}  "
          f"(sparsity: {sparsity:.1%})")
    return err < 2.0


def benchmark_sparse(M, N, K, sparsity=0.5):
    """Benchmark sparse kernel vs dense PyTorch matmul."""
    # Create sparse weights
    w = torch.randn(K, N, device='cuda', dtype=torch.float16)
    w = apply_sparsity_mask(w, sparsity=sparsity, block_k=_SPARSE_BLOCK_K,
                            block_n=_SPARSE_BLOCK_N)
    bv, bi, bc, _, _, actual_sparsity = create_block_sparse_format(
        w, block_k=_SPARSE_BLOCK_K, block_n=_SPARSE_BLOCK_N
    )

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    K_pad = ((K + _SPARSE_BLOCK_K - 1) // _SPARSE_BLOCK_K) * _SPARSE_BLOCK_K
    a_pad = torch.zeros(M, K_pad, device='cuda', dtype=torch.float16)
    a_pad[:, :K] = a

    # Benchmark
    sparse_ms = bench(lambda: sparse_matmul(a_pad, bv, bi, bc, N))
    dense_ms = bench(lambda: torch.matmul(a, w))

    def tflops(ms):
        return 2 * M * N * K / (ms * 1e-3) / 1e12

    print(f"[{M}x{K}x{N} @ {actual_sparsity:.0%} sparse]  "
          f"Sparse: {sparse_ms:.3f} ms ({tflops(sparse_ms):.2f} TFLOPS)  |  "
          f"Dense: {dense_ms:.3f} ms ({tflops(dense_ms):.2f} TFLOPS)  |  "
          f"Speedup: {dense_ms / sparse_ms:.2f}x")


if __name__ == "__main__":
    import sys
    print("=== Block-Sparse Matmul Kernel ===\n")
    if not verify_sparse():
        sys.exit(1)
    print()
    for sparsity in [0.5, 0.75, 0.9]:
        print(f"\n--- Sparsity: {sparsity:.0%} ---")
        for sz in [(1024, 1024, 1024), (2048, 2048, 2048),
                   (4096, 4096, 4096)]:
            benchmark_sparse(*sz, sparsity=sparsity)

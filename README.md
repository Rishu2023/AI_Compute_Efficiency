# AI Compute Efficiency — Custom Triton Kernels for NVIDIA T4

High-performance, specialized OpenAI Triton kernels for matrix multiplication
targeting the NVIDIA T4 GPU (Turing architecture, 64 KB shared memory,
FP16 Tensor Cores).

## Kernels

### 1. Optimized FP16 Matmul (`triton_py.py`)
Base Triton matmul kernel with:
- L2-friendly grouped tile ordering for cache locality
- Maskless inner K-loop via zero-padding (eliminates branch overhead)
- Aggressive autotuning across 11 tile/warp/stage configurations
- Software pipelining with multi-stage buffering

```bash
python triton_py.py
```

### 2. BitNet 1.58-bit Ternary Kernel (`bitnet_kernel.py`)
Specialized kernel for 1.58-bit quantized models (BitNet b1.58) where weights
are constrained to {-1, 0, +1}:
- **Zero multiplications**: replaces multiply-accumulate with conditional
  add/subtract
- **16x memory compression**: packs 4 ternary weights into a single byte
  (2-bit encoding)
- **Bandwidth-bound optimization**: 8x less weight data to read from memory
- Includes quantization utilities (absmean scaling per BitNet paper)

```bash
python bitnet_kernel.py
```

### 3. Block-Sparse Matmul (`sparse_matmul_kernel.py`)
Kernel that exploits block-level sparsity to skip zero blocks entirely:
- **Algorithmic innovation**: completely skips loading and computing zero blocks
- Precomputed block-sparse index format for fast lookup
- Scales linearly with actual non-zero content (not matrix dimensions)
- Effective for pruned models, sparse attention patterns, and MoE routing

```bash
python sparse_matmul_kernel.py
```

### 4. Fused Matmul + Bias + Activation (`fused_matmul_kernel.py`)
Three operations fused into a single kernel pass:
- **Eliminates 2 memory round-trips** (bias-add and activation intermediate
  buffers)
- Intermediate results stay in GPU registers (never hit global memory)
- Supports: ReLU, GELU (tanh approx), SiLU/Swish, Leaky ReLU
- Configurable: matmul-only, matmul+bias, matmul+act, or matmul+bias+act

```bash
python fused_matmul_kernel.py
```

### 5. Darwinian Kernel Evolution Arena (`kernel_arena.py`)
Evolutionary optimization framework for kernel configurations:
- **Genetic algorithm**: selection, crossover, mutation of tile sizes,
  warp counts, and pipeline stages
- Tournament selection with elitism (top configs always survive)
- Hardware constraint validation (64 KB smem, warp compatibility)
- Convergence tracking and JSON export
- Can run for thousands of generations to discover "alien" configurations

```bash
python kernel_arena.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the base matmul benchmark
python triton_py.py

# Run BitNet 1.58-bit kernel
python bitnet_kernel.py

# Run sparse matmul kernel
python sparse_matmul_kernel.py

# Run fused matmul+bias+activation kernel
python fused_matmul_kernel.py

# Run kernel evolution arena
python kernel_arena.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenAI Triton 2.1+
- NVIDIA GPU (optimized for T4, works on any CUDA GPU)

## Architecture

```
├── triton_py.py              # Optimized FP16 Tensor Core matmul
├── bitnet_kernel.py          # 1.58-bit ternary weight matmul
├── sparse_matmul_kernel.py   # Block-sparse matmul (skip zeros)
├── fused_matmul_kernel.py    # Fused matmul + bias + activation
├── kernel_arena.py           # Darwinian kernel evolution
├── requirements.txt          # Dependencies
└── Triton_py.ipynb           # Original Colab notebook
```

## Design Philosophy

1. **Specialize, don't generalize**: Each kernel targets a specific niche
   (ternary weights, sparsity, fused ops) rather than competing with
   general-purpose matmul
2. **Algorithmic discovery**: New mathematical shortcuts (skip zero blocks,
   replace mul with add/sub) rather than just tuning tile sizes
3. **Unconstrained evolution**: The arena framework can explore thousands
   of configurations to find non-obvious optimal points
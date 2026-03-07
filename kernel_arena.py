# -*- coding: utf-8 -*-
"""Darwinian Kernel Evolution Arena.

An evolutionary optimization framework that generates, mutates, benchmarks,
and selects the best Triton kernel configurations. Inspired by genetic
algorithms, this "arena" treats kernel configs as genomes and applies
selection pressure based on measured GPU performance.

Key innovations:
  - Population-based optimization of kernel tile sizes, warp counts, and stages
  - Tournament selection + crossover + mutation operators
  - Fitness = measured TFLOPS on actual GPU hardware
  - Elitism: top configs always survive to next generation
  - Constraint validation: ensures configs fit in T4's 64 KB shared memory
  - History tracking for convergence analysis
  - Can run indefinitely in the background ("unconstrained evolution")

This implements the "Darwinian Code Arena" concept from the optimization tips:
let the system generate thousands of kernel config mutations and select the
ones that actually run fastest on hardware.
"""

import copy
import json
import os
import random
import time

import torch
import triton
import triton.language as tl


# ── Kernel Config Genome ──────────────────────────────────────────────────────

# Valid parameter ranges for T4 GPU
PARAM_RANGES = {
    'BLOCK_M': [32, 64, 128, 256],
    'BLOCK_N': [32, 64, 128, 256],
    'BLOCK_K': [16, 32, 64, 128],
    'GROUP_M': [1, 2, 4, 8, 16],
    'num_warps': [2, 4, 8],
    'num_stages': [1, 2, 3, 4, 5],
}

# T4 shared memory limit (bytes)
T4_SMEM_LIMIT = 65536  # 64 KB


def estimate_smem_usage(block_m, block_n, block_k, num_stages,
                        dtype_bytes=2):
    """Estimate shared memory usage for a matmul config.

    smem = num_stages * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * dtype_bytes

    Args:
        block_m, block_n, block_k: tile dimensions
        num_stages: number of pipeline stages
        dtype_bytes: bytes per element (2 for fp16)

    Returns:
        Estimated shared memory usage in bytes
    """
    per_stage = (block_m * block_k + block_k * block_n) * dtype_bytes
    return per_stage * num_stages


def is_valid_config(config):
    """Check if a kernel config is valid for T4 GPU.

    Validates:
    - Shared memory fits in 64 KB
    - Parameters are in valid ranges
    - Warp count is compatible with tile size
    """
    bm = config['BLOCK_M']
    bn = config['BLOCK_N']
    bk = config['BLOCK_K']
    stages = config['num_stages']
    warps = config['num_warps']

    # Check smem
    smem = estimate_smem_usage(bm, bn, bk, stages)
    if smem > T4_SMEM_LIMIT:
        return False

    # Check that tile can be divided among warps
    # Each warp has 32 threads, and we need at least 1 element per thread
    tile_elements = bm * bn
    if tile_elements < warps * 32:
        return False

    # Check ranges
    if bm not in PARAM_RANGES['BLOCK_M']:
        return False
    if bn not in PARAM_RANGES['BLOCK_N']:
        return False
    if bk not in PARAM_RANGES['BLOCK_K']:
        return False
    if warps not in PARAM_RANGES['num_warps']:
        return False
    if stages not in PARAM_RANGES['num_stages']:
        return False

    return True


def random_config():
    """Generate a random valid kernel config."""
    for _ in range(100):
        config = {
            'BLOCK_M': random.choice(PARAM_RANGES['BLOCK_M']),
            'BLOCK_N': random.choice(PARAM_RANGES['BLOCK_N']),
            'BLOCK_K': random.choice(PARAM_RANGES['BLOCK_K']),
            'GROUP_M': random.choice(PARAM_RANGES['GROUP_M']),
            'num_warps': random.choice(PARAM_RANGES['num_warps']),
            'num_stages': random.choice(PARAM_RANGES['num_stages']),
        }
        if is_valid_config(config):
            return config
    # Fallback to known-good config
    return {
        'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
        'GROUP_M': 8, 'num_warps': 4, 'num_stages': 4,
    }


# ── Genetic Operators ─────────────────────────────────────────────────────────

def mutate(config, mutation_rate=0.3):
    """Mutate a kernel config by randomly changing parameters.

    Args:
        config: kernel config dict
        mutation_rate: probability of mutating each parameter

    Returns:
        New mutated config (original is not modified)
    """
    new_config = copy.deepcopy(config)

    for param, values in PARAM_RANGES.items():
        if random.random() < mutation_rate:
            new_config[param] = random.choice(values)

    # Validate and retry if invalid
    if not is_valid_config(new_config):
        return mutate(config, mutation_rate)  # retry with fresh mutation

    return new_config


def crossover(parent1, parent2):
    """Create a child config by combining two parents.

    Uses uniform crossover: each parameter randomly chosen from either parent.

    Args:
        parent1, parent2: kernel config dicts

    Returns:
        New child config
    """
    child = {}
    for param in PARAM_RANGES:
        child[param] = random.choice([parent1[param], parent2[param]])

    if not is_valid_config(child):
        # If invalid, try again or fall back to parent1
        for _ in range(10):
            child = {}
            for param in PARAM_RANGES:
                child[param] = random.choice([parent1[param], parent2[param]])
            if is_valid_config(child):
                return child
        return copy.deepcopy(parent1)

    return child


def tournament_select(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection.

    Args:
        population: list of configs
        fitnesses: list of fitness scores (higher = better)
        tournament_size: number of individuals in tournament

    Returns:
        Selected config
    """
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


# ── Benchmark Kernel (non-autotuned, uses explicit config) ────────────────────

@triton.jit
def arena_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Matmul kernel used by the arena (same logic, no autotune)."""
    pid        = tl.program_id(0)
    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_in_grp = GROUP_M * num_pid_n
    group_id   = pid // num_in_grp
    first_pm   = group_id * GROUP_M
    grp_sz     = min(num_pid_m - first_pm, GROUP_M)
    pid_m      = first_pm + (pid % grp_sz)
    pid_n      = (pid % num_in_grp) // grp_sz

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rk = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K // BLOCK_K):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (cm[:, None] < M) & (cn[None, :] < N)
    C_ptrs = C_ptr + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask)


def _pad_to(x, dim, multiple):
    """Pad tensor along dim to a multiple."""
    r = x.shape[dim] % multiple
    if r == 0:
        return x
    pad_size = multiple - r
    pad = torch.zeros(
        *x.shape[:dim], pad_size, *x.shape[dim + 1:],
        device=x.device, dtype=x.dtype
    )
    return torch.cat([x, pad], dim=dim)


def benchmark_config(config, M, N, K, warmup=10, rep=50):
    """Benchmark a single kernel config and return TFLOPS.

    Args:
        config: kernel config dict
        M, N, K: matrix dimensions
        warmup: warmup iterations
        rep: measurement iterations

    Returns:
        TFLOPS achieved, or 0.0 if the config fails
    """
    try:
        bm = config['BLOCK_M']
        bn = config['BLOCK_N']
        bk = config['BLOCK_K']
        gm = config['GROUP_M']
        nw = config['num_warps']
        ns = config['num_stages']

        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        a_p = _pad_to(a, dim=1, multiple=bk)
        b_p = _pad_to(b, dim=0, multiple=bk)
        K_pad = a_p.shape[1]

        c = torch.empty(M, N, device='cuda', dtype=torch.float16)

        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        # Warmup
        for _ in range(warmup):
            arena_matmul_kernel[grid](
                a_p, b_p, c, M, N, K_pad,
                a_p.stride(0), a_p.stride(1),
                b_p.stride(0), b_p.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                num_warps=nw, num_stages=ns,
            )

        # Measure
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(rep):
            arena_matmul_kernel[grid](
                a_p, b_p, c, M, N, K_pad,
                a_p.stride(0), a_p.stride(1),
                b_p.stride(0), b_p.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                num_warps=nw, num_stages=ns,
            )
        e.record()
        torch.cuda.synchronize()

        ms = s.elapsed_time(e) / rep
        tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

        # Verify correctness (spot check)
        ref = torch.matmul(a, b)
        err = (ref - c).abs().max().item()
        if err > 5.0:  # tolerance for fp16
            return 0.0

        return tflops

    except Exception:
        return 0.0


# ── Evolution Arena ───────────────────────────────────────────────────────────

class KernelEvolutionArena:
    """Darwinian evolution arena for kernel configuration optimization.

    Maintains a population of kernel configs and evolves them using
    genetic algorithm operators (selection, crossover, mutation).
    Fitness is measured as actual TFLOPS on the target GPU.

    Usage:
        arena = KernelEvolutionArena(population_size=50)
        best = arena.evolve(generations=100, M=4096, N=4096, K=4096)
    """

    def __init__(self, population_size=50, elite_count=5,
                 mutation_rate=0.3, crossover_rate=0.7,
                 tournament_size=3):
        """Initialize the arena.

        Args:
            population_size: number of configs in the population
            elite_count: number of top configs preserved each generation
            mutation_rate: probability of mutating each parameter
            crossover_rate: probability of crossover vs mutation-only
            tournament_size: tournament selection size
        """
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        # Initialize population
        self.population = [random_config() for _ in range(population_size)]
        self.fitnesses = [0.0] * population_size
        self.generation = 0
        self.history = []
        self.best_ever = None
        self.best_ever_fitness = 0.0

    def evaluate_population(self, M, N, K, warmup=10, rep=50):
        """Benchmark all configs in the population.

        Args:
            M, N, K: matrix dimensions to benchmark on
            warmup, rep: benchmark parameters
        """
        for i, config in enumerate(self.population):
            self.fitnesses[i] = benchmark_config(config, M, N, K,
                                                 warmup, rep)

    def evolve_one_generation(self, M, N, K, warmup=10, rep=50):
        """Run one generation of evolution.

        1. Evaluate fitness of all configs
        2. Select parents via tournament selection
        3. Create children via crossover and mutation
        4. Apply elitism (top configs survive)

        Args:
            M, N, K: matrix dimensions
            warmup, rep: benchmark parameters

        Returns:
            Dict with generation statistics
        """
        self.generation += 1

        # Evaluate
        self.evaluate_population(M, N, K, warmup, rep)

        # Track best
        gen_best_idx = max(range(len(self.fitnesses)),
                          key=lambda i: self.fitnesses[i])
        gen_best_fitness = self.fitnesses[gen_best_idx]
        gen_best_config = copy.deepcopy(self.population[gen_best_idx])

        if gen_best_fitness > self.best_ever_fitness:
            self.best_ever_fitness = gen_best_fitness
            self.best_ever = copy.deepcopy(gen_best_config)

        avg_fitness = sum(self.fitnesses) / len(self.fitnesses)

        stats = {
            'generation': self.generation,
            'best_tflops': gen_best_fitness,
            'avg_tflops': avg_fitness,
            'best_config': gen_best_config,
            'best_ever_tflops': self.best_ever_fitness,
        }
        self.history.append(stats)

        # Create next generation
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(self.fitnesses)),
                                key=lambda i: self.fitnesses[i],
                                reverse=True)

        new_population = []

        # Elitism: keep top configs
        for i in range(self.elite_count):
            new_population.append(
                copy.deepcopy(self.population[sorted_indices[i]])
            )

        # Fill rest with children
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                p1 = tournament_select(self.population, self.fitnesses,
                                       self.tournament_size)
                p2 = tournament_select(self.population, self.fitnesses,
                                       self.tournament_size)
                child = crossover(p1, p2)
            else:
                # Mutation only
                parent = tournament_select(self.population, self.fitnesses,
                                           self.tournament_size)
                child = mutate(parent, self.mutation_rate)

            # Always apply some mutation to children
            if random.random() < 0.5:
                child = mutate(child, self.mutation_rate * 0.5)

            new_population.append(child)

        self.population = new_population[:self.population_size]
        self.fitnesses = [0.0] * self.population_size

        return stats

    def evolve(self, generations=20, M=4096, N=4096, K=4096,
               warmup=10, rep=50, verbose=True):
        """Run multiple generations of evolution.

        Args:
            generations: number of generations to run
            M, N, K: matrix dimensions
            warmup, rep: benchmark parameters
            verbose: print progress

        Returns:
            Best config found
        """
        if verbose:
            print(f"\nStarting evolution: {generations} generations, "
                  f"population={self.population_size}, "
                  f"matrix=[{M}x{K}x{N}]")
            print("-" * 80)

        for gen in range(generations):
            stats = self.evolve_one_generation(M, N, K, warmup, rep)

            if verbose:
                cfg = stats['best_config']
                print(f"Gen {stats['generation']:3d} | "
                      f"Best: {stats['best_tflops']:.2f} TFLOPS | "
                      f"Avg: {stats['avg_tflops']:.2f} TFLOPS | "
                      f"All-time: {stats['best_ever_tflops']:.2f} TFLOPS | "
                      f"Config: BM={cfg['BLOCK_M']}, BN={cfg['BLOCK_N']}, "
                      f"BK={cfg['BLOCK_K']}, W={cfg['num_warps']}, "
                      f"S={cfg['num_stages']}")

        if verbose:
            print("-" * 80)
            print(f"\nBest config found ({self.best_ever_fitness:.2f} TFLOPS):")
            print(f"  {self.best_ever}")

        return self.best_ever

    def save_results(self, filepath):
        """Save evolution history to JSON.

        Args:
            filepath: output file path
        """
        results = {
            'best_config': self.best_ever,
            'best_tflops': self.best_ever_fitness,
            'total_generations': self.generation,
            'population_size': self.population_size,
            'history': self.history,
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

    @staticmethod
    def load_results(filepath):
        """Load evolution results from JSON.

        Args:
            filepath: input file path

        Returns:
            Dict with results
        """
        with open(filepath, 'r') as f:
            return json.load(f)


# ── Convenience Functions ─────────────────────────────────────────────────────

def quick_evolve(M=4096, N=4096, K=4096, generations=10,
                 population=20):
    """Quick evolution run with sensible defaults.

    Args:
        M, N, K: matrix dimensions
        generations: number of generations
        population: population size

    Returns:
        Best config dict
    """
    arena = KernelEvolutionArena(population_size=population)
    best = arena.evolve(generations=generations, M=M, N=N, K=K)
    return best


def generate_triton_config(config):
    """Convert an arena config dict to a triton.Config object.

    Args:
        config: dict with BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
                num_warps, num_stages

    Returns:
        triton.Config object
    """
    return triton.Config(
        {'BLOCK_M': config['BLOCK_M'],
         'BLOCK_N': config['BLOCK_N'],
         'BLOCK_K': config['BLOCK_K'],
         'GROUP_M': config['GROUP_M']},
        num_warps=config['num_warps'],
        num_stages=config['num_stages'],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Darwinian Kernel Evolution Arena ===\n")

    # Quick test with small population and few generations
    print("Running quick evolution (10 generations, 20 population)...\n")

    arena = KernelEvolutionArena(
        population_size=20,
        elite_count=3,
        mutation_rate=0.3,
        crossover_rate=0.7,
    )

    best_config = arena.evolve(
        generations=10,
        M=2048, N=2048, K=2048,
        warmup=5, rep=20,
    )

    print(f"\nBest config as triton.Config:")
    print(f"  {generate_triton_config(best_config)}")

    # Compare with PyTorch
    M, N, K = 2048, 2048, 2048
    tflops = benchmark_config(best_config, M, N, K, warmup=20, rep=100)
    print(f"\nFinal benchmark: {tflops:.2f} TFLOPS")

    # PyTorch reference
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    for _ in range(30):
        torch.matmul(a, b)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(100):
        torch.matmul(a, b)
    e.record()
    torch.cuda.synchronize()
    pt_ms = s.elapsed_time(e) / 100
    pt_tflops = 2 * M * N * K / (pt_ms * 1e-3) / 1e12
    print(f"PyTorch reference: {pt_tflops:.2f} TFLOPS")
    print(f"Ratio: {tflops / pt_tflops:.2f}x")

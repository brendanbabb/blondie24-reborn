"""
GPU Benchmark: measure the speedup from batched leaf evaluation.

Run on Badger-1 to see the difference the RTX 5060 makes:
    python -m analysis.benchmark_gpu

Compares:
  1. Single evaluation (one board at a time on CPU)
  2. Single evaluation (one board at a time on GPU)
  3. Batched evaluation (N boards at once on GPU)
  4. Full minimax search: CPU vs GPU with batched leaves
"""

import time
import torch
import numpy as np

from checkers.board import Board
from neural.network import CheckersNet
from neural.encoding import encode_board
from search.minimax import MinimaxAgent, _encode_board_fast
from config import SearchConfig, NetworkConfig
from utils import get_device, optimize_for_inference


def benchmark_forward_pass():
    """Benchmark raw forward pass throughput."""
    print("\n" + "=" * 60)
    print("  FORWARD PASS BENCHMARK")
    print("=" * 60)

    net = CheckersNet()
    board = Board()
    n_evals = 5000

    # --- CPU single ---
    net_cpu = CheckersNet()
    net_cpu.set_weight_vector(net.get_weight_vector())
    buf = torch.zeros(1, 32)

    t0 = time.perf_counter()
    for _ in range(n_evals):
        encoding = _encode_board_fast(board, 1.3)
        buf[0] = torch.from_numpy(encoding)
        _ = net_cpu(buf)
    cpu_single_time = time.perf_counter() - t0
    print(f"  CPU single:  {n_evals} evals in {cpu_single_time:.3f}s "
          f"({n_evals/cpu_single_time:.0f} evals/s)")

    # --- GPU single ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        net_gpu = CheckersNet()
        net_gpu.set_weight_vector(net.get_weight_vector())
        net_gpu = net_gpu.to(device)
        buf_gpu = torch.zeros(1, 32, device=device)

        # Warmup
        for _ in range(100):
            buf_gpu[0] = torch.from_numpy(_encode_board_fast(board, 1.3))
            _ = net_gpu(buf_gpu)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_evals):
            buf_gpu[0] = torch.from_numpy(_encode_board_fast(board, 1.3))
            _ = net_gpu(buf_gpu)
        torch.cuda.synchronize()
        gpu_single_time = time.perf_counter() - t0
        print(f"  GPU single:  {n_evals} evals in {gpu_single_time:.3f}s "
              f"({n_evals/gpu_single_time:.0f} evals/s)")

        # --- GPU batched ---
        for batch_size in [8, 16, 32, 64, 128, 256]:
            batch_buf = torch.zeros(batch_size, 32, device=device)
            encoding = _encode_board_fast(board, 1.3)
            for i in range(batch_size):
                batch_buf[i] = torch.from_numpy(encoding)

            n_batches = n_evals // batch_size

            # Warmup
            for _ in range(10):
                _ = net_gpu(batch_buf)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_batches):
                _ = net_gpu(batch_buf)
            torch.cuda.synchronize()
            gpu_batch_time = time.perf_counter() - t0
            total_evals = n_batches * batch_size
            print(f"  GPU batch={batch_size:3d}: {total_evals} evals in {gpu_batch_time:.3f}s "
                  f"({total_evals/gpu_batch_time:.0f} evals/s)")
    else:
        print("  (GPU not available — skipping GPU benchmarks)")


def benchmark_minimax_search():
    """Benchmark full minimax search at various depths."""
    print("\n" + "=" * 60)
    print("  MINIMAX SEARCH BENCHMARK")
    print("=" * 60)

    net = CheckersNet()
    board = Board()

    for depth in [2, 4, 6]:
        config = SearchConfig(depth=depth)

        # CPU
        agent_cpu = MinimaxAgent(net, config, torch.device("cpu"))
        n_games = 3
        cpu_nodes = 0

        t0 = time.perf_counter()
        for _ in range(n_games):
            b = Board()
            for move_num in range(10):  # 10 moves from opening
                move, score = agent_cpu.search(b)
                cpu_nodes += agent_cpu.nodes_evaluated
                if move is None:
                    break
                b = b.apply_move(move)
        cpu_time = time.perf_counter() - t0

        print(f"\n  Depth {depth} — CPU:")
        print(f"    {n_games}×10 moves in {cpu_time:.2f}s")
        print(f"    {cpu_nodes} nodes evaluated ({cpu_nodes/cpu_time:.0f} nodes/s)")

        # GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            net_gpu = CheckersNet()
            net_gpu.set_weight_vector(net.get_weight_vector())
            agent_gpu = MinimaxAgent(net_gpu, config, device)

            # Warmup
            agent_gpu.search(Board())
            gpu_nodes = 0
            gpu_batch_calls = 0

            t0 = time.perf_counter()
            for _ in range(n_games):
                b = Board()
                for move_num in range(10):
                    move, score = agent_gpu.search(b)
                    gpu_nodes += agent_gpu.nodes_evaluated
                    gpu_batch_calls += agent_gpu.batch_calls
                    if move is None:
                        break
                    b = b.apply_move(move)
            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - t0

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0

            print(f"  Depth {depth} — GPU (batched leaves):")
            print(f"    {n_games}×10 moves in {gpu_time:.2f}s")
            print(f"    {gpu_nodes} nodes ({gpu_nodes/gpu_time:.0f} nodes/s)")
            print(f"    Speedup: {speedup:.1f}x")
            print(f"    Batch calls: {gpu_batch_calls}"
                  f" (avg size {gpu_nodes/max(gpu_batch_calls,1):.1f})")


def benchmark_generation():
    """Benchmark a full generation (tournament) on CPU vs GPU."""
    print("\n" + "=" * 60)
    print("  FULL GENERATION BENCHMARK (pop=6, depth=4)")
    print("=" * 60)

    from evolution.population import Population
    from evolution.tournament import round_robin_tournament
    from config import EvolutionConfig

    evo_config = EvolutionConfig(population_size=6, games_per_individual=3)
    search_config = SearchConfig(depth=4)

    # CPU
    pop_cpu = Population(evo_config)
    t0 = time.perf_counter()
    round_robin_tournament(pop_cpu, search_config, evo_config, device="cpu")
    cpu_time = time.perf_counter() - t0
    print(f"  CPU: {cpu_time:.1f}s")

    # GPU
    if torch.cuda.is_available():
        pop_gpu = Population(evo_config)
        # Copy same weights so results are comparable
        for i, ind in enumerate(pop_gpu.individuals):
            ind.weights = pop_cpu.individuals[i].weights.copy()
            ind.sigmas = pop_cpu.individuals[i].sigmas.copy()

        t0 = time.perf_counter()
        round_robin_tournament(pop_gpu, search_config, evo_config, device="cuda")
        gpu_time = time.perf_counter() - t0
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"  GPU: {gpu_time:.1f}s (speedup: {speedup:.1f}x)")


if __name__ == "__main__":
    optimize_for_inference()
    print("Blondie24 Reborn — GPU Benchmark")
    print(f"PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — CPU-only benchmarks")

    benchmark_forward_pass()
    benchmark_minimax_search()
    benchmark_generation()

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)

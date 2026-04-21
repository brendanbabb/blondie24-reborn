"""Smoke test for PVS + aspiration JIT (1999)."""

import sys, time
import numpy as np

sys.path.insert(0, ".")
from checkers.board import Board
from search.fast_minimax_jit import FastAgentJit, warmup

print("warming...", flush=True)
t0 = time.perf_counter()
warmup()
print(f"compile: {time.perf_counter() - t0:.2f}s", flush=True)

rng = np.random.default_rng(0)
w = rng.standard_normal(1743).astype(np.float32) * 0.3
w[0] = 0.5; w[1] = 2.0

for d in (4, 6, 8):
    agent = FastAgentJit(w, depth=d)
    b = Board()
    t0 = time.perf_counter()
    move, score = agent.search(b)
    t1 = time.perf_counter()
    print(f"d={d} move={move} score={score:.5f} time={(t1 - t0) * 1000:.2f}ms", flush=True)
print("OK", flush=True)

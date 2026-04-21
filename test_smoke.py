"""
Smoke test: verify all core components work together.

Run from the project root:
    python -m test_smoke
"""

import sys
import numpy as np
import torch

# -- Test 1: Board creation and move generation --
print("Test 1: Board + move generation...", end=" ")
from checkers.board import Board, BLACK, WHITE
board = Board()
moves = board.get_legal_moves()
assert len(moves) == 7, f"Expected 7 opening moves for black, got {len(moves)}"
print(f"OK — {len(moves)} legal opening moves")

# -- Test 2: Apply a move and switch turns --
print("Test 2: Apply move...", end=" ")
new_board = board.apply_move(moves[0])
assert new_board.current_player == WHITE
assert new_board.squares is not board.squares  # immutability check
white_moves = new_board.get_legal_moves()
assert len(white_moves) == 7, f"Expected 7 opening moves for white, got {len(white_moves)}"
print(f"OK — turn switched, {len(white_moves)} white moves")

# -- Test 3: Neural network creation and forward pass --
print("Test 3: Neural network...", end=" ")
from neural.network import CheckersNet
from neural.encoding import encode_board
from config import NetworkConfig

net = CheckersNet()
n_weights = net.num_weights()
encoding = encode_board(board)
assert encoding.shape == (32,), f"Expected shape (32,), got {encoding.shape}"

score = net.evaluate_board(encoding)
assert -1.0 <= score <= 1.0, f"Score {score} out of range [-1, 1]"
print(f"OK — {n_weights} weights, eval={score:.4f}")

# -- Test 4: Weight vector round-trip --
print("Test 4: Weight vector get/set...", end=" ")
w = net.get_weight_vector()
assert len(w) == n_weights
net2 = CheckersNet()
net2.set_weight_vector(w)
w2 = net2.get_weight_vector()
assert np.allclose(w, w2), "Weight round-trip failed"
print(f"OK — {n_weights} weights round-tripped")

# -- Test 5: Minimax search --
print("Test 5: Minimax search (depth 2)...", end=" ")
from search.minimax import minimax_search
from config import SearchConfig

search_cfg = SearchConfig(depth=2)
best_move, best_score = minimax_search(board, net, search_cfg)
assert best_move is not None, "No move returned"
assert best_move in moves, "Returned move is not legal"
print(f"OK — chose move with score {best_score:.4f}")

# -- Test 6: Evolution --
print("Test 6: Evolution (mutation)...", end=" ")
from evolution.strategy import Individual, mutate, initialize_individual
from config import EvolutionConfig

evo_cfg = EvolutionConfig()
ind = initialize_individual(n_weights, evo_cfg)
offspring = mutate(ind, evo_cfg)
assert len(offspring.weights) == n_weights
assert not np.array_equal(ind.weights, offspring.weights), "Mutation produced identical weights"
print(f"OK — offspring differs in {np.sum(ind.weights != offspring.weights)} weights")

# -- Test 7: Population --
print("Test 7: Population management...", end=" ")
from evolution.population import Population

pop = Population(EvolutionConfig(population_size=4))
assert len(pop.individuals) == 4
assert pop.n_weights == n_weights

# Fake some fitness scores
pop.individuals[0].fitness = 3.0
pop.individuals[1].fitness = 1.0
pop.individuals[2].fitness = -1.0
pop.individuals[3].fitness = -3.0

stats = pop.stats()
assert stats["max_fitness"] == 3.0
assert stats["min_fitness"] == -3.0
print(f"OK — pop of 4, stats working")

# -- Test 8: Selection + Reproduction --
print("Test 8: Select + reproduce...", end=" ")
pop.select_and_reproduce()
assert len(pop.individuals) == 4
assert pop.generation == 1
print(f"OK — generation {pop.generation}, pop size {len(pop.individuals)}")

# -- Test A: Anaconda architecture (2001 paper) --
print("Test A: AnacondaNet (2001)...", end=" ")
from neural.anaconda_network import AnacondaNet
from neural.anaconda_windows import N_FILTERS, N_PP_WEIGHTS, WINDOWS
from neural.network import make_network, architecture_from_weight_count

assert N_FILTERS == 91
assert N_PP_WEIGHTS == 854
assert sum(len(w) for w in WINDOWS) == 854

anet = AnacondaNet()
assert anet.num_weights() == 5048
w = anet.get_weight_vector()
assert len(w) == 5048

# Round-trip with random weights
rng = np.random.default_rng(1)
w_rand = rng.standard_normal(5048).astype(np.float32) * 0.1
anet.set_weight_vector(w_rand)
w2 = anet.get_weight_vector()
assert np.allclose(w_rand, w2, atol=1e-6), "Anaconda weight roundtrip failed"

# Forward pass
score = anet.evaluate_board(encoding)
assert -1.0 <= score <= 1.0
print(f"OK — 91 filters, 854 pp-weights, 5048 total, eval={score:.4f}")

# -- Test B: Factory + weight-count inference --
print("Test B: make_network + arch inference...", end=" ")
from config import NetworkConfig as _NetCfg
net1999 = make_network(_NetCfg(architecture="checkersnet-1999"))
net2001 = make_network(_NetCfg(architecture="anaconda-2001"))
assert net1999.num_weights() == 1743
assert net2001.num_weights() == 5048
assert architecture_from_weight_count(1743) == "checkersnet-1999"
assert architecture_from_weight_count(5048) == "anaconda-2001"
print("OK")

# -- Test 9: Play a short game --
print("Test 9: Play a quick game (random vs random, max 20 moves)...", end=" ")
from checkers.game import play_game
import random

def random_agent(b):
    legal = b.get_legal_moves()
    return random.choice(legal)

result = play_game(random_agent, random_agent, max_moves=20)
print(f"OK — {result.moves} moves, result: {result.reason}")

# -- Test C: Anaconda JIT agent vs PyTorch forward parity + short game --
print("Test C: Anaconda JIT forward pass parity + quick game...", end=" ")
from neural.fast_eval_anaconda import (
    TOTAL as ANA_TOTAL, unpack_weights_anaconda, forward_batch_anaconda,
)
from search.fast_minimax_jit_anaconda import (
    FastAgentJitAnaconda, warmup_anaconda,
)

rng_c = np.random.default_rng(17)
w_ana = rng_c.standard_normal(ANA_TOTAL).astype(np.float32) * 0.1
anet.set_weight_vector(w_ana)

# PyTorch reference score from the raw 32-vector encoding.
enc_np = np.asarray(encoding, dtype=np.float32).reshape(1, 32)
W_pp, b_pp, W1, W2, b2, W3, b3, pd, kw = unpack_weights_anaconda(w_ana)
np_score = float(forward_batch_anaconda(enc_np, W_pp, b_pp, W1, W2, b2, W3, b3, pd)[0])
torch_score = float(anet.evaluate_board(encoding))
assert abs(np_score - torch_score) < 1e-4, (
    f"Anaconda numpy vs torch diverged: {np_score} vs {torch_score}"
)

# Warm + play one game: JIT agent vs random, max 40 moves.
warmup_anaconda()
jit_black = FastAgentJitAnaconda(w_ana, depth=2)
result_c = play_game(jit_black, random_agent, max_moves=40)
print(
    f"OK — np/torch diff={abs(np_score - torch_score):.2e}, "
    f"{result_c.moves} moves, result: {result_c.reason}"
)

# -- Test D: 1999 vs Anaconda dispatch via _fast_agent_class_for --
print("Test D: tournament fast-agent dispatch...", end=" ")
from evolution.tournament import _fast_agent_class_for
from search.fast_minimax_jit import FastAgentJit

assert _fast_agent_class_for("checkersnet-1999") is FastAgentJit
assert _fast_agent_class_for("anaconda-2001") is FastAgentJitAnaconda
print("OK")

print("\n" + "=" * 50)
print("  ALL SMOKE TESTS PASSED")
print("=" * 50)

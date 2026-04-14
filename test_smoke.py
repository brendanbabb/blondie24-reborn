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

# -- Test 9: Play a short game --
print("Test 9: Play a quick game (random vs random, max 20 moves)...", end=" ")
from checkers.game import play_game
import random

def random_agent(b):
    legal = b.get_legal_moves()
    return random.choice(legal)

result = play_game(random_agent, random_agent, max_moves=20)
print(f"OK — {result.moves} moves, result: {result.reason}")

print("\n" + "=" * 50)
print("  ALL SMOKE TESTS PASSED")
print("=" * 50)

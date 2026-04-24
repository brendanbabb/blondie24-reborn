"""Head-to-head match between two trained Anaconda checkpoints.

Each game is played at the given depth with quiescence on (the
play-strong default), with the two checkpoints alternating colors
across games. To break determinism (same-start games always play
the same way), each game starts after `--random-opening N` random
plies of opening play. Reports W/L/D from A's perspective.

Usage:
  python scripts/ai_vs_ai.py CKPT_A CKPT_B [--games N] [--depth D] [--random-opening K]

  # Quick 20-game comparison at depth 4, 6 random opening plies (~1 min):
  python scripts/ai_vs_ai.py checkpoints/best_gen0200.pt \\
      checkpoints/best_gen0900.pt --games 20 --depth 4 --random-opening 6

  # Full 40-game comparison at depth 6 (~10-20 min):
  python scripts/ai_vs_ai.py checkpoints/best_gen0200.pt \\
      checkpoints/best_gen0900.pt --games 40 --depth 6
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from checkers.board import Board, BLACK, WHITE  # noqa: E402
from checkers.game import play_game, GameResult  # noqa: E402
from search.fast_minimax_jit_anaconda import (  # noqa: E402
    FastAgentJitAnaconda, warmup_anaconda,
)


def random_opening_position(rng: np.random.Generator, plies: int) -> Board:
    """Play `plies` random legal moves from the start position. Returns the
    resulting Board. If the game ends early (no legal moves / max plies),
    returns the starting position instead so we don't try to play a
    finished game."""
    board = Board()
    for _ in range(plies):
        moves = board.get_legal_moves()
        over, _ = board.is_game_over()
        if not moves or over:
            return Board()  # restart from scratch if random walk dead-ends
        board = board.apply_move(moves[rng.integers(len(moves))])
    return board


def play_game_from(black_agent, white_agent, start_board: Board,
                   max_moves: int = 200) -> GameResult:
    """Like play_game but starts from a given board state (not Board())."""
    # Mirror the loop in checkers.game.play_game but with a custom starting
    # board. Simpler to inline than to refactor play_game.
    from checkers.board import Board as _Board  # noqa
    board = start_board
    move_count = 0
    state_counts = {}
    while move_count < max_moves:
        over, winner = board.is_game_over()
        if over:
            return GameResult(winner=winner, moves=move_count, reason="no_moves")
        state_key = board.squares.tobytes() + bytes([board.current_player + 2])
        state_counts[state_key] = state_counts.get(state_key, 0) + 1
        if state_counts[state_key] >= 3:
            return GameResult(winner=None, moves=move_count, reason="repetition")
        agent = black_agent if board.current_player == BLACK else white_agent
        move = agent(board)
        board = board.apply_move(move)
        move_count += 1
    return GameResult(winner=None, moves=move_count, reason="max_moves")


def load_weights(path: str) -> np.ndarray:
    """Load a .pt checkpoint and return the flat weight vector. Mirrors
    the dispatch in scripts/export_weights_to_js.py — checkpoints can
    have either a numpy array or a torch tensor under "weights"."""
    # weights_only=True blocks arbitrary pickle RCE during load. Our
    # checkpoints store the flat weights as a numpy array, so we narrowly
    # allowlist numpy's ndarray reconstructor — still safe (the allowlist
    # is a function-by-function opt-in, not arbitrary code execution).
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    ]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, list):
        ckpt = ckpt[0]
    w = ckpt["weights"]
    if isinstance(w, torch.Tensor):
        w = w.detach().cpu().numpy()
    return np.asarray(w, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Head-to-head Anaconda matchup.")
    p.add_argument("a", help="Checkpoint A (.pt)")
    p.add_argument("b", help="Checkpoint B (.pt)")
    p.add_argument("--games", type=int, default=20,
                   help="Number of games (will be balanced colors). Default 20.")
    p.add_argument("--depth", type=int, default=6,
                   help="Search depth in plies. Default 6 (matches play-strong).")
    p.add_argument("--max-moves", type=int, default=200,
                   help="Move cap before declaring draw. Default 200.")
    p.add_argument("--no-quiescence", action="store_true",
                   help="Disable quiescence search.")
    p.add_argument("--random-opening", type=int, default=8,
                   help="Random plies of opening play before the agents "
                        "take over. Without this, deterministic agents "
                        "from the start position always produce the same "
                        "game. Default 8.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the random openings.")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    use_q = not args.no_quiescence

    print(f"Warming JIT (depth {args.depth}, quiescence {'on' if use_q else 'off'})...",
          flush=True)
    warmup_anaconda()
    print("ready", flush=True)

    wa = load_weights(args.a)
    wb = load_weights(args.b)
    if len(wa) != 5048 or len(wb) != 5048:
        raise SystemExit(
            f"Both checkpoints must be Anaconda 2001 (5048 weights). "
            f"Got A={len(wa)}, B={len(wb)}."
        )

    agent_a = FastAgentJitAnaconda(wa, args.depth, use_quiescence=use_q)
    agent_b = FastAgentJitAnaconda(wb, args.depth, use_quiescence=use_q)

    print()
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"Playing {args.games} games at depth {args.depth} "
          f"(quiescence {'on' if use_q else 'off'}, "
          f"{args.random_opening} random opening plies)...")
    print()

    a_wins = 0
    b_wins = 0
    draws = 0
    total_moves = 0

    t0 = time.time()
    for i in range(args.games):
        a_is_black = (i % 2 == 0)
        if a_is_black:
            black, white = agent_a, agent_b
        else:
            black, white = agent_b, agent_a

        opening = random_opening_position(rng, args.random_opening)
        result = play_game_from(black, white, opening, max_moves=args.max_moves)
        total_moves += result.moves

        if result.winner is None:
            draws += 1
            outcome = "draw"
        elif (result.winner == BLACK) == a_is_black:
            a_wins += 1
            outcome = f"A wins ({'black' if a_is_black else 'white'})"
        else:
            b_wins += 1
            outcome = f"B wins ({'black' if not a_is_black else 'white'})"

        elapsed = time.time() - t0
        print(f"  game {i+1:3d}/{args.games}: A plays "
              f"{'BLACK' if a_is_black else 'WHITE'} -> {outcome:20s} "
              f"({result.moves} mv, {result.reason}, t={elapsed:.0f}s)",
              flush=True)

    elapsed = time.time() - t0
    print()
    print(f"=== {args.games} games in {elapsed:.0f}s "
          f"({total_moves/args.games:.0f} mv/game avg) ===")
    print(f"A: {a_wins:2d} W / {b_wins:2d} L / {draws:2d} D")
    print(f"B: {b_wins:2d} W / {a_wins:2d} L / {draws:2d} D")

    # Tournament-style score: +1 per win, 0.5 per draw
    a_score = a_wins + 0.5 * draws
    b_score = b_wins + 0.5 * draws
    print()
    print(f"A score (tournament-style, W=1, D=0.5): {a_score:.1f} / {args.games}")
    print(f"B score (tournament-style, W=1, D=0.5): {b_score:.1f} / {args.games}")

    if a_wins == b_wins:
        print(f"\nDecision: Tied on wins ({a_wins} apiece). "
              f"More games needed to discriminate.")
    elif a_wins > b_wins:
        margin = a_wins - b_wins
        print(f"\nDecision: A is stronger (wins by {margin}).")
    else:
        margin = b_wins - a_wins
        print(f"\nDecision: B is stronger (wins by {margin}).")


if __name__ == "__main__":
    main()

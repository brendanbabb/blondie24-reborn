"""
Play against the best evolved network in the terminal.

Usage:
    python -m play.human_vs_ai --checkpoint checkpoints/best_gen250.pt
    python -m play.human_vs_ai --checkpoint checkpoints/best_gen250.pt --depth 6
"""

import argparse
import torch
import numpy as np
from checkers.board import Board, BLACK, WHITE
from checkers.game import play_game, GameResult
from neural.network import CheckersNet
from search.minimax import make_agent
from config import SearchConfig, NetworkConfig


def human_agent(board: Board):
    """Interactive human agent — prints board and asks for move selection."""
    print("\n" + str(board))
    moves = board.get_legal_moves()
    
    print("\nAvailable moves:")
    for i, move in enumerate(moves):
        if isinstance(move, tuple):
            print(f"  {i}: {move[0]} → {move[1]}")
        else:
            # Jump sequence
            path = f"{move[0]}"
            for j in range(2, len(move), 2):
                path += f" → {move[j]}"
            captured = [move[k] for k in range(1, len(move), 2)]
            print(f"  {i}: {path} (captures: {captured})")
    
    while True:
        try:
            choice = int(input(f"\nYour move (0-{len(moves)-1}): "))
            if 0 <= choice < len(moves):
                return moves[choice]
            print("Invalid choice, try again.")
        except (ValueError, EOFError):
            print("Enter a number.")


def load_network(checkpoint_path: str, device: str = "cpu") -> CheckersNet:
    """Load a trained network from a checkpoint."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = CheckersNet(NetworkConfig())
    net.set_weight_vector(data["weights"])
    net = net.to(device)
    return net


def main():
    parser = argparse.ArgumentParser(description="Play checkers against Blondie24 Reborn")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best network checkpoint")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for AI")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--color", type=str, default="white", choices=["black", "white"],
                        help="Your color (white moves second)")
    args = parser.parse_args()
    
    # Load AI
    network = load_network(args.checkpoint, args.device)
    search_config = SearchConfig(depth=args.depth)
    ai_agent = make_agent(network, search_config, args.device)
    
    print("=" * 40)
    print("  BLONDIE24 REBORN")
    print(f"  AI search depth: {args.depth}")
    print(f"  You are playing: {args.color}")
    print("=" * 40)
    
    if args.color == "black":
        result = play_game(human_agent, ai_agent, verbose=False)
    else:
        result = play_game(ai_agent, human_agent, verbose=False)
    
    print("\n" + "=" * 40)
    if result.winner is None:
        print("  DRAW!")
    elif (result.winner == BLACK and args.color == "black") or \
         (result.winner == WHITE and args.color == "white"):
        print("  YOU WIN!")
    else:
        print("  BLONDIE24 WINS!")
    print(f"  {result.moves} moves, ended by: {result.reason}")
    print("=" * 40)


if __name__ == "__main__":
    main()

"""
Game loop: play a complete game of checkers between two agents.

An agent is any callable that takes a Board and returns a legal move.
"""

from checkers.board import Board, BLACK, WHITE
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class GameResult:
    """Result of a completed game."""
    winner: Optional[int]   # BLACK, WHITE, or None (draw)
    moves: int              # total number of moves played
    reason: str             # "no_moves", "max_moves", "no_pieces"


def play_game(
    black_agent: Callable,
    white_agent: Callable,
    max_moves: int = 200,
    verbose: bool = False,
) -> GameResult:
    """
    Play a complete game of checkers.
    
    Args:
        black_agent: callable(board) -> move for the black player
        white_agent: callable(board) -> move for the white player
        max_moves: maximum total moves before declaring a draw
        verbose: print board state each move
    
    Returns:
        GameResult with winner, move count, and termination reason
    """
    board = Board()
    move_count = 0
    
    # Track repeated states for draw detection
    state_counts: dict[bytes, int] = {}
    
    while move_count < max_moves:
        if verbose:
            print(f"\n--- Move {move_count + 1} ---")
            print(board)
        
        # Check game over
        game_over, winner = board.is_game_over()
        if game_over:
            return GameResult(winner=winner, moves=move_count, reason="no_moves")
        
        # Check for threefold repetition
        state_key = board.squares.tobytes() + bytes([board.current_player + 2])
        state_counts[state_key] = state_counts.get(state_key, 0) + 1
        if state_counts[state_key] >= 3:
            return GameResult(winner=None, moves=move_count, reason="repetition")
        
        # Get agent's move
        agent = black_agent if board.current_player == BLACK else white_agent
        move = agent(board)
        
        # Apply move
        board = board.apply_move(move)
        move_count += 1
    
    # Max moves reached — draw
    return GameResult(winner=None, moves=move_count, reason="max_moves")

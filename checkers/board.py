"""
Checkers board state representation.

Uses a compact representation: 32 playable squares on the 8x8 board.
Square numbering (standard checkers notation):

     0   1   2   3
   4   5   6   7
     8   9  10  11
  12  13  14  15
    16  17  18  19
  20  21  22  23
    24  25  26  27
  28  29  30  31

Even rows: squares are in columns 1,3,5,7
Odd rows:  squares are in columns 0,2,4,6

Piece encoding:
  0 = empty
  1 = black piece (moves "down" — toward higher row numbers)
  2 = black king
 -1 = white piece (moves "up" — toward lower row numbers)
 -2 = white king
"""

import numpy as np
from typing import Optional
from copy import deepcopy

# Constants
EMPTY = 0
BLACK_PIECE = 1
BLACK_KING = 2
WHITE_PIECE = -1
WHITE_KING = -2

BLACK = 1   # player identifier
WHITE = -1  # player identifier


def _build_adjacency():
    """
    Precompute adjacency lists for all 32 squares.
    Returns dict mapping square → list of (neighbor, jump_target) for each diagonal direction.
    
    Each square can have up to 4 diagonal neighbors.
    For each neighbor, we also store where a jump would land (if the neighbor is occupied).
    """
    adjacency = {i: [] for i in range(32)}
    
    for sq in range(32):
        row = sq // 4
        # Determine column based on row parity
        if row % 2 == 0:
            col = (sq % 4) * 2 + 1
        else:
            col = (sq % 4) * 2
        
        # Four diagonal directions: (drow, dcol)
        for dr, dc in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                # Convert (nr, nc) back to square index
                if nr % 2 == 0:
                    if nc % 2 == 1:
                        neighbor = nr * 4 + nc // 2
                    else:
                        continue  # not a playable square
                else:
                    if nc % 2 == 0:
                        neighbor = nr * 4 + nc // 2
                    else:
                        continue
                
                # Jump target
                jr, jc = row + 2 * dr, col + 2 * dc
                jump_target = None
                if 0 <= jr < 8 and 0 <= jc < 8:
                    if jr % 2 == 0:
                        if jc % 2 == 1:
                            jump_target = jr * 4 + jc // 2
                    else:
                        if jc % 2 == 0:
                            jump_target = jr * 4 + jc // 2
                
                adjacency[sq].append((neighbor, jump_target, dr))
    
    return adjacency


# Precomputed adjacency — built once at import time
ADJACENCY = _build_adjacency()


class Board:
    """
    Checkers board state.
    
    Attributes:
        squares: numpy array of 32 ints (piece encoding)
        current_player: BLACK (1) or WHITE (-1)
    """
    
    def __init__(self, squares: Optional[np.ndarray] = None, current_player: int = BLACK):
        if squares is not None:
            self.squares = squares.copy()
        else:
            self.squares = self._initial_position()
        self.current_player = current_player
    
    @staticmethod
    def _initial_position() -> np.ndarray:
        """Standard starting position: black on top (sq 0-11), white on bottom (sq 20-31)."""
        squares = np.zeros(32, dtype=np.int8)
        # Black pieces: squares 0-11
        for i in range(12):
            squares[i] = BLACK_PIECE
        # White pieces: squares 20-31
        for i in range(20, 32):
            squares[i] = WHITE_PIECE
        return squares
    
    def copy(self) -> "Board":
        return Board(self.squares.copy(), self.current_player)
    
    def get_pieces(self, player: int) -> list[int]:
        """Return list of square indices occupied by the given player."""
        if player == BLACK:
            return [i for i in range(32) if self.squares[i] in (BLACK_PIECE, BLACK_KING)]
        else:
            return [i for i in range(32) if self.squares[i] in (WHITE_PIECE, WHITE_KING)]
    
    def is_king(self, sq: int) -> bool:
        return self.squares[sq] in (BLACK_KING, WHITE_KING)
    
    def _get_jumps(self, sq: int, player: int) -> list[list[int]]:
        """
        Find all jump sequences starting from sq for the given player.
        Returns list of move paths, where each path is [start, captured1, land1, captured2, land2, ...].
        Multi-jumps are mandatory and explored recursively.
        """
        piece = self.squares[sq]
        is_king = piece in (BLACK_KING, WHITE_KING)
        
        results = []
        
        for neighbor, jump_target, dr in ADJACENCY[sq]:
            if jump_target is None:
                continue
            
            # Direction check: regular pieces can only move in one direction
            if not is_king:
                if player == BLACK and dr < 0:
                    continue  # black non-kings can only move down (dr > 0)
                if player == WHITE and dr > 0:
                    continue  # white non-kings can only move up (dr < 0)
            
            # Neighbor must be an opponent piece
            neighbor_piece = self.squares[neighbor]
            if player == BLACK and neighbor_piece not in (WHITE_PIECE, WHITE_KING):
                continue
            if player == WHITE and neighbor_piece not in (BLACK_PIECE, BLACK_KING):
                continue
            
            # Landing square must be empty
            if self.squares[jump_target] != EMPTY:
                continue
            
            # Execute the jump temporarily and look for multi-jumps
            old_sq = self.squares[sq]
            old_neighbor = self.squares[neighbor]
            old_target = self.squares[jump_target]
            
            self.squares[sq] = EMPTY
            self.squares[neighbor] = EMPTY
            self.squares[jump_target] = piece
            
            # Check for king promotion mid-jump
            promoted = False
            if not is_king:
                if player == BLACK and jump_target >= 28:
                    self.squares[jump_target] = BLACK_KING
                    promoted = True
                elif player == WHITE and jump_target <= 3:
                    self.squares[jump_target] = WHITE_KING
                    promoted = True
            
            # Recurse for multi-jumps (only if not just promoted — promotion ends turn)
            sub_jumps = []
            if not promoted:
                sub_jumps = self._get_jumps(jump_target, player)
            
            if sub_jumps:
                for sub in sub_jumps:
                    results.append([sq, neighbor, jump_target] + sub[1:])  # skip repeated start
            else:
                results.append([sq, neighbor, jump_target])
            
            # Undo temporary jump
            self.squares[sq] = old_sq
            self.squares[neighbor] = old_neighbor
            self.squares[jump_target] = old_target
        
        return results
    
    def _get_simple_moves(self, sq: int, player: int) -> list[tuple[int, int]]:
        """Get non-jump moves for a piece at sq."""
        piece = self.squares[sq]
        is_king = piece in (BLACK_KING, WHITE_KING)
        moves = []
        
        for neighbor, _, dr in ADJACENCY[sq]:
            if not is_king:
                if player == BLACK and dr < 0:
                    continue
                if player == WHITE and dr > 0:
                    continue
            
            if self.squares[neighbor] == EMPTY:
                moves.append((sq, neighbor))
        
        return moves
    
    def get_legal_moves(self) -> list:
        """
        Get all legal moves for the current player.
        
        Returns list of moves. Each move is either:
          - (from_sq, to_sq) for simple moves
          - [from_sq, captured1, land1, ...] for jump sequences
        
        If any jumps are available, ONLY jumps are legal (mandatory capture rule).
        """
        player = self.current_player
        pieces = self.get_pieces(player)
        
        # Check for jumps first (mandatory)
        all_jumps = []
        for sq in pieces:
            jumps = self._get_jumps(sq, player)
            all_jumps.extend(jumps)
        
        if all_jumps:
            return all_jumps
        
        # No jumps — return simple moves
        all_moves = []
        for sq in pieces:
            moves = self._get_simple_moves(sq, player)
            all_moves.extend(moves)
        
        return all_moves
    
    def apply_move(self, move) -> "Board":
        """
        Apply a move and return a NEW board state.
        Does NOT modify self.
        
        Move is either:
          - (from_sq, to_sq) for simple move
          - [from_sq, captured1, land1, captured2, land2, ...] for jump sequence
        """
        new_board = self.copy()
        
        if isinstance(move, tuple):
            # Simple move
            from_sq, to_sq = move
            new_board.squares[to_sq] = new_board.squares[from_sq]
            new_board.squares[from_sq] = EMPTY
            
            # Check for king promotion
            if new_board.squares[to_sq] == BLACK_PIECE and to_sq >= 28:
                new_board.squares[to_sq] = BLACK_KING
            elif new_board.squares[to_sq] == WHITE_PIECE and to_sq <= 3:
                new_board.squares[to_sq] = WHITE_KING
        else:
            # Jump sequence: [start, captured1, land1, captured2, land2, ...]
            piece = new_board.squares[move[0]]
            new_board.squares[move[0]] = EMPTY
            
            i = 1
            while i < len(move):
                captured_sq = move[i]
                land_sq = move[i + 1]
                new_board.squares[captured_sq] = EMPTY  # remove captured piece
                i += 2
            
            final_sq = move[-1]
            new_board.squares[final_sq] = piece
            
            # Check for king promotion
            if new_board.squares[final_sq] == BLACK_PIECE and final_sq >= 28:
                new_board.squares[final_sq] = BLACK_KING
            elif new_board.squares[final_sq] == WHITE_PIECE and final_sq <= 3:
                new_board.squares[final_sq] = WHITE_KING
        
        # Switch player
        new_board.current_player = -new_board.current_player
        return new_board
    
    def is_game_over(self) -> tuple[bool, Optional[int]]:
        """
        Check if game is over.
        Returns (is_over, winner) where winner is BLACK, WHITE, or None (draw).
        """
        moves = self.get_legal_moves()
        if not moves:
            # Current player has no moves — they lose
            return True, -self.current_player
        return False, None
    
    def piece_count(self) -> tuple[int, int]:
        """Return (black_count, white_count) including kings."""
        black = sum(1 for s in self.squares if s in (BLACK_PIECE, BLACK_KING))
        white = sum(1 for s in self.squares if s in (WHITE_PIECE, WHITE_KING))
        return black, white
    
    def __repr__(self) -> str:
        """ASCII representation of the board. Empty dark squares show their index."""
        symbols = {
            BLACK_PIECE: "b",
            BLACK_KING: "B",
            WHITE_PIECE: "w",
            WHITE_KING: "W",
        }
        lines = []
        lines.append("    " + " ".join(f"{c:2d}" for c in range(8)))
        for row in range(8):
            cells = []
            for col in range(8):
                if (row + col) % 2 == 1:
                    sq = row * 4 + col // 2
                    piece = self.squares[sq]
                    if piece == EMPTY:
                        cells.append(f"{sq:2d}")
                    else:
                        cells.append(f" {symbols[piece]}")
                else:
                    cells.append("  ")
            lines.append(f"{row}   " + " ".join(cells))
        player_name = "Black" if self.current_player == BLACK else "White"
        lines.append(f"  Turn: {player_name}")
        return "\n".join(lines)

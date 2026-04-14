"""
Numba-JIT board operations for the hot training loop.

Operates on raw numpy int8 arrays (32 squares) rather than Board instances,
which lets Numba compile the whole move-generation pipeline to native code.

Move representation for JIT:
  A move is a fixed-size int8 array of length 17:
    move[0]    = move kind (0 = simple, 1 = jump)
    move[1]    = number of captured squares (0 for simple)
    move[2]    = from square
    move[3]    = to square (for simple) or first landing square (for jumps)
    move[4..]  = alternating (captured, land) pairs for remaining jump steps

  For simple moves only positions 0..3 are meaningful.
  For a jump of K captures, positions 0..(2*K + 3) are used.
  Max path length in checkers is bounded — 17 is plenty.

Legal-moves output shape: (max_moves, 17) int8 + a count.
We return an (N, 17) array plus the count N so callers can slice.

Adjacency: precomputed at module import as two int8 arrays of shape (32, 4):
  NEIGHBORS[sq][d] = neighbor square in direction d, or -1 if no neighbor
  JUMP_TARGETS[sq][d] = landing square for a jump in direction d, or -1
  DIRS[d] = delta row sign (for non-king direction restriction): -1 or +1
"""

import numpy as np
from numba import njit, int8, int32

# Piece constants — mirror checkers/board.py
EMPTY = 0
BLACK_PIECE = 1
BLACK_KING = 2
WHITE_PIECE = -1
WHITE_KING = -2

BLACK = 1
WHITE = -1

# Move encoding
MOVE_SLOTS = 17
KIND_SIMPLE = 0
KIND_JUMP = 1


def _build_adjacency_arrays():
    """
    Precompute neighbor and jump-target arrays in the shape Numba wants.

    Directions: 0=up-left, 1=up-right, 2=down-left, 3=down-right.
    """
    neighbors = np.full((32, 4), -1, dtype=np.int8)
    jumps     = np.full((32, 4), -1, dtype=np.int8)
    dir_dr    = np.array([-1, -1, +1, +1], dtype=np.int8)

    for sq in range(32):
        row = sq // 4
        if row % 2 == 0:
            col = (sq % 4) * 2 + 1
        else:
            col = (sq % 4) * 2

        for d, (dr, dc) in enumerate([(-1, -1), (-1, +1), (+1, -1), (+1, +1)]):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                valid = (nr % 2 == 0 and nc % 2 == 1) or (nr % 2 == 1 and nc % 2 == 0)
                if valid:
                    neighbors[sq, d] = nr * 4 + nc // 2

            jr, jc = row + 2 * dr, col + 2 * dc
            if 0 <= jr < 8 and 0 <= jc < 8:
                valid = (jr % 2 == 0 and jc % 2 == 1) or (jr % 2 == 1 and jc % 2 == 0)
                if valid:
                    jumps[sq, d] = jr * 4 + jc // 2

    return neighbors, jumps, dir_dr


NEIGHBORS, JUMP_TARGETS, DIR_DR = _build_adjacency_arrays()


@njit(cache=False)
def _get_jumps_rec(
    squares, sq, player, neighbors, jump_targets, dir_dr,
    path, path_len, out_moves, out_count
):
    """
    Recursive jump search. Returns new out_count.

    path: (17,) int8 — working buffer, already contains [KIND_JUMP, n_caps, from, ...]
    path_len: current filled length of path
    out_moves: (M, 17) int8 — destination for completed jump paths
    out_count: current count of completed paths
    """
    piece = squares[sq]
    is_king = (piece == BLACK_KING) or (piece == WHITE_KING)

    any_extension = False

    for d in range(4):
        neighbor = neighbors[sq, d]
        if neighbor < 0:
            continue
        jump_target = jump_targets[sq, d]
        if jump_target < 0:
            continue

        if not is_king:
            dr = dir_dr[d]
            if player == BLACK and dr < 0:
                continue
            if player == WHITE and dr > 0:
                continue

        neighbor_piece = squares[neighbor]
        if player == BLACK:
            if neighbor_piece != WHITE_PIECE and neighbor_piece != WHITE_KING:
                continue
        else:
            if neighbor_piece != BLACK_PIECE and neighbor_piece != BLACK_KING:
                continue

        if squares[jump_target] != EMPTY:
            continue

        # Execute jump in place
        old_sq_val = squares[sq]
        old_neighbor = squares[neighbor]
        old_target = squares[jump_target]

        squares[sq] = EMPTY
        squares[neighbor] = EMPTY

        # King promotion mid-jump ends the sequence
        promoted = False
        if not is_king:
            if player == BLACK and jump_target >= 28:
                squares[jump_target] = BLACK_KING
                promoted = True
            elif player == WHITE and jump_target <= 3:
                squares[jump_target] = WHITE_KING
                promoted = True
        if not promoted:
            squares[jump_target] = piece

        # Append (captured, land) to path
        path[path_len] = neighbor
        path[path_len + 1] = jump_target
        new_path_len = path_len + 2
        # Increment capture count at position 1
        path[1] = path[1] + 1

        any_extension = True

        if promoted:
            # End of this path — emit
            for k in range(new_path_len):
                out_moves[out_count, k] = path[k]
            out_count += 1
        else:
            # Recurse to find multi-jump continuations
            prev_count = out_count
            out_count = _get_jumps_rec(
                squares, jump_target, player,
                neighbors, jump_targets, dir_dr,
                path, new_path_len, out_moves, out_count
            )
            if out_count == prev_count:
                # No further extensions — emit current path as-is
                for k in range(new_path_len):
                    out_moves[out_count, k] = path[k]
                out_count += 1

        # Undo
        squares[sq] = old_sq_val
        squares[neighbor] = old_neighbor
        squares[jump_target] = old_target
        path[1] = path[1] - 1

    return out_count


@njit(cache=False)
def get_legal_moves_fast(
    squares, current_player, neighbors, jump_targets, dir_dr,
    out_moves
):
    """
    Fill out_moves with all legal moves for current_player on squares.
    Returns the count of moves written.

    out_moves: (M, 17) int8 buffer, caller-allocated. Should be large enough
               (checkers position rarely has >48 legal moves).
    """
    # ── Pass 1: any jumps available? ──
    jump_count = 0
    path = np.zeros(MOVE_SLOTS, dtype=np.int8)

    for sq in range(32):
        piece = squares[sq]
        if piece == EMPTY:
            continue
        if current_player == BLACK:
            if piece != BLACK_PIECE and piece != BLACK_KING:
                continue
        else:
            if piece != WHITE_PIECE and piece != WHITE_KING:
                continue

        # Reset path header for this origin square
        path[0] = KIND_JUMP
        path[1] = 0
        path[2] = sq

        jump_count = _get_jumps_rec(
            squares, sq, current_player,
            neighbors, jump_targets, dir_dr,
            path, 3, out_moves, jump_count
        )

    if jump_count > 0:
        return jump_count

    # ── Pass 2: no jumps — emit simple moves ──
    simple_count = 0
    for sq in range(32):
        piece = squares[sq]
        if piece == EMPTY:
            continue
        if current_player == BLACK:
            if piece != BLACK_PIECE and piece != BLACK_KING:
                continue
        else:
            if piece != WHITE_PIECE and piece != WHITE_KING:
                continue
        is_king = (piece == BLACK_KING) or (piece == WHITE_KING)

        for d in range(4):
            neighbor = neighbors[sq, d]
            if neighbor < 0:
                continue
            if not is_king:
                dr = dir_dr[d]
                if current_player == BLACK and dr < 0:
                    continue
                if current_player == WHITE and dr > 0:
                    continue
            if squares[neighbor] != EMPTY:
                continue
            out_moves[simple_count, 0] = KIND_SIMPLE
            out_moves[simple_count, 1] = 0
            out_moves[simple_count, 2] = sq
            out_moves[simple_count, 3] = neighbor
            simple_count += 1

    return simple_count


@njit(cache=False)
def apply_move_fast(squares, move):
    """
    Apply move to a fresh copy of squares. Returns the new squares array
    and the new current player (caller flips).
    """
    new_squares = squares.copy()
    kind = move[0]
    from_sq = move[2]
    piece = new_squares[from_sq]

    if kind == KIND_SIMPLE:
        to_sq = move[3]
        new_squares[from_sq] = EMPTY
        new_squares[to_sq] = piece
        if piece == BLACK_PIECE and to_sq >= 28:
            new_squares[to_sq] = BLACK_KING
        elif piece == WHITE_PIECE and to_sq <= 3:
            new_squares[to_sq] = WHITE_KING
    else:
        # Jump: captures are at odd offsets (3, 5, 7, ...), lands at even (4, 6, ...)
        n_caps = move[1]
        new_squares[from_sq] = EMPTY
        final_land = from_sq
        for k in range(n_caps):
            captured = move[3 + 2 * k]
            land = move[4 + 2 * k]
            new_squares[captured] = EMPTY
            final_land = land
        new_squares[final_land] = piece
        if piece == BLACK_PIECE and final_land >= 28:
            new_squares[final_land] = BLACK_KING
        elif piece == WHITE_PIECE and final_land <= 3:
            new_squares[final_land] = WHITE_KING

    return new_squares


def get_legal_moves(squares, current_player):
    """Python wrapper: allocates buffer, calls JIT kernel, returns (N, 17) view."""
    buf = np.zeros((64, MOVE_SLOTS), dtype=np.int8)
    n = get_legal_moves_fast(squares, current_player, NEIGHBORS, JUMP_TARGETS, DIR_DR, buf)
    return buf[:n]


def apply_move(squares, move):
    """Python wrapper for apply_move_fast."""
    return apply_move_fast(squares, move)

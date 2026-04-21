"""Shared @njit move-ordering and killer-move primitives.

Used by both search/fast_minimax_jit.py (1999 CheckersNet) and
search/fast_minimax_jit_anaconda.py (2001 Anaconda). Keeping ordering
in one module avoids drift between the two duplicated alpha-beta bodies.

The killer helpers take the full 3D killers / 2D killers_valid arrays
plus an explicit ply index rather than sliced views — avoids a
Numba cache-reload ABI instability seen when 2D views of 3D arrays
were passed across the module boundary.
"""

import numpy as np
from numba import njit

from checkers.fast_board import MOVE_SLOTS


@njit(cache=True, inline='always')
def _moves_equal(buf, i, move_row):
    for k in range(MOVE_SLOTS):
        if buf[i, k] != move_row[k]:
            return False
    return True


@njit(cache=True, inline='always')
def _swap_rows(buf, a, b):
    if a == b:
        return
    for k in range(MOVE_SLOTS):
        tmp = buf[a, k]
        buf[a, k] = buf[b, k]
        buf[b, k] = tmp


@njit(cache=True, inline='always')
def _killer_matches(killers, ply, slot, buf, i):
    for k in range(MOVE_SLOTS):
        if killers[ply, slot, k] != buf[i, k]:
            return False
    return True


@njit(cache=True, inline='always')
def _killers_slots_equal(killers, ply):
    for k in range(MOVE_SLOTS):
        if killers[ply, 0, k] != killers[ply, 1, k]:
            return False
    return True


@njit(cache=True)
def order_moves(buf, n_moves, tt_move_idx,
                killers, killers_valid, ply,
                canonical_of):
    """Reorder buf[0..n_moves) in place: TT move -> killer1 -> killer2 -> rest.

    On entry buf is in canonical move-gen order and canonical_of[i] == i.
    On exit, buf is reordered and canonical_of[new_idx] gives the original
    canonical index — the caller needs this to translate the chosen best
    index back to canonical form before storing in the TT.

    tt_move_idx is the canonical index (-1 if none). killers has shape
    (n_ply, 2, MOVE_SLOTS) and killers_valid has shape (n_ply, 2) with
    killers_valid[ply, k] == 1 marking populated slots.
    """
    front = np.int64(0)

    if tt_move_idx >= 0 and tt_move_idx < n_moves:
        if tt_move_idx != front:
            _swap_rows(buf, front, tt_move_idx)
            tmp = canonical_of[front]
            canonical_of[front] = canonical_of[tt_move_idx]
            canonical_of[tt_move_idx] = tmp
        front += 1

    if killers_valid[ply, 0] == 1 and front < n_moves:
        found = np.int64(-1)
        for j in range(front, n_moves):
            if _killer_matches(killers, ply, 0, buf, j):
                found = j
                break
        if found >= 0:
            if found != front:
                _swap_rows(buf, front, found)
                tmp = canonical_of[front]
                canonical_of[front] = canonical_of[found]
                canonical_of[found] = tmp
            front += 1

    if killers_valid[ply, 1] == 1 and front < n_moves:
        # Skip if same as killer 1 (both already swapped to front).
        if killers_valid[ply, 0] == 1 and _killers_slots_equal(killers, ply):
            return
        found = np.int64(-1)
        for j in range(front, n_moves):
            if _killer_matches(killers, ply, 1, buf, j):
                found = j
                break
        if found >= 0:
            if found != front:
                _swap_rows(buf, front, found)
                tmp = canonical_of[front]
                canonical_of[front] = canonical_of[found]
                canonical_of[found] = tmp
            front += 1


@njit(cache=True, inline='always')
def update_killers(killers, killers_valid, ply, buf, i):
    """Record buf[i] as the primary killer at `ply`.

    If the move already equals the primary killer, no change. Otherwise
    shift primary -> secondary and install the new move as primary.
    """
    if killers_valid[ply, 0] == 1:
        same = True
        for k in range(MOVE_SLOTS):
            if killers[ply, 0, k] != buf[i, k]:
                same = False
                break
        if same:
            return
        for k in range(MOVE_SLOTS):
            killers[ply, 1, k] = killers[ply, 0, k]
        killers_valid[ply, 1] = np.int8(1)

    for k in range(MOVE_SLOTS):
        killers[ply, 0, k] = buf[i, k]
    killers_valid[ply, 0] = np.int8(1)

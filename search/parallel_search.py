"""
Generator-based multi-game batched minimax.

Runs many alpha-beta searches concurrently. Each search is a Python generator
that yields the boards it needs evaluated; a scheduler collects yielded boards
across every active search, groups by neural network, and fires one forward
pass per network per scheduler tick.

Effective GPU batch per tick ≈ (games where agent X is to move) × (boards
yielded per tick by one search). Combined with within-search two-level
expansion, typical batches climb from ~7 to several hundred, where the
RTX 5060 finally outruns CPU on this 1,743-param network.
"""

import numpy as np
import torch
from collections import defaultdict

from checkers.board import Board
from search.minimax import _encode_board_fast

INF = float("inf")


def ab_search_gen(board: Board, depth: int):
    """
    Generator: run alpha-beta on `board` at the given depth.

    Protocol: each `yield` sends out a list of boards; the scheduler
    must `.send()` back a list of floats of the same length (already
    sign-flipped to the root player's perspective). The generator
    returns `(best_move, best_score)` via StopIteration.value.
    """
    moves = board.get_legal_moves()
    if not moves:
        return (None, -INF)
    if len(moves) == 1:
        return (moves[0], 0.0)

    root_player = board.current_player
    best_move = moves[0]
    best_score = -INF

    # Match MinimaxAgent.search: each root move is searched with a fresh
    # (-INF, INF) window. No cross-root alpha propagation — this keeps the
    # non-best move scores exact, which matters for any tie-breaking that
    # might leak into other logic.
    for m in moves:
        child = board.apply_move(m)
        child_score = yield from _ab(child, depth - 1, -INF, INF, False, root_player)
        if child_score > best_score:
            best_score = child_score
            best_move = m

    return (best_move, best_score)


def _ab(board: Board, depth: int, alpha: float, beta: float,
        maximizing: bool, root_player: int):
    game_over, winner = board.is_game_over()
    if game_over:
        if winner == root_player:
            return INF
        if winner is None:
            return 0.0
        return -INF

    if depth == 0:
        scores = yield [board]
        return scores[0]

    moves = board.get_legal_moves()

    # Two-level expansion: yield all non-terminal grandchildren as one batch.
    if depth == 2:
        children = [board.apply_move(m) for m in moves]
        child_is_max = not maximizing
        child_scores = [None] * len(children)

        nonterm_gcs = []
        child_nt_idx = [[] for _ in children]
        child_terminal_gcs = [[] for _ in children]

        for ci, child in enumerate(children):
            go, w = child.is_game_over()
            if go:
                if w == root_player:
                    child_scores[ci] = INF
                elif w is None:
                    child_scores[ci] = 0.0
                else:
                    child_scores[ci] = -INF
                continue

            for gm in child.get_legal_moves():
                gc = child.apply_move(gm)
                go2, w2 = gc.is_game_over()
                if go2:
                    if w2 == root_player:
                        child_terminal_gcs[ci].append(INF)
                    elif w2 is None:
                        child_terminal_gcs[ci].append(0.0)
                    else:
                        child_terminal_gcs[ci].append(-INF)
                else:
                    child_nt_idx[ci].append(len(nonterm_gcs))
                    nonterm_gcs.append(gc)

        gc_scores = []
        if nonterm_gcs:
            gc_scores = yield nonterm_gcs

        for ci in range(len(children)):
            if child_scores[ci] is not None:
                continue
            sub = [gc_scores[idx] for idx in child_nt_idx[ci]]
            sub.extend(child_terminal_gcs[ci])
            if not sub:
                scores = yield [children[ci]]
                child_scores[ci] = scores[0]
                continue
            child_scores[ci] = max(sub) if child_is_max else min(sub)

        return max(child_scores) if maximizing else min(child_scores)

    # depth == 1 fallback (shallow searches: all children are leaves)
    if depth == 1:
        children = [board.apply_move(m) for m in moves]
        terminal = {}
        nonterm = []
        nonterm_idx = []
        for i, c in enumerate(children):
            go, w = c.is_game_over()
            if go:
                if w == root_player:
                    terminal[i] = INF
                elif w is None:
                    terminal[i] = 0.0
                else:
                    terminal[i] = -INF
            else:
                nonterm_idx.append(i)
                nonterm.append(c)

        batch_scores = []
        if nonterm:
            batch_scores = yield nonterm

        scores = [0.0] * len(children)
        for i, s in terminal.items():
            scores[i] = s
        for k, i in enumerate(nonterm_idx):
            scores[i] = batch_scores[k]

        return max(scores) if maximizing else min(scores)

    # Interior nodes (depth >= 3): standard alpha-beta with recursion
    if maximizing:
        value = -INF
        for m in moves:
            child = board.apply_move(m)
            child_val = yield from _ab(child, depth - 1, alpha, beta, False, root_player)
            value = max(value, child_val)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for m in moves:
            child = board.apply_move(m)
            child_val = yield from _ab(child, depth - 1, alpha, beta, True, root_player)
            value = min(value, child_val)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


class BatchedCheckersNets:
    """
    Stack weights from N CheckersNet instances into (N, ...) GPU tensors so
    that *all* networks can be evaluated in one kernel per layer. Per-example
    weight selection is done with torch.bmm(gathered_weights, input).

    Eliminates the scheduler's per-network forward-pass penalty: on an RTX
    5060 Laptop, one forward at batch 9900 takes ~0.4ms, but 100 separate
    forwards (one per network in the pop) cost ~35ms — pure kernel launch
    overhead. Stacking collapses that to a single launch per layer.
    """

    def __init__(self, networks, device: torch.device):
        self.device = device
        self.n_networks = len(networks)
        # Stack per-layer weights. Each network has the same shape.
        self.W1 = torch.stack([n.fc1.weight.data for n in networks]).to(device)  # (N, 40, 32)
        self.b1 = torch.stack([n.fc1.bias.data   for n in networks]).to(device)  # (N, 40)
        self.W2 = torch.stack([n.fc2.weight.data for n in networks]).to(device)  # (N, 10, 40)
        self.b2 = torch.stack([n.fc2.bias.data   for n in networks]).to(device)  # (N, 10)
        self.W3 = torch.stack([n.fc3.weight.data for n in networks]).to(device)  # (N, 1, 10)
        self.b3 = torch.stack([n.fc3.bias.data   for n in networks]).to(device)  # (N, 1)
        self.piece_diff = torch.stack(
            [n.piece_diff_weight.data.reshape(1) for n in networks]
        ).to(device).squeeze(-1)  # (N,)

    def forward(self, x: torch.Tensor, net_idx: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, 32) float tensor on self.device
        net_idx: (B,)    long tensor of population indices on self.device
        returns: (B,)    float tensor of evaluation scores in [-1, 1]
        """
        W1 = self.W1.index_select(0, net_idx)  # (B, 40, 32)
        b1 = self.b1.index_select(0, net_idx)  # (B, 40)
        W2 = self.W2.index_select(0, net_idx)  # (B, 10, 40)
        b2 = self.b2.index_select(0, net_idx)  # (B, 10)
        W3 = self.W3.index_select(0, net_idx)  # (B, 1, 10)
        b3 = self.b3.index_select(0, net_idx)  # (B, 1)
        pd = self.piece_diff.index_select(0, net_idx)  # (B,)

        h1 = torch.tanh(torch.bmm(W1, x.unsqueeze(-1)).squeeze(-1) + b1)
        h2 = torch.tanh(torch.bmm(W2, h1.unsqueeze(-1)).squeeze(-1) + b2)
        out = torch.bmm(W3, h2.unsqueeze(-1)).squeeze(-1) + b3  # (B, 1)
        bypass = x.sum(dim=-1, keepdim=True) * pd.unsqueeze(-1)
        return torch.tanh(out + bypass).squeeze(-1)


class ParallelSearchScheduler:
    """
    Drives many ab_search_gen instances in lockstep, batching their
    leaf eval requests across games. Batches are grouped by network
    because different agents in a tournament have different weights.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.profile = False
        self.prof = {"encode": 0.0, "h2d": 0.0, "fwd": 0.0, "d2h": 0.0,
                     "resume": 0.0, "ticks": 0, "boards": 0}

    def run_batched(self, boards, pop_indices, king_weights, depths,
                    root_players, batched_nets: "BatchedCheckersNets"):
        """
        Same as run(), but every tick does ONE forward pass through
        batched_nets instead of one per network. Requires all networks in the
        population to share architecture (true for CheckersNet). Much faster
        on GPU when pop is large and per-network batches are small.

        boards, pop_indices, king_weights, depths, root_players:
            parallel lists of length N (one per request)
        batched_nets: BatchedCheckersNets built from the full population
        """
        n = len(boards)
        if n == 0:
            return []

        gens = [ab_search_gen(b, d) for b, d in zip(boards, depths)]
        pending = [None] * n
        results = [None] * n

        for i, g in enumerate(gens):
            try:
                pending[i] = g.send(None)
            except StopIteration as e:
                results[i] = e.value
                pending[i] = None

        import time as _time
        prof = self.prof if self.profile else None

        while any(p is not None for p in pending):
            t0 = _time.perf_counter() if prof else 0.0
            all_boards = []
            slots = []
            for i, p in enumerate(pending):
                if p is None:
                    continue
                start = len(all_boards)
                all_boards.extend(p)
                slots.append((i, start, len(p)))

            total = len(all_boards)
            if total == 0:
                break

            stacked = np.empty((total, 32), dtype=np.float32)
            signs = np.empty(total, dtype=np.float32)
            idx = np.empty(total, dtype=np.int64)

            for (gi, start, count) in slots:
                kw = king_weights[gi]
                rp = root_players[gi]
                pi = pop_indices[gi]
                for j in range(count):
                    b = all_boards[start + j]
                    stacked[start + j] = _encode_board_fast(b, kw)
                    signs[start + j] = -1.0 if b.current_player != rp else 1.0
                    idx[start + j] = pi

            if prof is not None:
                t_encode = _time.perf_counter()
                prof["encode"] += t_encode - t0

            x = torch.from_numpy(stacked).to(self.device, non_blocking=True)
            net_idx = torch.from_numpy(idx).to(self.device, non_blocking=True)
            signs_t = torch.from_numpy(signs).to(self.device, non_blocking=True)
            if prof is not None:
                torch.cuda.synchronize()
                t_h2d = _time.perf_counter()
                prof["h2d"] += t_h2d - t_encode

            out = batched_nets.forward(x, net_idx) * signs_t
            if prof is not None:
                torch.cuda.synchronize()
                t_fwd = _time.perf_counter()
                prof["fwd"] += t_fwd - t_h2d

            all_scores = out.cpu().tolist()
            if prof is not None:
                t_d2h = _time.perf_counter()
                prof["d2h"] += t_d2h - t_fwd

            for (gi, start, count) in slots:
                scores_for_gen = all_scores[start:start + count]
                try:
                    pending[gi] = gens[gi].send(scores_for_gen)
                except StopIteration as e:
                    results[gi] = e.value
                    pending[gi] = None

            if prof is not None:
                t_resume = _time.perf_counter()
                prof["resume"] += t_resume - t_d2h
                prof["ticks"] += 1
                prof["boards"] += total

        return results

    def run(self, requests):
        """
        requests: list of (board, network, king_weight, depth, root_player)
        Returns: list of (best_move, best_score)
        """
        n = len(requests)
        if n == 0:
            return []

        boards = [r[0] for r in requests]
        networks = [r[1] for r in requests]
        king_weights = [r[2] for r in requests]
        depths = [r[3] for r in requests]
        root_players = [r[4] for r in requests]

        gens = [ab_search_gen(b, d) for b, d in zip(boards, depths)]
        pending = [None] * n
        results = [None] * n

        for i, g in enumerate(gens):
            try:
                pending[i] = g.send(None)
            except StopIteration as e:
                results[i] = e.value
                pending[i] = None

        while any(p is not None for p in pending):
            # Group pending yields by network (id, since networks are mutable
            # nn.Module objects and id() gives a stable handle per generation)
            flat_per_net: dict = defaultdict(list)
            slots_per_net: dict = defaultdict(list)  # [(gen_idx, start, count)]

            for i, p in enumerate(pending):
                if p is None:
                    continue
                nid = id(networks[i])
                start = len(flat_per_net[nid])
                flat_per_net[nid].extend(p)
                slots_per_net[nid].append((i, start, len(p)))

            scores_per_gen = [None] * n
            for nid, boards_flat in flat_per_net.items():
                slots = slots_per_net[nid]
                # All slots for this nid share the same network object
                net = networks[slots[0][0]]
                total = len(boards_flat)

                stacked = np.empty((total, 32), dtype=np.float32)
                signs = np.empty(total, dtype=np.float32)
                for gi, start, count in slots:
                    kw = king_weights[gi]
                    rp = root_players[gi]
                    for j in range(count):
                        b = boards_flat[start + j]
                        stacked[start + j] = _encode_board_fast(b, kw)
                        signs[start + j] = -1.0 if b.current_player != rp else 1.0

                tensor = torch.from_numpy(stacked).to(self.device, non_blocking=True)
                out = net(tensor).squeeze(-1)
                signs_t = torch.from_numpy(signs).to(self.device, non_blocking=True)
                out = out * signs_t
                all_scores = out.cpu().tolist()

                for gi, start, count in slots:
                    scores_per_gen[gi] = all_scores[start:start + count]

            for i in range(n):
                if pending[i] is None:
                    continue
                try:
                    pending[i] = gens[i].send(scores_per_gen[i])
                except StopIteration as e:
                    results[i] = e.value
                    pending[i] = None

        return results

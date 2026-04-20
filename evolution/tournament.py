"""
Tournament play: pit individuals against each other to determine fitness.

GPU OPTIMIZATION: We pre-build all MinimaxAgent objects at the start of
each generation (loading networks onto GPU once), then reuse them across
all games. This avoids repeated CPU→GPU weight transfers.
"""

import random
import numpy as np
import torch
from evolution.strategy import Individual
from evolution.population import Population
from checkers.board import Board, BLACK, WHITE
from checkers.game import play_game, GameResult
from search.minimax import MinimaxAgent
from search.parallel_search import ParallelSearchScheduler
from config import SearchConfig, EvolutionConfig


def _build_fast_agents(population: Population, search_config: SearchConfig):
    """Pre-build numpy/Numba agents for CPU tournament play.

    Numba is imported lazily so environments that can't load the numba
    DLLs (e.g. WDAC-blocked Windows) can still use the torch-based paths.
    """
    from search.fast_minimax_jit import FastAgentJit
    return [FastAgentJit(ind.weights, search_config.depth)
            for ind in population.individuals]


# ---------- multiprocessing worker ----------
#
# These are module-level so they're picklable. Each worker process re-imports
# this module, warms JIT on first call, and then chews through games.

def _mp_worker_init():
    """Warm Numba JIT inside a worker process so the first game isn't slow."""
    from search.fast_minimax_jit import warmup
    warmup()


def _mp_play_one(task):
    """
    Play a single game inside a worker. Task is a tuple:
        (pair_id, black_weights, white_weights, depth, max_moves)
    Returns (pair_id, winner_int, moves, reason) — a flat picklable tuple.
    winner_int: 1=black, -1=white, 0=draw.
    """
    from search.fast_minimax_jit import FastAgentJit
    pair_id, black_w, white_w, depth, max_moves = task
    black = FastAgentJit(black_w, depth)
    white = FastAgentJit(white_w, depth)
    result = play_game(black, white, max_moves=max_moves)
    w = result.winner
    winner_int = 0 if w is None else int(w)
    return (pair_id, winner_int, result.moves, result.reason)


def _build_agents(population: Population, search_config: SearchConfig,
                  device: torch.device) -> list[MinimaxAgent]:
    """Pre-build all agents with networks on GPU. Called once per generation."""
    agents = []
    for ind in population.individuals:
        net = population.get_network(ind, device)
        agent = MinimaxAgent(net, search_config, device)
        agents.append(agent)
    return agents


def random_pairing_tournament(
    population: Population,
    search_config: SearchConfig = SearchConfig(),
    evolution_config: EvolutionConfig = EvolutionConfig(),
    device: str = "cpu",
    verbose: bool = False,
    pool=None,
    max_moves: int = 200,
):
    """
    Each individual plays N games against randomly selected opponents
    (the paper-faithful tournament style from Fogel 1999/2001).

    On GPU, runs serially using the torch-based MinimaxAgent.
    On CPU, uses the numpy/Numba FastAgent path. If a multiprocessing
    Pool is provided, games are distributed across worker processes.
    """
    if isinstance(device, str):
        device = torch.device(device)

    pop = population.individuals
    n_games = evolution_config.games_per_individual
    population.reset_fitness()

    # Build pairings up front so CPU/GPU/pool paths share the same schedule.
    # Each entry: (player_idx, opponent_idx, player_is_black).
    pairings = []
    for i in range(len(pop)):
        opponents_idx = random.sample(
            [j for j in range(len(pop)) if j != i],
            min(n_games, len(pop) - 1)
        )
        for opp_idx in opponents_idx:
            pairings.append((i, opp_idx, random.random() < 0.5))

    # ── GPU path: serial MinimaxAgent ──
    if device.type != "cpu":
        agents = _build_agents(population, search_config, device)
        for i, opp_idx, i_is_black in pairings:
            if i_is_black:
                result = play_game(agents[i], agents[opp_idx], max_moves=max_moves)
                _update_fitness(pop[i], pop[opp_idx], result,
                                player_is_black=True, config=evolution_config)
            else:
                result = play_game(agents[opp_idx], agents[i], max_moves=max_moves)
                _update_fitness(pop[i], pop[opp_idx], result,
                                player_is_black=False, config=evolution_config)
            if verbose:
                winner_str = {1: "Black", -1: "White", None: "Draw"}[result.winner]
                print(f"  Game: {i} vs {opp_idx} → {winner_str} "
                      f"({result.moves} moves, {result.reason})")
        return

    # ── CPU path: serial FastAgentJit fallback ──
    if pool is None:
        agents = _build_fast_agents(population, search_config)
        for i, opp_idx, i_is_black in pairings:
            if i_is_black:
                result = play_game(agents[i], agents[opp_idx], max_moves=max_moves)
                _update_fitness(pop[i], pop[opp_idx], result,
                                player_is_black=True, config=evolution_config)
            else:
                result = play_game(agents[opp_idx], agents[i], max_moves=max_moves)
                _update_fitness(pop[i], pop[opp_idx], result,
                                player_is_black=False, config=evolution_config)
        return

    # ── CPU multiprocess: distribute games across worker pool ──
    tasks = []
    pair_map = []  # pair_id -> (player_idx, opponent_idx, player_is_black)
    depth = search_config.depth
    for pair_id, (i, opp_idx, i_is_black) in enumerate(pairings):
        if i_is_black:
            black_w, white_w = pop[i].weights, pop[opp_idx].weights
        else:
            black_w, white_w = pop[opp_idx].weights, pop[i].weights
        pair_map.append((i, opp_idx, i_is_black))
        tasks.append((pair_id, black_w, white_w, depth, max_moves))

    for pair_id, winner_int, _n_moves, _reason in pool.imap_unordered(
        _mp_play_one, tasks, chunksize=2
    ):
        i, opp_idx, i_is_black = pair_map[pair_id]
        player = pop[i]
        opponent = pop[opp_idx]
        player.games_played += 1
        opponent.games_played += 1

        if winner_int == 0:
            player.fitness += evolution_config.draw_score
            opponent.fitness += evolution_config.draw_score
            player.draws += 1
            opponent.draws += 1
        elif (winner_int == BLACK and i_is_black) or (winner_int == WHITE and not i_is_black):
            player.fitness += evolution_config.win_score
            opponent.fitness += evolution_config.loss_score
            player.wins += 1
            opponent.losses += 1
        else:
            player.fitness += evolution_config.loss_score
            opponent.fitness += evolution_config.win_score
            player.losses += 1
            opponent.wins += 1


def round_robin_tournament(
    population: Population,
    search_config: SearchConfig = SearchConfig(),
    evolution_config: EvolutionConfig = EvolutionConfig(),
    device: str = "cpu",
    verbose: bool = False,
    pool=None,
    max_moves: int = 200,
):
    """
    Full round-robin: every individual plays every other individual twice
    (once as each color). More expensive but more accurate fitness signal.

    On CPU, uses the numpy/Numba FastAgent path (no PyTorch in the hot loop).
    If a multiprocessing Pool is provided, games are distributed across
    worker processes. On GPU, uses the batched MinimaxAgent path.
    """
    if isinstance(device, str):
        device = torch.device(device)

    pop = population.individuals
    population.reset_fitness()

    # ── GPU path: unchanged ──
    if device.type != "cpu":
        agents = _build_agents(population, search_config, device)
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                result1 = play_game(agents[i], agents[j], max_moves=max_moves)
                _update_fitness(pop[i], pop[j], result1,
                                player_is_black=True, config=evolution_config)
                result2 = play_game(agents[j], agents[i], max_moves=max_moves)
                _update_fitness(pop[j], pop[i], result2,
                                player_is_black=True, config=evolution_config)
        return

    # ── CPU path: parallel over Pool, or serial fallback ──
    if pool is None:
        agents = _build_fast_agents(population, search_config)
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                result1 = play_game(agents[i], agents[j], max_moves=max_moves)
                _update_fitness(pop[i], pop[j], result1,
                                player_is_black=True, config=evolution_config)
                result2 = play_game(agents[j], agents[i], max_moves=max_moves)
                _update_fitness(pop[j], pop[i], result2,
                                player_is_black=True, config=evolution_config)
        return

    # Build picklable task list: (pair_id, black_w, white_w, depth, max_moves)
    tasks = []
    pair_map = []  # pair_id -> (black_idx, white_idx)
    depth = search_config.depth
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            pair_map.append((i, j))
            tasks.append((len(pair_map) - 1,
                          pop[i].weights, pop[j].weights,
                          depth, max_moves))
            pair_map.append((j, i))
            tasks.append((len(pair_map) - 1,
                          pop[j].weights, pop[i].weights,
                          depth, max_moves))

    # Use imap_unordered for slight throughput gain and to surface errors early.
    for pair_id, winner_int, n_moves, reason in pool.imap_unordered(
        _mp_play_one, tasks, chunksize=2
    ):
        black_idx, white_idx = pair_map[pair_id]
        black = pop[black_idx]
        white = pop[white_idx]
        black.games_played += 1
        white.games_played += 1
        if winner_int == 0:
            black.fitness += evolution_config.draw_score
            white.fitness += evolution_config.draw_score
            black.draws += 1
            white.draws += 1
        elif winner_int == BLACK:
            black.fitness += evolution_config.win_score
            white.fitness += evolution_config.loss_score
            black.wins += 1
            white.losses += 1
        else:
            white.fitness += evolution_config.win_score
            black.fitness += evolution_config.loss_score
            white.wins += 1
            black.losses += 1


def parallel_round_robin_tournament(
    population: Population,
    search_config: SearchConfig = SearchConfig(),
    evolution_config: EvolutionConfig = EvolutionConfig(),
    device: str = "cuda",
    verbose: bool = False,
    max_moves: int = 200,
):
    """
    Round-robin tournament that plays all games in lockstep on GPU.

    At each tick, every still-running game contributes its current
    minimax request to a shared batch; the scheduler runs one forward
    pass per network (not per game) and distributes results back.
    """
    if isinstance(device, str):
        device = torch.device(device)

    pop = population.individuals
    population.reset_fitness()

    networks = [population.get_network(ind, device) for ind in pop]
    for net in networks:
        net.eval()
    king_weights = [float(net.king_weight.item()) for net in networks]

    scheduler = ParallelSearchScheduler(device)
    # Stack all population weights onto GPU once per gen — lets the scheduler
    # do one batched forward per tick instead of one forward per network.
    from search.parallel_search import BatchedCheckersNets
    batched_nets = BatchedCheckersNets(networks, device)
    depth = search_config.depth

    # Build all pairings: every ordered pair (i black, j white), i != j.
    pairings = []
    n = len(pop)
    for i in range(n):
        for j in range(i + 1, n):
            pairings.append((i, j))
            pairings.append((j, i))

    # Active game state
    games = []
    for black_idx, white_idx in pairings:
        games.append({
            "board": Board(),
            "black_idx": black_idx,
            "white_idx": white_idx,
            "move_count": 0,
            "state_counts": {},
            "winner": "pending",   # int, None (draw), or "pending"
        })

    while True:
        active = [g for g in games if g["winner"] == "pending"]
        if not active:
            break

        req_boards = []
        req_pop_idx = []
        req_king_w = []
        req_depths = []
        req_roots = []
        request_games = []
        for g in active:
            board = g["board"]

            game_over, winner = board.is_game_over()
            if game_over:
                g["winner"] = winner
                continue

            state_key = board.squares.tobytes() + bytes([board.current_player + 2])
            g["state_counts"][state_key] = g["state_counts"].get(state_key, 0) + 1
            if g["state_counts"][state_key] >= 3:
                g["winner"] = None  # draw by repetition
                continue

            if g["move_count"] >= max_moves:
                g["winner"] = None  # draw by move cap
                continue

            if board.current_player == BLACK:
                agent_idx = g["black_idx"]
            else:
                agent_idx = g["white_idx"]

            req_boards.append(board)
            req_pop_idx.append(agent_idx)
            req_king_w.append(king_weights[agent_idx])
            req_depths.append(depth)
            req_roots.append(board.current_player)
            request_games.append(g)

        if not req_boards:
            continue

        results = scheduler.run_batched(
            req_boards, req_pop_idx, req_king_w, req_depths, req_roots, batched_nets,
        )

        for g, (move, _score) in zip(request_games, results):
            if move is None:
                # No legal moves — current player loses
                g["winner"] = -g["board"].current_player
                continue
            g["board"] = g["board"].apply_move(move)
            g["move_count"] += 1

    # Score games
    for g in games:
        black = pop[g["black_idx"]]
        white = pop[g["white_idx"]]
        black.games_played += 1
        white.games_played += 1

        w = g["winner"]
        if w is None:
            black.fitness += evolution_config.draw_score
            white.fitness += evolution_config.draw_score
            black.draws += 1
            white.draws += 1
        elif w == BLACK:
            black.fitness += evolution_config.win_score
            white.fitness += evolution_config.loss_score
            black.wins += 1
            white.losses += 1
        else:
            white.fitness += evolution_config.win_score
            black.fitness += evolution_config.loss_score
            white.wins += 1
            black.losses += 1

        if verbose:
            winner_str = {BLACK: "Black", WHITE: "White", None: "Draw"}[w]
            print(f"  {g['black_idx']} vs {g['white_idx']} → {winner_str} ({g['move_count']} moves)")


def _update_fitness(
    player: Individual,
    opponent: Individual,
    result: GameResult,
    player_is_black: bool,
    config: EvolutionConfig,
):
    """Update fitness scores for both players based on game result."""
    player.games_played += 1
    opponent.games_played += 1

    if result.winner is None:
        player.fitness += config.draw_score
        opponent.fitness += config.draw_score
        player.draws += 1
        opponent.draws += 1
    elif (result.winner == 1 and player_is_black) or (result.winner == -1 and not player_is_black):
        player.fitness += config.win_score
        opponent.fitness += config.loss_score
        player.wins += 1
        opponent.losses += 1
    else:
        player.fitness += config.loss_score
        opponent.fitness += config.win_score
        player.losses += 1
        opponent.wins += 1

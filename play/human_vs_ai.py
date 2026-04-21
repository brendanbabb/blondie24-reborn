"""
Play against the best evolved network in the terminal.

Usage:
    python -m play.human_vs_ai --checkpoint checkpoints/best_gen0450.pt
    python -m play.human_vs_ai --checkpoint checkpoints/best_gen0450.pt --depth 6
    python -m play.human_vs_ai --checkpoint checkpoints/best_gen0450.pt --no-color
"""

import argparse
import os
import time
import numpy as np
import torch
from checkers.board import (
    Board, BLACK, WHITE, EMPTY,
    BLACK_PIECE, BLACK_KING, WHITE_PIECE, WHITE_KING,
)
from neural.network import make_network, architecture_from_weight_count
from neural.encoding import encode_board
from search.minimax import make_agent
from config import SearchConfig, NetworkConfig


RESIGN = "RESIGN"
DRAW_OFFER = "DRAW_OFFER"


# ANSI escape sequences. Cell cells stay 2 chars wide — codes add no visual width.
class C:
    RESET      = "\033[0m"
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    BLACK_FG   = "\033[96m"      # bright cyan for black pieces
    WHITE_FG   = "\033[91m"      # bright red for white pieces
    LABEL      = "\033[93m"      # yellow labels
    MOVED_BG   = "\033[43;30m"   # yellow bg, black fg — from/to highlight
    CAPTURED_BG = "\033[41;97m"  # red bg, white fg — captured squares


USE_COLOR = True


def _paint(s: str, *codes: str) -> str:
    if not USE_COLOR or not codes:
        return s
    return "".join(codes) + s + C.RESET


def _enable_ansi_on_windows():
    """Activate VT100 processing on Windows cmd/PowerShell (no-op elsewhere)."""
    if os.name == "nt":
        os.system("")


def _describe_move(move) -> tuple[int, int, list[int], str]:
    """Return (from_sq, to_sq, captured_squares, path_string)."""
    if isinstance(move, tuple):
        from_sq, to_sq = move
        return from_sq, to_sq, [], f"{from_sq} -> {to_sq}"
    # Jump sequence: [from, cap1, land1, cap2, land2, ...]
    captured = [move[i] for i in range(1, len(move), 2)]
    path = str(move[0])
    for j in range(2, len(move), 2):
        path += f" -> {move[j]}"
    return move[0], move[-1], captured, path


def _render_board(
    board: Board,
    from_sq: int | None = None,
    to_sq: int | None = None,
    captured: list[int] | None = None,
) -> str:
    """Render the board with optional last-move highlights. Cells are 2 chars wide."""
    captured = captured or []
    pieces = {
        BLACK_PIECE: ("b", C.BLACK_FG),
        BLACK_KING:  ("B", C.BLACK_FG + C.BOLD),
        WHITE_PIECE: ("w", C.WHITE_FG),
        WHITE_KING:  ("W", C.WHITE_FG + C.BOLD),
    }

    lines = []
    header = "     " + "  ".join(_paint(str(c), C.LABEL) for c in range(8))
    lines.append(header)
    lines.append("    " + "-" * 25)

    for row in range(8):
        cells = []
        for col in range(8):
            if (row + col) % 2 == 1:
                sq = row * 4 + col // 2
                piece = board.squares[sq]
                if piece == EMPTY:
                    if sq == from_sq:
                        cells.append(_paint(" .", C.MOVED_BG))  # moved-from
                    elif sq in captured:
                        cells.append(_paint(" x", C.CAPTURED_BG))
                    else:
                        cells.append(_paint(f"{sq:2d}", C.DIM))
                else:
                    sym, color = pieces[piece]
                    if sq == to_sq:
                        cells.append(_paint(f" {sym}", C.MOVED_BG))
                    else:
                        cells.append(_paint(f" {sym}", color))
            else:
                cells.append("  ")
        row_label = _paint(str(row), C.LABEL)
        lines.append(f"  {row} | " + " ".join(cells))
    return "\n".join(lines)


def _format_move_option(i: int, move) -> str:
    from_sq, to_sq, captured, path = _describe_move(move)
    if captured:
        return f"  [{i:2d}] {path}   captures {captured}"
    return f"  [{i:2d}] {path}"


def _header(move_num: int, board: Board, human_color: str) -> str:
    black_n, white_n = board.piece_count()
    you = "Black" if human_color == "black" else "White"
    you_n = black_n if human_color == "black" else white_n
    blondie_n = white_n if human_color == "black" else black_n
    return (
        f"{'=' * 60}\n"
        f"  Move #{move_num}  |  You ({you}): {you_n} pcs"
        f"  |  Blondie: {blondie_n} pcs\n"
        f"{'=' * 60}"
    )


def human_move(board: Board) -> object:
    """Prompt the human for a move. Assumes the board/header were already printed."""
    moves = board.get_legal_moves()
    forced_jump = moves and not isinstance(moves[0], tuple)
    print()
    if forced_jump:
        print(_paint("  Capture is mandatory.", C.LABEL))
    print(_paint("  Your legal moves:", C.LABEL + C.BOLD))
    for i, m in enumerate(moves):
        print(_format_move_option(i, m))

    while True:
        try:
            choice = input(
                f"\n  Enter move number (0-{len(moves)-1}), 'draw', or 'resign': "
            ).strip().lower()
            if choice in ("resign", "r"):
                return RESIGN
            if choice in ("draw", "d"):
                return DRAW_OFFER
            idx = int(choice)
            if 0 <= idx < len(moves):
                return moves[idx]
            print(_paint("  Out of range. Try again.", C.WHITE_FG))
        except ValueError:
            print(_paint("  Not a number. Try again.", C.WHITE_FG))
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n  Game aborted.")


def _blondie_accepts_draw(network, board: Board) -> tuple[bool, float]:
    """
    Decide whether Blondie accepts a draw offer at the current position.

    Called when it's the human's turn, so the encoding is from the human's
    perspective. Blondie accepts unless it sees itself as clearly winning
    (its own eval > 0.3, equivalently human-perspective eval < -0.3).
    """
    king_w = float(network.king_weight.item())
    with torch.no_grad():
        human_eval = network.evaluate_board(encode_board(board, king_w))
    blondie_eval = -human_eval
    return blondie_eval <= 0.3, blondie_eval


def load_network(checkpoint_path: str, device: str = "cpu"):
    """Load a network from a checkpoint, auto-detecting its architecture."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weights = np.asarray(data["weights"])
    arch = data.get("architecture")
    if arch is None:
        arch = data.get("config", {}).get("network", {}).get("architecture")
    if arch is None:
        arch = architecture_from_weight_count(len(weights))
    net = make_network(NetworkConfig(architecture=arch))
    net.set_weight_vector(weights)
    return net.to(device)


def play(checkpoint: str, depth: int, device: str, human_color: str, max_moves: int = 200):
    network = load_network(checkpoint, device)
    ai_agent = make_agent(network, SearchConfig(depth=depth), device)

    board = Board()
    last_from, last_to, last_captured = None, None, []
    last_mover = None       # "You" or "Blondie"
    last_path = None
    last_think_s = None
    move_num = 0
    history: list[str] = []
    state_counts: dict[bytes, int] = {}

    arch_label = {
        "checkersnet-1999": "1999 CheckersNet (1,743 weights)",
        "anaconda-2001":    "2001 Anaconda (5,048 weights)",
    }.get(getattr(network, "config", NetworkConfig()).architecture, "unknown arch")
    print(_paint("\n" + "=" * 60, C.LABEL))
    print(_paint(f"  BLONDIE24 REBORN  |  checkpoint: {os.path.basename(checkpoint)}", C.LABEL + C.BOLD))
    print(_paint(f"  Architecture: {arch_label}", C.LABEL))
    print(_paint(f"  You are {human_color.upper()}  |  AI depth: {depth}", C.LABEL))
    print(_paint("  b/B = black (Blondie's pieces if you're white)   w/W = white", C.DIM))
    print(_paint("  Capitals are kings.  Numbers mark empty playable squares.", C.DIM))
    print(_paint("  Type 'draw' or 'resign' at the move prompt to end the game.", C.DIM))
    print(_paint("=" * 60, C.LABEL))

    winner = None
    reason = "max moves"

    while move_num < max_moves:
        # Game-over check
        game_over, w = board.is_game_over()
        if game_over:
            winner = w
            reason = "no moves"
            break

        # Threefold repetition
        key = board.squares.tobytes() + bytes([board.current_player + 2])
        state_counts[key] = state_counts.get(key, 0) + 1
        if state_counts[key] >= 3:
            winner = None
            reason = "threefold repetition"
            break

        human_turn = (
            (board.current_player == BLACK and human_color == "black") or
            (board.current_player == WHITE and human_color == "white")
        )

        print("\n" + _header(move_num + 1, board, human_color))

        # Report the previous move, if any
        if last_mover is not None:
            mover_color = C.BLACK_FG if last_mover == "Blondie" else C.WHITE_FG
            extra = f" (captured {last_captured})" if last_captured else ""
            timing = f"   [{last_think_s:.1f}s thinking]" if last_think_s is not None else ""
            print(_paint(f"  {last_mover}: {last_path}{extra}{timing}", mover_color + C.BOLD))
        else:
            print(_paint("  (opening position)", C.DIM))

        print()
        print(_render_board(board, from_sq=last_from, to_sq=last_to, captured=last_captured))

        if human_turn:
            move = human_move(board)
            if move == RESIGN:
                winner = WHITE if human_color == "black" else BLACK
                reason = "you resigned"
                break
            if move == DRAW_OFFER:
                accept, bl_eval = _blondie_accepts_draw(network, board)
                if accept:
                    print(_paint(
                        f"\n  Blondie accepts the draw.  [eval {bl_eval:+.2f}]",
                        C.BLACK_FG + C.BOLD,
                    ))
                    winner = None
                    reason = "draw agreed"
                    break
                print(_paint(
                    f"\n  Blondie declines — it likes its position.  [eval {bl_eval:+.2f}]",
                    C.BLACK_FG + C.BOLD,
                ))
                continue
            mover = "You"
            last_think_s = None
        else:
            print(_paint("\n  Blondie is thinking...", C.BLACK_FG + C.BOLD), flush=True)
            t0 = time.time()
            move = ai_agent(board)
            last_think_s = time.time() - t0
            mover = "Blondie"

        from_sq, to_sq, captured, path = _describe_move(move)
        history.append(f"{mover[0]}:{path}")
        board = board.apply_move(move)
        last_from, last_to, last_captured = from_sq, to_sq, captured
        last_mover, last_path = mover, path
        move_num += 1

    # Final board + result
    print("\n" + _header(move_num, board, human_color))
    if last_mover:
        mover_color = C.BLACK_FG if last_mover == "Blondie" else C.WHITE_FG
        extra = f" (captured {last_captured})" if last_captured else ""
        print(_paint(f"  {last_mover}: {last_path}{extra}", mover_color + C.BOLD))
    print()
    print(_render_board(board, from_sq=last_from, to_sq=last_to, captured=last_captured))

    print(_paint("\n" + "=" * 60, C.LABEL))
    if winner is None:
        print(_paint("  DRAW", C.LABEL + C.BOLD))
    elif (winner == BLACK and human_color == "black") or (winner == WHITE and human_color == "white"):
        print(_paint("  YOU WIN!", C.LABEL + C.BOLD))
    else:
        print(_paint("  BLONDIE WINS", C.BLACK_FG + C.BOLD))
    print(_paint(f"  {move_num} moves  |  ended by: {reason}", C.DIM))
    print(_paint("=" * 60, C.LABEL))


def main():
    parser = argparse.ArgumentParser(description="Play checkers against Blondie24 Reborn")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best network checkpoint")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for AI")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--color", type=str, default="white", choices=["black", "white"],
                        help="Your color (black moves first)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    global USE_COLOR
    USE_COLOR = not args.no_color
    if USE_COLOR:
        _enable_ansi_on_windows()

    play(args.checkpoint, args.depth, args.device, args.color)


if __name__ == "__main__":
    main()

"""
Export a trained checkpoint's flat weight vector as a little-endian float32
binary for the browser demo to load.

Works for both the 1999 CheckersNet (1,743 weights) and the 2001 Anaconda
(5,048 weights). The JS side detects which architecture to use based on the
file's byte length — no format tag needed.

Usage:
    python scripts/export_weights_to_js.py <checkpoint.pt> <output.bin> [--arch auto|1999|2001]

    # Also emits a small JSON fixture file with (board, score) pairs the JS
    # tests can cross-check against:
    python scripts/export_weights_to_js.py ... --fixtures <fixtures.json>

    # If you don't have a trained checkpoint yet and just want the pipeline
    # end-to-end:
    python scripts/export_weights_to_js.py --init-random 2001 docs/weights/anaconda.bin
        --fixtures docs/weights/anaconda-fixtures.json
"""

import argparse
import json
import sys
import struct
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neural.network import CheckersNet  # noqa: E402
from neural.anaconda_network import AnacondaNet  # noqa: E402
from config import NetworkConfig  # noqa: E402


N_1999 = 1743
N_2001 = 5048


def load_checkpoint_weights(path: Path) -> np.ndarray:
    """Load a .pt checkpoint and return the flat weight vector."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Two supported shapes:
    #   1) {"weights": tensor, "sigmas": tensor}                      (best_genN.pt)
    #   2) list of such dicts (population_genN.pt) — we take [0]
    if isinstance(ckpt, list):
        ckpt = ckpt[0]
    if "weights" not in ckpt:
        raise SystemExit(f"Unexpected checkpoint format in {path}: keys={list(ckpt.keys())}")
    w = ckpt["weights"]
    return w.detach().cpu().numpy().astype(np.float32)


def init_random_weights(arch: str, seed: int = 1729) -> np.ndarray:
    """Produce a deterministic random initialization so JS tests can pin
    down exact values without needing a trained checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arch == "1999":
        net = CheckersNet(NetworkConfig())
        for p in net.parameters():
            p.data.normal_(0.0, 0.10)
        net.king_weight.data.fill_(2.0)
        return net.get_weight_vector().astype(np.float32)
    elif arch == "2001":
        net = AnacondaNet(NetworkConfig())
        vec = np.random.default_rng(seed).normal(0.0, 0.10, size=N_2001).astype(np.float32)
        # king weight slot is the last one
        vec[-1] = 2.0
        net.set_weight_vector(vec)
        return net.get_weight_vector().astype(np.float32)
    else:
        raise SystemExit(f"unknown arch: {arch}")


def write_bin(path: Path, weights: np.ndarray):
    arr = np.ascontiguousarray(weights, dtype=np.float32)
    if sys.byteorder != "little":
        arr = arr.byteswap().view(arr.dtype)  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.tobytes())
    print(f"wrote {path} ({len(arr)} weights, {path.stat().st_size} bytes)")


def make_fixtures(arch: str, weights: np.ndarray) -> list:
    """Run the Python network on a set of representative boards and record
    (board, score) pairs. JS reads these and asserts equivalence."""
    if arch == "1999":
        net = CheckersNet(NetworkConfig())
    else:
        net = AnacondaNet(NetworkConfig())
    net.set_weight_vector(weights)

    fixtures = []

    def record(label, squares_32, current_player):
        # Board encoding follows the ±1 / ±K convention from the current
        # side-to-move's perspective — same encoder as the JS side uses.
        king_weight = float(net.king_weight.data.item())
        x = np.zeros(32, dtype=np.float32)
        for i, p in enumerate(squares_32):
            if p == 0:
                continue
            is_king = (abs(p) == 2)
            mag = king_weight if is_king else 1.0
            x[i] = mag if (p * current_player) > 0 else -mag
        with torch.no_grad():
            score = net.forward(torch.from_numpy(x)).item()
        fixtures.append({
            "label": label,
            "squares": list(int(p) for p in squares_32),
            "currentPlayer": int(current_player),
            "score": float(score),
        })

    # 1. Starting position, black to move
    start = [0] * 32
    for i in range(12):
        start[i] = 1  # black men
    for i in range(20, 32):
        start[i] = -1  # white men
    record("starting-position-black-to-move", start, +1)

    # 2. Same position, white to move (should be symmetric under the encoder)
    record("starting-position-white-to-move", start, -1)

    # 3. Material-advantage (black up by 2 pieces)
    adv = start.copy()
    adv[20] = 0
    adv[21] = 0
    record("black-up-two-pieces-black-to-move", adv, +1)

    # 4. A position with kings on both sides
    kings = [0] * 32
    kings[0] = 2    # black king
    kings[31] = -2  # white king
    kings[15] = 1   # black man midboard
    kings[16] = -1  # white man midboard
    record("two-kings-sparse-black-to-move", kings, +1)

    # 5. Endgame: single black king vs. single white man
    endgame = [0] * 32
    endgame[12] = 2
    endgame[28] = -1
    record("endgame-black-king-vs-white-man", endgame, +1)

    return fixtures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="path to .pt checkpoint (or omit with --init-random)")
    ap.add_argument("output", help="path to output .bin")
    ap.add_argument(
        "--arch", choices=["auto", "1999", "2001"], default="auto",
        help="which architecture the checkpoint belongs to; 'auto' infers from weight count",
    )
    ap.add_argument(
        "--init-random", choices=["1999", "2001"],
        help="skip loading a checkpoint; use a deterministic random init of the given arch",
    )
    ap.add_argument(
        "--fixtures", type=Path,
        help="also write a JSON of (board, score) fixtures for JS cross-check",
    )
    args = ap.parse_args()

    if args.init_random:
        arch = args.init_random
        weights = init_random_weights(arch)
        print(f"initialized random {arch} weights ({len(weights)} floats)")
    else:
        if not args.input:
            ap.error("input checkpoint required unless --init-random is given")
        weights = load_checkpoint_weights(Path(args.input))
        if args.arch == "auto":
            if len(weights) == N_1999:
                arch = "1999"
            elif len(weights) == N_2001:
                arch = "2001"
            else:
                raise SystemExit(
                    f"Unknown weight count {len(weights)}: expected {N_1999} (1999) or {N_2001} (2001)."
                )
        else:
            arch = args.arch
        print(f"loaded {args.input}: {len(weights)} weights, arch={arch}")

    out = Path(args.output)
    write_bin(out, weights)

    # Always write a sidecar .meta.json next to the .bin so the play-strong
    # page can show provenance ("850 gens from checkpoint X") to the user.
    meta_path = out.with_suffix(".meta.json")
    src_label = (
        f"random init (seed 1729, --init-random {args.init_random})"
        if args.init_random
        else f"checkpoint: {Path(args.input).name}"
    )
    meta_path.write_text(json.dumps({
        "source": src_label,
        "architecture": "Anaconda (2001)" if arch == "2001" else "CheckersNet (1999)",
        "arch_key": arch,
        "nWeights": int(len(weights)),
        "generations": None if args.init_random else _gens_from_name(Path(args.input).name),
        "note": (
            "Untrained random-init weights shipped as a placeholder. "
            "To get a real opponent, train locally and re-run this export."
            if args.init_random
            else "Trained weights exported from a checkpoint."
        ),
    }, indent=2))
    print(f"wrote {meta_path}")

    if args.fixtures:
        fixtures = make_fixtures(arch, weights)
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        meta = {"arch": arch, "nWeights": int(len(weights)), "fixtures": fixtures}
        args.fixtures.write_text(json.dumps(meta, indent=2))
        print(f"wrote {args.fixtures} ({len(fixtures)} test positions)")


def _gens_from_name(fname: str) -> int | None:
    """Best-effort extraction of 'gen1450' from 'best_gen1450.pt', etc."""
    import re
    m = re.search(r"gen(\d+)", fname)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    main()

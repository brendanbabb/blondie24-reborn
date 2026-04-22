/*
 * Checkers engine, 32-square dark-squares-only representation.
 *
 * Squares are numbered 0..31 in row-major order over the dark squares:
 *
 *     col: 0  1  2  3  4  5  6  7
 * row 0 :  .  0  .  1  .  2  .  3
 * row 1 :  4  .  5  .  6  .  7  .
 * row 2 :  .  8  .  9  . 10  . 11
 * row 3 : 12  . 13  . 14  . 15  .
 * row 4 :  . 16  . 17  . 18  . 19
 * row 5 : 20  . 21  . 22  . 23  .
 * row 6 :  . 24  . 25  . 26  . 27
 * row 7 : 28  . 29  . 30  . 31  .
 *
 * Black starts on squares 0..11 (top) and moves DOWN the board (increasing row).
 * White starts on squares 20..31 (bottom) and moves UP (decreasing row).
 * Kings move in both directions.
 *
 * Rules encoded:
 *  - Standard American/English draughts: 8x8, single-square diagonal moves,
 *    mandatory jumps, multi-jumps, king promotion on last rank, king stops on
 *    promotion (no continued jump after becoming king this turn).
 *  - Game over: a side with no legal moves LOSES.
 *  - Draws declared at the game-loop level: threefold repetition or move cap.
 */

(function (global) {
  "use strict";

  const BLACK = 1;
  const WHITE = -1;

  const EMPTY = 0;
  const BLACK_PIECE = 1;
  const BLACK_KING = 2;
  const WHITE_PIECE = -1;
  const WHITE_KING = -2;

  // Precompute neighbors. For each square and each of 4 diagonal directions,
  // the immediate neighbor and the landing square (2 diagonals away) — or -1
  // if off-board. Directions indexed 0..3 as:
  //   0 = up-left, 1 = up-right, 2 = down-left, 3 = down-right
  // "up" = toward row 0, "down" = toward row 7.
  const NEIGHBORS = new Int8Array(32 * 4);
  const JUMPS = new Int8Array(32 * 4);

  function rowColOf(sq) {
    const row = (sq / 4) | 0;
    const colOffset = (row % 2 === 0) ? 1 : 0;
    const col = (sq % 4) * 2 + colOffset;
    return [row, col];
  }

  function sqOf(row, col) {
    if (row < 0 || row > 7 || col < 0 || col > 7) return -1;
    if ((row + col) % 2 === 0) return -1; // light square
    return row * 4 + ((col - (row % 2 === 0 ? 1 : 0)) >> 1);
  }

  (function buildTables() {
    for (let sq = 0; sq < 32; sq++) {
      const [row, col] = rowColOf(sq);
      // 0 up-left, 1 up-right, 2 down-left, 3 down-right
      const dirs = [
        [-1, -1], [-1, +1], [+1, -1], [+1, +1],
      ];
      for (let d = 0; d < 4; d++) {
        const [dr, dc] = dirs[d];
        NEIGHBORS[sq * 4 + d] = sqOf(row + dr, col + dc);
        JUMPS[sq * 4 + d]     = sqOf(row + 2 * dr, col + 2 * dc);
      }
    }
  })();

  function initialSquares() {
    const s = new Int8Array(32);
    for (let i = 0; i < 12; i++) s[i] = BLACK_PIECE;
    for (let i = 20; i < 32; i++) s[i] = WHITE_PIECE;
    return s;
  }

  function makeBoard() {
    return {
      squares: initialSquares(),
      currentPlayer: BLACK,
      moveCount: 0,
    };
  }

  function cloneBoard(b) {
    return {
      squares: new Int8Array(b.squares),
      currentPlayer: b.currentPlayer,
      moveCount: b.moveCount,
    };
  }

  function owner(piece) {
    if (piece === BLACK_PIECE || piece === BLACK_KING) return BLACK;
    if (piece === WHITE_PIECE || piece === WHITE_KING) return WHITE;
    return 0;
  }

  function isKing(piece) { return piece === BLACK_KING || piece === WHITE_KING; }

  // Return list of legal moves for currentPlayer.
  // Each move is either:
  //   [from, to]                     simple sliding move
  //   [from, cap1, land1, cap2, land2, ...]  jump sequence (len >= 3, odd)
  // Forced jumps: if any jumps exist, ONLY jumps are returned.
  function getLegalMoves(board) {
    const sq = board.squares;
    const side = board.currentPlayer;
    const jumps = [];
    const slides = [];

    for (let from = 0; from < 32; from++) {
      const piece = sq[from];
      if (piece === EMPTY || owner(piece) !== side) continue;
      collectJumpsFrom(sq, from, piece, [from], jumps);
    }

    if (jumps.length > 0) return jumps;

    for (let from = 0; from < 32; from++) {
      const piece = sq[from];
      if (piece === EMPTY || owner(piece) !== side) continue;
      collectSlidesFrom(sq, from, piece, slides);
    }
    return slides;
  }

  // Directions a piece may move in (index into NEIGHBORS 0..3).
  function allowedDirs(piece) {
    if (piece === BLACK_PIECE)  return [2, 3];      // black men go down
    if (piece === WHITE_PIECE)  return [0, 1];      // white men go up
    if (piece === BLACK_KING || piece === WHITE_KING) return [0, 1, 2, 3];
    return [];
  }

  function collectSlidesFrom(sq, from, piece, out) {
    const dirs = allowedDirs(piece);
    for (let i = 0; i < dirs.length; i++) {
      const to = NEIGHBORS[from * 4 + dirs[i]];
      if (to === -1) continue;
      if (sq[to] === EMPTY) out.push([from, to]);
    }
  }

  // Recursive capture search. `path` always has odd length ending at the
  // current landing square. Captures follow American draughts rules: mandatory,
  // multi-jumps allowed, kinging ends the sequence (men promoting stop this
  // turn even if more jumps would be available as a king).
  function collectJumpsFrom(sq, current, piece, path, out) {
    const dirs = allowedDirs(piece);
    let extended = false;
    for (let i = 0; i < dirs.length; i++) {
      const d = dirs[i];
      const mid = NEIGHBORS[current * 4 + d];
      const land = JUMPS[current * 4 + d];
      if (mid === -1 || land === -1) continue;

      // mid must hold an enemy piece that hasn't been captured in this path,
      // and landing must be empty OR the start square (a piece can "return"
      // since it vacated). Simpler check: landing empty is enough for most
      // engines because the piece is no longer at `from` mid-jump.
      const midPiece = sq[mid];
      if (midPiece === EMPTY) continue;
      if (owner(midPiece) === owner(piece)) continue;

      // Skip squares already captured in this path.
      let alreadyCaptured = false;
      for (let p = 1; p < path.length; p += 2) {
        if (path[p] === mid) { alreadyCaptured = true; break; }
      }
      if (alreadyCaptured) continue;

      // Landing must be empty, OR equal to the starting square (the jumper
      // has moved, leaving that square free).
      if (land !== path[0] && sq[land] !== EMPTY) continue;

      // Execute the hop in a virtual board state (path only).
      path.push(mid, land);

      // If this hop promotes a man, stop extending (American rule).
      let becomesKing = false;
      let nextPiece = piece;
      if (piece === BLACK_PIECE && land >= 28) {
        becomesKing = true;
        nextPiece = BLACK_KING;
      } else if (piece === WHITE_PIECE && land <= 3) {
        becomesKing = true;
        nextPiece = WHITE_KING;
      }

      if (!becomesKing) {
        // Try to continue the chain.
        const before = out.length;
        collectJumpsFrom(sq, land, nextPiece, path, out);
        if (out.length === before) {
          // No extension found: this is a terminal jump sequence.
          out.push(path.slice());
        } else {
          extended = true;
        }
      } else {
        // Kinged this turn — always a terminal sequence.
        out.push(path.slice());
      }

      path.pop(); path.pop();
    }
    return extended;
  }

  // Apply a move to a cloned board and return the new board (pure function).
  function applyMove(board, move) {
    const next = cloneBoard(board);
    const sq = next.squares;
    const from = move[0];
    const piece = sq[from];
    sq[from] = EMPTY;

    if (move.length === 2) {
      const to = move[1];
      sq[to] = maybePromote(piece, to);
    } else {
      // jump sequence
      let last = from;
      let currentPiece = piece;
      for (let i = 1; i < move.length; i += 2) {
        const cap = move[i];
        const land = move[i + 1];
        sq[cap] = EMPTY;
        currentPiece = maybePromote(currentPiece, land);
        last = land;
      }
      sq[last] = currentPiece;
    }

    next.currentPlayer = -next.currentPlayer;
    next.moveCount = board.moveCount + 1;
    return next;
  }

  function maybePromote(piece, land) {
    if (piece === BLACK_PIECE && land >= 28) return BLACK_KING;
    if (piece === WHITE_PIECE && land <= 3)  return WHITE_KING;
    return piece;
  }

  function pieceCount(board) {
    let b = 0, w = 0;
    for (let i = 0; i < 32; i++) {
      const p = board.squares[i];
      if (p === BLACK_PIECE || p === BLACK_KING) b++;
      else if (p === WHITE_PIECE || p === WHITE_KING) w++;
    }
    return [b, w];
  }

  // Returns [gameOver, winner] where winner is BLACK, WHITE, or null for draw.
  // Only checks the win-by-no-moves condition; draw rules (repetition,
  // move cap) live at the game-loop level.
  function isGameOver(board) {
    const moves = getLegalMoves(board);
    if (moves.length === 0) {
      // Current player has no moves → they lose.
      return [true, -board.currentPlayer];
    }
    return [false, null];
  }

  // Hash for repetition detection: squares bytes + currentPlayer.
  function stateKey(board) {
    // TypedArray.toString for small arrays is fine; alt: encode as base64.
    // We keep it plain for readability.
    return Array.prototype.join.call(board.squares, ",") + "|" + board.currentPlayer;
  }

  global.Checkers = {
    BLACK, WHITE,
    EMPTY, BLACK_PIECE, BLACK_KING, WHITE_PIECE, WHITE_KING,
    rowColOf, sqOf,
    makeBoard, cloneBoard,
    getLegalMoves, applyMove,
    owner, isKing, pieceCount,
    isGameOver, stateKey,
  };
})(typeof self !== "undefined" ? self : this);

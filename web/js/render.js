/*
 * Canvas renderer for the checkers board.
 *
 * Responsibilities:
 *   - Draw board, pieces, last-move highlights, legal-move hints.
 *   - Translate canvas (x,y) clicks into the 32-square index the caller cares about.
 *   - No game state owned here — render() is called with the current board and hints.
 */

(function (global) {
  "use strict";

  const C = global.Checkers;

  const SQUARE_PX = 60;   // 8 × 60 = 480 canvas
  const PIECE_R = 22;
  const PIECE_BORDER = 3;

  const COLOR = {
    bgLight: "#c9a778",
    bgDark:  "#6a4a2a",
    label:   "#3a2a18",
    blackPiece: "#1a1a1a",
    blackPieceEdge: "#000",
    whitePiece: "#e8e8e8",
    whitePieceEdge: "#888",
    kingCrown: "#f3c14a",
    hlFrom:    "rgba(243, 193, 74, 0.55)",
    hlTo:      "rgba(243, 193, 74, 0.85)",
    hlLegal:   "rgba(80, 200, 120, 0.55)",
    hlCapture: "rgba(230, 90, 90, 0.60)",
    hlSelect:  "rgba(100, 180, 255, 0.55)",
  };

  function draw(ctx, board, opts) {
    opts = opts || {};
    const lastFrom = opts.lastFrom != null ? opts.lastFrom : -1;
    const lastTo = opts.lastTo != null ? opts.lastTo : -1;
    const captured = opts.captured || [];
    const legalTargets = opts.legalTargets || []; // squares the human can click to move INTO
    const selectedFrom = opts.selectedFrom != null ? opts.selectedFrom : -1;
    const captureTargets = opts.captureTargets || [];
    const humanColor = opts.humanColor || "white"; // "white" => white on bottom

    ctx.clearRect(0, 0, 8 * SQUARE_PX, 8 * SQUARE_PX);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const viewR = humanColor === "black" ? 7 - r : r;
        const viewC = humanColor === "black" ? 7 - c : c;
        const x = viewC * SQUARE_PX;
        const y = viewR * SQUARE_PX;
        const isDark = (r + c) % 2 === 1;

        ctx.fillStyle = isDark ? COLOR.bgDark : COLOR.bgLight;
        ctx.fillRect(x, y, SQUARE_PX, SQUARE_PX);

        if (!isDark) continue;
        const sq = C.sqOf(r, c);
        if (sq === -1) continue;

        // Highlights on the square background
        if (sq === lastFrom || sq === lastTo) {
          ctx.fillStyle = sq === lastTo ? COLOR.hlTo : COLOR.hlFrom;
          ctx.fillRect(x, y, SQUARE_PX, SQUARE_PX);
        }
        if (sq === selectedFrom) {
          ctx.fillStyle = COLOR.hlSelect;
          ctx.fillRect(x, y, SQUARE_PX, SQUARE_PX);
        }
        if (captured.indexOf(sq) !== -1) {
          ctx.fillStyle = COLOR.hlCapture;
          ctx.fillRect(x, y, SQUARE_PX, SQUARE_PX);
        }

        // Square number label (1-indexed for display; matches standard
        // checkers notation where dark squares are numbered 1-32).
        ctx.fillStyle = "rgba(0,0,0,0.35)";
        ctx.font = "10px -apple-system, Segoe UI, sans-serif";
        ctx.fillText(String(sq + 1), x + 4, y + 12);

        const piece = board.squares[sq];
        if (piece !== C.EMPTY) {
          drawPiece(ctx, x + SQUARE_PX / 2, y + SQUARE_PX / 2, piece);
        }

        // Legal-move target dots on top
        if (legalTargets.indexOf(sq) !== -1) {
          ctx.beginPath();
          ctx.fillStyle = captureTargets.indexOf(sq) !== -1 ? COLOR.hlCapture : COLOR.hlLegal;
          ctx.arc(x + SQUARE_PX / 2, y + SQUARE_PX / 2, 10, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  function drawPiece(ctx, cx, cy, piece) {
    const isBlack = piece === C.BLACK_PIECE || piece === C.BLACK_KING;
    const isKing  = piece === C.BLACK_KING || piece === C.WHITE_KING;

    ctx.beginPath();
    ctx.arc(cx, cy, PIECE_R, 0, Math.PI * 2);
    ctx.fillStyle = isBlack ? COLOR.blackPiece : COLOR.whitePiece;
    ctx.fill();
    ctx.strokeStyle = isBlack ? COLOR.blackPieceEdge : COLOR.whitePieceEdge;
    ctx.lineWidth = PIECE_BORDER;
    ctx.stroke();

    if (isKing) {
      ctx.beginPath();
      ctx.arc(cx, cy, PIECE_R - 7, 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.kingCrown;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = COLOR.kingCrown;
      ctx.font = "bold 14px serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("K", cx, cy + 1);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }
  }

  // Convert a canvas click to a board square index (0..31), or -1 if outside.
  function clickToSquare(canvas, ev, humanColor) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (ev.clientX - rect.left) * scaleX;
    const y = (ev.clientY - rect.top) * scaleY;
    let c = Math.floor(x / SQUARE_PX);
    let r = Math.floor(y / SQUARE_PX);
    if (humanColor === "black") { c = 7 - c; r = 7 - r; }
    if (r < 0 || r > 7 || c < 0 || c > 7) return -1;
    return C.sqOf(r, c);
  }

  // Tiny board used for the self-play replay panel. No highlights, no hints,
  // just squares + pieces, scaled to the given canvas size. Accepts a raw
  // Int8Array(32) of squares so it can render historical snapshots.
  //
  // opts.highlightSq: 0..31 to draw a vivid ring around that square (used by
  //   the AI's-plan strip to mark the piece that just moved); -1/undefined skips it.
  // opts.pieceScale: piece radius as fraction of cell size. Defaults to 0.38
  //   (matches the live-evolve demo's mini-replay panel). The play-strong
  //   AI's-plan strip passes a smaller value so pieces don't crowd the cell
  //   at the smaller mini-board scale.
  function drawMini(ctx, squares, opts) {
    opts = opts || {};
    const canvasSize = opts.size != null ? opts.size : 200;
    const highlightSq = opts.highlightSq != null ? opts.highlightSq : -1;
    const pieceScale = opts.pieceScale != null ? opts.pieceScale : 0.38;
    const cell = canvasSize / 8;
    const pr = cell * pieceScale;

    ctx.clearRect(0, 0, canvasSize, canvasSize);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const x = c * cell;
        const y = r * cell;
        const isDark = (r + c) % 2 === 1;
        ctx.fillStyle = isDark ? COLOR.bgDark : COLOR.bgLight;
        ctx.fillRect(x, y, cell, cell);
        if (!isDark) continue;
        const sq = C.sqOf(r, c);
        if (sq === -1) continue;

        // Square-level highlight (drawn behind the piece) for the moved-to
        // square — gives a glow under the piece even when the piece itself
        // is small. Uses blue (not the board's yellow accent) so it doesn't
        // collide with the king-crown ring or the white piece edges.
        if (sq === highlightSq) {
          ctx.fillStyle = "rgba(80, 160, 255, 0.55)";
          ctx.fillRect(x, y, cell, cell);
        }

        const piece = squares[sq];
        if (piece === C.EMPTY) continue;
        const cx = x + cell / 2;
        const cy = y + cell / 2;
        const isBlack = piece === C.BLACK_PIECE || piece === C.BLACK_KING;
        const isKing  = piece === C.BLACK_KING || piece === C.WHITE_KING;
        // Mini-board pieces are tiny (~3-4px radius). The main-board edge
        // colors (#000 around black pieces, #888 around white) wash out at
        // this scale — black-on-dark-brown becomes nearly invisible. Use
        // bright white edges so every piece reads against either square
        // color, and bump the line width relative to the small piece.
        ctx.beginPath();
        ctx.arc(cx, cy, pr, 0, Math.PI * 2);
        ctx.fillStyle = isBlack ? COLOR.blackPiece : COLOR.whitePiece;
        ctx.fill();
        ctx.strokeStyle = isBlack ? "#f0f0f0" : "#1a1410";
        ctx.lineWidth = 1.8;
        ctx.stroke();
        if (isKing) {
          ctx.beginPath();
          ctx.arc(cx, cy, pr * 0.55, 0, Math.PI * 2);
          ctx.strokeStyle = COLOR.kingCrown;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        // Bright blue ring around the just-moved piece, drawn last so it
        // sits on top of the piece outline. Blue keeps it visually distinct
        // from the yellow king-crown and the white piece edge.
        if (sq === highlightSq) {
          ctx.beginPath();
          ctx.arc(cx, cy, pr + 2, 0, Math.PI * 2);
          ctx.strokeStyle = "#4aa0ff";
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }
      }
    }
  }

  global.Render = { draw, drawMini, clickToSquare };
})(typeof self !== "undefined" ? self : this);

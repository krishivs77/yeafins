import { Chess, Move, Square } from "chess.js";

export type PlayerColour = "white" | "black";
export type ColourChoice = PlayerColour | "random";
export type PromotionPiece = "q" | "r" | "b" | "n";

export type GameOutcome = {
  result: "1-0" | "0-1" | "1/2-1/2";
  winner: "Visitor" | "Yeafins" | "Draw";
  reason: string;
};

export function parseUci(uci: string) {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) return null;
  return {
    from: uci.slice(0, 2) as Square,
    to: uci.slice(2, 4) as Square,
    promotion: (uci[4] as PromotionPiece | undefined) ?? undefined,
  };
}

export function applyUciMove(game: Chess, uci: string): Move | null {
  const parsed = parseUci(uci);
  if (!parsed) return null;
  try {
    return game.move(parsed);
  } catch {
    return null;
  }
}

export function legalDestinations(game: Chess, square: Square): Square[] {
  return game.moves({ square, verbose: true }).map((move) => move.to);
}

export function needsPromotion(game: Chess, from: Square, to: Square): boolean {
  return game
    .moves({ square: from, verbose: true })
    .some((move) => move.to === to && Boolean(move.promotion));
}

export function groupedHistory(game: Chess): Array<{ number: number; white?: Move; black?: Move }> {
  const moves = game.history({ verbose: true });
  const rows: Array<{ number: number; white?: Move; black?: Move }> = [];
  for (let index = 0; index < moves.length; index += 2) {
    rows.push({ number: index / 2 + 1, white: moves[index], black: moves[index + 1] });
  }
  return rows;
}

export function gameOutcome(game: Chess, visitorColour: PlayerColour): GameOutcome | null {
  if (!game.isGameOver()) return null;
  if (game.isCheckmate()) {
    const winnerColour = game.turn() === "w" ? "black" : "white";
    return {
      result: winnerColour === "white" ? "1-0" : "0-1",
      winner: winnerColour === visitorColour ? "Visitor" : "Yeafins",
      reason: "Checkmate",
    };
  }
  const reason = game.isStalemate()
    ? "Stalemate"
    : game.isThreefoldRepetition()
      ? "Threefold repetition"
      : game.isInsufficientMaterial()
        ? "Insufficient material"
        : game.isDrawByFiftyMoves()
          ? "Fifty-move rule"
          : "Draw";
  return { result: "1/2-1/2", winner: "Draw", reason };
}

export function gamePgn(game: Chess, visitorColour: PlayerColour): string {
  const copy = new Chess();
  for (const move of game.history({ verbose: true })) {
    copy.move({ from: move.from, to: move.to, promotion: move.promotion });
  }
  const today = new Date();
  const date = `${today.getFullYear()}.${String(today.getMonth() + 1).padStart(2, "0")}.${String(
    today.getDate(),
  ).padStart(2, "0")}`;
  const outcome = gameOutcome(game, visitorColour);
  copy.setHeader("Event", "Play against Yeafins");
  copy.setHeader("Site", "public website");
  copy.setHeader("Date", date);
  copy.setHeader("White", visitorColour === "white" ? "Visitor" : "Yeafins");
  copy.setHeader("Black", visitorColour === "black" ? "Visitor" : "Yeafins");
  copy.setHeader("Result", outcome?.result ?? "*");
  return copy.pgn();
}

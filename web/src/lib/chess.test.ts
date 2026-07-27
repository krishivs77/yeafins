import { Chess } from "chess.js";
import { describe, expect, it } from "vitest";
import {
  applyUciMove,
  gameOutcome,
  groupedHistory,
  needsPromotion,
  parseUci,
} from "@/lib/chess";

describe("chess helpers", () => {
  it("applies a legal UCI move and rejects malformed or illegal moves", () => {
    const game = new Chess();
    expect(applyUciMove(game, "e2e4")?.san).toBe("e4");
    expect(applyUciMove(game, "e2e5")).toBeNull();
    expect(parseUci("oops")).toBeNull();
  });

  it("groups SAN move history into numbered rows", () => {
    const game = new Chess();
    game.move("e4");
    game.move("e5");
    game.move("Nf3");
    expect(groupedHistory(game).map((row) => [row.number, row.white?.san, row.black?.san])).toEqual([
      [1, "e4", "e5"],
      [2, "Nf3", undefined],
    ]);
  });

  it("detects promotion and checkmate outcomes", () => {
    const promotion = new Chess("8/P7/8/8/8/8/7p/4K2k w - - 0 1");
    expect(needsPromotion(promotion, "a7", "a8")).toBe(true);

    const mate = new Chess();
    for (const move of ["f3", "e5", "g4", "Qh4#"]) mate.move(move);
    expect(gameOutcome(mate, "white")).toEqual({
      result: "0-1",
      winner: "Yeafins",
      reason: "Checkmate",
    });
  });
});

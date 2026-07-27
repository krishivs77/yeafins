import { fireEvent, render, screen } from "@testing-library/react";
import { Chess } from "chess.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MoveHistory } from "@/components/move-history";
import { groupedHistory } from "@/lib/chess";

describe("MoveHistory", () => {
  beforeEach(() => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    URL.createObjectURL = vi.fn(() => "blob:game");
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  });

  it("shows compact SAN history and supports PGN copy and download", () => {
    const game = new Chess();
    game.move("e4");
    game.move("e5");
    const pgn = '[Event "Play against Yeafins"]\n\n1. e4 e5 *';

    render(
      <MoveHistory
        rows={groupedHistory(game)}
        visitorColour="white"
        pgn={pgn}
      />,
    );

    expect(screen.getByText("e4")).toBeInTheDocument();
    expect(screen.getByText("e5")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Copy PGN" }));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(pgn);
    fireEvent.click(screen.getByRole("button", { name: "Download" }));
    expect(URL.createObjectURL).toHaveBeenCalledOnce();
  });
});

import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EngineMoveResponse } from "@/lib/api";
import { useYeafinsGame } from "@/hooks/use-yeafins-game";

function response(uci: string, san: string): EngineMoveResponse {
  return {
    fen: "request-fen",
    selected_move_uci: uci,
    selected_move_san: san,
    phase: "opening",
    resolved_style_weight: 0.2,
    mode: "blended",
    top_k: 16,
    stockfish_elo: 2000,
    candidates: [
      {
        move_uci: uci,
        move_san: san,
        model_rank: 1,
        model_probability: 0.4,
        stockfish_cp: 20,
        selected: true,
      },
    ],
    game_over: false,
    outcome: null,
  };
}

describe("useYeafinsGame", () => {
  it("accepts legal visitor moves and applies the Yeafins response", async () => {
    const requester = vi.fn().mockResolvedValue(response("e7e5", "e5"));
    const { result } = renderHook(() => useYeafinsGame(requester));

    act(() => result.current.startGame("white"));
    expect(result.current.visitorColour).toBe("white");
    expect(result.current.makeHumanMove("e2", "e5")).toBe(false);
    act(() => {
      expect(result.current.makeHumanMove("e2", "e4")).toBe(true);
    });

    await waitFor(() => expect(result.current.history[0]?.black?.san).toBe("e5"));
    expect(result.current.history[0]?.white?.san).toBe("e4");
    expect(result.current.engineData?.selected_move_uci).toBe("e7e5");
    expect(requester).toHaveBeenCalledTimes(1);
  });

  it("requests exactly one opening engine move when the visitor chooses Black", async () => {
    const requester = vi.fn().mockResolvedValue(response("e2e4", "e4"));
    const { result } = renderHook(() => useYeafinsGame(requester));

    act(() => result.current.startGame("black"));

    await waitFor(() => expect(result.current.history[0]?.white?.san).toBe("e4"));
    expect(result.current.visitorColour).toBe("black");
    expect(requester).toHaveBeenCalledTimes(1);
  });

  it("resets the game and aborts stale engine responses", async () => {
    let resolveRequest: (value: EngineMoveResponse) => void = () => undefined;
    const requester = vi.fn(
      () =>
        new Promise<EngineMoveResponse>((resolve) => {
          resolveRequest = resolve;
        }),
    );
    const { result } = renderHook(() => useYeafinsGame(requester));

    act(() => result.current.startGame("black"));
    await waitFor(() => expect(requester).toHaveBeenCalledTimes(1));
    act(() => result.current.restart());
    await act(async () => resolveRequest(response("e2e4", "e4")));

    expect(result.current.started).toBe(false);
    expect(result.current.history).toHaveLength(0);
    expect(result.current.fen).toContain(" w KQkq ");
  });

  it("marks resignation as game over and cancels further play", () => {
    const requester = vi.fn().mockResolvedValue(response("e7e5", "e5"));
    const { result } = renderHook(() => useYeafinsGame(requester));
    act(() => result.current.startGame("white"));
    act(() => result.current.resign());

    expect(result.current.outcome).toEqual({
      result: "0-1",
      winner: "Yeafins",
      reason: "Visitor resigned",
    });
    expect(result.current.makeHumanMove("e2", "e4")).toBe(false);
  });
});

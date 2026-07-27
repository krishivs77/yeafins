import { afterEach, describe, expect, it, vi } from "vitest";
import {
  EngineApiError,
  PUBLIC_ENGINE_CONFIG,
  getHealth,
  requestMove,
} from "@/lib/api";

const moveResponse = {
  fen: "start-fen",
  selected_move_uci: "e7e5",
  selected_move_san: "e5",
  phase: "opening",
  resolved_style_weight: 0.2,
  mode: "blended",
  top_k: 16,
  stockfish_elo: 2000,
  candidates: [
    {
      move_uci: "e7e5",
      move_san: "e5",
      model_rank: 1,
      model_probability: 0.42,
      stockfish_cp: 31,
      selected: true,
    },
  ],
  game_over: false,
  outcome: null,
};

afterEach(() => vi.unstubAllGlobals());

describe("engine API", () => {
  it("parses a successful health response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ status: "ok", model_loaded: true, stockfish_available: true }),
          { status: 200 },
        ),
      ),
    );
    await expect(getHealth()).resolves.toEqual({
      status: "ok",
      model_loaded: true,
      stockfish_available: true,
    });
  });

  it("parses a successful move and sends exact public parameters", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(moveResponse), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(requestMove("start-fen")).resolves.toMatchObject({
      selected_move_uci: "e7e5",
      phase: "opening",
    });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(init.body))).toEqual({
      fen: "start-fen",
      ...PUBLIC_ENGINE_CONFIG,
    });
    expect(PUBLIC_ENGINE_CONFIG).toEqual({
      top_k: 16,
      mode: "blended",
      stockfish_elo: 2000,
      depth: null,
      time_limit_seconds: 1.5,
      style_weight: null,
    });
  });

  it("surfaces the backend structured error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: { code: "service_unavailable", message: "Engine warming up." },
          }),
          { status: 503 },
        ),
      ),
    );
    const error = await requestMove("start-fen").catch((caught) => caught);
    expect(error).toBeInstanceOf(EngineApiError);
    expect(error).toMatchObject({
      code: "service_unavailable",
      message: "Engine warming up.",
      status: 503,
    });
  });
});

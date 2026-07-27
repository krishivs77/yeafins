export type EngineHealth = {
  status: "ok" | "unhealthy";
  model_loaded: boolean;
  stockfish_available: boolean;
};

export type EngineCandidate = {
  move_uci: string;
  move_san: string;
  model_rank: number;
  model_probability: number;
  stockfish_cp: number;
  selected: boolean;
};

export type EngineMoveResponse = {
  fen: string;
  selected_move_uci: string;
  selected_move_san: string;
  phase: "opening" | "middlegame" | "endgame";
  resolved_style_weight: number;
  mode: "blended";
  top_k: number;
  stockfish_elo: number;
  candidates: EngineCandidate[];
  game_over: boolean;
  outcome: {
    result: string;
    winner: "white" | "black" | null;
    termination: string;
  } | null;
};

export const PUBLIC_ENGINE_CONFIG = {
  top_k: 16,
  mode: "blended" as const,
  stockfish_elo: 2000,
  depth: null,
  time_limit_seconds: 1.5,
  style_weight: null,
};

export type MoveRequest = { fen: string } & typeof PUBLIC_ENGINE_CONFIG;

export class EngineApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "EngineApiError";
  }
}

const baseUrl = () => {
  const configured =
    process.env.NEXT_PUBLIC_ENGINE_API_URL?.trim() || "http://127.0.0.1:8000";
  return configured.replace(/\/+$/, "");
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function fetchJson(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<unknown> {
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort("timeout"), timeoutMs);
  const externalSignal = init.signal;

  const abortFromExternal = () => timeout.abort(externalSignal?.reason);
  externalSignal?.addEventListener("abort", abortFromExternal, { once: true });

  try {
    const response = await fetch(`${baseUrl()}${path}`, {
      ...init,
      signal: timeout.signal,
    });
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const error = isRecord(body) && isRecord(body.error) ? body.error : null;
      throw new EngineApiError(
        typeof error?.message === "string"
          ? error.message
          : "The Yeafins engine could not complete the request.",
        typeof error?.code === "string" ? error.code : "request_failed",
        response.status,
      );
    }
    return body;
  } catch (error) {
    if (error instanceof EngineApiError) throw error;
    if (timeout.signal.aborted) {
      if (externalSignal?.aborted) {
        throw new EngineApiError("The request was cancelled.", "aborted", 0);
      }
      throw new EngineApiError(
        "The engine took too long to respond. You can safely retry.",
        "timeout",
        0,
      );
    }
    throw new EngineApiError(
      "Could not reach the Yeafins engine. Check the connection and try again.",
      "network_error",
      0,
    );
  } finally {
    clearTimeout(timer);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}

function parseHealth(value: unknown): EngineHealth {
  if (
    !isRecord(value) ||
    (value.status !== "ok" && value.status !== "unhealthy") ||
    typeof value.model_loaded !== "boolean" ||
    typeof value.stockfish_available !== "boolean"
  ) {
    throw new EngineApiError("The engine returned an invalid health response.", "bad_response", 0);
  }
  return value as EngineHealth;
}

function parseMoveResponse(value: unknown): EngineMoveResponse {
  if (
    !isRecord(value) ||
    typeof value.fen !== "string" ||
    typeof value.selected_move_uci !== "string" ||
    typeof value.selected_move_san !== "string" ||
    !["opening", "middlegame", "endgame"].includes(String(value.phase)) ||
    typeof value.resolved_style_weight !== "number" ||
    value.mode !== "blended" ||
    !Array.isArray(value.candidates) ||
    !value.candidates.every(
      (candidate) =>
        isRecord(candidate) &&
        typeof candidate.move_uci === "string" &&
        typeof candidate.move_san === "string" &&
        typeof candidate.model_rank === "number" &&
        typeof candidate.model_probability === "number" &&
        typeof candidate.stockfish_cp === "number" &&
        typeof candidate.selected === "boolean",
    )
  ) {
    throw new EngineApiError("The engine returned an invalid move response.", "bad_response", 0);
  }
  return value as EngineMoveResponse;
}

export async function getHealth(signal?: AbortSignal): Promise<EngineHealth> {
  return parseHealth(await fetchJson("/health", { method: "GET", signal }, 8_000));
}

export async function requestMove(
  fen: string,
  signal?: AbortSignal,
): Promise<EngineMoveResponse> {
  const payload: MoveRequest = { fen, ...PUBLIC_ENGINE_CONFIG };
  return parseMoveResponse(
    await fetchJson(
      "/move",
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
        signal,
      },
      30_000,
    ),
  );
}

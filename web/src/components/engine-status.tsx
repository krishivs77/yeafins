"use client";

import { useCallback, useEffect, useState } from "react";
import { EngineHealth, getHealth } from "@/lib/api";

export type ConnectionState = "connecting" | "connected" | "unavailable";

export function useEngineHealth() {
  const [state, setState] = useState<ConnectionState>("connecting");
  const [health, setHealth] = useState<EngineHealth | null>(null);

  const check = useCallback(async (signal?: AbortSignal) => {
    setState("connecting");
    try {
      const result = await getHealth(signal);
      setHealth(result);
      setState(result.status === "ok" ? "connected" : "unavailable");
    } catch {
      if (!signal?.aborted) {
        setHealth(null);
        setState("unavailable");
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getHealth(controller.signal)
      .then((result) => {
        setHealth(result);
        setState(result.status === "ok" ? "connected" : "unavailable");
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setHealth(null);
          setState("unavailable");
        }
      });
    return () => controller.abort();
  }, []);

  return { state, health, retry: () => void check() };
}

export function EngineStatus({
  state,
  retry,
}: {
  state: ConnectionState;
  retry: () => void;
}) {
  const copy = {
    connecting: ["Connecting…", "Checking engine availability."],
    connected: ["Engine ready", "Ready to play."],
    unavailable: ["Engine unavailable", "Try again in a moment."],
  }[state];

  return (
    <div className={`connection connection--${state}`} role="status" aria-live="polite">
      <span className="connection__dot" aria-hidden="true" />
      <div>
        <strong>{copy[0]}</strong>
        <span>{copy[1]}</span>
      </div>
      {state === "unavailable" && (
        <button className="text-button" type="button" onClick={retry}>
          Retry
        </button>
      )}
    </div>
  );
}

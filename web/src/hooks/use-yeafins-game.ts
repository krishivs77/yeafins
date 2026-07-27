"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess, Move, Square } from "chess.js";
import { EngineMoveResponse, requestMove } from "@/lib/api";
import {
  ColourChoice,
  GameOutcome,
  PlayerColour,
  PromotionPiece,
  applyUciMove,
  gameOutcome,
  gamePgn,
  groupedHistory,
  needsPromotion,
} from "@/lib/chess";

type PendingPromotion = { from: Square; to: Square } | null;
type MoveRequester = typeof requestMove;

function cloneGame(source: Chess): Chess {
  const copy = new Chess();
  for (const move of source.history({ verbose: true })) {
    copy.move({ from: move.from, to: move.to, promotion: move.promotion });
  }
  return copy;
}

export function useYeafinsGame(moveRequester: MoveRequester = requestMove) {
  const gameRef = useRef(new Chess());
  const requestVersion = useRef(0);
  const controllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);
  const [game, setGame] = useState(() => new Chess());
  const fen = game.fen();
  const [started, setStarted] = useState(false);
  const [visitorColour, setVisitorColour] = useState<PlayerColour>("white");
  const [thinking, setThinking] = useState(false);
  const [resigned, setResigned] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMove, setLastMove] = useState<{ from: Square; to: Square } | null>(null);
  const [engineData, setEngineData] = useState<EngineMoveResponse | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<PendingPromotion>(null);

  const sync = useCallback(() => {
    setGame(cloneGame(gameRef.current));
  }, []);

  const cancelRequest = useCallback(() => {
    requestVersion.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setThinking(false);
  }, []);

  const requestEngineMove = useCallback(async () => {
    const game = gameRef.current;
    if (!started || resigned || game.isGameOver()) return;
    const engineColour = visitorColour === "white" ? "b" : "w";
    if (game.turn() !== engineColour || controllerRef.current) return;

    const expectedFen = game.fen();
    const version = ++requestVersion.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    setThinking(true);
    setError(null);

    try {
      const response = await moveRequester(expectedFen, controller.signal);
      if (
        !mountedRef.current ||
        version !== requestVersion.current ||
        gameRef.current.fen() !== expectedFen ||
        controller.signal.aborted
      ) {
        return;
      }
      const move = applyUciMove(gameRef.current, response.selected_move_uci);
      if (!move) {
        setError("The engine returned a move that is not legal in this position.");
        return;
      }
      setLastMove({ from: move.from, to: move.to });
      setEngineData(response);
      sync();
    } catch (caught) {
      if (!controller.signal.aborted && version === requestVersion.current) {
        setError(caught instanceof Error ? caught.message : "The engine request failed.");
      }
    } finally {
      if (version === requestVersion.current) {
        controllerRef.current = null;
        setThinking(false);
      }
    }
  }, [moveRequester, resigned, started, sync, visitorColour]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const startGame = useCallback(
    (choice: ColourChoice) => {
      cancelRequest();
      gameRef.current = new Chess();
      const colour =
        choice === "random" ? (Math.random() < 0.5 ? "white" : "black") : choice;
      setVisitorColour(colour);
      setStarted(true);
      setResigned(false);
      setError(null);
      setLastMove(null);
      setEngineData(null);
      setPendingPromotion(null);
      setGame(new Chess());
    },
    [cancelRequest],
  );

  useEffect(() => {
    if (started && visitorColour === "black" && gameRef.current.history().length === 0) {
      void requestEngineMove();
    }
  }, [requestEngineMove, started, visitorColour]);

  const completeHumanMove = useCallback(
    (from: Square, to: Square, promotion?: PromotionPiece) => {
      if (!started || thinking || resigned || gameRef.current.isGameOver()) return false;
      const visitorTurn = visitorColour === "white" ? "w" : "b";
      if (gameRef.current.turn() !== visitorTurn) return false;
      let move: Move;
      try {
        move = gameRef.current.move({ from, to, promotion });
      } catch {
        return false;
      }
      setLastMove({ from: move.from, to: move.to });
      setPendingPromotion(null);
      setError(null);
      sync();
      queueMicrotask(() => void requestEngineMove());
      return true;
    },
    [requestEngineMove, resigned, started, sync, thinking, visitorColour],
  );

  const makeHumanMove = useCallback(
    (from: Square, to: Square) => {
      if (needsPromotion(gameRef.current, from, to)) {
        setPendingPromotion({ from, to });
        return false;
      }
      return completeHumanMove(from, to);
    },
    [completeHumanMove],
  );

  const choosePromotion = useCallback(
    (piece: PromotionPiece) => {
      if (!pendingPromotion) return;
      completeHumanMove(pendingPromotion.from, pendingPromotion.to, piece);
    },
    [completeHumanMove, pendingPromotion],
  );

  const restart = useCallback(() => {
    cancelRequest();
    gameRef.current = new Chess();
    setStarted(false);
    setResigned(false);
    setError(null);
    setEngineData(null);
    setLastMove(null);
    setPendingPromotion(null);
    setGame(new Chess());
  }, [cancelRequest]);

  const resign = useCallback(() => {
    cancelRequest();
    setResigned(true);
    setError(null);
  }, [cancelRequest]);

  const outcome: GameOutcome | null = resigned
    ? {
        result: visitorColour === "white" ? "0-1" : "1-0",
        winner: "Yeafins",
        reason: "Visitor resigned",
      }
    : gameOutcome(game, visitorColour);

  return {
    game,
    fen,
    started,
    visitorColour,
    thinking,
    resigned,
    error,
    lastMove,
    engineData,
    pendingPromotion,
    outcome,
    history: groupedHistory(game),
    pgn: gamePgn(game, visitorColour),
    startGame,
    makeHumanMove,
    choosePromotion,
    cancelPromotion: () => setPendingPromotion(null),
    requestEngineMove,
    restart,
    resign,
  };
}

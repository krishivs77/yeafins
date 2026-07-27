"use client";

import { useState } from "react";
import { ColourChoice } from "@/lib/chess";
import { useYeafinsGame } from "@/hooks/use-yeafins-game";
import { YeafinsBoard } from "@/components/chess-board";
import { EngineStatus, useEngineHealth } from "@/components/engine-status";
import { GameResult } from "@/components/game-result";
import { GameSetup } from "@/components/game-setup";
import { MoveHistory } from "@/components/move-history";
import { PlayerCard } from "@/components/player-card";
import { PromotionDialog } from "@/components/promotion-dialog";

export function GameShell() {
  const [choice, setChoice] = useState<ColourChoice>("white");
  const health = useEngineHealth();
  const game = useYeafinsGame();
  const visitorTurn = game.game.turn() === (game.visitorColour === "white" ? "w" : "b");
  const canInteract =
    game.started &&
    visitorTurn &&
    !game.thinking &&
    !game.outcome &&
    health.state === "connected";

  return (
    <>
      <div className="game-layout">
        <main className="board-column">
          <div className="board-meta">
            <strong aria-live="polite">
              {game.started
                ? game.thinking
                  ? "Yeafins is thinking…"
                  : game.outcome
                    ? "Game over"
                    : visitorTurn
                      ? "Your turn"
                      : "Yeafins to move"
                : "Choose a side to begin"}
            </strong>
          </div>
          <YeafinsBoard
            game={game.game}
            fen={game.fen}
            orientation={game.visitorColour}
            interactive={Boolean(canInteract)}
            lastMove={game.lastMove}
            onMove={game.makeHumanMove}
          />
          <p className="board-hint" aria-live="polite">
            {!game.started
              ? "Choose a colour to begin."
              : game.thinking
                ? "Board locked while Yeafins chooses a move."
                : canInteract
                  ? "Drag a piece or select a piece, then its destination."
                  : game.outcome
                    ? `${game.outcome.winner}: ${game.outcome.reason}`
                    : "Waiting for Yeafins."}
          </p>
        </main>

        <aside className="game-sidebar" aria-label="Game controls">
          <EngineStatus state={health.state} retry={health.retry} />

          {!game.started ? (
            <GameSetup
              choice={choice}
              onChoice={setChoice}
              onStart={() => game.startGame(choice)}
              disabled={health.state !== "connected"}
            />
          ) : (
            <>
              <section className="players" aria-label="Players">
                <PlayerCard
                  name="Yeafins"
                  detail={game.visitorColour === "white" ? "Black" : "White"}
                  active={!visitorTurn && !Boolean(game.outcome)}
                  thinking={game.thinking}
                  engine
                />
                <PlayerCard
                  name="Visitor"
                  detail={game.visitorColour === "white" ? "White" : "Black"}
                  active={visitorTurn && !Boolean(game.outcome)}
                />
              </section>

              {game.error && (
                <div className="game-error" role="alert">
                  <strong>Yeafins could not respond</strong>
                  <p>{game.error}</p>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void game.requestEngineMove()}
                    disabled={game.thinking || visitorTurn}
                  >
                    Retry move
                  </button>
                </div>
              )}

              {game.outcome ? (
                <GameResult outcome={game.outcome} onRestart={game.restart} />
              ) : (
                <div className="game-controls">
                  <button className="secondary-button" type="button" onClick={game.restart}>
                    Restart
                  </button>
                  <button
                    className="danger-button"
                    type="button"
                    onClick={game.resign}
                  >
                    Resign
                  </button>
                </div>
              )}
            </>
          )}

          <MoveHistory
            rows={game.history}
            visitorColour={game.visitorColour}
            pgn={game.pgn}
          />
        </aside>
      </div>
      {game.pendingPromotion && (
        <PromotionDialog onChoose={game.choosePromotion} onCancel={game.cancelPromotion} />
      )}
    </>
  );
}

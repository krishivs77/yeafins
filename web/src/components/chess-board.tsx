"use client";

import { useMemo, useState } from "react";
import { PieceSymbol, Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import { PlayerColour, legalDestinations } from "@/lib/chess";

export function YeafinsBoard({
  game,
  fen,
  orientation,
  interactive,
  lastMove,
  onMove,
}: {
  game: import("chess.js").Chess;
  fen: string;
  orientation: PlayerColour;
  interactive: boolean;
  lastMove: { from: Square; to: Square } | null;
  onMove: (from: Square, to: Square) => boolean;
}) {
  const [selected, setSelected] = useState<Square | null>(null);
  const destinations = useMemo(
    () => (selected ? legalDestinations(game, selected) : []),
    [game, selected],
  );

  const squareStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    if (lastMove) {
      styles[lastMove.from] = { background: "rgba(224, 183, 92, 0.35)" };
      styles[lastMove.to] = { background: "rgba(224, 183, 92, 0.5)" };
    }
    if (selected) {
      styles[selected] = {
        boxShadow: "inset 0 0 0 4px rgba(238, 201, 112, .95)",
      };
      for (const destination of destinations) {
        styles[destination] = {
          ...styles[destination],
          background:
            game.get(destination) === undefined
              ? "radial-gradient(circle, rgba(20,30,26,.55) 0 16%, transparent 18%)"
              : "radial-gradient(circle, transparent 0 60%, rgba(20,30,26,.48) 62%)",
        };
      }
    }
    return styles;
  }, [destinations, game, lastMove, selected]);

  const visitorPiece = orientation === "white" ? "w" : "b";
  const selectSquare = (square: Square, pieceType?: PieceSymbol, pieceColour?: string) => {
    if (!interactive) return;
    if (selected && destinations.includes(square)) {
      onMove(selected, square);
      setSelected(null);
      return;
    }
    if (pieceType && pieceColour === visitorPiece) {
      setSelected(square);
    } else {
      setSelected(null);
    }
  };

  return (
    <div className="board-frame" aria-label="Interactive chessboard">
      <Chessboard
        options={{
          id: "yeafins-board",
          position: fen,
          boardOrientation: orientation,
          allowDragging: interactive,
          allowDrawingArrows: false,
          showNotation: true,
          animationDurationInMs: 220,
          lightSquareStyle: { backgroundColor: "#d8ccb3" },
          darkSquareStyle: { backgroundColor: "#667466" },
          boardStyle: { borderRadius: "4px" },
          squareStyles,
          canDragPiece: ({ piece }) => interactive && piece.pieceType[0] === visitorPiece,
          onPieceDrop: ({ sourceSquare, targetSquare }) => {
            setSelected(null);
            return targetSquare ? onMove(sourceSquare as Square, targetSquare as Square) : false;
          },
          onSquareClick: ({ square, piece }) =>
            selectSquare(
              square as Square,
              piece?.pieceType?.[1]?.toLowerCase() as PieceSymbol | undefined,
              piece?.pieceType?.[0],
            ),
        }}
      />
    </div>
  );
}

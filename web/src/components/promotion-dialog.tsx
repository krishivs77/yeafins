import { useEffect, useRef } from "react";
import { PromotionPiece } from "@/lib/chess";

const pieces: Array<{ value: PromotionPiece; label: string; glyph: string }> = [
  { value: "q", label: "Queen", glyph: "♛" },
  { value: "r", label: "Rook", glyph: "♜" },
  { value: "b", label: "Bishop", glyph: "♝" },
  { value: "n", label: "Knight", glyph: "♞" },
];

export function PromotionDialog({
  onChoose,
  onCancel,
}: {
  onChoose: (piece: PromotionPiece) => void;
  onCancel: () => void;
}) {
  const firstButton = useRef<HTMLButtonElement>(null);
  useEffect(() => firstButton.current?.focus(), []);

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <div
        className="promotion-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="promotion-title"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Escape") onCancel();
        }}
      >
        <span className="eyebrow">Pawn promotion</span>
        <h2 id="promotion-title">Choose a piece</h2>
        <div className="promotion-options">
          {pieces.map((piece, index) => (
            <button
              ref={index === 0 ? firstButton : undefined}
              key={piece.value}
              type="button"
              onClick={() => onChoose(piece.value)}
              aria-label={`Promote to ${piece.label}`}
            >
              <span aria-hidden="true">{piece.glyph}</span>
              {piece.label}
            </button>
          ))}
        </div>
        <button className="text-button" type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}

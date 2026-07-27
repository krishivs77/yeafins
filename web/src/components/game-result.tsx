import { GameOutcome } from "@/lib/chess";

export function GameResult({
  outcome,
  onRestart,
}: {
  outcome: GameOutcome;
  onRestart: () => void;
}) {
  return (
    <section className="result-card" aria-labelledby="result-title" aria-live="assertive">
      <span className="eyebrow">Game complete · {outcome.result}</span>
      <h2 id="result-title">{outcome.winner === "Draw" ? "Drawn game" : `${outcome.winner} wins`}</h2>
      <p>{outcome.reason}</p>
      <button className="primary-button" type="button" onClick={onRestart}>
        Play again
      </button>
    </section>
  );
}

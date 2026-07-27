import { EngineCandidate } from "@/lib/api";

export function formatEngineScore(cp: number): string {
  if (Math.abs(cp) >= 90_000) return cp > 0 ? "decisive +" : "decisive −";
  const score = cp / 100;
  return `${score >= 0 ? "+" : "−"}${Math.abs(score).toFixed(2)}`;
}

export function CandidatePanel({
  candidates,
  styleWeight,
}: {
  candidates: EngineCandidate[];
  styleWeight: number | null;
}) {
  return (
    <details className="panel candidate-panel" open>
      <summary>
        <span>
          <span className="eyebrow">Last Yeafins decision</span>
          <strong>Personalized candidates</strong>
        </span>
        <span className="summary-meta">{candidates.length || 16} moves</span>
      </summary>
      <p className="candidate-explainer">
        Yeafins selects from moves its policy model considers characteristic of Krishiv,
        then balances those preferences with Stockfish evaluation. Scores use the side
        to move in the position Yeafins received.
      </p>
      {candidates.length ? (
        <div className="candidate-table" role="table" aria-label="Yeafins candidate moves">
          <div className="candidate-row candidate-row--head" role="row">
            <span role="columnheader">Rank</span>
            <span role="columnheader">Move</span>
            <span role="columnheader">Policy</span>
            <span role="columnheader">Engine score</span>
          </div>
          {candidates.map((candidate) => (
            <div
              className={`candidate-row ${candidate.selected ? "is-selected" : ""}`}
              role="row"
              key={`${candidate.model_rank}-${candidate.move_uci}`}
            >
              <span role="cell">#{candidate.model_rank}</span>
              <strong role="cell">
                {candidate.move_san}
                {candidate.selected && <em>Selected</em>}
              </strong>
              <span role="cell">{(candidate.model_probability * 100).toFixed(1)}%</span>
              <span role="cell">{formatEngineScore(candidate.stockfish_cp)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="empty-state">Candidate analysis appears after Yeafins makes a move.</p>
      )}
      {styleWeight !== null && (
        <div className="weight-note">
          Resolved style weight <strong>{styleWeight.toFixed(2)}</strong>
        </div>
      )}
    </details>
  );
}

export function PlayerCard({
  name,
  detail,
  active,
  thinking = false,
  engine = false,
}: {
  name: string;
  detail: string;
  active: boolean;
  thinking?: boolean;
  engine?: boolean;
}) {
  return (
    <div className={`player-card ${active ? "is-active" : ""}`}>
      <div className={`player-avatar ${engine ? "player-avatar--engine" : ""}`} aria-hidden="true">
        {engine ? "Y" : "V"}
      </div>
      <div>
        <strong>{name}</strong>
        <span>{detail}</span>
      </div>
      <div className="player-state" aria-live="polite">
        {thinking ? (
          <>
            <span className="thinking-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            Thinking
          </>
        ) : active ? (
          "To move"
        ) : (
          "Waiting"
        )}
      </div>
    </div>
  );
}

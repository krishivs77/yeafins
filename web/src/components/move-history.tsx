import { Move } from "chess.js";

type Row = { number: number; white?: Move; black?: Move };

export function MoveHistory({
  rows,
  visitorColour,
  pgn,
}: {
  rows: Row[];
  visitorColour: "white" | "black";
  pgn: string;
}) {
  const copyPgn = async () => navigator.clipboard.writeText(pgn);
  const downloadPgn = () => {
    const url = URL.createObjectURL(new Blob([pgn], { type: "application/x-chess-pgn" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "visitor-vs-yeafins.pgn";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="panel history-panel" aria-labelledby="history-title">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Game record</span>
          <h2 id="history-title">Move history</h2>
        </div>
        <div className="panel-actions">
          <button type="button" onClick={() => void copyPgn()} disabled={!rows.length}>
            Copy PGN
          </button>
          <button type="button" onClick={downloadPgn} disabled={!rows.length}>
            Download
          </button>
        </div>
      </div>
      <div className="history-labels" aria-hidden="true">
        <span />
        <span>White</span>
        <span>Black</span>
      </div>
      <div className="history-list" role="list" aria-label="Moves in SAN">
        {rows.length ? (
          rows.map((row, index) => (
            <div className="history-row" role="listitem" key={row.number}>
              <span>{row.number}.</span>
              <span className={visitorColour === "white" ? "visitor-move" : "engine-move"}>
                {row.white?.san ?? "—"}
              </span>
              <span className={visitorColour === "black" ? "visitor-move" : "engine-move"}>
                {row.black?.san ?? (index === rows.length - 1 ? "…" : "—")}
              </span>
            </div>
          ))
        ) : (
          <p className="empty-state">Your moves will appear here once the game begins.</p>
        )}
      </div>
    </section>
  );
}

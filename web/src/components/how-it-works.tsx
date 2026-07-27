const steps = [
  ["01", "Predict", "A ResNet policy model predicts legal moves that resemble Krishiv’s historical play."],
  ["02", "Propose", "The model supplies its top 16 personalized legal candidates—never the whole Stockfish tree."],
  ["03", "Evaluate", "A Stockfish evaluator targeted around 2000 Elo scores only those proposed moves."],
  ["04", "Blend", "Style and engine quality combine with phase-aware weights: 0.20 / 0.10 / 0.20."],
];

const stats = [
  ["1,842", "accepted games"],
  ["56,255", "move positions"],
  ["≈31.3%", "held-out top-1"],
  ["≈52.9%", "held-out top-3"],
  ["≈63.2%", "held-out top-5"],
  ["≈73.0%", "top-8 coverage"],
];

export function HowItWorks() {
  return (
    <section className="how-section" id="how-it-works" aria-labelledby="how-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Under the board</span>
          <h2 id="how-title">Style proposes. Strength checks.</h2>
        </div>
        <p>
          Yeafins is a personalized hybrid, not a Stockfish wrapper. Its candidate set
          always begins with the learned policy.
        </p>
      </div>
      <div className="steps-grid">
        {steps.map(([number, title, body]) => (
          <article key={number}>
            <span>{number}</span>
            <h3>{title}</h3>
            <p>{body}</p>
          </article>
        ))}
      </div>
      <div className="stats-block">
        <div className="stats-intro">
          <span className="eyebrow">Model notebook</span>
          <h2>Historical move prediction</h2>
          <p>Legal-masked results on held-out positions from Krishiv’s game history.</p>
        </div>
        <div className="stats-grid">
          {stats.map(([value, label]) => (
            <div key={label}>
              <strong>{value}</strong>
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
      <p className="limitation">
        <strong>Reality check.</strong> Target evaluator strength is not equivalent to a measured
        playing rating for the combined engine. Games are not stored, and the current API uses one
        serialized evaluator worker.
      </p>
    </section>
  );
}

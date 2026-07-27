import { GameShell } from "@/components/game-shell";
import { HowItWorks } from "@/components/how-it-works";

export default function Home() {
  return (
    <>
      <header className="site-header">
        <a className="brand" href="#" aria-label="Yeafins home">
          <span className="brand-mark" aria-hidden="true">
            Y
          </span>
          <span>
            <strong>YEAFINS</strong>
            <small>Personal chess intelligence</small>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#play">Play</a>
          <a href="#how-it-works">How it works</a>
          <span className="tech-badge">Policy model + Stockfish hybrid</span>
        </nav>
      </header>

      <div className="page-grid" aria-hidden="true" />
      <section className="hero" id="play" aria-labelledby="hero-title">
        <div className="hero-copy">
          <span className="eyebrow">A personal AI chess experiment</span>
          <h1 id="hero-title">
            Play the moves
            <br />
            <em>Krishiv might make.</em>
          </h1>
          <p>
            Yeafins learned move preferences from 1,842 historical games, then pairs
            that style with a Stockfish evaluator targeted around 2000 Elo.
          </p>
        </div>
        <div className="hero-signal" aria-label="Engine pipeline">
          <span>Policy</span>
          <i />
          <span>16 candidates</span>
          <i />
          <span>Blend</span>
        </div>
      </section>

      <GameShell />
      <HowItWorks />

      <footer>
        <a className="brand" href="#" aria-label="Back to Yeafins top">
          <span className="brand-mark" aria-hidden="true">
            Y
          </span>
          <span>
            <strong>YEAFINS</strong>
            <small>Built from real games, not a persona prompt.</small>
          </span>
        </a>
        <p>Complete FEN in. Personalized move out. No games stored.</p>
      </footer>
    </>
  );
}

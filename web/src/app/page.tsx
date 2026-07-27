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
            <small>Play chess</small>
          </span>
        </a>
        <a className="header-link" href="#about">What is Yeafins?</a>
      </header>

      <section className="hero" id="play" aria-labelledby="hero-title">
        <h1 id="hero-title">Play Yeafins</h1>
        <p>Play against a chess engine trained on Krishiv&apos;s games.</p>
      </section>

      <GameShell />
      <HowItWorks />

      <footer>
        <strong>YEAFINS</strong>
        <p>Your game stays in this browser.</p>
      </footer>
    </>
  );
}

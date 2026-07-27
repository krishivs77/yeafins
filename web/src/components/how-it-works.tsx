export function HowItWorks() {
  return (
    <section className="about-section" id="about" aria-labelledby="about-title">
      <details>
        <summary id="about-title">What is Yeafins?</summary>
        <p>
          Yeafins is a personalized chess engine trained on Krishiv&apos;s historical
          games. It learns which moves he tends to consider, then uses a chess engine
          to choose a strong move from those options. The result is designed to play
          somewhat like Krishiv while remaining competitive.
        </p>
      </details>
    </section>
  );
}

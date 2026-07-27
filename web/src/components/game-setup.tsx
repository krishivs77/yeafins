import { ColourChoice } from "@/lib/chess";

const choices: Array<{ value: ColourChoice; label: string; symbol: string }> = [
  { value: "white", label: "Play as White", symbol: "♙" },
  { value: "black", label: "Play as Black", symbol: "♟" },
  { value: "random", label: "Random", symbol: "◐" },
];

export function GameSetup({
  choice,
  onChoice,
  onStart,
  disabled,
}: {
  choice: ColourChoice;
  onChoice: (choice: ColourChoice) => void;
  onStart: () => void;
  disabled: boolean;
}) {
  return (
    <section className="setup-card" aria-labelledby="setup-title">
      <h2 id="setup-title">Choose your side</h2>
      <p>Choose a colour, then start the game.</p>
      <div className="colour-options" role="radiogroup" aria-label="Choose your colour">
        {choices.map((item) => (
          <button
            key={item.value}
            className={`colour-option ${choice === item.value ? "is-selected" : ""}`}
            type="button"
            role="radio"
            aria-checked={choice === item.value}
            onClick={() => onChoice(item.value)}
          >
            <span aria-hidden="true">{item.symbol}</span>
            {item.label}
          </button>
        ))}
      </div>
      <button className="primary-button" type="button" onClick={onStart} disabled={disabled}>
        {disabled ? "Waiting for engine" : "Start Game"}
        <span aria-hidden="true">→</span>
      </button>
    </section>
  );
}

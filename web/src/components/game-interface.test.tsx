import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EngineStatus } from "@/components/engine-status";
import { GameResult } from "@/components/game-result";
import { GameSetup } from "@/components/game-setup";
import { HowItWorks } from "@/components/how-it-works";

describe("simplified game interface", () => {
  it("supports colour selection and disables start while the engine is unavailable", () => {
    const onChoice = vi.fn();
    const onStart = vi.fn();
    const { rerender } = render(
      <GameSetup
        choice="white"
        onChoice={onChoice}
        onStart={onStart}
        disabled
      />,
    );

    expect(screen.getByRole("button", { name: "Waiting for engine" })).toBeDisabled();
    fireEvent.click(screen.getByRole("radio", { name: "Play as Black" }));
    expect(onChoice).toHaveBeenCalledWith("black");

    rerender(
      <GameSetup
        choice="black"
        onChoice={onChoice}
        onStart={onStart}
        disabled={false}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Start Game" }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it("uses simple engine and game-over language", () => {
    const retry = vi.fn();
    const restart = vi.fn();
    render(
      <>
        <EngineStatus state="unavailable" retry={retry} />
        <GameResult
          outcome={{ result: "0-1", winner: "Yeafins", reason: "Checkmate" }}
          onRestart={restart}
        />
      </>,
    );

    expect(screen.getByText("Engine unavailable")).toBeInTheDocument();
    expect(screen.getByText("Yeafins wins")).toBeInTheDocument();
    expect(screen.getByText("Checkmate")).toBeInTheDocument();
  });

  it("keeps technical candidates, ratings, and metrics out of the visitor copy", () => {
    render(<HowItWorks />);

    expect(screen.getByText("What is Yeafins?")).toBeInTheDocument();
    expect(screen.queryByText(/candidate/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/2000/)).not.toBeInTheDocument();
    expect(screen.queryByText(/top-1/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/stockfish/i)).not.toBeInTheDocument();
  });
});

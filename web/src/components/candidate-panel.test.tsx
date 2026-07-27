import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CandidatePanel, formatEngineScore } from "@/components/candidate-panel";

describe("CandidatePanel", () => {
  it("marks the selected candidate and formats engine scores", () => {
    render(
      <CandidatePanel
        styleWeight={0.2}
        candidates={[
          {
            move_uci: "e7e5",
            move_san: "e5",
            model_rank: 1,
            model_probability: 0.42,
            stockfish_cp: 31,
            selected: true,
          },
        ]}
      />,
    );
    expect(screen.getByText("Selected")).toBeInTheDocument();
    expect(screen.getByText("42.0%")).toBeInTheDocument();
    expect(screen.getByText("+0.31")).toBeInTheDocument();
    expect(formatEngineScore(100_000)).toBe("decisive +");
  });
});

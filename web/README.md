# Yeafins web

The public Yeafins interface is a Next.js App Router application where visitors play a
complete chess game against the personalized engine. The browser owns legal game state
with `chess.js`; the separately deployed FastAPI service remains authoritative for every
Yeafins move.

## Stack

- Next.js, React, TypeScript, and Tailwind CSS
- `chess.js` for legal moves, outcomes, FEN, history, and PGN
- `react-chessboard` for responsive drag-and-drop and click interaction
- Vitest and Testing Library

## Local setup

Start the backend from the repository root:

```sh
YEAFINS_CHECKPOINT_PATH=runs/resnet_baseline/best.pt \
STOCKFISH_PATH=/opt/homebrew/bin/stockfish \
ALLOWED_ORIGINS=http://localhost:3000 \
yeafins-api
```

Then start the frontend:

```sh
cd web
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

`NEXT_PUBLIC_ENGINE_API_URL` is required in hosted environments and should contain the
FastAPI origin, such as `https://engine.example.com`. `NEXT_PUBLIC_` values are compiled
into browser code and visible to visitors; this URL is configuration, not a secret.

## Engine contract

Every public move request sends this fixed configuration:

```json
{
  "top_k": 16,
  "mode": "blended",
  "stockfish_elo": 2000,
  "depth": 10,
  "time_limit_seconds": null,
  "style_weight": null
}
```

The null style weight resolves on the backend to opening `0.20`, middlegame `0.10`, and
endgame `0.20`. The combined engine does not have a measured 2000 rating; that number is
the configured target for its Stockfish evaluator.

## Checks and production build

```sh
npm run lint
npm run typecheck
npm run test
npm run build
```

## Vercel

Import the repository in Vercel, set the Root Directory to `web`, and add
`NEXT_PUBLIC_ENGINE_API_URL` for Preview and Production. Use the standard Next.js build
with no custom server. The backend must be deployed separately on a container-capable
platform, and its `ALLOWED_ORIGINS` must include each exact Vercel frontend origin that
should call it.

Current limitations: no authentication, accounts, saved games, public undo, or persistent
sessions. Engine requests use a complete FEN and games remain only in browser memory.

# Yeafins API deployment

## Architecture

Yeafins uses a stateless Next.js/TypeScript frontend on Vercel and a separate
containerized FastAPI engine. The browser sends the complete FEN to the API. PyTorch
inference and Stockfish run only in the backend container because they require a
persistent, native runtime and are unsuitable for browser code or Vercel serverless
functions.

The frontend must set `NEXT_PUBLIC_ENGINE_API_URL` to the public HTTPS URL of this API.

## Local startup

Install the package, then run:

```sh
YEAFINS_CHECKPOINT_PATH=runs/resnet_baseline/best.pt \
STOCKFISH_PATH=/opt/homebrew/bin/stockfish \
ALLOWED_ORIGINS=http://localhost:3000 \
yeafins-api
```

Required deployment configuration:

- `YEAFINS_CHECKPOINT_PATH`: readable trained checkpoint path.
- `STOCKFISH_PATH`: Stockfish executable path. If omitted, `stockfish` is resolved from
  `PATH`.
- `ALLOWED_ORIGINS`: comma-separated exact frontend origins.

Optional configuration:

- `STOCKFISH_THREADS` (default `1`)
- `STOCKFISH_HASH_MB` (default `128`)
- `HOST` (default `0.0.0.0`)
- `PORT` (default `8000`)
- `LOG_LEVEL` (default `info`)

Local CORS defaults allow `http://localhost:3000` and `http://127.0.0.1:3000`.
Production should specify only its deployed frontend origins.

## Request and health check

```sh
curl http://localhost:8000/health

curl -X POST http://localhost:8000/move \
  -H 'Content-Type: application/json' \
  -d '{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "mode": "blended",
    "top_k": 16,
    "stockfish_elo": 2000,
    "depth": 10,
    "time_limit_seconds": null,
    "style_weight": null
  }'
```

`/health` returns HTTP 200 only when both the checkpoint-backed model and Stockfish
are ready. It returns HTTP 503 with readiness flags otherwise.

The public frontend defaults are `mode=blended`, `top_k=16`,
`stockfish_elo=2000`, `depth=10`, and `style_weight=null`. A null style weight uses
the existing phase-aware values: opening `0.20`, middlegame `0.10`, and endgame
`0.20`. Candidates in responses are ordered by ascending model rank.

## Docker

Build the runtime image:

```sh
docker build -t yeafins-api .
```

The checkpoint is intentionally not copied into the public build context. Mount it:

```sh
docker run --rm -p 8000:8000 \
  -e ALLOWED_ORIGINS=https://your-app.vercel.app \
  -v /absolute/private/path/best.pt:/app/models/best.pt:ro \
  yeafins-api
```

Alternatively, a private deployment build may copy the checkpoint to
`/app/models/best.pt` in a private derived image. Hosting-platform object storage or
a startup download to a persistent/private volume is also suitable.

## Current limits and production guidance

Version 1 uses one serialized Stockfish worker and typically performs CPU inference
in containers. It has no authentication, persistent game sessions, or server-side
game state. Add HTTPS, platform or reverse-proxy rate limiting, a request timeout,
restricted CORS, monitoring, and centralized logs before public exposure. Scale with
multiple container replicas when one serialized worker is insufficient.

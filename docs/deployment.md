# Yeafins API deployment

## Architecture

Yeafins uses a stateless Next.js frontend and a separate containerized FastAPI
service. The browser sends a complete FEN to the API. One application-wide PyTorch
policy model proposes moves, and one application-wide Stockfish process evaluates
them behind a serialized request lock.

The frontend must set `NEXT_PUBLIC_ENGINE_API_URL` to the public HTTPS origin of this
service. The backend must include the exact frontend origins in `ALLOWED_ORIGINS`.

## Why the first Render image exceeded 512 MB

The original production install used the default Linux PyTorch wheel and installed the
project's full data/training dependency set. That allowed pip to include CUDA, NVIDIA,
and Triton packages even though the service uses CPU inference. API startup also
imported the training dataset module through a shared device helper, loading pandas and
PyArrow into the server process. Finally, Stockfish reserved a 128 MB hash table.

The Render process completed FastAPI startup and bound Uvicorn successfully, then was
killed after its resident memory exceeded the free instance's 512 MB limit.

The production path now:

- installs the official CPU-only PyTorch 2.6.0 wheel;
- installs only API runtime dependencies;
- keeps training/data modules out of the API import graph;
- loads an inference-only checkpoint with `weights_only=True`;
- uses one model, one Stockfish process, and one Uvicorn worker;
- defaults Stockfish to one thread and a 16 MB hash table.

These changes materially reduce image size and resident memory, but memory remains
platform-dependent. Free-tier success is not guaranteed until verified on a real
Render deployment under startup and repeated move traffic.

## Runtime dependencies

The production package retains:

- `torch` from the official CPU wheel index;
- `numpy` for board tensors and legal-move masks;
- `chess` for positions, move encoding, and Stockfish UCI communication;
- `fastapi` and `pydantic` for HTTP and schemas;
- `uvicorn` without its optional development/performance extras.

The Docker image does not install pandas, PyArrow, scikit-learn, tqdm, httpx, PyYAML,
pytest, Ruff, or mypy. Local data, training, and development installations remain
available as:

```sh
pip install -e '.[data]'
pip install -e '.[training]'
pip install -e '.[dev]'
```

## Checkpoint strategy

`models/best.pt` is the resumable training checkpoint. It contains model weights and
configuration plus optimizer state, scheduler state, epoch metadata, training
configuration, and history.

The container uses `models/best_inference.pt`, which contains only:

- `format_version`;
- `model_config`;
- `model_state_dict`.

Regenerate it deterministically without overwriting the training checkpoint:

```sh
python scripts/export_inference_checkpoint.py \
  models/best.pt \
  models/best_inference.pt
```

Inference loads with `weights_only=True`, calls `model.eval()`, and runs forward passes
inside `torch.inference_mode()`.

## Build and verify the image

```sh
docker build -t yeafins-api .
docker run --rm yeafins-api python scripts/verify_cpu_runtime.py
```

The Docker build runs the same dependency verification and fails if an installed
distribution is named `triton` or begins with `nvidia-` or `cuda-`. It also asserts
that `torch.cuda.is_available()` is false.

Docker was not available in the development environment used for this optimization,
so these commands must be run in CI, Render, or another Docker-capable host.

## Render Free deployment

Create a Render Web Service from the repository and select Docker as the runtime. The
repository `Dockerfile` is the build and start definition. It explicitly launches:

```sh
uvicorn yeafins.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1
```

Do not add a second start command, development reload, or `WEB_CONCURRENCY` greater
than one. Each worker would load another policy model and start another Stockfish
process.

Recommended Render variables:

```text
YEAFINS_CHECKPOINT_PATH=/app/models/best_inference.pt
ALLOWED_ORIGINS=https://your-frontend.vercel.app
STOCKFISH_THREADS=1
STOCKFISH_HASH_MB=16
LOG_LEVEL=info
YEAFINS_LOG_MEMORY=true
```

Render supplies `PORT`. Keep `YEAFINS_LOG_MEMORY=true` during the first deployment and
load test, then disable it if the additional logs are no longer useful. A 16 MB
Stockfish hash reduces evaluator cache capacity but is safer within 512 MB.

CPU-only PyTorch and model initialization can still cause a noticeable free-tier cold
start. The health check allows a 60-second startup period.

## Memory diagnostics

Memory logging is opt-in and never changes API responses. With
`YEAFINS_LOG_MEMORY=true`, logs include checkpoints after:

- API module import;
- before engine startup;
- policy model load;
- Stockfish startup;
- completed engine startup;
- every completed move request.

For a local or container measurement using production settings:

```sh
python scripts/measure_api_memory.py \
  --checkpoint models/best_inference.pt \
  --stockfish-path /usr/games/stockfish \
  --requests 5
```

The script reports RSS at process start, after API imports, after model/Stockfish
startup, after the first move, and after repeated sequential moves. On Linux it reads
current RSS from `/proc/self/status`; other platforms use `resource`.

## Health and move verification

Inspect Render logs for:

- one Uvicorn worker;
- one policy-model load;
- one Stockfish startup;
- `Application startup complete`;
- memory checkpoints remaining below the instance limit;
- no CUDA/NVIDIA/Triton dependency-check failures.

Then verify readiness:

```sh
curl https://your-api.onrender.com/health
```

`/health` returns HTTP 200 only when both the policy model and Stockfish are ready. It
returns HTTP 503 with the existing readiness flags otherwise.

Verify a move without changing the public contract:

```sh
curl -X POST https://your-api.onrender.com/move \
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

The phase-aware style weights remain opening `0.20`, middlegame `0.10`, and endgame
`0.20`. Candidate ordering and structured errors are unchanged.

## Local startup

Install development dependencies, then run:

```sh
pip install -e '.[dev]'

YEAFINS_CHECKPOINT_PATH=models/best_inference.pt \
STOCKFISH_PATH=/opt/homebrew/bin/stockfish \
ALLOWED_ORIGINS=http://localhost:3000 \
STOCKFISH_THREADS=1 \
STOCKFISH_HASH_MB=16 \
yeafins-api
```

Local CORS defaults continue to allow `http://localhost:3000` and
`http://127.0.0.1:3000`.

## Remaining production guidance

The service still has no authentication or rate limiting. Use platform or reverse
proxy limits, restricted CORS, monitoring, and centralized logs before broad public
exposure. Render Free CPU availability and cold-start behavior vary. If measured RSS
still exceeds 512 MB after these changes, the next justified step is evaluating an
ONNX Runtime CPU export with numerical and candidate-equivalence tests—not weakening
health semantics or deferring an inevitable first-request OOM.

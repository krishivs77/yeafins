FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    STOCKFISH_PATH=/usr/games/stockfish \
    STOCKFISH_THREADS=1 \
    STOCKFISH_HASH_MB=16 \
    YEAFINS_CHECKPOINT_PATH=/app/models/best_inference.pt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts/verify_cpu_runtime.py ./scripts/verify_cpu_runtime.py

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 \
    && pip install --no-cache-dir . \
    && python scripts/verify_cpu_runtime.py \
    && python -c "import torch; assert not torch.cuda.is_available(); print(torch.__version__)"

RUN addgroup --system yeafins \
    && adduser --system --ingroup yeafins yeafins \
    && mkdir -p /app/models \
    && chown -R yeafins:yeafins /app

COPY --chown=yeafins:yeafins models/best_inference.pt /app/models/best_inference.pt

USER yeafins
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn yeafins.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

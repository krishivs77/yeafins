FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    STOCKFISH_PATH=/usr/games/stockfish \
    YEAFINS_CHECKPOINT_PATH=/app/models/best.pt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl stockfish \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN mkdir -p /app/models
COPY models/best.pt /app/models/best.pt

RUN pip install --no-cache-dir .

RUN addgroup --system yeafins \
    && adduser --system --ingroup yeafins yeafins \
    && mkdir -p /app/models \
    && chown -R yeafins:yeafins /app

USER yeafins
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${PORT:-8000}/health" > /dev/null || exit 1

CMD ["sh", "-c", "uvicorn yeafins.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

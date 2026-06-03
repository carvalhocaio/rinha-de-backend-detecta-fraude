# =========== builder ==============
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
RUN uv run --no-dev python -m src.preprocess

# ============ runtime =============
FROM python:3.13-slim AS runtime

WORKDIR /app

ENV MALLOC_ARENA_MAX=2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

COPY --from=builder /app/data /app/data
COPY --from=builder /app/resources/normalization.json /app/resources/normalization.json
COPY --from=builder /app/resources/mcc_risk.json /app/resources/mcc_risk.json

EXPOSE 9999

CMD [ "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "9999", "--workers", "1", "src.app:app" ]

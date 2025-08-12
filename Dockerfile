FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev


COPY pyproject.toml .
COPY uv.lock .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# RUN uv sync --no-dev 
# COPY . /app
COPY . .

FROM python:3.12-slim-bookworm
WORKDIR /app/app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
RUN which uvicorn && uvicorn --version

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
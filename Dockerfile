FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-package torch

COPY src/ src/
RUN uv pip install --no-deps -e .

ENTRYPOINT ["esmfold"]

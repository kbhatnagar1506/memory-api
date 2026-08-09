# syntax=docker/dockerfile:1
# Multi-stage: the build stage carries compilers, the runtime stage does not.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[gemini]"

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user: a container that does not need root should not have it.
RUN useradd --create-home --uid 10001 mapi
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER mapi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "mapi.main:app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

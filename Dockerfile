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
# The migrations travel with the code that expects them. Without these the
# image could not migrate its own database, so every schema change had to be
# run from a laptop checkout that might not match the deployed build -- and
# since staging and production now refuse to start on an unmigrated schema,
# "the image" and "the migration it needs" must be one artifact:
#   docker run --rm --env-file mapi.env <image> alembic upgrade head
# Owned by root and read-only to the runtime user: nothing at runtime writes
# to them.
COPY alembic.ini ./
COPY migrations ./migrations
USER mapi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status==200 else 1)"

# $PORT, not a literal, because Cloud Run injects it (8080) and refuses to route
# to a container listening anywhere else -- a hardcoded port is the single most
# common reason a working image fails to start there. Shell form so the variable
# expands; `exec` so uvicorn is PID 1 and receives SIGTERM directly, which is
# what makes Cloud Run's shutdown graceful instead of a 10-second kill. The
# default keeps `docker run -p 8000:8000` working locally.
CMD ["sh", "-c", "exec uvicorn mapi.main:app --factory --host 0.0.0.0 --port ${PORT:-8000}"]

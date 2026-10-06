FROM python:3.11-slim

# Required for git clone in repo ingestion
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Opt-in only. The default image has neither fastembed nor its model: the provider is optional,
# the wheels are large and the model is a download. Build with
#   docker build --build-arg INSTALL_FASTEMBED=1 .
# to install it and bake the model into the image, then run with
# CODEATLAS_EMBEDDING_PROVIDER=fastembed. Without the build arg this line copies one small file
# and the install is skipped, so the default build is unchanged.
ARG INSTALL_FASTEMBED=0
ARG FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
ENV FASTEMBED_CACHE_PATH=/app/.fastembed
COPY requirements-fastembed.txt .
RUN if [ "$INSTALL_FASTEMBED" = "1" ]; then \
      pip install --no-cache-dir -r requirements-fastembed.txt && \
      python -c "from fastembed import TextEmbedding; TextEmbedding('$FASTEMBED_MODEL')"; \
    fi

COPY codeatlas ./codeatlas
# Retrieval eval results, read at startup into the codeatlas_retrieval_* gauges on /metrics.
COPY eval/results ./eval/results
COPY README.md .

EXPOSE 8000

# Render sets $PORT. Fall back to 8000 locally.
# --no-proxy-headers: the app reads X-Forwarded-For itself (CODEATLAS_TRUSTED_PROXY_HOPS), so
# uvicorn must not rewrite the client address first.
CMD ["sh", "-c", "uvicorn codeatlas.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers"]

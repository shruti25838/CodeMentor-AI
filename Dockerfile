FROM python:3.11-slim

# Required for git clone in repo ingestion
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY codeatlas ./codeatlas
COPY README.md .

EXPOSE 8000

# Render sets $PORT. Fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn codeatlas.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

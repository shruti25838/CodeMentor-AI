FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY codeatlas ./codeatlas
COPY README.md .

EXPOSE 8000

CMD ["uvicorn", "codeatlas.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

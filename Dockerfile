# ---- Churn prediction API ----
# Build:  docker build -t churn-api .
# Run:    docker run -p 8000:8000 churn-api      (docs at http://localhost:8000/docs)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models PREDICTION_LOG=/app/logs/predictions.jsonl PORT=8000

RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# The trained pipeline + model card (built by `python -m churn.train`)
COPY models ./models
RUN mkdir -p /app/logs && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')" || exit 1
CMD ["sh", "-c", "uvicorn churn.api:app --host 0.0.0.0 --port ${PORT}"]

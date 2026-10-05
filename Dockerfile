# Slim, PyTorch-free image for the demo. The model is pre-trained (deploy/serve/), so nothing heavy runs at startup.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user deploy/serve ./deploy/serve
COPY --chown=user data/processed/results.csv data/processed/qualifying.csv data/processed/sprints.csv ./data/processed/
COPY --chown=user reports/backtest_scores.csv reports/metrics_overall.csv ./reports/
COPY --chown=user predictions ./predictions

EXPOSE 7860
# Hosts like Render inject $PORT; default to 7860 elsewhere.
CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-7860}"]

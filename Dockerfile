# Hugging Face Spaces (Docker SDK) / any container host. Serves the demo on port 7860.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small (the server never uses a GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user data/processed/results.csv data/processed/qualifying.csv data/processed/sprints.csv ./data/processed/
COPY --chown=user reports/backtest_scores.csv reports/metrics_overall.csv reports/final_hyperparams.json ./reports/
COPY --chown=user predictions ./predictions

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]

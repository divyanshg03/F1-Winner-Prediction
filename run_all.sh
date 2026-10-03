#!/usr/bin/env bash
# Reproduce everything: data -> tests -> backtest -> figures -> next-race prediction.
set -euo pipefail
python -m pip install -r requirements.txt
(cd src && python -m f1pred.ingest 2014 "$(date +%Y)")
python -m pytest tests -q
python scripts/run_backtest.py
python scripts/holdout_report.py
python scripts/make_figures.py
python scripts/predict_next.py

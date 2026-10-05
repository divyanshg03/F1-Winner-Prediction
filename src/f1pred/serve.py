"""Torch-free scoring for the demo/deployment: loads weights saved by scripts/build_serve_artifacts.py.

score = 0.5 * conditional-logit utility + 0.5 * LightGBM log-odds (same blend as the live model in the demo).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ServeModel:
    def __init__(self, folder: Path):
        import lightgbm as lgb

        a = json.loads((folder / "logit.json").read_text(encoding="utf8"))
        self.cols = a["cols"]
        self.mu, self.sd, self.w = (np.array(a[k], dtype=float) for k in ("mu", "sd", "w"))
        self.gbm = lgb.Booster(model_file=str(folder / "gbm.txt"))

    def score(self, X: np.ndarray) -> np.ndarray:
        z = np.nan_to_num((X - self.mu) / self.sd, nan=0.0) @ self.w
        p = np.clip(self.gbm.predict(X), 1e-6, 1 - 1e-6)
        return 0.5 * z + 0.5 * np.log(p / (1 - p))

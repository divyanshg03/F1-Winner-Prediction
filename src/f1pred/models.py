"""Models that turn per-driver features into a *race-level* winner distribution.

A race has exactly one winner, so we always model P(driver i wins | field) with a
softmax across the drivers of that race rather than 20 independent yes/no problems.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def race_softmax(score: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Softmax of `score` within each race."""
    out = np.zeros_like(score, dtype=float)
    for r in pd.unique(race_ids):
        m = race_ids == r
        s = score[m] - score[m].max()
        e = np.exp(s)
        out[m] = e / e.sum()
    return out


def race_normalise(p: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=float)
    for r in pd.unique(race_ids):
        m = race_ids == r
        out[m] = p[m] / p[m].sum()
    return out


class _Std:
    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def __call__(self, X):
        return np.nan_to_num((X - self.mu) / self.sd, nan=0.0)


class ConditionalLogit:
    """Multinomial/conditional logit: score_i = w.x_i ; P(i wins) = softmax over the race.

    Fit by L-BFGS on the race-level log-likelihood with an L2 penalty. Interpretable
    (the weights *are* the story) and hard to overfit on ~200 races.
    """

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    @staticmethod
    def _pack(X, y, race_ids):
        races = pd.unique(race_ids)
        dmax = max((race_ids == r).sum() for r in races)
        F = X.shape[1]
        Xp = np.zeros((len(races), dmax, F))
        mask = np.zeros((len(races), dmax), dtype=bool)
        win = np.zeros(len(races), dtype=int)
        for i, r in enumerate(races):
            idx = np.where(race_ids == r)[0]
            Xp[i, : len(idx)] = X[idx]
            mask[i, : len(idx)] = True
            if y is not None:
                w = np.where(y[idx] == 1)[0]
                win[i] = w[0] if len(w) else 0
        return Xp, mask, win

    def fit(self, X, y, race_ids):
        import torch  # only training needs torch; serving does not

        self.std = _Std().fit(X)
        Xp, mask, win = self._pack(self.std(X), y, race_ids)
        Xt = torch.tensor(Xp, dtype=torch.float64)
        mt = torch.tensor(mask)
        wt = torch.tensor(win)
        w = torch.zeros(Xt.shape[2], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            s = (Xt @ w).masked_fill(~mt, -1e9)
            nll = -(s.gather(1, wt[:, None]).squeeze(1) - torch.logsumexp(s, dim=1)).mean()
            loss = nll + self.l2 * (w ** 2).sum() / len(wt)
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach().numpy()
        return self

    def score(self, X):
        return self.std(X) @ self.w

    def predict(self, X, race_ids):
        return race_softmax(self.score(X), race_ids)


class BoostedRace:
    """LightGBM binary classifier, then renormalised so each race sums to 1."""

    def __init__(self, seed: int = 7, **kw):
        import lightgbm as lgb

        params = dict(n_estimators=220, learning_rate=0.03, num_leaves=6, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=8.0,
                      random_state=seed, verbose=-1, n_jobs=4)
        params.update(kw)
        self.m = lgb.LGBMClassifier(**params)

    def fit(self, X, y, race_ids):
        self.m.fit(X, y.astype(int))
        return self

    def score(self, X):
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, X, race_ids):
        return race_softmax(self.score(X), race_ids)


def temperature_fit(scores_by_race: list[tuple[np.ndarray, int]]) -> float:
    """Single temperature T minimising winner NLL: p = softmax(score / T)."""
    if len(scores_by_race) < 30:
        return 1.0
    from scipy.optimize import minimize_scalar

    def nll(logT):
        T = np.exp(logT)
        tot = 0.0
        for s, w in scores_by_race:
            z = s / T
            z = z - z.max()
            tot -= z[w] - np.log(np.exp(z).sum())
        return tot / len(scores_by_race)

    return float(np.exp(minimize_scalar(nll, bounds=(-1.0, 1.5), method="bounded").x))

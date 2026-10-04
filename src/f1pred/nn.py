"""Neural race model: a small MLP scores each driver; softmax over the field gives win probabilities.

Same race-level cross-entropy as the conditional logit, but the score function is non-linear.
Several seeds are averaged in score space.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import ConditionalLogit, _Std, race_softmax


class NeuralRace:
    def __init__(self, hidden: int = 32, dropout: float = 0.2, wd: float = 1e-3, epochs: int = 40, lr: float = 3e-3, seeds: int = 5):
        self.h, self.p, self.wd, self.epochs, self.lr, self.seeds = hidden, dropout, wd, epochs, lr, seeds

    def _net(self, f: int) -> nn.Module:
        return nn.Sequential(nn.Linear(f, self.h), nn.ReLU(), nn.Dropout(self.p), nn.Linear(self.h, self.h), nn.ReLU(),
                             nn.Dropout(self.p), nn.Linear(self.h, 1))

    def fit(self, X, y, race_ids):
        self.std = _Std().fit(X)
        Xp, mask, win = ConditionalLogit._pack(self.std(X), y, race_ids)
        Xt, mt, wt = torch.tensor(Xp, dtype=torch.float32), torch.tensor(mask), torch.tensor(win)
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = self._net(Xt.shape[2])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                net.train(); opt.zero_grad()
                s = net(Xt).squeeze(-1).masked_fill(~mt, -1e4)
                loss = -(s.gather(1, wt[:, None]).squeeze(1) - torch.logsumexp(s, 1)).mean()
                loss.backward(); opt.step()
            self.nets.append(net.eval())
        return self

    def score(self, X):
        x = torch.tensor(self.std(X), dtype=torch.float32)
        with torch.no_grad():
            return np.mean([n(x).squeeze(-1).numpy() for n in self.nets], axis=0)

    def predict(self, X, race_ids):
        return race_softmax(self.score(X), race_ids)

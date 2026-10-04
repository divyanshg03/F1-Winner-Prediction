"""Models and features that use more of each race than just the winner.

PlackettLuce  : trains on the whole finishing order (top-K positions), not only who won. Torch, runs on the GPU.
add_ratings   : Elo-style driver + team ratings updated after every race from the full finishing order.
Chaos         : P(win) = P(finishes) x P(win | finishes), via an offset in the conditional logit.
SkBinary      : wraps decision tree / random forest / CatBoost so they give race-level win probabilities.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .features import RESET_SEASONS, SEASON_CARRYOVER
from .models import ConditionalLogit, _Std, race_softmax

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELO = ["drv_elo", "con_elo", "elo_sum", "elo_rel", "elo_logp"]


# --------------------------------------------------------------------------- Plackett-Luce
def _pack_sorted(X, pos, race_ids):
    races = pd.unique(race_ids)
    dmax = max((race_ids == r).sum() for r in races)
    Xp = np.zeros((len(races), dmax, X.shape[1]), dtype=np.float32)
    mask = np.zeros((len(races), dmax), dtype=bool)
    for i, r in enumerate(races):
        idx = np.where(race_ids == r)[0]
        order = idx[np.argsort(np.where(np.isnan(pos[idx]), 1e9, pos[idx]), kind="stable")]
        Xp[i, : len(idx)] = X[order]
        mask[i, : len(idx)] = True
    return Xp, mask


class PlackettLuce:
    """Score each driver; P(order) = prod_k softmax over the drivers not yet placed. Only the top-K places are used
    (lower places are mostly chaos: retirements, lapped cars). K=1 is the plain winner-only model."""

    def __init__(self, K=6, l2=3.0, hidden=0, epochs=150, wd=1e-3, dropout=0.2, seeds=5):
        self.K, self.l2, self.hidden, self.epochs, self.wd, self.dropout = K, l2, hidden, epochs, wd, dropout
        self.seeds = seeds if hidden else 1

    def _net(self, f):
        if not self.hidden:
            return nn.Linear(f, 1, bias=False)
        h = self.hidden
        return nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(h, h), nn.ReLU(),
                             nn.Dropout(self.dropout), nn.Linear(h, 1))

    def fit(self, X, pos, race_ids):
        self.std = _Std().fit(X)
        Xp, mask = _pack_sorted(self.std(X).astype(np.float32), pos, race_ids)
        Xt, mt = torch.tensor(Xp, device=DEV), torch.tensor(mask, device=DEV)
        n = mt.sum(1)
        kk = torch.arange(Xt.shape[1], device=DEV)[None, :]
        use = (kk < self.K) & mt & (kk < (n[:, None] - 1))
        R = Xt.shape[0]

        def nll(net):
            s = net(Xt).squeeze(-1).masked_fill(~mt, -1e4)
            lse = torch.flip(torch.logcumsumexp(torch.flip(s, [1]), 1), [1])
            return -((s - lse) * use).sum(1).mean()

        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = self._net(Xt.shape[2]).to(DEV)
            if not self.hidden:
                opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=150, tolerance_grad=1e-7, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad()
                    loss = nll(net) + self.l2 * sum((p ** 2).sum() for p in net.parameters()) / R
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=self.wd)
                for _ in range(self.epochs):
                    net.train(); opt.zero_grad()
                    nll(net).backward(); opt.step()
            self.nets.append(net.eval())
        return self

    def score(self, X):
        x = torch.tensor(self.std(X).astype(np.float32), device=DEV)
        with torch.no_grad():
            return np.mean([n(x).squeeze(-1).cpu().numpy() for n in self.nets], axis=0)


# --------------------------------------------------------------------------- Elo ratings
def add_ratings(df: pd.DataFrame, kd: float = 0.08, kc: float = 0.15) -> pd.DataFrame:
    """Pre-race driver and team ratings (logit units), updated from the full finishing order of every earlier race."""
    df = df.sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
    rd, rc = defaultdict(float), defaultdict(float)
    out = np.zeros((len(df), 3))
    last = None
    for key, g in df.groupby("race_key", sort=False):
        season = int(g.season.iloc[0])
        if season != last:  # new season: drivers keep most of their rating, teams regress (more at a regulation reset)
            f = RESET_SEASONS.get(season, SEASON_CARRYOVER)
            for k in list(rc):
                rc[k] *= f
            for k in list(rd):
                rd[k] *= 0.9
            last = season
        d, c = g.driver_id.to_numpy(), g.constructor_id.to_numpy()
        r_d = np.array([rd[x] for x in d]); r_c = np.array([rc[x] for x in c])
        out[g.index.to_numpy()] = np.c_[r_d, r_c, r_d + r_c]
        pos = g.pos.to_numpy(float)
        if np.isnan(pos).all():
            continue
        s = r_d + r_c
        n = len(g)
        P = 1 / (1 + np.exp(-(s[:, None] - s[None, :])))
        np.fill_diagonal(P, 0)
        exp_ = P.sum(1) / (n - 1)
        act = (pos[:, None] < pos[None, :]).sum(1) / (n - 1)
        delta = act - exp_
        wd = np.where(g["mech"].to_numpy() == 1, 0.25, 1.0)  # a mechanical failure is mostly the car's fault, not the driver's
        for i in range(n):
            rd[d[i]] += kd * delta[i] * wd[i]
            rc[c[i]] += kc * delta[i]
    df["drv_elo"], df["con_elo"], df["elo_sum"] = out[:, 0], out[:, 1], out[:, 2]
    df["elo_rel"] = df.elo_sum - df.groupby("race_key").elo_sum.transform("max")
    ex = np.exp(df.elo_sum - df.groupby("race_key").elo_sum.transform("max"))
    df["elo_logp"] = np.log(ex / ex.groupby(df.race_key).transform("sum"))
    return df


# --------------------------------------------------------------------------- chaos decomposition
class OffsetLogit:
    """Conditional logit with a fixed per-driver offset added to the score."""

    def __init__(self, l2=3.0):
        self.l2 = l2

    def fit(self, X, y, race_ids, offset):
        self.std = _Std().fit(X)
        Xp, mask, win = ConditionalLogit._pack(np.c_[self.std(X), offset], y, race_ids)
        Xt, mt, wt = torch.tensor(Xp, dtype=torch.float64), torch.tensor(mask), torch.tensor(win)
        w = torch.zeros(Xt.shape[2] - 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            s = (Xt[..., :-1] @ w + Xt[..., -1]).masked_fill(~mt, -1e9)
            loss = -(s.gather(1, wt[:, None]).squeeze(1) - torch.logsumexp(s, 1)).mean() + self.l2 * (w ** 2).sum() / len(wt)
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach().numpy()
        return self

    def score(self, X, offset):
        return self.std(X) @ self.w + offset


class Chaos:
    """P(win) proportional to P(finishes) x exp(skill utility): the utility is trained with the finish risk held fixed."""

    DNF = ["drv_dnf", "con_mech", "grid_rank_sq", "front_row", "drv_rookie"]

    def __init__(self, cols, l2=3.0):
        self.cols, self.l2 = cols, l2

    def _off(self, d):
        return np.log(np.clip(1 - self.dm.predict_proba(d[self.DNF].fillna(0).to_numpy(float))[:, 1], 1e-3, 1))

    def fit(self, d):
        self.dm = LogisticRegression(C=1.0, max_iter=500).fit(d[self.DNF].fillna(0).to_numpy(float), d["dnf"].astype(int))
        self.ol = OffsetLogit(self.l2).fit(d[self.cols].to_numpy(float), d.won.to_numpy(), d.race_key.to_numpy(), self._off(d))
        return self

    def score(self, d):
        return self.ol.score(d[self.cols].to_numpy(float), self._off(d))


# --------------------------------------------------------------------------- tree models
class SkBinary:
    """decision tree / random forest / CatBoost -> log-odds score; race softmax gives win probabilities."""

    def __init__(self, est, nan_fill=True):
        self.est, self.nan_fill = est, nan_fill

    def fit(self, X, y, race_ids):
        self.est.fit(np.nan_to_num(X) if self.nan_fill else X, y.astype(int))
        return self

    def score(self, X):
        p = np.clip(self.est.predict_proba(np.nan_to_num(X) if self.nan_fill else X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, X, race_ids):
        return race_softmax(self.score(X), race_ids)

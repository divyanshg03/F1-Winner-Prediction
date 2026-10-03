"""Walk-forward backtest: before every race we refit on strictly earlier races only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from .models import BoostedRace, ConditionalLogit, race_softmax, temperature_fit

PRE = F.FORM_FEATS
POST = F.FORM_FEATS + F.QUALI_FEATS
GRID_ONLY = ["grid", "grid_rank_sq"]

MODELS = {  # name -> (estimator factory, feature list)
    "grid_logit": (lambda: ConditionalLogit(l2=1.0), GRID_ONLY),
    "logit_pre": (lambda: ConditionalLogit(l2=3.0), PRE),
    "logit_post": (lambda: ConditionalLogit(l2=3.0), POST),
    "gbm_pre": (lambda: BoostedRace(), PRE),
    "gbm_post": (lambda: BoostedRace(), POST),
}


def walk_forward(df: pd.DataFrame, first_test_season: int = 2018, last_test_season: int = 2100, refit_every: int = 1, verbose=True) -> pd.DataFrame:
    """Return df of out-of-sample rows with a score column per model plus calibrated ensembles."""
    df = df[df["won"].notna()].sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
    race_order = df.drop_duplicates("race_key")[["race_key", "season", "date"]].reset_index(drop=True)
    test_races = race_order[(race_order.season >= first_test_season) & (race_order.season <= last_test_season)].race_key.tolist()
    rid = df["race_key"].to_numpy()
    out = []
    fitted: dict[str, object] = {}
    since = refit_every
    for k, race in enumerate(test_races):
        te = rid == race
        tr = df["date"].to_numpy() < df.loc[te, "date"].iloc[0]
        # races of same date never happen, so strict < is safe
        rec = df.loc[te, ["race_key", "season", "round", "race_name", "driver", "driver_id", "constructor",
                          "grid", "won"]].copy()
        if since >= refit_every:
            for name, (mk, cols) in MODELS.items():
                fitted[name] = mk().fit(df.loc[tr, cols].to_numpy(float), df.loc[tr, "won"].to_numpy(), rid[tr])
            since = 0
        since += 1
        for name, (mk, cols) in MODELS.items():
            rec[f"s_{name}"] = fitted[name].score(df.loc[te, cols].to_numpy(float))
        out.append(rec)
        if verbose and k % 25 == 0:
            print(f"  walk-forward {k}/{len(test_races)} {race}", flush=True)
    res = pd.concat(out, ignore_index=True)
    return add_ensembles(res)


def add_ensembles(res: pd.DataFrame) -> pd.DataFrame:
    """Blend logit+gbm scores; temperature is fit ONLY on earlier out-of-sample races."""
    res = res.copy()
    # logit scores are log-odds-like utilities; gbm scores are log-odds; both sit on a comparable scale
    res["s_ens_pre"] = 0.5 * res["s_logit_pre"] + 0.5 * res["s_gbm_pre"]
    res["s_ens_post"] = 0.5 * res["s_logit_post"] + 0.5 * res["s_gbm_post"]
    keys = res.drop_duplicates("race_key")["race_key"].tolist()
    for name in ["ens_pre", "ens_post"]:
        hist, cal = [], np.zeros(len(res))
        for key in keys:
            m = (res["race_key"] == key).to_numpy()
            T = temperature_fit(hist)
            cal[m] = res.loc[m, f"s_{name}"].to_numpy() / T
            s = res.loc[m, f"s_{name}"].to_numpy()
            w = int(np.argmax(res.loc[m, "won"].to_numpy()))
            hist.append((s, w))
        res[f"s_{name}_cal"] = cal
    return res


def probs(res: pd.DataFrame, name: str) -> np.ndarray:
    col = f"s_{name}"
    return race_softmax(res[col].to_numpy(float), res["race_key"].to_numpy())


def race_metrics(res: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Per-race winner log-loss, top-1, top-3, multiclass Brier."""
    d = res[["race_key", "won"]].copy()
    d["p"] = p
    rows = []
    for key, g in d.groupby("race_key", sort=False):
        w = int(np.argmax(g["won"].to_numpy()))
        order = np.argsort(-g["p"].to_numpy())
        rows.append(dict(race_key=key, logloss=-np.log(max(g["p"].iloc[w], 1e-9)),
                         top1=float(order[0] == w), top3=float(w in order[:3]),
                         brier=float(((g["p"].to_numpy() - g["won"].to_numpy()) ** 2).sum()),
                         p_winner=float(g["p"].iloc[w]), p_max=float(g["p"].max())))
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, names: list[str], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for nm in names:
        m = race_metrics(res, probs(res, nm))
        idx = rng.integers(0, len(m), (n_boot, len(m)))
        ll = m["logloss"].to_numpy()[idx].mean(1)
        t1 = m["top1"].to_numpy()[idx].mean(1)
        rows.append(dict(model=nm, races=len(m), top1=m.top1.mean(), top1_lo=np.percentile(t1, 2.5),
                         top1_hi=np.percentile(t1, 97.5), top3=m.top3.mean(), logloss=m.logloss.mean(),
                         ll_lo=np.percentile(ll, 2.5), ll_hi=np.percentile(ll, 97.5), brier=m.brier.mean()))
    return pd.DataFrame(rows)


def paired_diff(res: pd.DataFrame, a: str, b: str, mask=None, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean per-race log-loss advantage of model `a` over `b` (positive = a better) with a bootstrap CI."""
    sel = res if mask is None else res[mask]
    ma = race_metrics(sel, probs(sel, a)).set_index("race_key")["logloss"]
    mb = race_metrics(sel, probs(sel, b)).set_index("race_key")["logloss"]
    d = (mb - ma).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return dict(a=a, b=b, races=len(d), diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

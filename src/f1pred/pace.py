"""Race-pace model: predict each driver's pace deficit (fraction of a lap behind the fastest car)
from this weekend's qualifying and *earlier* races' lap data. Also a retirement-risk model.

The residual spread of this model is the "pace distribution" the simulator samples from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .features import RESET_SEASONS, SEASON_CARRYOVER, _Ewma

PRIOR_GAP = 0.010
PACE_FEATS = ["qual_gap", "con_qual_best", "drv_pace_s", "drv_pace_l", "con_pace_s", "con_pace_l", "drv_rookie"]
QUALI_ONLY = ["qual_gap"]


def add_pace_history(df: pd.DataFrame, lap_driver: pd.DataFrame) -> pd.DataFrame:
    """Add leak-free EWMA race-pace features and the realised target `y_gap` (NaN if unknown)."""
    ld = lap_driver.dropna(subset=["pace_rel"]).copy()
    ld["y"] = ld["pace_rel"] - ld.groupby("race_key")["pace_rel"].transform("min")
    ymap = ld.set_index(["race_key", "driver_id"])["y"].to_dict()
    D = dict(s=_Ewma(4, PRIOR_GAP), l=_Ewma(12, PRIOR_GAP))
    C = dict(s=_Ewma(3, PRIOR_GAP), l=_Ewma(10, PRIOR_GAP))
    out, last_season = [], None
    for key in df.drop_duplicates("race_key").sort_values("date")["race_key"]:
        r = df[df.race_key == key].copy()
        season = int(r.season.iloc[0])
        if season != last_season:
            f = RESET_SEASONS.get(season, SEASON_CARRYOVER)
            for e in C.values():
                for k in list(e.den):
                    e.discount(k, f)
            last_season = season
        r["drv_pace_s"] = [D["s"].get(k) for k in r.driver_id]
        r["drv_pace_l"] = [D["l"].get(k) for k in r.driver_id]
        r["con_pace_s"] = [C["s"].get(k) for k in r.constructor_id]
        r["con_pace_l"] = [C["l"].get(k) for k in r.constructor_id]
        r["y_gap"] = [ymap.get((key, d), np.nan) for d in r.driver_id]
        out.append(r)
        known = r.dropna(subset=["y_gap"])
        if len(known):
            cy = known.groupby("constructor_id")["y_gap"].mean()
            for row in known.itertuples():
                D["s"].update(row.driver_id, row.y_gap)
                D["l"].update(row.driver_id, row.y_gap)
            for c, v in cy.items():
                C["s"].update(c, v)
                C["l"].update(c, v)
    return pd.concat(out, ignore_index=True)


class PaceModel:
    def __init__(self, feats=PACE_FEATS, alpha=3.0):
        self.feats, self.alpha = feats, alpha

    def fit(self, d: pd.DataFrame):
        d = d.dropna(subset=["y_gap"])
        self.sc = StandardScaler().fit(d[self.feats])
        self.m = Ridge(alpha=self.alpha).fit(self.sc.transform(d[self.feats]), d["y_gap"])
        res = d["y_gap"] - self.m.predict(self.sc.transform(d[self.feats]))
        self.sigma = float(res.std())
        self.n = len(d)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        p = self.m.predict(self.sc.transform(d[self.feats]))
        return np.clip(p, 0.0, None)


class RetireModel:
    """P(driver does not finish) from recent personal and team retirement rates."""

    FEATS = ["drv_dnf", "con_mech"]

    def fit(self, d: pd.DataFrame):
        d = d.dropna(subset=self.FEATS + ["dnf"])
        self.m = LogisticRegression(C=2.0).fit(d[self.FEATS], d["dnf"].astype(int))
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(d[self.FEATS].fillna(0.1))[:, 1]

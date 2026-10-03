"""Walk-forward driver for the race simulator (setup building, calibration, scoring)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import features as F
from . import lapdata
from .pace import PACE_FEATS, PaceModel, RetireModel, add_pace_history
from .sim import Params, RaceSetup, laplace, simulate


class Context:
    """Everything the simulator needs, indexed so that 'history' is strictly earlier races."""

    def __init__(self, df: pd.DataFrame | None = None, rebuild: bool = False):
        df = F.make() if df is None else df
        D, R = lapdata.build_all(force=rebuild)
        df = df[df.won.notna() | df.won.isna()]
        self.df = add_pace_history(df, D)
        self.D = D.set_index(["race_key", "driver_id"])
        self.R = R.set_index("race_key")
        order = self.df.drop_duplicates("race_key").sort_values("date")
        self.keys = order.race_key.tolist()
        self.date = dict(zip(order.race_key, order.date))
        self.circuit = dict(zip(order.race_key, order.circuit_id))
        # 2021 Belgian GP (3 laps behind the safety car, half points) has no racing to simulate
        self.lap_keys = [k for k in self.keys if k in self.R.index and self.R.loc[k, "total_laps"] >= 20]

    def prior(self, key: str) -> list[str]:
        return [k for k in self.lap_keys if self.date[k] < self.date[key]]

    # ------------------------------------------------------------------------------------
    def setup(self, key: str, force_sc=None, feats=PACE_FEATS) -> tuple[RaceSetup, dict]:
        r = self.df[self.df.race_key == key].reset_index(drop=True)
        prior_keys = self.prior(key)
        train = self.df[self.df.race_key.isin(prior_keys)]
        pm = PaceModel(feats).fit(train)
        rm = RetireModel().fit(self.df[(self.df.date < self.date[key]) & self.df.won.notna()])
        pace = pm.predict(r)
        circ_prior = [k for k in prior_keys if self.circuit[k] == self.circuit[key]]
        Rp = self.R.loc[prior_keys]
        if key in self.R.index:
            total = int(self.R.loc[key, "total_laps"])
        elif circ_prior:
            total = int(self.R.loc[circ_prior[-1], "total_laps"])
        else:
            total = int(Rp.total_laps.median())
        base = float(self.R.loc[circ_prior[-1], "base_lap_s"]) if circ_prior else float(Rp.base_lap_s.median())
        deg = float(np.clip(Rp.deg.median(), 0.0, 0.002))
        pl = self.R.loc[circ_prior, "pit_loss"].dropna() if circ_prior else pd.Series(dtype=float)
        pit_loss = float(pl.median()) if len(pl) else float(Rp.pit_loss.median())
        # strategies of finishers at this circuit (fallback: everyone)
        def strategies(keys):
            out = []
            for k in keys:
                sub = self.D.loc[k]
                for _, row in sub.iterrows():
                    if row.last_lap >= 0.9 * self.R.loc[k, "total_laps"]:
                        out.append(json.loads(row.pit_fracs))
            return out
        strat = strategies(circ_prior) if circ_prior else []
        if len(strat) < 10:
            strat = strategies(prior_keys[-40:])
        sc_counts = list(self.R.loc[circ_prior, "n_sc"]) if len(circ_prior) >= 3 else []
        sc_counts = sc_counts + list(Rp.n_sc.tail(60)) if len(sc_counts) < 6 else sc_counts
        starts, lens = [], []
        for k, row in Rp.tail(80).iterrows():
            starts += json.loads(row.sc_starts)
            lens += json.loads(row.sc_lens)
        setup = RaceSetup(
            drivers=r.driver_id.tolist(), grid=r.grid.rank(method="first").to_numpy(), pace=pace, pace_sigma=pm.sigma,
            p_retire=rm.predict(r), total_laps=total, base_lap_s=base, deg=deg, pit_loss=pit_loss,
            strategies=strat, sc_counts=sc_counts or [0, 1], sc_starts=starts or [0.3], sc_lens=lens or [4],
            force_sc=force_sc,
        )
        return setup, dict(pace_model=pm, rows=r)

    # ------------------------------------------------------------------------------------
    def run(self, key: str, prm: Params, seed: int = 0, **kw) -> pd.DataFrame:
        setup, aux = self.setup(key, **kw)
        out = simulate(setup, prm, seed)
        r = aux["rows"]
        return pd.DataFrame({
            "race_key": key, "driver_id": r.driver_id, "p_sim_raw": out["win"], "p_sim": laplace(out["win"], prm.s),
            "p_podium": out["podium"], "exp_pos": out["exp_pos"], "p_sc": out["had_sc"],
        })


def logloss_of(df: pd.DataFrame, won: pd.DataFrame, col="p_sim") -> float:
    m = df.merge(won, on=["race_key", "driver_id"])
    w = m[m.won == 1]
    return float(-np.log(w[col]).mean())

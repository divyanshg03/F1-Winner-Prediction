"""Leak-free feature engineering.

Everything about a race is computed from sessions that finished *before* it.
The one exception is the same weekend's qualifying/sprint, which is known before
lights-out and is only used by the "post-quali" feature set.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .ingest import PROC

# Seasons with a big regulation reset: constructor form is heavily discounted.
RESET_SEASONS = {2014: 0.25, 2017: 0.6, 2022: 0.25, 2026: 0.25}
SEASON_CARRYOVER = 0.7  # constructor form carried into any other new season

PRIOR = dict(perf=0.45, win=0.05, podium=0.15, qgap=0.012, gain=0.0, dnf=0.12, mech=0.06)

# feature groups -----------------------------------------------------------------
QUALI_FEATS = ["grid", "grid_rank_sq", "front_row", "qual_gap", "qual_pos", "sprint_rank",
               "is_pole", "grid_pen", "con_qual_best", "con_qual_mean", "con_qual_best_rel"]
FORM_FEATS = [
    "drv_perf_s", "drv_perf_l", "drv_win", "drv_podium", "drv_qgap", "drv_gain", "drv_dnf",
    "con_perf_s", "con_perf_l", "con_win", "con_podium", "con_qgap", "con_mech",
    "drv_circ", "con_circ", "drv_rookie", "races_since_reset",
    "con_perf_xs", "con_qgap_xs", "con_win_xs",
    "drv_perf_s_rel", "con_perf_s_rel", "con_win_rel", "drv_win_rel", "drv_qgap_rel", "con_qgap_rel",
]


def _secs(t) -> float:
    if not isinstance(t, str) or not t:
        return np.nan
    m = re.match(r"^(\d+):(\d+\.\d+)$", t)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else np.nan


_DRIVER_FAULT = re.compile(r"collision|accident|spun|damage|puncture|disqualified|withdrew|did not", re.I)


def _status_flags(status: str) -> tuple[int, int]:
    """(dnf, mechanical_dnf). Lapped/finished are not DNFs."""
    s = (status or "").strip()
    if s == "Finished" or s.startswith("+") or s == "Lapped":
        return 0, 0
    mech = 0 if _DRIVER_FAULT.search(s) else 1
    return 1, mech


class _Ewma:
    """Exponentially decayed mean with half-life measured in observations."""

    def __init__(self, half_life: float, prior: float):
        self.d = 0.5 ** (1.0 / half_life)
        self.prior = prior
        self.num = defaultdict(float)
        self.den = defaultdict(float)

    def get(self, k):
        # one pseudo-observation of the prior keeps tiny samples sane
        return (self.num[k] + self.prior) / (self.den[k] + 1.0)

    def update(self, k, x):
        self.num[k] = self.num[k] * self.d + x
        self.den[k] = self.den[k] * self.d + 1.0

    def discount(self, k, f):
        self.num[k] *= f
        self.den[k] *= f


def load_raw():
    res = pd.read_csv(PROC / "results.csv")
    qua = pd.read_csv(PROC / "qualifying.csv")
    spr = pd.read_csv(PROC / "sprints.csv")
    return res, qua, spr


def _best_time(row) -> float:
    ts = [_secs(x) for x in (row["q1"], row["q2"], row["q3"])]
    ts = [t for t in ts if np.isfinite(t)]
    return min(ts) if ts else np.nan


def build(res: pd.DataFrame, qua: pd.DataFrame, spr: pd.DataFrame, include_future: bool = False) -> pd.DataFrame:
    """One row per (race, driver) with pre-race features and the `won` label.

    include_future=True also emits rows for weekends that have qualifying but no
    result yet (label NaN) so the same code path serves real predictions.
    """
    res = res.copy()
    res["date"] = pd.to_datetime(res["date"])
    res["pos"] = pd.to_numeric(res["position"], errors="coerce")
    res["grid_n"] = pd.to_numeric(res["grid"], errors="coerce")
    flags = res["status"].map(_status_flags)
    res["dnf"] = [f[0] for f in flags]
    res["mech"] = [f[1] for f in flags]

    qua = qua.copy()
    qua["date"] = pd.to_datetime(qua["date"])
    qua["best"] = qua.apply(_best_time, axis=1)
    qua["qpos"] = pd.to_numeric(qua["position"], errors="coerce")
    spr = spr.copy()
    spr["spos"] = pd.to_numeric(spr["position"], errors="coerce")
    qbest_map = qua.set_index(["season", "round", "driver_id"])["best"].to_dict()
    qpos_map = qua.set_index(["season", "round", "driver_id"])["qpos"].to_dict()
    spos_map = spr.set_index(["season", "round", "driver_id"])["spos"].to_dict()

    cols = ["season", "round", "date", "race_name", "circuit_id"]
    done = res[cols].drop_duplicates(["season", "round"])
    if include_future:
        q_races = qua[cols].drop_duplicates(["season", "round"])
        q_races = q_races.merge(done[["season", "round"]], how="left", on=["season", "round"], indicator=True)
        q_races = q_races[q_races["_merge"] == "left_only"].drop(columns="_merge")
        done = pd.concat([done, q_races])
    races = done.sort_values(["date", "round"]).reset_index(drop=True)

    P = PRIOR
    D = dict(
        perf_s=_Ewma(5, P["perf"]), perf_l=_Ewma(20, P["perf"]), win=_Ewma(15, P["win"]),
        podium=_Ewma(10, P["podium"]), qgap=_Ewma(6, P["qgap"]), gain=_Ewma(12, P["gain"]),
        dnf=_Ewma(15, P["dnf"]),
    )
    C = dict(
        perf_xs=_Ewma(3, P["perf"]), qgap_xs=_Ewma(3, P["qgap"]), win_xs=_Ewma(3, P["win"]),
        perf_s=_Ewma(6, P["perf"]), perf_l=_Ewma(30, P["perf"]), win=_Ewma(20, P["win"]),
        podium=_Ewma(15, P["podium"]), qgap=_Ewma(8, P["qgap"]), mech=_Ewma(20, P["mech"]),
    )
    circ_d = defaultdict(lambda: [0.0, 0])
    circ_c = defaultdict(lambda: [0.0, 0])
    n_prior = defaultdict(int)
    last_season, since_reset = None, 0
    out = []

    for _, race in races.iterrows():
        s, rnd = int(race["season"]), int(race["round"])
        if s != last_season:  # new season: discount constructor memory
            f = RESET_SEASONS.get(s, SEASON_CARRYOVER)
            for e in C.values():
                for k in list(e.den):
                    e.discount(k, f)
            if s in RESET_SEASONS:
                since_reset = 0
            last_season = s
        rows = res[(res.season == s) & (res["round"] == rnd)].copy()
        if rows.empty:  # future race: build the entry list from qualifying
            q = qua[(qua.season == s) & (qua["round"] == rnd)]
            rows = q[["season", "round", "race_name", "date", "circuit_id", "driver_id", "driver",
                      "constructor_id", "constructor"]].copy()
            rows["pos"], rows["dnf"], rows["mech"] = np.nan, 0, 0
            rows["grid_n"] = q["qpos"].values  # proxy: qualifying order (penalties unknown)
        n = len(rows)
        circuit = race["circuit_id"]

        # --- weekend (known before lights-out) ---------------------------------
        keys = [(s, rnd, d) for d in rows.driver_id]
        qb = np.array([qbest_map.get(k, np.nan) for k in keys], dtype=float)
        pole = np.nanmin(qb) if np.isfinite(qb).any() else np.nan
        gap = pd.Series(qb / pole - 1.0, index=rows.index)
        fill = gap.quantile(0.95) if gap.notna().any() else 0.05
        rows["qual_gap"] = gap.fillna(fill)
        rows["qual_pos"] = pd.Series([qpos_map.get(k, np.nan) for k in keys], index=rows.index)
        rows["qual_pos"] = rows["qual_pos"].fillna(rows["grid_n"]).fillna(n)
        g = rows["grid_n"].where(rows["grid_n"] > 0, n)  # pit-lane start -> back of grid
        rows["grid"] = g.fillna(rows["qual_pos"])
        rows["grid_rank_sq"] = np.sqrt(rows["grid"])
        rows["front_row"] = (rows["grid"] <= 2).astype(int)
        rows["is_pole"] = (rows["qual_pos"] == 1).astype(int)
        rows["grid_pen"] = (rows["grid"] - rows["qual_pos"]).clip(-10, 10)
        rows["con_qual_best"] = rows.groupby("constructor_id")["qual_gap"].transform("min")
        rows["con_qual_mean"] = rows.groupby("constructor_id")["qual_gap"].transform("mean")
        rows["con_qual_best_rel"] = rows["con_qual_best"] - rows["con_qual_best"].min()
        sp = np.array([spos_map.get(k, np.nan) for k in keys], dtype=float)
        rows["sprint_rank"] = np.where(np.isnan(sp), rows["grid"], sp)

        # --- history-based features (state BEFORE this race) -------------------
        for pre, E, key in (("drv", D, "driver_id"), ("con", C, "constructor_id")):
            for name, e in E.items():
                rows[f"{pre}_{name}"] = [e.get(k) for k in rows[key]]
        rows["drv_circ"] = [(circ_d[(d, circuit)][0] + 0.9) / (circ_d[(d, circuit)][1] + 2) for d in rows.driver_id]
        rows["con_circ"] = [(circ_c[(c, circuit)][0] + 1.35) / (circ_c[(c, circuit)][1] + 3) for c in rows.constructor_id]
        rows["drv_rookie"] = [int(n_prior[d] < 8) for d in rows.driver_id]
        rows["races_since_reset"] = since_reset
        for c in ["drv_perf_s", "con_perf_s", "con_win", "drv_win"]:
            rows[f"{c}_rel"] = rows[c] - rows[c].max()
        for c in ["drv_qgap", "con_qgap"]:
            rows[f"{c}_rel"] = rows[c] - rows[c].min()
        rows["won"] = np.where(rows["pos"].isna(), np.nan, (rows["pos"] == 1).astype(float))
        rows["race_key"] = f"{s}-{rnd:02d}"
        out.append(rows)

        # --- update states with this race's outcome (only if it has happened) --
        if rows["pos"].notna().any():
            fin = rows["pos"].fillna(n).to_numpy()
            perf = (n - fin) / max(n - 1, 1)  # 1.0 winner .. 0.0 last
            perf = np.where(rows["dnf"].to_numpy() == 1, np.minimum(perf, 0.15), perf)
            con_perf = pd.Series(perf).groupby(rows["constructor_id"].to_numpy()).mean()
            for i, r in enumerate(rows.itertuples(index=False)):
                d, c = r.driver_id, r.constructor_id
                D["perf_s"].update(d, perf[i]); D["perf_l"].update(d, perf[i])
                D["win"].update(d, float(fin[i] == 1)); D["podium"].update(d, float(fin[i] <= 3))
                D["qgap"].update(d, r.qual_gap); D["gain"].update(d, (r.grid - fin[i]) / n)
                D["dnf"].update(d, r.dnf)
                C["perf_xs"].update(c, perf[i]); C["win_xs"].update(c, float(fin[i] == 1)); C["qgap_xs"].update(c, r.qual_gap)
                C["perf_s"].update(c, perf[i]); C["perf_l"].update(c, perf[i])
                C["win"].update(c, float(fin[i] == 1)); C["podium"].update(c, float(fin[i] <= 3))
                C["qgap"].update(c, r.qual_gap); C["mech"].update(c, r.mech)
                circ_d[(d, circuit)][0] += perf[i]; circ_d[(d, circuit)][1] += 1
                circ_c[(c, circuit)][0] += con_perf[c]; circ_c[(c, circuit)][1] += 1
                n_prior[d] += 1
            since_reset += 1
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make(include_future: bool = False) -> pd.DataFrame:
    return build(*load_raw(), include_future=include_future)

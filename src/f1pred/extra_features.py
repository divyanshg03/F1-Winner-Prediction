"""Extra feature groups from the domain checklist. All history-based values use strictly earlier races.

A  circuit        : geometry (length, altitude, twistiness, straights, corners) + learned circuit history
                    (overtaking difficulty, pole conversion, safety-car rate, DNF rate, tyre degradation)
B  driver / team  : experience, constructor championship standing, teammate comparison
C  tyre           : circuit degradation x constructor race-vs-qualifying pace gap (a tyre-management proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import lapdata
from .ingest import PROC
from .pace import add_pace_history

A_STATIC = ["length_m", "altitude_m", "mean_curv", "straight_frac", "tight_frac", "corners"]
A_HIST = ["circ_move", "circ_pole", "circ_sc", "circ_dnf", "circ_deg", "grid_x_pole", "grid_x_move", "grid_x_straight"]
B = ["drv_exp", "con_pts_share", "con_rank", "qual_vs_mate", "drv_vs_mate"]
C = ["circ_deg_x_tyre", "con_tyre_proxy"]
GROUPS = {"A": A_STATIC + A_HIST, "B": B, "C": C}


def _expanding(vals: dict, order: list, circ: dict, shrink: float = 2.0) -> dict:
    """Per race: shrunken mean of `vals` over earlier races at the same circuit (NaN-aware)."""
    out, hist, allv = {}, {}, []
    for k in order:
        prior_all = float(np.mean(allv)) if allv else np.nan
        h = hist.get(circ[k], [])
        out[k] = (np.nansum(h) + shrink * prior_all) / (len(h) + shrink) if len(h) and not np.isnan(prior_all) else prior_all
        v = vals.get(k, np.nan)
        if not np.isnan(v):
            hist.setdefault(circ[k], []).append(v)
            allv.append(v)
    return out


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    D, R = lapdata.build_all()
    R = R.set_index("race_key")
    df = add_pace_history(df, D).sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
    order = df.drop_duplicates("race_key")["race_key"].tolist()
    circ = dict(zip(df.race_key, df.circuit_id))

    # ---- race-level realised stats (used only via earlier races) ------------------------
    stat = {"move": {}, "pole": {}, "sc": {}, "dnf": {}, "deg": {}}
    for k, g in df[df.won.notna()].groupby("race_key"):
        fin = g[g.dnf == 0]
        stat["move"][k] = float((fin.grid - fin.pos).abs().mean()) if len(fin) else np.nan
        stat["pole"][k] = float(g.loc[g.grid.idxmin(), "won"]) if g.grid.notna().any() else np.nan
        stat["dnf"][k] = float(g.dnf.mean())
        if k in R.index:
            stat["sc"][k] = float(R.loc[k, "n_sc"] > 0)
            stat["deg"][k] = float(R.loc[k, "deg"]) if pd.notna(R.loc[k, "deg"]) else np.nan
    for name, vals in stat.items():
        e = _expanding(vals, order, circ)
        df["circ_" + name] = df.race_key.map(e)
    for c in ["circ_move", "circ_pole", "circ_sc", "circ_dnf", "circ_deg"]:
        df[c] = df[c].fillna(df[c].mean())
    df["grid_x_pole"] = df.grid_rank_sq * (df.circ_pole - df.circ_pole.mean())
    df["grid_x_move"] = df.grid_rank_sq * (df.circ_move - df.circ_move.mean())

    # ---- static geometry ------------------------------------------------------------------
    geo = pd.read_csv(PROC / "circuits.csv").set_index("circuit_id")
    geo["corners"] = geo["corners"].where(geo["corners"] >= 5)
    for c in A_STATIC:
        df[c] = df.circuit_id.map(geo[c])
        df[c] = df[c].fillna(geo[c].median())
    df["grid_x_straight"] = df.grid_rank_sq * (df.straight_frac - df.straight_frac.mean())

    # ---- B: experience, standings, teammate ------------------------------------------------
    df["drv_exp"] = np.log1p(df.groupby("driver_id").cumcount().where(df.won.notna() | True) * 0 + df.groupby("driver_id")["race_key"].transform(lambda s: pd.Series(range(len(s)), index=s.index)))
    res = pd.read_csv(PROC / "results.csv")
    pts = res.assign(points=pd.to_numeric(res.points, errors="coerce")).groupby(["season", "round", "constructor_id"]).points.sum().reset_index()
    pts = pts.sort_values(["season", "round"])
    pts["cum_before"] = pts.groupby(["season", "constructor_id"]).points.cumsum() - pts.points
    df = df.merge(pts[["season", "round", "constructor_id", "cum_before"]], on=["season", "round", "constructor_id"], how="left")
    df["cum_before"] = df.cum_before.fillna(0.0)
    mx = df.groupby("race_key").cum_before.transform("max")
    df["con_pts_share"] = np.where(mx > 0, df.cum_before / mx.where(mx > 0, 1), 0.5)
    df["con_rank"] = df.groupby("race_key").cum_before.rank(ascending=False, method="min")
    df["qual_vs_mate"] = df.qual_gap - df.groupby(["race_key", "constructor_id"]).qual_gap.transform("mean")
    df["drv_vs_mate"] = df.drv_perf_s - df.con_perf_s

    # ---- C: tyre proxy ---------------------------------------------------------------------
    df["con_tyre_proxy"] = df.con_pace_l - df.con_qgap
    df["circ_deg_x_tyre"] = (df.circ_deg - df.circ_deg.mean()) / (df.circ_deg.std() + 1e-9) * df.con_tyre_proxy
    return df

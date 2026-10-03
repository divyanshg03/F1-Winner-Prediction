"""Turn raw lap data into the quantities the race simulator needs.

Per driver-race : pace_rel (fraction slower than the field median on clean green laps),
                  stop count and pit-lap fractions, retirement lap.
Per race        : base lap time, tyre degradation, pit-stop time loss, safety-car / VSC events.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import PROC

LAPS = PROC / "laps"


def load_laps(key: str) -> pd.DataFrame:
    d = pd.read_csv(LAPS / f"{key}.csv.gz", dtype={"track_status": str})
    d["track_status"] = d["track_status"].fillna("1").str.replace(".0", "", regex=False)
    return d


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start_index, length) of each contiguous run of True."""
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            out.append((i, j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def summarise(key: str) -> tuple[pd.DataFrame, dict]:
    d = load_laps(key)
    total = int(d["total_laps"].iloc[0])
    green = d["track_status"] == "1"
    d["green"] = green
    clean = green & (d.pit_in == 0) & (d.pit_out == 0) & (d.lap > 1) & d.lap_s.notna()
    med_all = d[clean].groupby("lap")["lap_s"].agg(["median", "size"])
    med = med_all[med_all["size"] >= 5]["median"]
    d["med"] = d["lap"].map(med)
    clean &= d["med"].notna() & (d["lap_s"] < 1.07 * d["med"])
    d["delta"] = d["lap_s"] / d["med"] - 1.0
    c = d[clean].copy()

    # --- per driver pace ----------------------------------------------------
    g = c.groupby("driver_id")["delta"]
    pace = pd.DataFrame({"pace_rel": g.apply(lambda s: s.clip(upper=s.quantile(0.9)).mean()), "n_clean": g.size()})
    pace.loc[pace.n_clean < 8, "pace_rel"] = np.nan

    # --- tyre degradation (within driver-stint, pooled) ---------------------
    c = c[c.tyre_life.notna()].copy()
    c["_k"] = c["driver_id"] + "_" + c["stint"].astype(str)
    x = c["tyre_life"] - c.groupby("_k")["tyre_life"].transform("mean")
    y = c["delta"] - c.groupby("_k")["delta"].transform("mean")
    deg = float((x * y).sum() / max((x * x).sum(), 1e-9)) if len(c) > 50 else np.nan

    # --- stops --------------------------------------------------------------
    stops = d[d.pit_in == 1].groupby("driver_id")["lap"].apply(lambda s: [round(v / total, 4) for v in s])
    pace["n_stops"] = stops.reindex(pace.index).map(lambda v: len(v) if isinstance(v, list) else 0)
    pace["pit_fracs"] = stops.reindex(pace.index).map(lambda v: json.dumps(v) if isinstance(v, list) else "[]")
    last_lap = d.groupby("driver_id")["lap"].max()
    pace["last_lap"] = last_lap.reindex(pace.index)

    # pit loss: in-lap + out-lap excess over field median, green stops only
    losses = []
    for drv, grp in d.groupby("driver_id"):
        grp = grp.set_index("lap")
        for lap in grp.index[grp.pit_in == 1]:
            nxt = lap + 1
            if nxt in grp.index and grp.loc[lap, "green"] and grp.loc[nxt, "green"] and pd.notna(grp.loc[nxt, "lap_s"]):
                a, b = grp.loc[lap], grp.loc[nxt]
                if pd.notna(a.lap_s) and pd.notna(a.med) and pd.notna(b.med):
                    losses.append((a.lap_s - a.med) + (b.lap_s - b.med))
    pit_loss = float(np.median(losses)) if len(losses) >= 5 else np.nan

    # --- safety car / VSC ---------------------------------------------------
    per_lap = d.groupby("lap")["track_status"]
    sc = per_lap.apply(lambda s: np.mean(s.str.contains("4"))).reindex(range(1, total + 1)).fillna(0).to_numpy() > 0.5
    vsc = per_lap.apply(lambda s: np.mean(s.str.contains("6|7"))).reindex(range(1, total + 1)).fillna(0).to_numpy() > 0.5
    red = per_lap.apply(lambda s: np.mean(s.str.contains("5"))).reindex(range(1, total + 1)).fillna(0).to_numpy() > 0.5
    sc_runs, vsc_runs = _runs(sc), _runs(vsc)
    race = dict(
        race_key=key, total_laps=total,
        base_lap_s=float(med.median()) if len(med) else np.nan,
        deg=deg, pit_loss=pit_loss,
        n_sc=len(sc_runs), sc_laps=int(sc.sum()), sc_starts=json.dumps([round((s + 1) / total, 3) for s, _ in sc_runs]),
        sc_lens=json.dumps([l for _, l in sc_runs]),
        n_vsc=len(vsc_runs), red=int(red.any()),
    )
    pace = pace.reset_index()
    pace.insert(0, "race_key", key)
    return pace, race


def build_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_d, out_r = PROC / "lap_driver.csv", PROC / "lap_race.csv"
    if out_d.exists() and out_r.exists() and not force:
        return pd.read_csv(out_d), pd.read_csv(out_r)
    dr, rr = [], []
    for f in sorted(LAPS.glob("20??-??.csv.gz")):
        p, r = summarise(f.name.replace(".csv.gz", ""))
        dr.append(p)
        rr.append(r)
    D, R = pd.concat(dr, ignore_index=True), pd.DataFrame(rr)
    D.to_csv(out_d, index=False)
    R.to_csv(out_r, index=False)
    return D, R


if __name__ == "__main__":
    D, R = build_all(force=True)
    print(len(R), "races summarised")
    print(R.describe().round(3).T[["mean", "min", "max"]])

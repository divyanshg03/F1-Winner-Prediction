"""Circuit geometry table: length, altitude, shape (curvature, straightness), corner count.

Sources: bacinger/f1-circuits GeoJSON (layout, length, altitude), Jolpica (lat/long for matching),
FastF1 circuit_info (corner count; manually annotated upstream, so approximate).
Output: data/processed/circuits.csv keyed by Jolpica circuit_id.
"""
from __future__ import annotations

import json
import warnings

import fastf1
import numpy as np
import pandas as pd
import requests

from .ingest import PROC, RAW

warnings.filterwarnings("ignore")


def _hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def shape(coords) -> dict:
    xy = np.array([c[:2] for c in coords], dtype=float)
    lat0 = xy[:, 1].mean()
    x = (xy[:, 0] - xy[0, 0]) * 111.32 * np.cos(lat0 * np.pi / 180) * 1000
    y = (xy[:, 1] - xy[0, 1]) * 110.57 * 1000
    d = np.hypot(np.diff(x), np.diff(y))
    keep = d > 1.0
    heading = np.unwrap(np.arctan2(np.diff(y)[keep], np.diff(x)[keep]))
    seg = d[keep]
    turn = np.abs(np.diff(heading))
    ds = (seg[1:] + seg[:-1]) / 2
    curv = turn / np.maximum(ds, 1e-6)  # rad per metre
    L = seg.sum()
    return dict(
        geo_length_m=float(L),
        mean_curv=float(turn.sum() / L * 1000),  # rad/km: how twisty
        straight_frac=float(ds[curv < 0.002].sum() / ds.sum()),  # share of the lap that is near-straight
        tight_frac=float(ds[curv > 0.02].sum() / ds.sum()),  # share in slow, tight turns
    )


def build() -> pd.DataFrame:
    out = PROC / "circuits.csv"
    gj = json.load(open(RAW / "geo" / "circuits_geo.geojson"))["features"]
    jc = requests.get("https://api.jolpi.ca/ergast/f1/circuits.json?limit=200", timeout=30).json()["MRData"]["CircuitTable"]["Circuits"]
    used = set(pd.read_csv(PROC / "results.csv").circuit_id)
    rows = []
    for c in jc:
        if c["circuitId"] not in used:
            continue
        lat, lon = float(c["Location"]["lat"]), float(c["Location"]["long"])
        best, bd = None, 1e9
        for f in gj:
            co = f["geometry"]["coordinates"]
            co = co[0] if isinstance(co[0][0], list) else co
            cl = np.mean([p[1] for p in co]); cn = np.mean([p[0] for p in co])
            dist = _hav(lat, lon, cl, cn)
            if dist < bd:
                best, bd = f, dist
        if bd > 8:  # not the same place: no geometry rather than a wrong one
            print("no geometry match:", c["circuitId"], round(bd, 1), "km")
            rows.append(dict(circuit_id=c["circuitId"], matched=0))
            continue
        co = best["geometry"]["coordinates"]; co = co[0] if isinstance(co[0][0], list) else co
        p = best["properties"]
        rows.append(dict(circuit_id=c["circuitId"], matched=1, geo_id=p["id"], length_m=p.get("length"), altitude_m=p.get("altitude"), **shape(co)))
    df = pd.DataFrame(rows)

    # corner counts from FastF1 (latest race at each circuit since 2018, reusing the lap-data caches)
    res = pd.read_csv(PROC / "results.csv").drop_duplicates(["season", "round"]).sort_values("season")
    res = res[res.season >= 2018]
    last = res.groupby("circuit_id").tail(1).set_index("circuit_id")
    corners = {}
    for cid, r in last.iterrows():
        yr = int(r.season)
        rng = "2018_2020" if yr <= 2020 else "2021_2023" if yr <= 2023 else "2024_2026"
        try:
            fastf1.Cache.enable_cache(str(RAW / f"fastf1_{rng}"))
            s = fastf1.get_session(yr, int(r["round"]), "R")
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            corners[cid] = len(s.get_circuit_info().corners)
        except Exception as e:  # noqa: BLE001
            print("corners failed", cid, repr(e)[:80])
    df["corners"] = df.circuit_id.map(corners)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.drop(columns="geo_id").round(3).to_string(index=False))

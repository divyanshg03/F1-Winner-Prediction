"""Practice-session pace and race-day weather from FastF1.

Usage: python -m f1pred.ingest_sessions 2018 2026 [shard n_shards]
Writes data/processed/sessions/<season>-<round>.json with
  weather: mean air/track temp, humidity, wind, rain flag (race session)
  practice: per-driver {best, longrun} from FP2 (FP1 if no FP2, e.g. sprint weekends), seconds
"""
from __future__ import annotations

import json
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd

from .ingest import PROC, RAW

warnings.filterwarnings("ignore")
OUT = PROC / "sessions"


def practice(season: int, rnd: int) -> dict:
    for name in ("FP2", "FP1"):
        try:
            s = fastf1.get_session(season, rnd, name)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception:  # noqa: BLE001
            continue
        L = s.laps
        ids = s.results.set_index("Abbreviation")["DriverId"].to_dict()
        L = L[L["LapTime"].notna() & (L["PitOutTime"].isna()) & (L["PitInTime"].isna())].copy()
        L["t"] = L["LapTime"].dt.total_seconds()
        if L.empty:
            continue
        best = L["t"].min()
        out = {}
        for drv, g in L.groupby("Driver"):
            lr = g[(g["t"] < 1.10 * best) & (g["TyreLife"].fillna(0) >= 4)]["t"]
            out[ids.get(drv, drv)] = dict(best=float(g["t"].min() / best - 1), longrun=float(lr.median() / best - 1) if len(lr) >= 3 else None)
        return dict(session=name, drivers=out)
    return {}


def weather(season: int, rnd: int) -> dict:
    s = fastf1.get_session(season, rnd, "R")
    s.load(laps=False, telemetry=False, weather=True, messages=False)
    w = s.weather_data
    return dict(air=float(w.AirTemp.mean()), track=float(w.TrackTemp.mean()), humidity=float(w.Humidity.mean()),
                wind=float(w.WindSpeed.mean()), rain=int(bool(w.Rainfall.any())))


def main(first, last, shard=0, n_shards=1):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"fastf1_sess_s{shard}"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    races = pd.read_csv(PROC / "results.csv")[["season", "round"]].drop_duplicates()
    races = races[(races.season >= first) & (races.season <= last)].sort_values(["season", "round"])
    todo = [r for r in races.itertuples() if not (OUT / f"{r.season}-{r.round:02d}.json").exists()][shard::n_shards]
    for r in todo:
        rec = {}
        for k, fn in (("weather", weather), ("practice", practice)):
            try:
                rec[k] = fn(r.season, r.round)
            except Exception as e:  # noqa: BLE001
                rec[k] = {}
                print("FAIL", k, r.season, r.round, repr(e)[:100], flush=True)
        (OUT / f"{r.season}-{r.round:02d}.json").write_text(json.dumps(rec))
        print("ok", r.season, r.round, flush=True)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])

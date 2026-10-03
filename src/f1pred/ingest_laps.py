"""Lap-level race data from FastF1 (lap times, tyres, pit stops, track status).

Usage:  python -m f1pred.ingest_laps 2018 2020      # one process per year range
Writes data/processed/laps/<season>-<round>.csv.gz, one file per race. Resumable:
existing files are skipped, failures are logged to data/processed/laps/_failed.txt.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import fastf1
import pandas as pd

from .ingest import PROC, RAW

warnings.filterwarnings("ignore")
OUT = PROC / "laps"
KEEP = ["driver_id", "lap", "lap_s", "position", "compound", "tyre_life", "stint", "pit_in", "pit_out", "track_status", "accurate"]


def race_frame(season: int, rnd: int) -> pd.DataFrame:
    s = fastf1.get_session(season, rnd, "R")
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    ids = s.results.set_index("Abbreviation")["DriverId"].to_dict()
    L = s.laps
    df = pd.DataFrame({
        "driver_id": L["Driver"].map(ids),
        "lap": L["LapNumber"],
        "lap_s": L["LapTime"].dt.total_seconds(),
        "position": L["Position"],
        "compound": L["Compound"],
        "tyre_life": L["TyreLife"],
        "stint": L["Stint"],
        "pit_in": L["PitInTime"].notna().astype(int),
        "pit_out": L["PitOutTime"].notna().astype(int),
        "track_status": L["TrackStatus"].astype(str),
        "accurate": L["IsAccurate"].astype(int),
    })
    df["total_laps"] = int(L["LapNumber"].max())
    return df.dropna(subset=["driver_id", "lap"])


def main(first: int, last: int, shard: int = 0, n_shards: int = 1) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"fastf1_{first}_{last}" if n_shards == 1 else RAW / f"fastf1_{first}_{last}_s{shard}"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    races = pd.read_csv(PROC / "results.csv")[["season", "round"]].drop_duplicates()
    races = races[(races.season >= first) & (races.season <= last)].sort_values(["season", "round"])
    todo = [r for r in races.itertuples() if not (OUT / f"{r.season}-{r.round:02d}.csv.gz").exists()]
    if n_shards > 1:  # extra workers take the *remaining* races from the far end
        todo = todo[::-1][shard::n_shards]
    for r in todo:
        f = OUT / f"{r.season}-{r.round:02d}.csv.gz"
        if f.exists():
            continue
        try:
            race_frame(r.season, r.round).to_csv(f, index=False)
            print("ok", r.season, r.round, flush=True)
        except Exception as e:  # noqa: BLE001 - keep going, log for retry
            print("FAIL", r.season, r.round, repr(e)[:160], flush=True)
            with open(OUT / "_failed.txt", "a") as fh:
                fh.write(f"{r.season}-{r.round}\t{repr(e)[:160]}\n")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])

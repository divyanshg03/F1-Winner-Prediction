"""Download race, qualifying and sprint results from the Jolpica (Ergast-compatible) API.

Raw JSON pages are cached on disk so re-runs are free and polite to the API.
Output: data/processed/{results,qualifying,sprints}.csv  (one row per driver per session)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PAGE = 100


def _get(url: str, cache: Path, refresh: bool = False) -> dict:
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf8"))
    for attempt in range(6):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(r.text, encoding="utf8")
        time.sleep(0.35)
        return r.json()
    raise RuntimeError(f"rate limited: {url}")


def fetch_all(season: int, endpoint: str, refresh: bool = False) -> list[dict]:
    """Return the list of Race dicts for a season/endpoint, paging through the API."""
    races: dict[str, dict] = {}
    offset, total = 0, None
    while total is None or offset < total:
        url = f"{BASE}/{season}/{endpoint}.json?limit={PAGE}&offset={offset}"
        d = _get(url, RAW / endpoint / f"{season}_{offset}.json", refresh)["MRData"]
        total = int(d["total"])
        for race in d["RaceTable"]["Races"]:
            key = race["round"]
            if key not in races:
                races[key] = {k: v for k, v in race.items() if not k.endswith("Results")}
                races[key]["rows"] = []
            for k, v in race.items():
                if k.endswith("Results"):
                    races[key]["rows"].extend(v)
        offset += PAGE
    return list(races.values())


def _flat(race: dict, row: dict, session: str) -> dict:
    drv, con = row["Driver"], row["Constructor"]
    out = {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_name": race["raceName"],
        "date": race["date"],
        "circuit_id": race["Circuit"]["circuitId"],
        "session": session,
        "driver_id": drv["driverId"],
        "driver": f'{drv["givenName"]} {drv["familyName"]}',
        "constructor_id": con["constructorId"],
        "constructor": con["name"],
        "position": row.get("position"),
        "position_text": row.get("positionText"),
    }
    if session in ("race", "sprint"):
        out.update(
            grid=row.get("grid"),
            points=row.get("points"),
            laps=row.get("laps"),
            status=row.get("status"),
        )
    else:  # qualifying
        out.update(q1=row.get("Q1"), q2=row.get("Q2"), q3=row.get("Q3"))
    return out


def build(first: int = 2014, last: int = 2026, refresh_current: bool = True) -> dict[str, pd.DataFrame]:
    PROC.mkdir(parents=True, exist_ok=True)
    specs = {"results": ("race", "results"), "qualifying": ("qualifying", "qualifying"), "sprints": ("sprint", "sprint")}
    frames = {}
    for name, (session, endpoint) in specs.items():
        rows = []
        for season in range(first, last + 1):
            # the in-progress season changes every race weekend, so always re-pull it
            refresh = refresh_current and season == last
            for race in fetch_all(season, endpoint, refresh=refresh):
                rows += [_flat(race, r, session) for r in race["rows"]]
            print(f"{name} {season}: {len(rows)} cumulative rows", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(PROC / f"{name}.csv", index=False)
        frames[name] = df
    return frames


if __name__ == "__main__":
    import sys

    a = [int(x) for x in sys.argv[1:3]] or [2014, 2026]
    build(*a)

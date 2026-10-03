"""Score a frozen prediction against the real result and append it to the same JSON.

Usage:  python scripts/score_prediction.py 2026-16
The original probabilities are never modified; a `result` block is added.
"""
import json
import math
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
key = sys.argv[1]
season, rnd = key.split("-")
path = ROOT / "predictions" / f"{key}.json"
rec = json.loads(path.read_text(encoding="utf8"))

races = requests.get(f"https://api.jolpi.ca/ergast/f1/{int(season)}/{int(rnd)}/results.json", timeout=30).json()["MRData"]["RaceTable"]["Races"]
if not races:
    sys.exit("No result published yet.")
winner = next(r for r in races[0]["Results"] if r["position"] == "1")["Driver"]
name = f'{winner["givenName"]} {winner["familyName"]}'
probs = {p["driver"]: p["p_win"] for p in rec["probabilities"]}
p = probs.get(name)
ranked = [p_["driver"] for p_ in sorted(rec["probabilities"], key=lambda x: -x["p_win"])]
rec["result"] = dict(
    winner=name,
    p_assigned=p,
    model_rank_of_winner=ranked.index(name) + 1 if name in ranked else None,
    log_loss=round(-math.log(max(p or 1e-6, 1e-6)), 3),
    top_pick=ranked[0],
    top_pick_won=ranked[0] == name,
)
path.write_text(json.dumps(rec, indent=2), encoding="utf8")
print(json.dumps(rec["result"], indent=2))
